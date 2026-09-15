#!/usr/bin/env python
"""
Causal abstraction experiment for the verse-long task.

Hypothesis under test: by the end of the prompt the model holds a summary of the question
(TOPIC) separate from the requested format (FORMAT), and generation reads that summary rather
than re-deriving the topic from the question tokens.

Everything is measured teacher-forced: under a (possibly patched) prompt, score the four
reference answers of the pair and read off two contrasts,

    topic  = mean logp(q2_*) - mean logp(q1_*)      > 0 means the run favours Q2
    format = mean logp(*_verse) - mean logp(*_prose) > 0 means the run favours verse

--experiments selects the arms:

  patch       Patch the base run's residual stream from a source run at (layer, position set).
              Three conditions:
                topic    base (Q1, verse)  <- source (Q2, verse)
                format   base (Q1, verse)  <- source (Q1, prose)   [needs format_aligned pairs]
                compete  base (Q2, verse)  <- source (Q1, verse)   [topic run in reverse]
              Position sets: fmt, question, suffix, last, prompt, and 'concisely' as a control.
              'last' is the competition test: the Q1 question tokens are still in context and
              still attended to by every scored continuation token, so the summary only wins if
              it is genuinely a bottleneck.  'question' vs 'last' separates re-reading from a
              pre-generation summary; if only 'question' moves the topic, the bottleneck claim
              is dead.

  transplant  Sufficiency without a competing question.  Take the base run's residual at
              last/suffix and transplant it into "Respond in verse and concisely." -- the same
              template with no question at all -- then score the same four references.  Donors
              Q1 and Q2 are both run, so the contrast between them is the effect; the
              untransplanted blank prompt is the baseline.

  generate    Optional confirmation.  Patch the prefill only, then decode greedily with no
              further intervention, and dump the text for judging.  If one prompt-time patch
              carries a whole poem, topic is committed at the prompt boundary.

Records are appended (fsync'd) per example and skipped on restart, so the job is requeue-safe.
--plot_only rebuilds figures and summary.json from the records without loading a model.

Tested against nnsight 0.4.11 / transformers 4.53.3 (tuple decoder-layer outputs).
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import verse_long_utils as U  # noqa: E402

CONDITIONS = {
    "topic":   dict(base=("q1", "verse"), source=("q2", "verse"), moves="topic", holds="format"),
    "format":  dict(base=("q1", "verse"), source=("q1", "prose"), moves="format", holds="topic",
                    needs_format_aligned=True),
    "compete": dict(base=("q2", "verse"), source=("q1", "verse"), moves="topic", holds="format"),
}
DEFAULT_POSITIONS = ["fmt", "question", "suffix", "last", "prompt", "concisely"]


def _val(p):
    return getattr(p, "value", p)


# ============================================================================ model wrapper
class Patcher:
    def __init__(self, model_name, dtype, layer_batch, max_ref_tokens):
        import torch
        from nnsight import LanguageModel

        self.torch = torch
        self.model = LanguageModel(model_name, device_map="auto",
                                   torch_dtype=getattr(torch, dtype), dispatch=True)
        self.tok = self.model.tokenizer
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if not self.tok.is_fast:
            raise RuntimeError("need a fast tokenizer for offset mapping")
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.layer_batch = layer_batch
        self.max_ref_tokens = max_ref_tokens
        self.tuple_out = self._detect_tuple_output()
        print(f"[patcher] {model_name}: {self.n_layers} layers, decoder-layer output is "
              f"{'tuple' if self.tuple_out else 'tensor'}", flush=True)

    def _detect_tuple_output(self):
        x = self.torch.tensor([[self.tok.eos_token_id]])
        with self.torch.no_grad(), self.model.trace(x):
            out = self.layers[0].output.save()
        return isinstance(_val(out), tuple)

    def _resid(self, layer):
        out = self.layers[layer].output
        return out[0] if self.tuple_out else out

    def anatomy(self, fmt, question=None):
        return U.Anatomy(self.tok, fmt, question)

    def batch(self, anatomy, refs):
        return U.ReferenceBatch(self.tok, anatomy.ids, refs, self.max_ref_tokens, self.model.device)

    def _inputs(self, ids, mask):
        return {"input_ids": ids, "attention_mask": mask}

    def cache_acts(self, anatomy, positions):
        """Residual at `positions` for every layer, from an unpatched run of this prompt."""
        torch = self.torch
        ids = torch.tensor([anatomy.ids], device=self.model.device)
        mask = torch.ones_like(ids)
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            acts = [self._resid(l)[0, positions, :].save() for l in range(self.n_layers)]
        return [_val(a) for a in acts]

    def _write(self, layer, rows, positions, acts):
        """resid[rows, positions, :] = acts, with the value materialized at exactly the target
        shape.

        Written as two equal-length advanced index tensors rather than
        `resid[row_slice, positions, :] = acts`. The slice-plus-list form relies on the value
        broadcasting up from [P, d] to [R, P, d], and under nnsight that expansion is done
        against the wrong shape: it produced a value 5120x (one d_model) too large and tripped
        an internal assert in index_put_. Adjacent index tensors of equal length with a trailing
        slice give a target of exactly [R*P, d], so nothing broadcasts.
        """
        torch = self.torch
        R, P = len(rows), len(positions)
        if tuple(acts.shape) != (P, acts.shape[-1]):
            raise ValueError(f"patch value for layer {layer} has shape {tuple(acts.shape)}, "
                             f"expected ({P}, d_model)")
        dev = acts.device
        row_idx = torch.tensor(rows, device=dev, dtype=torch.long).repeat_interleave(P)
        col_idx = torch.tensor(positions, device=dev, dtype=torch.long).repeat(R)
        self._resid(layer)[row_idx, col_idx, :] = acts.repeat(R, 1)

    def score(self, batch, patches=None):
        """Unpatched (patches=None) or single-layer-per-block scoring.

        patches: list of (layer, base_positions, acts) applied to every row of the batch.
        """
        torch = self.torch
        ids, mask, tgt, smask = batch.tile(1)
        rows = list(range(ids.shape[0]))
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            for layer, pos, acts in (patches or []):
                self._write(layer, rows, pos, acts)
            g = torch.log_softmax(self.model.lm_head.output.float(), -1) \
                     .gather(-1, tgt.unsqueeze(-1)).squeeze(-1).save()
        return batch.reduce(_val(g), smask, n_blocks=1)[0]

    def layer_sweep(self, batch, base_positions, acts_by_layer):
        """One batched forward per chunk of layers; block b patches layer chunk[b]."""
        torch = self.torch
        R = len(batch.keys)
        out = [None] * self.n_layers
        for s in range(0, self.n_layers, self.layer_batch):
            chunk = list(range(s, min(s + self.layer_batch, self.n_layers)))
            ids, mask, tgt, smask = batch.tile(len(chunk))
            with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
                for b, layer in enumerate(chunk):
                    self._write(layer, list(range(b * R, (b + 1) * R)),
                                base_positions, acts_by_layer[layer])
                g = torch.log_softmax(self.model.lm_head.output.float(), -1) \
                         .gather(-1, tgt.unsqueeze(-1)).squeeze(-1).save()
            for b, layer in enumerate(zip(chunk, batch.reduce(_val(g), smask, len(chunk)))):
                out[layer[0]] = layer[1]
        return out

    def generate(self, anatomy, patches, max_new_tokens):
        """Patch the prefill only, then greedy-decode from the resulting KV cache."""
        torch = self.torch
        ids = torch.tensor([anatomy.ids], device=self.model.device)
        mask = torch.ones_like(ids)
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            for layer, pos, acts in (patches or []):
                self._write(layer, [0], pos, acts)
            out = self.model.output.save()
        out = _val(out)
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        hf = self.model._model
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        toks = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                t = int(nxt)
                if t in eos:
                    break
                toks.append(t)
                o = hf(input_ids=nxt, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        return self.tok.decode(toks)


def debug_shapes(P, pairs, rows):
    """One pair, one layer: print every shape involved in a patch and perform the write.

    Costs a model load and two forwards. Run this first after any environment change -- it
    fails in a minute instead of after a queue wait."""
    pair = pairs[0]
    refs = refs_for(pair, rows)
    base = P.anatomy("verse", rows[pair["q1"]]["question"])
    source = P.anatomy("verse", rows[pair["q2"]]["question"])
    batch = P.batch(base, refs)
    ids, mask, tgt, smask = batch.tile(2)
    print(f"pair {pair['pair_id']}  base_tokens={len(base.ids)}  source_tokens={len(source.ids)}")
    print(f"references={batch.keys}  ref_lens={batch.ref_lens}")
    print(f"tiled batch: input_ids={tuple(ids.shape)} target={tuple(tgt.shape)} "
          f"(2 layer blocks x {len(batch.keys)} references)")
    union = sorted({i for ps in P_SETS(base, source) for i in source.positions[ps]})
    acts = P.cache_acts(source, union)
    print(f"cache_acts: {P.n_layers} layers, each {tuple(acts[0].shape)} "
          f"(expected ({len(union)}, d_model))")
    for ps in P_SETS(base, source):
        bp, sp = base.positions[ps], source.positions[ps]
        idx = P.torch.tensor([union.index(i) for i in sp])
        a = acts[0][idx.to(acts[0].device)]
        print(f"  {ps:<10} base={len(bp)} source={len(sp)} acts={tuple(a.shape)} "
              f"align={base.check_alignment(source, ps) or 'ok'}")
    ps = "last"
    a = acts[0][P.torch.tensor([union.index(i) for i in source.positions[ps]]).to(acts[0].device)]
    with P.torch.no_grad(), P.model.trace(P._inputs(ids, mask)):
        h = P._resid(0).shape.save()
        P._write(0, [0, 1], base.positions[ps], a)
        out = P.model.lm_head.output.shape.save()
    print(f"resid shape inside trace: {tuple(_val(h))}   lm_head out: {tuple(_val(out))}")
    print("write to layer 0 succeeded")


def P_SETS(base, source):
    return [ps for ps in DEFAULT_POSITIONS if ps in base.positions and ps in source.positions]


# ============================================================================ arms
def refs_for(pair, rows):
    i, j = pair["q1"], pair["q2"]
    return {"q1_verse": rows[i]["verse_answer"], "q1_prose": rows[i]["prose_answer"],
            "q2_verse": rows[j]["verse_answer"], "q2_prose": rows[j]["prose_answer"]}


def pack(scores):
    return dict({k: round(v["logp_mean"], 5) for k, v in scores.items()},
                **{k: round(v, 5) for k, v in U.contrasts(scores).items()})


def done_keys(path):
    """pair_id -> set of already-written keys. Resume is per key, not per pair: a condition can
    be legitimately absent for some pairs (e.g. 'format' needs format_aligned), so requiring a
    full key set per pair would rerun those pairs forever."""
    done = {}
    if path.exists():
        for r in U.load_jsonl(path):
            done.setdefault(r["pair_id"], set()).add(r["key"])
    return done


def emit(fout, record):
    fout.write(json.dumps(record) + "\n")
    fout.flush()
    os.fsync(fout.fileno())


def run_patch(P, pairs, rows, position_sets, out_dir):
    path = out_dir / "records_patch.jsonl"
    conds = [c for c in CONDITIONS if c in position_sets]
    done = done_keys(path)

    def applicable(pair):
        """Keys this pair can actually produce; a condition needing format_aligned produces
        none on a pair whose verse and prose prompts differ in length."""
        return {f"{c}|{ps}" for c in conds for ps in position_sets[c]
                if not (CONDITIONS[c].get("needs_format_aligned")
                        and not pair.get("format_aligned", False))}

    wanted = lambda p: applicable(p) - done.get(p["pair_id"], set())
    todo = [p for p in pairs if wanted(p)]
    print(f"[patch] {len(pairs)} pairs, {len(todo)} with work left, conditions={conds}", flush=True)

    with open(path, "a") as fout:
        for n, pair in enumerate(todo):
            have = done.get(pair["pair_id"], set())
            refs = refs_for(pair, rows)
            anat = {("q1", "verse"): P.anatomy("verse", rows[pair["q1"]]["question"]),
                    ("q2", "verse"): P.anatomy("verse", rows[pair["q2"]]["question"]),
                    ("q1", "prose"): P.anatomy("prose", rows[pair["q1"]]["question"])}
            for cond in conds:
                spec = CONDITIONS[cond]
                if spec.get("needs_format_aligned") and not pair.get("format_aligned", False):
                    continue
                base, source = anat[spec["base"]], anat[spec["source"]]
                sets = [ps for ps in position_sets[cond]
                        if ps in base.positions and ps in source.positions
                        and f"{cond}|{ps}" not in have]
                if not sets:
                    continue
                bad = {ps: base.check_alignment(source, ps) for ps in sets}
                sets = [ps for ps in sets if not bad[ps]]
                if not sets:
                    print(f"  pair {pair['pair_id']} {cond}: no aligned position sets ({bad})")
                    continue
                base_batch = P.batch(base, refs)
                base_scores = P.score(base_batch)
                source_scores = P.score(P.batch(source, refs))
                union = sorted({i for ps in sets for i in source.positions[ps]})
                where = {i: k for k, i in enumerate(union)}
                acts = P.cache_acts(source, union)
                for ps in sets:
                    idx = P.torch.tensor([where[i] for i in source.positions[ps]])
                    sliced = [a[idx.to(a.device)] for a in acts]
                    curve = P.layer_sweep(base_batch, base.positions[ps], sliced)
                    emit(fout, dict(pair_id=pair["pair_id"], key=f"{cond}|{ps}", experiment="patch",
                                    condition=cond, position_set=ps, q1=pair["q1"], q2=pair["q2"],
                                    moves=spec["moves"], holds=spec["holds"],
                                    n_positions=len(base.positions[ps]),
                                    baseline_base=pack(base_scores),
                                    baseline_source=pack(source_scores),
                                    layers=[pack(s) for s in curve]))
                print(f"[patch] {n + 1}/{len(todo)} pair {pair['pair_id']} {cond}: "
                      f"base topic={U.contrasts(base_scores)['topic']:+.3f} "
                      f"source topic={U.contrasts(source_scores)['topic']:+.3f} sets={sets}",
                      flush=True)


def run_transplant(P, pairs, rows, donor_positions, out_dir):
    path = out_dir / "records_transplant.jsonl"
    done = done_keys(path)
    wanted = lambda p: {f"{d}|{ps}" for d in ("q1", "q2") for ps in donor_positions} - done.get(p["pair_id"], set())
    todo = [p for p in pairs if wanted(p)]
    print(f"[transplant] {len(pairs)} pairs, {len(todo)} with work left", flush=True)

    blank = P.anatomy("verse")
    with open(path, "a") as fout:
        for n, pair in enumerate(todo):
            have = done.get(pair["pair_id"], set())
            refs = refs_for(pair, rows)
            blank_batch = P.batch(blank, refs)
            blank_scores = P.score(blank_batch)
            for donor in ("q1", "q2"):
                anatomy = P.anatomy("verse", rows[pair[donor]]["question"])
                sets = [ps for ps in donor_positions
                        if len(anatomy.positions[ps]) == len(blank.positions[ps])
                        and f"{donor}|{ps}" not in have]
                if not sets:
                    continue
                if len(sets) < len(donor_positions):
                    print(f"  pair {pair['pair_id']} donor {donor}: dropped "
                          f"{set(donor_positions) - set(sets)} (span length differs from blank)")
                union = sorted({i for ps in sets for i in anatomy.positions[ps]})
                where = {i: k for k, i in enumerate(union)}
                acts = P.cache_acts(anatomy, union)
                for ps in sets:
                    idx = P.torch.tensor([where[i] for i in anatomy.positions[ps]])
                    sliced = [a[idx.to(a.device)] for a in acts]
                    curve = P.layer_sweep(blank_batch, blank.positions[ps], sliced)
                    emit(fout, dict(pair_id=pair["pair_id"], key=f"{donor}|{ps}",
                                    experiment="transplant", donor=donor, position_set=ps,
                                    q1=pair["q1"], q2=pair["q2"],
                                    baseline_blank=pack(blank_scores),
                                    layers=[pack(s) for s in curve]))
            print(f"[transplant] {n + 1}/{len(todo)} pair {pair['pair_id']} "
                  f"blank topic={U.contrasts(blank_scores)['topic']:+.3f}", flush=True)


def run_generate(P, pairs, rows, gen_layers, gen_positions, max_new_tokens, out_dir):
    path = out_dir / "generations.jsonl"
    done = done_keys(path)
    wanted = lambda p: ({f"{ps}|{l}" for ps in gen_positions for l in gen_layers} | {"unpatched"}) \
        - done.get(p["pair_id"], set())
    todo = [p for p in pairs if wanted(p)]
    print(f"[generate] {len(todo)} pairs x {len(gen_positions)} x {len(gen_layers)} layers", flush=True)

    with open(path, "a") as fout:
        for n, pair in enumerate(todo):
            have = done.get(pair["pair_id"], set())
            base = P.anatomy("verse", rows[pair["q1"]]["question"])
            source = P.anatomy("verse", rows[pair["q2"]]["question"])
            common = dict(pair_id=pair["pair_id"], q1=pair["q1"], q2=pair["q2"],
                          q1_question=rows[pair["q1"]]["question"],
                          q2_question=rows[pair["q2"]]["question"])
            if "unpatched" not in have:
                emit(fout, dict(common, key="unpatched", condition="topic", position_set=None,
                                layer=None, text=P.generate(base, None, max_new_tokens)))
            sets = [ps for ps in gen_positions if not base.check_alignment(source, ps)
                    and any(f"{ps}|{l}" not in have for l in gen_layers)]
            union = sorted({i for ps in sets for i in source.positions[ps]})
            where = {i: k for k, i in enumerate(union)}
            acts = P.cache_acts(source, union)
            for ps in sets:
                idx = P.torch.tensor([where[i] for i in source.positions[ps]])
                for layer in gen_layers:
                    if f"{ps}|{layer}" in have:
                        continue
                    a = acts[layer][idx.to(acts[layer].device)]
                    text = P.generate(base, [(layer, base.positions[ps], a)], max_new_tokens)
                    emit(fout, dict(common, key=f"{ps}|{layer}", condition="topic",
                                    position_set=ps, layer=layer, text=text))
            print(f"[generate] {n + 1}/{len(todo)} pair {pair['pair_id']} done", flush=True)


# ============================================================================ aggregation
def aggregate(out_dir, min_denominator=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = {}

    def latest(path):
        """Last write per (pair_id, key) wins. Reruns with different flags append rather than
        overwrite, so without this a pair can be counted twice and the CIs shrink for free."""
        seen = {}
        for r in U.load_jsonl(path):
            seen[(r["pair_id"], r["key"])] = r
        return list(seen.values())

    def curves(rows, field):
        return np.array([[l[field] for l in r["layers"]] for r in rows])

    def band(ax, x, arr, label, color=None):
        mean = arr.mean(0)
        ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(mean)
        line, = ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, color=line.get_color())
        return mean, ci, line.get_color()

    # ---------------- patch
    path = out_dir / "records_patch.jsonl"
    if path.exists():
        recs = latest(path)
        by = {}
        for r in recs:
            by.setdefault(r["condition"], {}).setdefault(r["position_set"], []).append(r)
        for cond, per_ps in by.items():
            moves = per_ps[list(per_ps)[0]][0]["moves"]
            holds = per_ps[list(per_ps)[0]][0]["holds"]
            order = [p for p in DEFAULT_POSITIONS if p in per_ps]
            fig, axes = plt.subplots(2, len(order), figsize=(3.5 * len(order), 7), squeeze=False)
            for col, ps in enumerate(order):
                rows = per_ps[ps]
                denom = np.array([r["baseline_source"][moves] - r["baseline_base"][moves] for r in rows])
                keep = np.abs(denom) > min_denominator
                if keep.sum() < len(rows):
                    print(f"[aggregate] {cond}/{ps}: dropped {int((~keep).sum())} pairs with a "
                          f"degenerate {moves} denominator")
                rows = [r for r, k in zip(rows, keep) if k]
                if not rows:
                    continue
                denom = denom[keep][:, None]
                x = np.arange(len(rows[0]["layers"]))
                base_m = np.array([r["baseline_base"][moves] for r in rows])[:, None]
                norm = (curves(rows, moves) - base_m) / denom
                ax = axes[0][col]
                mean, ci, _ = band(ax, x, norm, f"normalized {moves}")
                ax.axhline(0, color="grey", lw=0.5)
                ax.axhline(1, color="grey", lw=0.5, ls=":")
                ax.set_ylim(-0.35, 1.35)
                ax.set_title(f"{ps}  (n={len(rows)})")
                if col == 0:
                    ax.set_ylabel(f"normalized {moves}\n0 = base run, 1 = source run")
                ax2 = axes[1][col]
                raw_hold = curves(rows, holds)
                hm, hci, _ = band(ax2, x, raw_hold, f"raw {holds} (side effect)")
                b_hold = float(np.mean([r["baseline_base"][holds] for r in rows]))
                s_hold = float(np.mean([r["baseline_source"][holds] for r in rows]))
                ax2.axhline(b_hold, ls="--", lw=1, color="k", label="base run")
                ax2.axhline(s_hold, ls=":", lw=1, color="tab:red", label="source run")
                ax2.set_xlabel("layer")
                if col == 0:
                    ax2.set_ylabel(f"{holds} contrast (nats/token)")
                ax.legend(fontsize=7)
                ax2.legend(fontsize=7)
                peak = int(np.argmax(mean))
                summary.setdefault("patch", {}).setdefault(cond, {})[ps] = dict(
                    n=len(rows), moves=moves, holds=holds,
                    normalized_mean=mean.tolist(), normalized_ci95=ci.tolist(),
                    peak_layer=peak, peak_value=float(mean[peak]),
                    first_layer_normalized_ge_half=(
                        int(np.argmax(mean >= 0.5)) if (mean >= 0.5).any() else None),
                    side_effect_raw_mean=hm.tolist(),
                    side_effect_max_shift=float(np.max(np.abs(hm - b_hold))),
                    base_level=b_hold, source_level=s_hold)
            fig.suptitle(f"verse-long patching — condition '{cond}': "
                         f"base {CONDITIONS[cond]['base']} <- source {CONDITIONS[cond]['source']}\n"
                         f"top: the variable this condition should move; "
                         f"bottom: the one it should leave alone")
            fig.tight_layout()
            fig.savefig(out_dir / f"patch_{cond}.png", dpi=150)
            plt.close(fig)

    # ---------------- transplant
    path = out_dir / "records_transplant.jsonl"
    if path.exists():
        recs = latest(path)
        by = {}
        for r in recs:
            by.setdefault(r["position_set"], {}).setdefault(r["donor"], []).append(r)
        for ps, per_donor in by.items():
            fig, ax = plt.subplots(figsize=(7, 4.5))
            means = {}
            for donor, color in (("q1", "tab:blue"), ("q2", "tab:orange")):
                rows = per_donor.get(donor, [])
                if not rows:
                    continue
                x = np.arange(len(rows[0]["layers"]))
                m, _, _ = band(ax, x, curves(rows, "topic"), f"donor = {donor} summary", color)
                means[donor] = m
            base = float(np.mean([r["baseline_blank"]["topic"]
                                  for rows in per_donor.values() for r in rows]))
            ax.axhline(base, ls="--", lw=1, color="k", label="blank prompt, no transplant")
            ax.set_xlabel("layer")
            ax.set_ylabel("topic contrast (nats/token)\n>0 favours Q2, <0 favours Q1")
            ax.set_title(f"Transplant into a question-free prompt — donor residual at '{ps}'")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"transplant_{ps}.png", dpi=150)
            plt.close(fig)
            if {"q1", "q2"} <= set(means):
                sep = means["q2"] - means["q1"]
                summary.setdefault("transplant", {})[ps] = dict(
                    n=len(per_donor["q1"]), blank_topic=base,
                    donor_q1_mean=means["q1"].tolist(), donor_q2_mean=means["q2"].tolist(),
                    separation=sep.tolist(), peak_layer=int(np.argmax(sep)),
                    peak_separation=float(np.max(sep)))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f)
    for cond, per_ps in summary.get("patch", {}).items():
        for ps, d in per_ps.items():
            print(f"[aggregate] patch/{cond}/{ps}: n={d['n']} peak normalized {d['moves']}="
                  f"{d['peak_value']:.2f} @ layer {d['peak_layer']}, "
                  f"max {d['holds']} side effect={d['side_effect_max_shift']:.3f} nats")
    for ps, d in summary.get("transplant", {}).items():
        print(f"[aggregate] transplant/{ps}: n={d['n']} peak donor separation="
              f"{d['peak_separation']:.3f} nats @ layer {d['peak_layer']}")
    print(f"[aggregate] wrote plots + summary.json to {out_dir}")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--data_dir", default=None, help="default: <repo>/data/<model_name>/verse-long")
    ap.add_argument("--pairs", default=None,
                    help="default: <repo>/data/<model_name>/verse-long-abstraction/pairs.jsonl")
    ap.add_argument("--out_dir", default=None,
                    help="default: <repo>/results/<model_name>/verse-long-abstraction")
    ap.add_argument("--experiments", nargs="+", default=["patch", "transplant"],
                    choices=["patch", "transplant", "generate"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS, choices=DEFAULT_POSITIONS)
    ap.add_argument("--transplant_positions", nargs="+", default=["last", "suffix"],
                    choices=["last", "suffix"])
    ap.add_argument("--gen_layers", type=int, nargs="+", default=None,
                    help="layers to free-generate at (default: the peak layer per position set "
                         "from summary.json, else the middle layer)")
    ap.add_argument("--gen_positions", nargs="+", default=["last", "question"])
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--max_ref_tokens", type=int, default=128)
    ap.add_argument("--layer_batch", type=int, default=4, help="layers per batched forward")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--min_denominator", type=float, default=0.05,
                    help="drop pairs whose source-minus-base contrast is smaller than this, "
                         "since the normalized curve divides by it")
    ap.add_argument("--debug_shapes", action="store_true",
                    help="dump every shape involved in a patch for one pair, then exit")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    data_root = REPO_ROOT / "data" / model_name / "verse-long-abstraction"
    pairs_path = Path(args.pairs) if args.pairs else data_root / "pairs.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / model_name / "verse-long-abstraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        try:
            from determinism import enable_determinism
            enable_determinism()
        except ImportError:
            pass
        verse_path, prose_path = U.resolve_data_paths(REPO_ROOT, model_name, args.data_dir)
        rows = U.load_rows(verse_path, prose_path)
        pairs = U.load_jsonl(pairs_path)
        for k, p in enumerate(pairs):
            p.setdefault("pair_id", f"{p['q1']}_{p['q2']}")
        if args.max_pairs:
            pairs = pairs[: args.max_pairs]
        print(f"{len(pairs)} pairs from {pairs_path}")

        P = Patcher(args.model, args.dtype, args.layer_batch, args.max_ref_tokens)
        if args.debug_shapes:
            debug_shapes(P, pairs, rows)
            return
        if "patch" in args.experiments:
            run_patch(P, pairs, rows, {c: args.positions for c in args.conditions}, out_dir)
        if "transplant" in args.experiments:
            run_transplant(P, pairs, rows, args.transplant_positions, out_dir)
        if "generate" in args.experiments:
            layers = args.gen_layers or [P.n_layers // 2]
            run_generate(P, pairs, rows, layers, args.gen_positions, args.max_new_tokens, out_dir)

    aggregate(out_dir, args.min_denominator)


if __name__ == "__main__":
    main()