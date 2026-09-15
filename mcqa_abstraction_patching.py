#!/usr/bin/env python
"""
Residual-stream activation patching for the verse-MCQA causal abstraction experiment.

For every (clean, corrupted) pair from generate_data/verse_mcqa_abstraction/build_counterfactuals.py
and every layer L, run the CLEAN prompt with the residual stream (decoder-layer output) at the
chosen token position(s) overwritten by the CORRUPTED run's activation at layer L, then read the
answer logits at the final prompt position.

  step 1 (task)      metric = logit(prose letter)            - logit(clean verse letter)
  step 2 (position)  metric = logit(corrupted verse letter)  - logit(clean verse letter)
  step 3 (letter)    metric = logit(alt letter, e.g. S)      - logit(clean verse letter)
                     aux    = logit(std letter at that index, e.g. D) - logit(clean verse letter)

Positions (--positions):
  paren   the token containing the "(" that ends the user turn (the site in the plan)
  final   the last prompt token ("\n" after "<|im_start|>assistant"), where the answer is read
  suffix  every token from "(" to the end of the prompt (identical template tokens in both runs)

The answer letter is predicted at `final`, five template tokens after "(" for Qwen1.5, so a patch at
`paren` in the last few layers cannot reach the output: expect the paren curve to fall back toward
the clean baseline at the top of the network even if the variable is still there.

Per-example results are appended (fsync'd) to records_step{k}.jsonl and skipped on restart, so the
job can be requeued. Plots and summary.json are rebuilt from the records at the end (--plot_only
to rebuild without the model).

Tested against nnsight 0.4.11 + transformers 4.53.3 (tuple decoder-layer outputs); the layer
output type is detected at startup, and saved proxies are unwrapped with getattr(p, "value", p).
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

ANSWER_SUFFIX = "\nAnswer: ("
STEP_NAMES = {1: "task (verse → prose)", 2: "position (options permuted)", 3: "letter (A–D → P–S)"}
POSITION_CHOICES = ("paren", "final", "suffix")


def _val(p):
    return getattr(p, "value", p)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================================ model-side
class Patcher:
    def __init__(self, model_name, dtype, layer_batch, logits_to_keep=True):
        import torch
        from nnsight import LanguageModel

        self.torch = torch
        self.model = LanguageModel(model_name, device_map="auto",
                                   torch_dtype=getattr(torch, dtype), dispatch=True)
        self.tok = self.model.tokenizer
        if not self.tok.is_fast:
            raise RuntimeError("need a fast tokenizer (offset mapping) to locate the '(' token")
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.layer_batch = layer_batch
        self.trace_kwargs = {"logits_to_keep": 1} if logits_to_keep else {}
        self.tuple_out = self._detect_tuple_output()
        print(f"[patcher] {model_name}: {self.n_layers} layers, "
              f"decoder-layer output is {'tuple' if self.tuple_out else 'tensor'}", flush=True)

    def _detect_tuple_output(self):
        x = self.torch.tensor([[self.tok.eos_token_id]])
        with self.torch.no_grad(), self.model.trace(x, **self.trace_kwargs):
            out = self.layers[0].output.save()
        return isinstance(_val(out), tuple)

    def _resid(self, layer):
        out = self.layers[layer].output
        return out[0] if self.tuple_out else out

    def letter_id(self, letter):
        ids = self.tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"answer letter {letter!r} is not a single token: {ids}")
        return ids[0]

    def encode(self, messages):
        """Token ids exactly as data_handler builds them, plus the index of the '(' token."""
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        start = text.rfind(ANSWER_SUFFIX)
        if start < 0:
            raise ValueError("prompt does not end its user turn with 'Answer: ('")
        char = start + len(ANSWER_SUFFIX) - 1
        hits = [t for t, (s, e) in enumerate(offsets) if s <= char < e]
        if len(hits) != 1:
            raise ValueError(f"could not map '(' to a unique token (hits={hits})")
        paren = hits[0]
        if "(" not in self.tok.decode([ids[paren]]):
            raise ValueError(f"token at paren index decodes to {self.tok.decode([ids[paren]])!r}")
        return ids, paren

    @staticmethod
    def positions(name, n_tokens, paren):
        if name == "paren":
            return [paren]
        if name == "final":
            return [n_tokens - 1]
        if name == "suffix":
            return list(range(paren, n_tokens))
        raise ValueError(name)

    def cache_run(self, ids, positions, token_ids):
        """Unpatched run: residuals at `positions` for every layer + answer-token logits."""
        torch = self.torch
        x = torch.tensor([ids])
        with torch.no_grad(), self.model.trace(x, **self.trace_kwargs):
            acts = [self._resid(l)[0, positions, :].save() for l in range(self.n_layers)]
            logits = self.model.lm_head.output[0, -1, :].save()
        logits = _val(logits).float()
        return [_val(a) for a in acts], logits[token_ids].cpu(), int(logits.argmax())

    def patched_runs(self, ids, positions, acts, token_ids):
        """One batched forward per chunk of layers; batch row i patches layer chunk[i]."""
        torch = self.torch
        gathered = [None] * self.n_layers
        top1 = [None] * self.n_layers
        for s in range(0, self.n_layers, self.layer_batch):
            chunk = list(range(s, min(s + self.layer_batch, self.n_layers)))
            x = torch.tensor([ids] * len(chunk))
            with torch.no_grad(), self.model.trace(x, **self.trace_kwargs):
                for i, layer in enumerate(chunk):
                    self._resid(layer)[i, positions, :] = acts[layer]
                logits = self.model.lm_head.output[:, -1, :].save()
            logits = _val(logits).float()
            g, a = logits[:, token_ids].cpu(), logits.argmax(-1).cpu()
            for i, layer in enumerate(chunk):
                gathered[layer] = g[i]
                top1[layer] = int(a[i])
        return gathered, top1


def run_step(patcher, step, pairs, position_names, out_dir, require_baselines):
    torch = patcher.torch
    rec_path = out_dir / f"records_step{step}.jsonl"
    done = set()
    if rec_path.exists():
        for r in load_jsonl(rec_path):
            if r.get("skipped") or set(position_names) <= set(r.get("positions", {})):
                done.add(r["id"])
    todo = [p for p in pairs if p["id"] not in done]
    print(f"[step {step}] {len(pairs)} pairs, {len(done)} already done, {len(todo)} to run", flush=True)
    max_sanity_err = 0.0

    with open(rec_path, "a") as fout:
        for n, pair in enumerate(todo):
            c_ids, c_paren = patcher.encode(pair["clean_prompt"])
            k_ids, k_paren = patcher.encode(pair["corrupted_prompt"])
            if c_ids[c_paren + 1:] != k_ids[k_paren + 1:]:
                raise RuntimeError(f"id {pair['id']}: post-'(' template tokens differ between runs")

            clean_tok = patcher.letter_id(pair["clean_answer"])
            metrics = {"primary": patcher.letter_id(pair["corrupted_answer"])}
            for name, letter in pair.get("aux_answers", {}).items():
                metrics[name] = patcher.letter_id(letter)
            names = list(metrics)
            token_ids = [clean_tok] + [metrics[m] for m in names]

            clean_pos = {p: patcher.positions(p, len(c_ids), c_paren) for p in position_names}
            corr_pos = {p: patcher.positions(p, len(k_ids), k_paren) for p in position_names}
            union = sorted({i for v in corr_pos.values() for i in v})
            where = {i: j for j, i in enumerate(union)}

            corr_acts, corr_g, corr_top = patcher.cache_run(k_ids, union, token_ids)
            _, clean_g, clean_top = patcher.cache_run(c_ids, [c_paren], token_ids)

            base = dict(id=pair["id"], step=step, clean_answer=pair["clean_answer"],
                        corrupted_answer=pair["corrupted_answer"],
                        aux_answers=pair.get("aux_answers", {}),
                        token_ids=dict(clean=clean_tok, **metrics),
                        n_tokens=dict(clean=len(c_ids), corrupted=len(k_ids)),
                        paren_index=dict(clean=c_paren, corrupted=k_paren),
                        baseline_top1=dict(clean=clean_top, corrupted=corr_top))
            base_ok = clean_top == clean_tok and corr_top == metrics["primary"]
            if require_baselines and not base_ok:
                base["skipped"] = "baseline prediction does not match label"
                fout.write(json.dumps(base) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                print(f"[step {step}] id {pair['id']}: baseline mismatch, skipped", flush=True)
                continue

            def lds(g):
                return {m: float(g[1 + j] - g[0]) for j, m in enumerate(names)}

            record = dict(base, ld_clean=lds(clean_g), ld_corrupted=lds(corr_g), positions={})
            for pname in position_names:
                idx = torch.tensor([where[i] for i in corr_pos[pname]])
                acts = [a[idx.to(a.device)] for a in corr_acts]
                gathered, top1 = patcher.patched_runs(c_ids, clean_pos[pname], acts, token_ids)
                curve = {m: [float(g[1 + j] - g[0]) for g in gathered] for j, m in enumerate(names)}
                record["positions"][pname] = dict(ld=curve, top1=top1)
                if pname in ("final", "suffix"):
                    # Patching the final position after the last layer must reproduce the corrupted run.
                    err = abs(curve["primary"][-1] - record["ld_corrupted"]["primary"])
                    max_sanity_err = max(max_sanity_err, err)
                    if err > 0.1:
                        print(f"[step {step}] WARNING id {pair['id']}: last-layer {pname} patch differs "
                              f"from corrupted run by {err:.3f} logits", flush=True)

            fout.write(json.dumps(record) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            print(f"[step {step}] {n + 1}/{len(todo)} id={pair['id']} "
                  f"ld_clean={record['ld_clean']['primary']:.2f} "
                  f"ld_corrupted={record['ld_corrupted']['primary']:.2f}", flush=True)
    if any(p in position_names for p in ("final", "suffix")) and todo:
        print(f"[step {step}] sanity: max |last-layer final patch - corrupted| = {max_sanity_err:.4f}", flush=True)


# ============================================================================ aggregation
def aggregate(out_dir, steps, position_names, intersect):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    recs = {}
    for s in steps:
        path = out_dir / f"records_step{s}.jsonl"
        recs[s] = {r["id"]: r for r in load_jsonl(path) if not r.get("skipped")} if path.exists() else {}
    summary = {"steps": {}, "intersect_ids": intersect}
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}

    for pname in position_names:
        valid = {s: {i for i, r in recs[s].items() if pname in r["positions"]} for s in steps}
        common = set.intersection(*valid.values()) if intersect and valid else None
        fig_s, (ax_n, ax_f) = plt.subplots(1, 2, figsize=(13, 4.5))
        for s in steps:
            ids = sorted(common if common is not None else valid[s])
            if not ids:
                print(f"[aggregate] step {s} / {pname}: no records")
                continue
            rows = [recs[s][i] for i in ids]
            metrics = list(rows[0]["ld_clean"])
            n_layers = len(rows[0]["positions"][pname]["ld"]["primary"])
            layers = np.arange(n_layers)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            step_sum = {"n": len(ids), "ids": ids, "metrics": {}}
            for m in metrics:
                curves = np.array([r["positions"][pname]["ld"][m] for r in rows])
                mean = curves.mean(0)
                ci = 1.96 * curves.std(0, ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else 0 * mean
                clean = np.mean([r["ld_clean"][m] for r in rows])
                corr = np.mean([r["ld_corrupted"][m] for r in rows])
                label = ("logit(corrupted answer) − logit(clean answer)" if m == "primary"
                         else f"logit({m} letter) − logit(clean answer)")
                line, = ax.plot(layers, mean, label=label)
                ax.fill_between(layers, mean - ci, mean + ci, alpha=0.2, color=line.get_color())
                ax.axhline(clean, ls="--", lw=1, color=line.get_color(), alpha=0.7)
                if m == "primary":
                    ax.axhline(corr, ls=":", lw=1.2, color="k", label="corrupted run (primary)")
                step_sum["metrics"][m] = dict(mean=mean.tolist(), ci95=np.broadcast_to(ci, mean.shape).tolist(),
                                              clean=float(clean), corrupted=float(corr))
                if m == "primary":
                    norm = (mean - clean) / (corr - clean)
                    step_sum["normalized_primary"] = norm.tolist()
                    crossing = np.nonzero(norm >= 0.5)[0]
                    step_sum["first_layer_normalized_ge_0.5"] = int(crossing[0]) if len(crossing) else None
                    ax_n.plot(layers, norm, color=colors.get(s), label=f"step {s}: {STEP_NAMES[s]}")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xlabel("layer")
            ax.set_ylabel("logit difference")
            ax.set_title(f"Step {s} — {STEP_NAMES[s]} — patch '{pname}' residual (n={len(ids)})\n"
                         "dashed = unpatched clean, dotted = corrupted run")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"ld_step{s}_{pname}.png", dpi=150)
            plt.close(fig)

            hit = np.array([[t == r["token_ids"]["primary"] for t in r["positions"][pname]["top1"]] for r in rows])
            ax_f.plot(layers, hit.mean(0), color=colors.get(s), label=f"step {s}: top-1 = corrupted answer")
            step_sum["top1_is_corrupted_answer"] = hit.mean(0).tolist()
            for m in metrics:
                if m != "primary":
                    aux = np.array([[t == r["token_ids"][m] for t in r["positions"][pname]["top1"]] for r in rows])
                    ax_f.plot(layers, aux.mean(0), color=colors.get(s), ls="--",
                              label=f"step {s}: top-1 = {m} letter")
                    step_sum[f"top1_is_{m}"] = aux.mean(0).tolist()
            summary["steps"].setdefault(str(s), {})[pname] = step_sum

        ax_n.axhline(0, color="grey", lw=0.5)
        ax_n.axhline(1, color="grey", lw=0.5, ls=":")
        ax_n.set_xlabel("layer")
        ax_n.set_ylabel("(LD_patched − LD_clean) / (LD_corrupted − LD_clean)")
        ax_n.set_title(f"Normalized patching effect — '{pname}'")
        ax_n.legend(fontsize=8)
        ax_f.set_xlabel("layer")
        ax_f.set_ylabel("fraction of examples")
        ax_f.set_ylim(-0.02, 1.02)
        ax_f.set_title(f"Top-1 prediction after patching — '{pname}'")
        ax_f.legend(fontsize=8)
        fig_s.tight_layout()
        fig_s.savefig(out_dir / f"summary_{pname}.png", dpi=150)
        plt.close(fig_s)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f)
    for s, per_pos in summary["steps"].items():
        for pname, d in per_pos.items():
            print(f"[aggregate] step {s} / {pname}: n={d['n']}, "
                  f"first layer with normalized effect >= 0.5: {d.get('first_layer_normalized_ge_0.5')}")
    print(f"[aggregate] wrote plots + summary.json to {out_dir}")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--data_dir", default=None, help="default: <repo>/data/<model_name>/verse-mcqa-abstraction")
    ap.add_argument("--out_dir", default=None, help="default: <repo>/results/<model_name>/verse-mcqa-abstraction")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--positions", nargs="+", default=["paren", "final"], choices=POSITION_CHOICES)
    ap.add_argument("--layer_batch", type=int, default=8, help="layers patched per batched forward")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--no_intersect", action="store_true",
                    help="use every passing pair per step instead of ids shared by all requested steps")
    ap.add_argument("--keep_baseline_failures", action="store_true",
                    help="do not skip pairs whose unpatched predictions disagree with the labels")
    ap.add_argument("--no_logits_to_keep", action="store_true")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data" / model_name / "verse-mcqa-abstraction"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / model_name / "verse-mcqa-abstraction"
    out_dir.mkdir(parents=True, exist_ok=True)
    intersect = not args.no_intersect

    if not args.plot_only:
        try:
            from determinism import enable_determinism
            enable_determinism()
        except ImportError:
            pass
        pairs = {s: load_jsonl(data_dir / f"step{s}-pairs.jsonl") for s in args.steps}
        if intersect:
            common = set.intersection(*({p["id"] for p in v} for v in pairs.values()))
            pairs = {s: [p for p in v if p["id"] in common] for s, v in pairs.items()}
        if args.max_examples:
            keep = sorted({p["id"] for v in pairs.values() for p in v})[: args.max_examples]
            pairs = {s: [p for p in v if p["id"] in set(keep)] for s, v in pairs.items()}
        patcher = Patcher(args.model, args.dtype, args.layer_batch, logits_to_keep=not args.no_logits_to_keep)
        for s in args.steps:
            run_step(patcher, s, pairs[s], args.positions, out_dir,
                     require_baselines=not args.keep_baseline_failures)

    aggregate(out_dir, args.steps, args.positions, intersect)


if __name__ == "__main__":
    main()