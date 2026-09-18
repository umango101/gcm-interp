#!/usr/bin/env python
"""
Per-layer verse/prose residual patching: copy the verse run's residual stream into the prose
run and measure how far the run moves toward the verse answer.

For each layer i, run the VERSE prompt (source) and the PROSE prompt (receiver) on the same
held-out item, then overwrite the receiver's residual stream at layer i, at the chosen
positions, with the source's residual stream at the corresponding positions.  Nothing is added
and nothing is rescaled: the receiver's activations at those sites are replaced outright.

Accuracy is teacher-forced and needs no judge.  Each verse-long question has both a verse and a
prose reference answer, so for a held-out item the patched prose prompt is scored on both and
counted correct when the verse answer wins on mean per-token log-probability.  That gives the
curve a real floor and ceiling, both drawn on the plot:

    floor    unpatched prose prompt   (should sit near 0)
    ceiling  unpatched verse prompt   (the same measurement when the prompt just asks for verse)

Because the patch is a replacement rather than a perturbation, the raw/normed scaling sweep of
the steering version is gone -- there is no vector whose magnitude has to be chosen.  What
replaces it is --patch_positions, which is the axis that actually changes the claim:

    last     replace only the final prompt token.  The sharp bottleneck test: every answer
             token has to reach the patched state through attention.
    prompt   replace the whole prompt.  At layer 0 this is by construction identical to just
             running the verse prompt, so the curve starts pinned at the ceiling; the
             informative part is where it departs.
    answer   replace the reference-answer positions.  Read this one with suspicion: at late
             layers those residuals are close to the verse run's next-token predictions, which
             is nearly the quantity being scored.
    all      prompt + answer.

TOKEN ALIGNMENT.  The verse and prose prompts are not the same length, so receiver position p
is matched to source position p + (len(source_prompt) - len(receiver_prompt)) -- i.e. the two
sequences are right-aligned at the end of the prompt.  This is correct for the shared chat
template suffix and for the answer positions (both runs append the identical reference tokens,
so those line up exactly by offset).  It is an assumption about the earlier prompt tokens,
where the two instructions differ; the shift is recorded in every record as prompt_shift, and
receiver positions with no source counterpart are dropped.

Outputs in results/<model>/verse-patch/:
    accuracy.json                       per-layer accuracy per patch site, plus baselines
    accuracy_by_layer.png               the requested graph
    records_accuracy.jsonl              resumable per-item records
    generations.jsonl                   optional, for the existing verse/prose judge
    logit_lens.json / .txt / .png       optional (--lens), the difference-in-means readout

Tested against nnsight 0.4.11 / transformers 4.53.3.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import verse_long_utils as U  # noqa: E402

SOURCE_POSITIONS = ("last", "answer", "prompt")
PATCH_POSITIONS = ("last", "all", "prompt", "answer")


def _val(p):
    return getattr(p, "value", p)


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
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.layer_batch = layer_batch
        self.max_ref_tokens = max_ref_tokens
        self._meta = {}
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

    def _layer_meta(self, layer):
        """Device and dtype of this layer's own parameters.

        Reading the parameters rather than assuming self.model.device keeps writes correct when
        device_map='auto' shards layers across GPUs, where the target device varies by layer.
        Cached source activations are already on the right shard, so the .to() in _write is a
        no-op for them; it matters for the float32/CPU vectors the optional logit lens builds.
        """
        if layer not in self._meta:
            p = next(self.layers[layer].parameters())
            self._meta[layer] = (p.device, p.dtype)
        return self._meta[layer]

    def _write(self, layer, rows, positions, src, alpha):
        """resid[rows, positions, :] <- src, with index tensors of equal length.

        Same shape discipline as the patching script this is modelled on: a slice-plus-list
        index relying on the value to broadcast expands wrongly under nnsight, so both index
        tensors are built explicitly and the value is materialized at the exact target shape.

        src has shape [len(rows), len(positions), d] and is flattened row-major to match the
        (repeat_interleave rows, repeat columns) index order.  alpha < 1 interpolates between
        the receiver's own activation and the source's; alpha == 1 is the plain swap.
        """
        torch = self.torch
        R, P = len(rows), len(positions)
        dev, dt = self._layer_meta(layer)
        src = src.to(device=dev, dtype=dt).reshape(R * P, -1)
        row_idx = torch.tensor(rows, device=dev, dtype=torch.long).repeat_interleave(P)
        col_idx = torch.tensor(positions, device=dev, dtype=torch.long).repeat(R)
        h = self._resid(layer)
        if alpha >= 1.0:
            h[row_idx, col_idx, :] = src
        else:
            h[row_idx, col_idx, :] = (1.0 - alpha) * h[row_idx, col_idx, :] + alpha * src

    def anatomy(self, fmt, question):
        return U.Anatomy(self.tok, fmt, question)

    def batch(self, anatomy, refs):
        return U.ReferenceBatch(self.tok, anatomy.ids, refs, self.max_ref_tokens, self.model.device)

    def _inputs(self, ids, mask):
        return {"input_ids": ids, "attention_mask": mask}

    # ---------------------------------------------------------------- patch geometry
    @staticmethod
    def patch_index(anatomy, patch_positions, total_len):
        """Receiver positions the patch is written to."""
        if patch_positions == "last":
            return list(anatomy.positions["last"])
        if patch_positions == "prompt":
            return list(anatomy.positions["prompt"])
        if patch_positions == "answer":
            return list(range(len(anatomy.ids), total_len))
        return list(range(total_len))

    @staticmethod
    def align(recv_anat, src_anat, recv_positions):
        """Right-align the two runs at the end of the prompt.

        Returns (receiver positions kept, matching source positions, shift).  Receiver
        positions whose counterpart would fall before the start of the source sequence are
        dropped -- that happens only when the receiver prompt is the longer of the two and the
        request reaches into its leading tokens.
        """
        shift = len(src_anat.ids) - len(recv_anat.ids)
        keep = [(p, p + shift) for p in recv_positions if p + shift >= 0]
        return [p for p, _ in keep], [q for _, q in keep], shift

    # ---------------------------------------------------------------- source capture
    def cache_source(self, ids, mask, positions):
        """Residual stream of the source run at `positions`, per layer, shape [R, P, d].

        Left on its own shard in the model's dtype: the write targets the same layer and so the
        same device, and for --patch_positions all/prompt the full-sequence cache is large
        enough that a CPU round trip per layer is not worth paying.
        """
        torch = self.torch
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            acts = [self._resid(l)[:, positions, :].save() for l in range(self.n_layers)]
        return [_val(a).detach() for a in acts]

    def cache_source_batch(self, src_batch, positions):
        ids, mask, _, _ = src_batch.tile(1)
        return self.cache_source(ids, mask, positions)

    def cache_source_prompt(self, src_anat, positions):
        torch = self.torch
        ids = torch.tensor([src_anat.ids], device=self.model.device)
        return self.cache_source(ids, torch.ones_like(ids), positions)

    # ---------------------------------------------------------------- patching sweep
    def patch_sweep(self, batch, src_cache, recv_positions, alpha):
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
                                recv_positions, src_cache[layer], alpha)
                g = torch.log_softmax(self.model.lm_head.output.float(), -1) \
                         .gather(-1, tgt.unsqueeze(-1)).squeeze(-1).save()
            for b, scores in enumerate(batch.reduce(_val(g), smask, len(chunk))):
                out[chunk[b]] = scores
        return out

    def score(self, batch):
        torch = self.torch
        ids, mask, tgt, smask = batch.tile(1)
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            g = torch.log_softmax(self.model.lm_head.output.float(), -1) \
                     .gather(-1, tgt.unsqueeze(-1)).squeeze(-1).save()
        return batch.reduce(_val(g), smask, n_blocks=1)[0]

    # ---------------------------------------------------------------- free generation
    def generate(self, recv_anat, src_anat, layer, patch_positions, max_new_tokens, alpha):
        """Patch the prefill, then decode unpatched.

        Unlike the steering version there is no hook during decoding: a generated token has no
        counterpart in the source run, so there is nothing to copy into it.  Patch sites that
        fall entirely in the answer segment therefore do nothing here, and the caller is warned.
        """
        torch = self.torch
        n_prompt = len(recv_anat.ids)
        ids = torch.tensor([recv_anat.ids], device=self.model.device)
        mask = torch.ones_like(ids)
        recv_pos = [i for i in self.patch_index(recv_anat, patch_positions, n_prompt)
                    if i < n_prompt]
        recv_pos, src_pos, _ = self.align(recv_anat, src_anat, recv_pos)
        cache = self.cache_source_prompt(src_anat, src_pos) if (layer is not None and recv_pos) \
            else None
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            if cache is not None:
                self._write(layer, [0], recv_pos, cache[layer], alpha)
            out = self.model.output.save()
        del cache
        out = _val(out)
        cache_kv, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        hf = self.model._model
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        toks = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                t = int(nxt)
                if t in eos:
                    break
                toks.append(t)
                o = hf(input_ids=nxt, past_key_values=cache_kv, use_cache=True)
                cache_kv, nxt = o.past_key_values, o.logits[:, -1].argmax(-1, keepdim=True)
        return self.tok.decode(toks)


# ============================================================================ optional lens
def build_direction(S, rows, train_ids, source_position):
    """verse-mean minus prose-mean, per layer, over the train split.

    Used only by --lens.  It plays no part in the patching measurement, which needs no
    estimated direction at all -- it reads the source activations off the source run directly.
    """
    import torch

    acc_v = acc_p = None
    norms = None
    for n, rid in enumerate(train_ids):
        q = rows[rid]["question"]
        v = residual_summary(S, S.anatomy("verse", q), source_position, rows[rid]["verse_answer"])
        p = residual_summary(S, S.anatomy("prose", q), source_position, rows[rid]["prose_answer"])
        if acc_v is None:
            acc_v = [torch.zeros_like(x) for x in v]
            acc_p = [torch.zeros_like(x) for x in p]
            norms = [0.0] * len(v)
        for l in range(len(v)):
            acc_v[l] += v[l]
            acc_p[l] += p[l]
            norms[l] += float(p[l].norm())
        if (n + 1) % 10 == 0 or n + 1 == len(train_ids):
            print(f"[direction] {n + 1}/{len(train_ids)} train items", flush=True)
    k = len(train_ids)
    vecs = [(acc_v[l] - acc_p[l]) / k for l in range(len(acc_v))]
    return vecs, [x / k for x in norms]


def residual_summary(S, anatomy, source_position, answer=None):
    """Mean residual at the requested site, per layer, in float32 on CPU."""
    torch = S.torch
    if source_position == "answer":
        if answer is None:
            raise ValueError("source_position='answer' needs the answer text")
        ref_ids = S.tok(answer, add_special_tokens=False)["input_ids"][: S.max_ref_tokens]
        ids = torch.tensor([list(anatomy.ids) + ref_ids], device=S.model.device)
        positions = list(range(len(anatomy.ids), ids.shape[1]))
    else:
        ids = torch.tensor([anatomy.ids], device=S.model.device)
        positions = (anatomy.positions["last"] if source_position == "last"
                     else anatomy.positions["prompt"])
    mask = torch.ones_like(ids)
    with torch.no_grad(), S.model.trace(S._inputs(ids, mask)):
        acts = [S._resid(l)[0, positions, :].mean(0).save() for l in range(S.n_layers)]
    return [_val(a).float().cpu() for a in acts]


def lens(S, vec, top_k, apply_norm=True):
    """Unembed one direction. The final RMSNorm's learned per-channel weight changes the token
    ranking, so it is applied by default; the RMS rescale itself is a positive scalar and is
    rank-neutral, included only to keep magnitudes comparable across layers."""
    torch = S.torch
    W = S.model.lm_head.weight
    v = vec.to(W.device, W.dtype)
    if apply_norm:
        norm = S.model.model.norm
        eps = getattr(norm, "variance_epsilon", 1e-6)
        v = v * torch.rsqrt(v.float().pow(2).mean() + eps).to(v.dtype)
        v = v * norm.weight
    with torch.no_grad():
        logits = (v @ W.T).float()
    logits = logits - logits.mean()
    top = torch.topk(logits, top_k)
    bot = torch.topk(-logits, top_k)
    fmt = lambda idx, val: [dict(token=S.tok.decode([int(t)]), token_id=int(t),
                                 logit=round(float(s), 4))
                            for t, s in zip(idx, val)]
    return dict(top=fmt(top.indices, top.values), bottom=fmt(bot.indices, -bot.values))


def run_lens(S, args, rows, train_ids, out_dir):
    vecs, resid_norms = build_direction(S, rows, train_ids, args.source_position)
    entries = []
    for l, v in enumerate(vecs):
        entry = dict(layer=l, vector_norm=round(float(v.norm()), 4),
                     mean_resid_norm=round(resid_norms[l], 4),
                     **lens(S, v, args.top_k, apply_norm=not args.no_lens_norm))
        if l > 0:
            a, b = vecs[l - 1], v
            entry["cos_with_prev_layer"] = round(float((a @ b) / (a.norm() * b.norm() + 1e-9)), 4)
        entries.append(entry)
    with open(out_dir / "logit_lens.json", "w") as f:
        json.dump(dict(model=args.model, source_position=args.source_position,
                       lens_norm_applied=not args.no_lens_norm, top_k=args.top_k,
                       train_ids=train_ids, layers=entries), f, indent=2)
    with open(out_dir / "logit_lens.txt", "w") as f:
        f.write(f"verse - prose direction, difference in means at '{args.source_position}', "
                f"n_train={len(train_ids)}\n")
        f.write(f"{'layer':>5} {'|v|':>9} {'cos prev':>9}  top tokens (verse end) | "
                f"bottom tokens (prose end)\n")
        for e in entries:
            top = " ".join(repr(t["token"]) for t in e["top"])
            bot = " ".join(repr(t["token"]) for t in e["bottom"])
            f.write(f"{e['layer']:>5} {e['vector_norm']:>9.3f} "
                    f"{e.get('cos_with_prev_layer', float('nan')):>9.3f}  {top} | {bot}\n")
    print(f"[lens] wrote {out_dir}/logit_lens.txt")


# ============================================================================ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--data_dir", default=None, help="default: <repo>/data/<model_name>/verse-long")
    ap.add_argument("--out_dir", default=None,
                    help="default: <repo>/results/<model_name>/verse-patch")
    ap.add_argument("--patch_positions", nargs="+", default=["last", "all"],
                    choices=PATCH_POSITIONS,
                    help="swept: one accuracy curve per entry, all in the same job")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="1.0 replaces the receiver activation outright; below 1.0 interpolates "
                         "between receiver and source at the patch sites")
    ap.add_argument("--train_frac", type=float, default=0.5,
                    help="only used with --lens; without it every item is a test item, since "
                         "patching estimates nothing from a train split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--max_ref_tokens", type=int, default=128)
    ap.add_argument("--layer_batch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--generate_layers", type=int, nargs="+", default=None,
                    help="also dump patched generations at these layers for the verse/prose judge")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--plot_only", action="store_true")
    # optional difference-in-means readout, independent of the patching measurement
    ap.add_argument("--lens", action="store_true",
                    help="also compute the verse-prose difference-in-means logit lens")
    ap.add_argument("--source_position", default="last", choices=SOURCE_POSITIONS,
                    help="--lens only: where the difference-in-means is taken")
    ap.add_argument("--top_k", type=int, default=10, help="--lens only")
    ap.add_argument("--no_lens_norm", action="store_true",
                    help="--lens only: unembed the raw vector without the final norm's weight")
    args = ap.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / model_name / "verse-patch"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        try:
            from determinism import enable_determinism
            enable_determinism()
        except ImportError:
            pass
        import torch

        verse_path, prose_path = U.resolve_data_paths(REPO_ROOT, model_name, args.data_dir)
        rows = U.load_rows(verse_path, prose_path)
        ids = sorted(rows)
        if args.max_items:
            ids = ids[: args.max_items]
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        if args.lens:
            n_train = max(1, int(round(args.train_frac * len(ids))))
            train_ids, test_ids = sorted(ids[:n_train]), sorted(ids[n_train:])
            if not test_ids:
                raise ValueError("empty test split; lower --train_frac or raise --max_items")
            print(f"{len(rows)} rows; {len(train_ids)} train / {len(test_ids)} test "
                  f"(seed {args.seed})")
        else:
            train_ids, test_ids = [], sorted(ids)
            print(f"{len(rows)} rows; {len(test_ids)} test (no train split needed)")

        S = Patcher(args.model, args.dtype, args.layer_batch, args.max_ref_tokens)
        if args.lens:
            run_lens(S, args, rows, train_ids, out_dir)

        records = []
        rec_path = out_dir / "records_accuracy.jsonl"
        done = set()
        if rec_path.exists():
            for r in U.load_jsonl(rec_path):
                if "patch_position" not in r:
                    continue  # steering-era record; different experiment, skip it
                done.add((r["id"], r["patch_position"]))
                records.append(r)
        warned_shift = set()
        with open(rec_path, "a") as fout:
            for n, rid in enumerate(test_ids):
                q = rows[rid]["question"]
                refs = {"verse": rows[rid]["verse_answer"], "prose": rows[rid]["prose_answer"]}
                prose_anat = S.anatomy("prose", q)     # receiver
                verse_anat = S.anatomy("verse", q)     # source
                prose_batch = S.batch(prose_anat, refs)
                verse_batch = S.batch(verse_anat, refs)
                floor = S.score(prose_batch)
                ceiling = S.score(verse_batch)
                margin = lambda sc: sc["verse"]["logp_mean"] - sc["prose"]["logp_mean"]
                for where in args.patch_positions:
                    if (rid, where) in done:
                        continue
                    want = S.patch_index(prose_anat, where, prose_batch.input_ids.shape[1])
                    recv_pos, src_pos, shift = S.align(prose_anat, verse_anat, want)
                    if not recv_pos:
                        raise ValueError(f"id={rid}: no alignable positions for '{where}'")
                    if shift and where not in warned_shift:
                        warned_shift.add(where)
                        print(f"[align] prompts differ by {shift:+d} tokens; right-aligned at "
                              f"the end of the prompt, {len(want) - len(recv_pos)} receiver "
                              f"position(s) dropped for '{where}'", flush=True)
                    cache = S.cache_source_batch(verse_batch, src_pos)
                    curve = S.patch_sweep(prose_batch, cache, recv_pos, args.alpha)
                    del cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    rec = dict(id=rid, patch_position=where, alpha=args.alpha,
                               prompt_shift=shift, n_patched=len(recv_pos),
                               floor_margin=round(margin(floor), 5),
                               ceiling_margin=round(margin(ceiling), 5),
                               margins=[round(margin(c), 5) for c in curve])
                    records.append(rec)
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    os.fsync(fout.fileno())
                print(f"[patch] {n + 1}/{len(test_ids)} id={rid} "
                      f"floor={margin(floor):+.3f} ceiling={margin(ceiling):+.3f}", flush=True)

        # ---------------- optional generations
        if args.generate_layers:
            gen_path = out_dir / "generations.jsonl"
            with open(gen_path, "a") as fout:
                for rid in test_ids:
                    q = rows[rid]["question"]
                    recv, src = S.anatomy("prose", q), S.anatomy("verse", q)
                    base = dict(id=rid, question=q)
                    fout.write(json.dumps(dict(base, layer=None, patch_position=None,
                                               text=S.generate(recv, src, None, "last",
                                                               args.max_new_tokens,
                                                               args.alpha))) + "\n")
                    for where in args.patch_positions:
                        if where == "answer":
                            print("[generate] skipping 'answer': generated tokens have no "
                                  "source counterpart to copy from", flush=True)
                            continue
                        for l in args.generate_layers:
                            txt = S.generate(recv, src, l, where, args.max_new_tokens, args.alpha)
                            fout.write(json.dumps(dict(base, layer=l, patch_position=where,
                                                       text=txt)) + "\n")
                    fout.flush()
                    os.fsync(fout.fileno())
                    print(f"[generate] id={rid} done", flush=True)
            print(f"[generate] wrote {gen_path}")

    aggregate(out_dir)


def aggregate(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rec_path = out_dir / "records_accuracy.jsonl"
    if not rec_path.exists():
        print(f"[aggregate] no records at {rec_path}")
        return
    seen = {}
    for r in U.load_jsonl(rec_path):
        if "patch_position" not in r:
            continue
        seen[(r["id"], r["patch_position"])] = r
    if not seen:
        print(f"[aggregate] no patching records at {rec_path}")
        return
    by = {}
    for r in seen.values():
        by.setdefault(r["patch_position"], []).append(r)

    wheres = [w for w in PATCH_POSITIONS if w in by]
    colors = dict(zip(PATCH_POSITIONS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
    summary = {}

    floor_acc = float(np.mean([r["floor_margin"] > 0 for r in seen.values()]))
    ceil_acc = float(np.mean([r["ceiling_margin"] > 0 for r in seen.values()]))
    floor_m = float(np.mean([r["floor_margin"] for r in seen.values()]))
    ceil_m = float(np.mean([r["ceiling_margin"] for r in seen.values()]))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for where in wheres:
        rs = by[where]
        m = np.array([r["margins"] for r in rs])
        x = np.arange(m.shape[1])
        acc = (m > 0).mean(0)
        mean = m.mean(0)
        ci = 1.96 * m.std(0, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.zeros_like(mean)
        ax.plot(x, acc, color=colors[where], label=f"patch at '{where}' (n={len(rs)})")
        ax2.plot(x, mean, color=colors[where], label=f"patch at '{where}'")
        ax2.fill_between(x, mean - ci, mean + ci, color=colors[where], alpha=0.2)
        peak = int(np.argmax(acc))
        peak_m = int(np.argmax(mean))
        summary[where] = dict(
            n=len(rs), accuracy=acc.tolist(), margin_mean=mean.tolist(),
            peak_layer=peak, peak_accuracy=float(acc[peak]),
            peak_margin_layer=peak_m, peak_margin=float(mean[peak_m]))
    for a, lo, hi in ((ax, floor_acc, ceil_acc), (ax2, floor_m, ceil_m)):
        a.axhline(lo, ls="--", lw=1, color="k", label="unpatched prose prompt (floor)")
        a.axhline(hi, ls=":", lw=1.2, color="grey", label="unpatched verse prompt (ceiling)")
        a.set_xlabel("layer index")
        a.legend(fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("accuracy: verse answer scored above prose answer")
    ax.set_title("Patching effectiveness by layer (verse run -> prose run)")
    ax2.set_ylabel("logp(verse) - logp(prose), nats/token")
    ax2.set_title("Same measurement, before thresholding")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_layer.png", dpi=150)
    plt.close(fig)

    summary["baselines"] = dict(floor_accuracy=floor_acc, ceiling_accuracy=ceil_acc,
                                floor_margin=floor_m, ceiling_margin=ceil_m)
    lens_path = out_dir / "logit_lens.json"
    if lens_path.exists():
        with open(lens_path) as f:
            entries = json.load(f)["layers"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot([e["layer"] for e in entries], [e["vector_norm"] for e in entries],
                label="||verse - prose||")
        ax.plot([e["layer"] for e in entries], [e["mean_resid_norm"] for e in entries],
                label="mean residual norm")
        ax.set_yscale("log")
        ax.set_xlabel("layer index")
        ax.set_ylabel("norm")
        ax.set_title("Difference-in-means norm vs residual norm (diagnostic only)")
        for e in entries[::max(1, len(entries) // 12)]:
            ax.annotate(e["top"][0]["token"].strip()[:10], (e["layer"], e["vector_norm"]),
                        fontsize=7, rotation=45)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "logit_lens_top1.png", dpi=150)
        plt.close(fig)

    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(summary, f)
    print(f"[aggregate] floor accuracy {floor_acc:.2f}, ceiling accuracy {ceil_acc:.2f}")
    for where, d in summary.items():
        if where == "baselines":
            continue
        print(f"[aggregate] patch at '{where}': n={d['n']} "
              f"peak accuracy {d['peak_accuracy']:.2f} at layer {d['peak_layer']}; "
              f"peak margin {d['peak_margin']:+.3f} nats at layer {d['peak_margin_layer']}")
    print(f"[aggregate] wrote plots + accuracy.json to {out_dir}")


if __name__ == "__main__":
    main()