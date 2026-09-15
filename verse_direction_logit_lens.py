#!/usr/bin/env python
"""
Per-layer verse/prose direction: logit lens readout and steering effectiveness.

For each layer i, take the difference-in-means between the verse and prose residual streams on
a train split, then

  (1) push that vector through the final norm and the unembedding and report the top-k tokens,
  (2) add it to the residual stream of a held-out PROSE prompt and measure how far the run
      moves toward the verse answer.

Accuracy is teacher-forced and needs no judge.  Each verse-long question has both a verse and a
prose reference answer, so for a held-out item the steered prose prompt is scored on both and
counted correct when the verse answer wins on mean per-token log-probability.  That gives the
curve a real floor and ceiling, both drawn on the plot:

    floor    unsteered prose prompt   (should sit near 0)
    ceiling  unsteered verse prompt   (the same measurement when the prompt just asks for verse)

Two scalings are swept, because they answer different questions and disagree by construction:

    raw     add the difference vector as-is.  Its norm grows with depth along with the residual
            stream, so a peak here can mean "this layer carries the direction" or merely "the
            vector is large here".
    normed  rescale to --resid_frac of the layer's mean residual norm, so every layer gets a
            perturbation of the same relative size.  This is the one to read for "which layer".

Outputs in results/<model>/verse-direction/:
    logit_lens.json / logit_lens.txt   top-k tokens per layer, both directions
    accuracy.json                       per-layer accuracy, both scalings, plus baselines
    accuracy_by_layer.png               the requested graph
    logit_lens_top1.png                 rank-1 token per layer, as a quick companion view
    generations.jsonl                   optional, for the existing verse/prose judge

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
STEER_POSITIONS = ("all", "prompt", "answer")


def _val(p):
    return getattr(p, "value", p)


class Steerer:
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
        self.tuple_out = self._detect_tuple_output()
        print(f"[steerer] {model_name}: {self.n_layers} layers, decoder-layer output is "
              f"{'tuple' if self.tuple_out else 'tensor'}", flush=True)

    def _detect_tuple_output(self):
        x = self.torch.tensor([[self.tok.eos_token_id]])
        with self.torch.no_grad(), self.model.trace(x):
            out = self.layers[0].output.save()
        return isinstance(_val(out), tuple)

    def _resid(self, layer):
        out = self.layers[layer].output
        return out[0] if self.tuple_out else out

    def _add(self, layer, rows, positions, vec):
        """resid[rows, positions, :] += vec, with index tensors of equal length.

        Same shape discipline as the patching script: a slice-plus-list index relying on the
        value to broadcast expands wrongly under nnsight, so both index tensors are built
        explicitly and the value is materialized at the exact target shape.
        """
        torch = self.torch
        R, P = len(rows), len(positions)
        dev = vec.device
        row_idx = torch.tensor(rows, device=dev, dtype=torch.long).repeat_interleave(P)
        col_idx = torch.tensor(positions, device=dev, dtype=torch.long).repeat(R)
        h = self._resid(layer)
        h[row_idx, col_idx, :] = h[row_idx, col_idx, :] + vec.unsqueeze(0).repeat(R * P, 1)

    def anatomy(self, fmt, question):
        return U.Anatomy(self.tok, fmt, question)

    def batch(self, anatomy, refs):
        return U.ReferenceBatch(self.tok, anatomy.ids, refs, self.max_ref_tokens, self.model.device)

    def _inputs(self, ids, mask):
        return {"input_ids": ids, "attention_mask": mask}

    # ---------------------------------------------------------------- direction estimation
    def residual_summary(self, anatomy, source_position, answer=None):
        """Mean residual at the requested site, per layer, plus the mean residual norm there."""
        torch = self.torch
        if source_position == "answer":
            if answer is None:
                raise ValueError("source_position='answer' needs the answer text")
            ref_ids = self.tok(answer, add_special_tokens=False)["input_ids"][: self.max_ref_tokens]
            ids = torch.tensor([list(anatomy.ids) + ref_ids], device=self.model.device)
            positions = list(range(len(anatomy.ids), ids.shape[1]))
        else:
            ids = torch.tensor([anatomy.ids], device=self.model.device)
            positions = (anatomy.positions["last"] if source_position == "last"
                         else anatomy.positions["prompt"])
        mask = torch.ones_like(ids)
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            acts = [self._resid(l)[0, positions, :].mean(0).save() for l in range(self.n_layers)]
        return [_val(a).float().cpu() for a in acts]

    # ---------------------------------------------------------------- logit lens
    def lens(self, vec, top_k, apply_norm=True):
        """Unembed one direction. The final RMSNorm's learned per-channel weight changes the
        token ranking, so it is applied by default; the RMS rescale itself is a positive scalar
        and is rank-neutral, included only to keep magnitudes comparable across layers."""
        torch = self.torch
        head = self.model.lm_head
        W = head.weight
        v = vec.to(W.device, W.dtype)
        if apply_norm:
            norm = self.model.model.norm
            eps = getattr(norm, "variance_epsilon", 1e-6)
            v = v * torch.rsqrt(v.float().pow(2).mean() + eps).to(v.dtype)
            v = v * norm.weight
        with torch.no_grad():
            logits = (v @ W.T).float()
        logits = logits - logits.mean()
        top = torch.topk(logits, top_k)
        bot = torch.topk(-logits, top_k)
        fmt = lambda idx, val: [dict(token=self.tok.decode([int(t)]), token_id=int(t),
                                     logit=round(float(s), 4))
                                for t, s in zip(idx, val)]
        return dict(top=fmt(top.indices, top.values), bottom=fmt(bot.indices, -bot.values))

    # ---------------------------------------------------------------- steering sweep
    def steer_sweep(self, batch, anatomy, vecs, steer_positions):
        """One batched forward per chunk of layers; block b steers layer chunk[b]."""
        torch = self.torch
        R = len(batch.keys)
        if steer_positions == "all":
            positions = list(range(batch.input_ids.shape[1]))
        elif steer_positions == "prompt":
            positions = anatomy.positions["prompt"]
        else:
            positions = list(range(len(anatomy.ids), batch.input_ids.shape[1]))
        out = [None] * self.n_layers
        for s in range(0, self.n_layers, self.layer_batch):
            chunk = list(range(s, min(s + self.layer_batch, self.n_layers)))
            ids, mask, tgt, smask = batch.tile(len(chunk))
            with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
                for b, layer in enumerate(chunk):
                    self._add(layer, list(range(b * R, (b + 1) * R)), positions,
                              vecs[layer].to(self.model.device))
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

    def generate(self, anatomy, layer, vec, steer_positions, max_new_tokens):
        torch = self.torch
        ids = torch.tensor([anatomy.ids], device=self.model.device)
        mask = torch.ones_like(ids)
        pos = anatomy.positions["prompt"] if steer_positions in ("all", "prompt") else []
        with torch.no_grad(), self.model.trace(self._inputs(ids, mask)):
            if vec is not None and pos:
                self._add(layer, [0], pos, vec.to(self.model.device))
            out = self.model.output.save()
        out = _val(out)
        cache, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        hf = self.model._model
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        toks = []
        steer_gen = vec is not None and steer_positions in ("all", "answer")
        handle = None
        if steer_gen:
            v = vec.to(self.model.device)

            def hook(mod, inp, out_):
                if isinstance(out_, tuple):
                    return (out_[0] + v.to(out_[0].dtype),) + out_[1:]
                return out_ + v.to(out_.dtype)

            handle = hf.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    t = int(nxt)
                    if t in eos:
                        break
                    toks.append(t)
                    o = hf(input_ids=nxt, past_key_values=cache, use_cache=True)
                    cache, nxt = o.past_key_values, o.logits[:, -1].argmax(-1, keepdim=True)
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(toks)


# ============================================================================ driver
def build_direction(S, rows, train_ids, source_position):
    """verse-mean minus prose-mean, per layer, over the train split."""
    import torch

    acc_v = acc_p = None
    norms = None
    for n, rid in enumerate(train_ids):
        q = rows[rid]["question"]
        v = S.residual_summary(S.anatomy("verse", q), source_position, rows[rid]["verse_answer"])
        p = S.residual_summary(S.anatomy("prose", q), source_position, rows[rid]["prose_answer"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--data_dir", default=None, help="default: <repo>/data/<model_name>/verse-long")
    ap.add_argument("--out_dir", default=None,
                    help="default: <repo>/results/<model_name>/verse-direction")
    ap.add_argument("--source_position", default="last", choices=SOURCE_POSITIONS,
                    help="where the difference-in-means is taken: the final prompt token, the "
                         "answer tokens, or the whole prompt")
    ap.add_argument("--steer_positions", default="all", choices=STEER_POSITIONS)
    ap.add_argument("--scalings", nargs="+", default=["raw", "normed"], choices=["raw", "normed"])
    ap.add_argument("--resid_frac", type=float, default=0.1,
                    help="'normed' rescales the vector to this fraction of the layer's mean "
                         "residual norm")
    ap.add_argument("--alpha", type=float, default=1.0, help="multiplies both scalings")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--max_ref_tokens", type=int, default=128)
    ap.add_argument("--layer_batch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no_lens_norm", action="store_true",
                    help="unembed the raw vector without the final norm's learned weight")
    ap.add_argument("--generate_layers", type=int, nargs="+", default=None,
                    help="also dump steered generations at these layers for the verse/prose judge")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / model_name / "verse-direction"
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
        n_train = max(1, int(round(args.train_frac * len(ids))))
        train_ids, test_ids = sorted(ids[:n_train]), sorted(ids[n_train:])
        if not test_ids:
            raise ValueError("empty test split; lower --train_frac or raise --max_items")
        print(f"{len(rows)} rows; {len(train_ids)} train / {len(test_ids)} test (seed {args.seed})")

        S = Steerer(args.model, args.dtype, args.layer_batch, args.max_ref_tokens)
        vecs, resid_norms = build_direction(S, rows, train_ids, args.source_position)

        # ---------------- (1) logit lens
        lens = []
        for l, v in enumerate(vecs):
            entry = dict(layer=l, vector_norm=round(float(v.norm()), 4),
                         mean_resid_norm=round(resid_norms[l], 4),
                         **S.lens(v, args.top_k, apply_norm=not args.no_lens_norm))
            if l > 0:
                a, b = vecs[l - 1], v
                entry["cos_with_prev_layer"] = round(
                    float((a @ b) / (a.norm() * b.norm() + 1e-9)), 4)
            lens.append(entry)
        with open(out_dir / "logit_lens.json", "w") as f:
            json.dump(dict(model=args.model, source_position=args.source_position,
                           lens_norm_applied=not args.no_lens_norm, top_k=args.top_k,
                           train_ids=train_ids, layers=lens), f, indent=2)
        with open(out_dir / "logit_lens.txt", "w") as f:
            f.write(f"verse - prose direction, difference in means at '{args.source_position}', "
                    f"n_train={len(train_ids)}\n")
            f.write(f"{'layer':>5} {'|v|':>9} {'cos prev':>9}  top tokens (verse end) | "
                    f"bottom tokens (prose end)\n")
            for e in lens:
                top = " ".join(repr(t["token"]) for t in e["top"])
                bot = " ".join(repr(t["token"]) for t in e["bottom"])
                f.write(f"{e['layer']:>5} {e['vector_norm']:>9.3f} "
                        f"{e.get('cos_with_prev_layer', float('nan')):>9.3f}  {top} | {bot}\n")
        print(f"[lens] wrote {out_dir}/logit_lens.txt")

        # ---------------- (2) steering accuracy
        scaled = {}
        for mode in args.scalings:
            if mode == "raw":
                scaled[mode] = [args.alpha * v for v in vecs]
            else:
                scaled[mode] = [args.alpha * args.resid_frac * resid_norms[l] * v / (v.norm() + 1e-9)
                                for l, v in enumerate(vecs)]

        records = []
        rec_path = out_dir / "records_accuracy.jsonl"
        done = set()
        if rec_path.exists():
            for r in U.load_jsonl(rec_path):
                done.add((r["id"], r["scaling"]))
                records.append(r)
        with open(rec_path, "a") as fout:
            for n, rid in enumerate(test_ids):
                q = rows[rid]["question"]
                refs = {"verse": rows[rid]["verse_answer"], "prose": rows[rid]["prose_answer"]}
                prose_anat = S.anatomy("prose", q)
                verse_anat = S.anatomy("verse", q)
                prose_batch = S.batch(prose_anat, refs)
                floor = S.score(prose_batch)
                ceiling = S.score(S.batch(verse_anat, refs))
                margin = lambda sc: sc["verse"]["logp_mean"] - sc["prose"]["logp_mean"]
                for mode in args.scalings:
                    if (rid, mode) in done:
                        continue
                    curve = S.steer_sweep(prose_batch, prose_anat, scaled[mode],
                                          args.steer_positions)
                    rec = dict(id=rid, scaling=mode, floor_margin=round(margin(floor), 5),
                               ceiling_margin=round(margin(ceiling), 5),
                               margins=[round(margin(c), 5) for c in curve])
                    records.append(rec)
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    os.fsync(fout.fileno())
                print(f"[steer] {n + 1}/{len(test_ids)} id={rid} "
                      f"floor={margin(floor):+.3f} ceiling={margin(ceiling):+.3f}", flush=True)

        # ---------------- optional generations
        if args.generate_layers:
            gen_path = out_dir / "generations.jsonl"
            with open(gen_path, "a") as fout:
                for rid in test_ids:
                    q = rows[rid]["question"]
                    anat = S.anatomy("prose", q)
                    base = dict(id=rid, question=q, steer_positions=args.steer_positions)
                    fout.write(json.dumps(dict(base, layer=None, scaling=None,
                                               text=S.generate(anat, 0, None, args.steer_positions,
                                                               args.max_new_tokens))) + "\n")
                    for mode in args.scalings:
                        for l in args.generate_layers:
                            txt = S.generate(anat, l, scaled[mode][l], args.steer_positions,
                                             args.max_new_tokens)
                            fout.write(json.dumps(dict(base, layer=l, scaling=mode, text=txt)) + "\n")
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
        seen[(r["id"], r["scaling"])] = r
    by = {}
    for r in seen.values():
        by.setdefault(r["scaling"], []).append(r)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    summary = {}
    for mode, color in (("raw", "tab:blue"), ("normed", "tab:orange")):
        rs = by.get(mode)
        if not rs:
            continue
        m = np.array([r["margins"] for r in rs])
        acc = (m > 0).mean(0)
        x = np.arange(m.shape[1])
        ax.plot(x, acc, color=color, label=f"{mode} scaling")
        mean = m.mean(0)
        ci = 1.96 * m.std(0, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.zeros_like(mean)
        ax2.plot(x, mean, color=color, label=f"{mode} scaling")
        ax2.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)
        peak = int(np.argmax(acc))
        summary[mode] = dict(n=len(rs), accuracy=acc.tolist(), margin_mean=mean.tolist(),
                             peak_layer=peak, peak_accuracy=float(acc[peak]))
    floor_acc = float(np.mean([r["floor_margin"] > 0 for r in seen.values()]))
    ceil_acc = float(np.mean([r["ceiling_margin"] > 0 for r in seen.values()]))
    floor_m = float(np.mean([r["floor_margin"] for r in seen.values()]))
    ceil_m = float(np.mean([r["ceiling_margin"] for r in seen.values()]))
    for a, lo, hi in ((ax, floor_acc, ceil_acc), (ax2, floor_m, ceil_m)):
        a.axhline(lo, ls="--", lw=1, color="k", label="unsteered prose prompt (floor)")
        a.axhline(hi, ls=":", lw=1.2, color="tab:green", label="unsteered verse prompt (ceiling)")
        a.set_xlabel("layer index")
        a.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("accuracy: verse answer scored above prose answer")
    ax.set_title("Steering effectiveness by layer")
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
            lens = json.load(f)["layers"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot([e["layer"] for e in lens], [e["vector_norm"] for e in lens],
                label="||verse - prose||")
        ax.plot([e["layer"] for e in lens], [e["mean_resid_norm"] for e in lens],
                label="mean residual norm")
        ax.set_yscale("log")
        ax.set_xlabel("layer index")
        ax.set_ylabel("norm")
        ax.set_title("Direction norm vs residual norm — why 'raw' and 'normed' differ")
        for e in lens[::max(1, len(lens) // 12)]:
            ax.annotate(e["top"][0]["token"].strip()[:10], (e["layer"], e["vector_norm"]),
                        fontsize=7, rotation=45)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "logit_lens_top1.png", dpi=150)
        plt.close(fig)

    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(summary, f)
    print(f"[aggregate] floor accuracy {floor_acc:.2f}, ceiling accuracy {ceil_acc:.2f}")
    for mode, d in summary.items():
        if mode != "baselines":
            print(f"[aggregate] {mode}: n={d['n']} peak accuracy {d['peak_accuracy']:.2f} "
                  f"at layer {d['peak_layer']}")
    print(f"[aggregate] wrote plots + accuracy.json to {out_dir}")


if __name__ == "__main__":
    main()