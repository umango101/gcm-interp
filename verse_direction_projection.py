#!/usr/bin/env python
"""
Observational read of the verse/prose direction: project real representations onto it, then
unembed.

This is the read-only counterpart to verse_direction_logit_lens.py. Nothing is intervened on.
The direction v is still the difference in means (fit on a train split, unit-normalized to
v_hat), but here it is used as a measuring stick applied to representations the model produces
on its own.

For every held-out item, both the verse-prompted and the prose-prompted run are forwarded with
their own reference answer teacher-forced, and at every layer and token position we record

    c    = h . v_hat          how much of the direction the representation actually contains
    rms  = rms(h)             the scale the final norm would divide by

Two things come out of that.

(1) PRESENCE.  Paired accuracy per layer -- the fraction of items where the verse run carries
    more of the direction than the prose run at the same position. Reported separately at the
    final prompt token and across answer positions, which is the observational version of the
    question the steering sweep raised: last-token steering barely moved the margin while
    all-position steering saturated, and the projection curves say whether that is because the
    direction is simply not present at the final prompt token.

(2) UNEMBEDDING.  The final norm is linear in h at fixed rms, so the logit lens decomposes
    exactly:

        logits(h) = W_U(w * h) / rms(h)
                  = (c / rms) * W_U(w * v_hat)   +   W_U(w * h_perp) / rms

    The first term is the direction's additive contribution to the logits, in nats, at the
    magnitude the model actually has. That is what gets reported: top-k tokens by contribution,
    scaled by the observed c/rms rather than by an arbitrary vector norm. The second term is the
    same lens with the direction removed, so the two together show what the direction adds to
    the model's real prediction at each layer.

    One thing to be aware of when reading the token lists: c * v_hat is a scalar multiple of
    v_hat, so the token *ranking* of the projection is fixed by sign(c) and does not vary per
    item or per layer beyond that. Only the magnitudes vary. Per-layer differences in the lists
    come from v_hat itself changing with layer, not from the projection step.

Outputs in results/<model>/verse-projection/:
    records_projection.jsonl    per (item, condition) projections
    projection.json             per-layer summary
    projection_by_layer.png     presence curves + paired accuracy
    projection_heatmap.png      layer x position-in-answer
    direction_lens.txt/.json    top-k tokens by logit contribution, and the removal view

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
N_BINS = 10
N_FIRST = 16


def _val(p):
    return getattr(p, "value", p)


class Projector:
    def __init__(self, model_name, dtype, max_ref_tokens):
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
        self.max_ref_tokens = max_ref_tokens
        self._meta = {}
        self.norm = self.model.model.norm
        self.eps = getattr(self.norm, "variance_epsilon", 1e-6)
        self.tuple_out = self._detect_tuple_output()
        print(f"[projector] {model_name}: {self.n_layers} layers, decoder-layer output is "
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
        if layer not in self._meta:
            p = next(self.layers[layer].parameters())
            self._meta[layer] = (p.device, p.dtype)
        return self._meta[layer]

    def anatomy(self, fmt, question):
        return U.Anatomy(self.tok, fmt, question)

    def _inputs(self, ids, mask):
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, anatomy, answer):
        ref = self.tok(answer, add_special_tokens=False)["input_ids"][: self.max_ref_tokens]
        return list(anatomy.ids) + ref, len(anatomy.ids)

    # ------------------------------------------------------------------ direction
    def mean_residual(self, ids, positions):
        torch = self.torch
        x = torch.tensor([ids], device=self.model.device)
        mask = torch.ones_like(x)
        with torch.no_grad(), self.model.trace(self._inputs(x, mask)):
            acts = [self._resid(l)[0, positions, :].float().mean(0).save()
                    for l in range(self.n_layers)]
        return [_val(a).float().cpu() for a in acts]

    # ------------------------------------------------------------------ projection
    def project(self, ids, n_prompt, v_hat):
        """c = h . v_hat and rms(h) at every position, per layer, plus h at the last prompt
        token for the removal lens."""
        torch = self.torch
        x = torch.tensor([ids], device=self.model.device)
        mask = torch.ones_like(x)
        projs, rmss, lasts = [], [], []
        with torch.no_grad(), self.model.trace(self._inputs(x, mask)):
            for l in range(self.n_layers):
                dev, _ = self._layer_meta(l)
                vh = v_hat[l].to(device=dev, dtype=torch.float32)
                h = self._resid(l)[0].float()
                projs.append((h @ vh).save())
                rmss.append(h.pow(2).mean(-1).add(self.eps).sqrt().save())
                lasts.append(h[n_prompt - 1, :].save())
        return ([_val(p).cpu() for p in projs], [_val(r).cpu() for r in rmss],
                [_val(h).cpu() for h in lasts])

    # ------------------------------------------------------------------ unembedding
    def direction_logits(self, v_hat_layer):
        """W_U(w * v_hat): the per-token logit change per unit of c/rms."""
        torch = self.torch
        W = self.model.lm_head.weight
        x = (v_hat_layer.to(device=W.device, dtype=torch.float32) *
             self.norm.weight.float())
        with torch.no_grad():
            return (x.to(W.dtype) @ W.T).float().cpu()

    def full_logits(self, h):
        torch = self.torch
        W = self.model.lm_head.weight
        hh = h.to(device=W.device, dtype=torch.float32)
        hn = hh * torch.rsqrt(hh.pow(2).mean() + self.eps)
        with torch.no_grad():
            return ((hn * self.norm.weight.float()).to(W.dtype) @ W.T).float().cpu()


def binned(values, n_bins):
    """Mean of `values` within each of n_bins equal spans of relative position."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return [float("nan")] * n_bins
    edges = np.linspace(0, len(v), n_bins + 1)
    out = []
    for b in range(n_bins):
        lo, hi = int(edges[b]), max(int(edges[b + 1]), int(edges[b]) + 1)
        out.append(float(v[lo:min(hi, len(v))].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None,
                    help="default: <repo>/results/<model_name>/verse-projection")
    ap.add_argument("--source_position", default="last", choices=SOURCE_POSITIONS,
                    help="where the direction is fit; keep this equal to the steering run so "
                         "the two experiments share a direction")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--max_ref_tokens", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / model_name / "verse-projection"
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
            raise ValueError("empty test split")
        print(f"{len(rows)} rows; {len(train_ids)} train / {len(test_ids)} test (seed {args.seed})")

        P = Projector(args.model, args.dtype, args.max_ref_tokens)

        # ---------------- direction, fit on train only
        acc_v = acc_p = None
        for n, rid in enumerate(train_ids):
            q = rows[rid]["question"]
            out = {}
            for fmt, ans in (("verse", rows[rid]["verse_answer"]), ("prose", rows[rid]["prose_answer"])):
                a = P.anatomy(fmt, q)
                if args.source_position == "answer":
                    tok_ids, n_prompt = P.encode(a, ans)
                    pos = list(range(n_prompt, len(tok_ids)))
                else:
                    tok_ids = list(a.ids)
                    pos = a.positions["last"] if args.source_position == "last" else a.positions["prompt"]
                out[fmt] = P.mean_residual(tok_ids, pos)
            if acc_v is None:
                acc_v = [torch.zeros_like(x) for x in out["verse"]]
                acc_p = [torch.zeros_like(x) for x in out["prose"]]
            for l in range(P.n_layers):
                acc_v[l] += out["verse"][l]
                acc_p[l] += out["prose"][l]
            if (n + 1) % 10 == 0 or n + 1 == len(train_ids):
                print(f"[direction] {n + 1}/{len(train_ids)} train items", flush=True)
        vecs = [(acc_v[l] - acc_p[l]) / len(train_ids) for l in range(P.n_layers)]
        v_hat = [v / (v.norm() + 1e-9) for v in vecs]

        # ---------------- projections on held-out items
        rec_path = out_dir / "records_projection.jsonl"
        done = set()
        if rec_path.exists():
            for r in U.load_jsonl(rec_path):
                done.add((r["id"], r["condition"]))
        mean_last = {"verse": [torch.zeros(len(v_hat[0])) for _ in range(P.n_layers)],
                     "prose": [torch.zeros(len(v_hat[0])) for _ in range(P.n_layers)]}
        counts = {"verse": 0, "prose": 0}
        with open(rec_path, "a") as fout:
            for n, rid in enumerate(test_ids):
                q = rows[rid]["question"]
                for fmt, ans in (("verse", rows[rid]["verse_answer"]),
                                 ("prose", rows[rid]["prose_answer"])):
                    a = P.anatomy(fmt, q)
                    tok_ids, n_prompt = P.encode(a, ans)
                    projs, rmss, lasts = P.project(tok_ids, n_prompt, v_hat)
                    for l in range(P.n_layers):
                        mean_last[fmt][l] += lasts[l].float()
                    counts[fmt] += 1
                    if (rid, fmt) in done:
                        continue
                    rec = dict(id=rid, condition=fmt, n_prompt=n_prompt,
                               n_answer=len(tok_ids) - n_prompt,
                               last_proj=[round(float(p[n_prompt - 1]), 5) for p in projs],
                               last_rms=[round(float(r[n_prompt - 1]), 5) for r in rmss],
                               answer_proj_bins=[[round(x, 5) for x in
                                                  binned(p[n_prompt:].tolist(), N_BINS)]
                                                 for p in projs],
                               answer_rms_bins=[[round(x, 5) for x in
                                                 binned(r[n_prompt:].tolist(), N_BINS)]
                                                for r in rmss],
                               answer_proj_first=[[round(float(x), 5)
                                                   for x in p[n_prompt:n_prompt + N_FIRST]]
                                                  for p in projs])
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    os.fsync(fout.fileno())
                print(f"[project] {n + 1}/{len(test_ids)} id={rid}", flush=True)

        # ---------------- unembedding
        recs = U.load_jsonl(rec_path)
        by_cond = {"verse": [r for r in recs if r["condition"] == "verse"],
                   "prose": [r for r in recs if r["condition"] == "prose"]}
        lens_entries = []
        for l in range(P.n_layers):
            u = P.direction_logits(v_hat[l])
            u = u - u.mean()
            scale = {}
            for fmt in ("verse", "prose"):
                rs = by_cond[fmt]
                scale[fmt] = (sum(r["last_proj"][l] / max(r["last_rms"][l], 1e-6) for r in rs)
                              / max(len(rs), 1))
            top = u.topk(args.top_k)
            bot = (-u).topk(args.top_k)
            mk = lambda idx, val: [dict(token=P.tok.decode([int(t)]), token_id=int(t),
                                        per_unit_logit=round(float(s), 4),
                                        nats_verse=round(float(s) * scale["verse"], 4),
                                        nats_prose=round(float(s) * scale["prose"], 4))
                                   for t, s in zip(idx, val)]
            hv = mean_last["verse"][l] / max(counts["verse"], 1)
            full = P.full_logits(hv)
            c = float(hv @ v_hat[l])
            rms = float((hv.pow(2).mean() + P.eps).sqrt())
            contrib = (c / rms) * u
            removed = full - contrib
            entry = dict(layer=l, direction_norm=round(float(vecs[l].norm()), 4),
                         mean_c_over_rms=dict(verse=round(scale["verse"], 5),
                                              prose=round(scale["prose"], 5)),
                         top=mk(top.indices, top.values), bottom=mk(bot.indices, -bot.values),
                         mean_verse_argmax=P.tok.decode([int(full.argmax())]),
                         mean_verse_argmax_without_direction=P.tok.decode([int(removed.argmax())]),
                         argmax_changes=bool(int(full.argmax()) != int(removed.argmax())))
            drop = (full - removed).topk(args.top_k)
            entry["largest_logit_gain_from_direction"] = [
                dict(token=P.tok.decode([int(t)]), nats=round(float(s), 4))
                for t, s in zip(drop.indices, drop.values)]
            lens_entries.append(entry)
        with open(out_dir / "direction_lens.json", "w") as f:
            json.dump(dict(model=args.model, source_position=args.source_position,
                           train_ids=train_ids, layers=lens_entries), f, indent=2)
        with open(out_dir / "direction_lens.txt", "w") as f:
            f.write(f"verse-prose direction, fit at '{args.source_position}', "
                    f"n_train={len(train_ids)}\n")
            f.write("nats = logit contribution at the observed c/rms of the mean verse run, "
                    "final prompt token\n\n")
            f.write(f"{'layer':>5} {'c/rms(v)':>9} {'c/rms(p)':>9}  top tokens (nats) | "
                    f"bottom tokens\n")
            for e in lens_entries:
                top = " ".join(f"{t['token']!r}:{t['nats_verse']:+.2f}" for t in e["top"][:6])
                bot = " ".join(f"{t['token']!r}" for t in e["bottom"][:6])
                f.write(f"{e['layer']:>5} {e['mean_c_over_rms']['verse']:>9.3f} "
                        f"{e['mean_c_over_rms']['prose']:>9.3f}  {top} | {bot}\n")
        print(f"[lens] wrote {out_dir}/direction_lens.txt")

    aggregate(out_dir)


def aggregate(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rec_path = out_dir / "records_projection.jsonl"
    if not rec_path.exists():
        print(f"[aggregate] no records at {rec_path}")
        return
    seen = {}
    for r in U.load_jsonl(rec_path):
        seen[(r["id"], r["condition"])] = r
    ids = sorted({k[0] for k in seen})
    ids = [i for i in ids if (i, "verse") in seen and (i, "prose") in seen]
    if not ids:
        print("[aggregate] no complete verse/prose pairs")
        return
    V = [seen[(i, "verse")] for i in ids]
    Pr = [seen[(i, "prose")] for i in ids]
    n_layers = len(V[0]["last_proj"])
    x = np.arange(n_layers)

    def arr(rows, field):
        return np.array([r[field] for r in rows], dtype=float)

    last_v, last_p = arr(V, "last_proj"), arr(Pr, "last_proj")
    rms_v, rms_p = arr(V, "last_rms"), arr(Pr, "last_rms")
    ans_v = arr(V, "answer_proj_bins").mean(-1)
    ans_p = arr(Pr, "answer_proj_bins").mean(-1)
    arms_v = arr(V, "answer_rms_bins").mean(-1)
    arms_p = arr(Pr, "answer_rms_bins").mean(-1)

    acc_last = (last_v > last_p).mean(0)
    acc_ans = (ans_v > ans_p).mean(0)

    def d_prime(a, b):
        s = np.sqrt(0.5 * (a.var(0, ddof=1) + b.var(0, ddof=1))) + 1e-9
        return (a.mean(0) - b.mean(0)) / s

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0][0]
    ax.plot(x, acc_last, label="final prompt token")
    ax.plot(x, acc_ans, label="answer positions (mean)")
    ax.axhline(0.5, ls="--", lw=1, color="k", label="chance")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("paired accuracy: verse run projects higher")
    ax.set_title("Is the direction present in the real representation?")
    ax = axes[0][1]
    for name, a, b in (("final prompt token", last_v, last_p),
                       ("answer positions", ans_v, ans_p)):
        ax.plot(x, d_prime(a, b), label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("d' between verse and prose runs")
    ax.set_title("Separation, in pooled standard deviations")
    ax = axes[1][0]
    for name, a, r, ls in (("verse, final prompt token", last_v, rms_v, "-"),
                           ("prose, final prompt token", last_p, rms_p, "-"),
                           ("verse, answer positions", ans_v, arms_v, "--"),
                           ("prose, answer positions", ans_p, arms_p, "--")):
        m = (a / np.maximum(r, 1e-6)).mean(0)
        ax.plot(x, m, ls, label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("mean c / rms(h)   (scale-free)")
    ax.set_title("Projection magnitude, normalized by the scale the lens divides by")
    for a in axes.flat[:3]:
        a.set_xlabel("layer index")
        a.legend(fontsize=7)
    ax = axes[1][1]
    diff = (arr(V, "answer_proj_bins") / np.maximum(arr(V, "answer_rms_bins"), 1e-6)
            - arr(Pr, "answer_proj_bins") / np.maximum(arr(Pr, "answer_rms_bins"), 1e-6)).mean(0)
    im = ax.imshow(diff.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(),
                   extent=[0, n_layers - 1, 0, 1])
    ax.set_xlabel("layer index")
    ax.set_ylabel("relative position in answer")
    ax.set_title("(verse - prose) c/rms, by layer and position in answer")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "projection_by_layer.png", dpi=150)
    plt.close(fig)

    onset = lambda a: int(np.argmax(a >= 0.9)) if (a >= 0.9).any() else None
    summary = dict(n_pairs=len(ids), n_layers=n_layers,
                   accuracy_last=acc_last.tolist(), accuracy_answer=acc_ans.tolist(),
                   d_prime_last=d_prime(last_v, last_p).tolist(),
                   d_prime_answer=d_prime(ans_v, ans_p).tolist(),
                   c_over_rms_last=dict(verse=(last_v / rms_v).mean(0).tolist(),
                                        prose=(last_p / rms_p).mean(0).tolist()),
                   c_over_rms_answer=dict(verse=(ans_v / arms_v).mean(0).tolist(),
                                          prose=(ans_p / arms_p).mean(0).tolist()),
                   first_layer_accuracy_ge_0p9=dict(last=onset(acc_last), answer=onset(acc_ans)),
                   peak_layer=dict(last=int(np.argmax(acc_last)), answer=int(np.argmax(acc_ans))))
    with open(out_dir / "projection.json", "w") as f:
        json.dump(summary, f)
    print(f"[aggregate] n={len(ids)} pairs")
    print(f"[aggregate] final prompt token: peak accuracy {acc_last.max():.2f} "
          f"at layer {int(np.argmax(acc_last))}, "
          f"first layer >=0.9: {summary['first_layer_accuracy_ge_0p9']['last']}")
    print(f"[aggregate] answer positions:   peak accuracy {acc_ans.max():.2f} "
          f"at layer {int(np.argmax(acc_ans))}, "
          f"first layer >=0.9: {summary['first_layer_accuracy_ge_0p9']['answer']}")
    print(f"[aggregate] wrote plots + projection.json to {out_dir}")


if __name__ == "__main__":
    main()