#!/usr/bin/env python
"""Visualize and compare ATP head-attribution maps.

These are attribution (indirect-effect) scores per (layer, head), NOT attention
patterns. Signed: positive means patching that head's source activation pushes
the metric L = LL(undesired) - LL(desired) up, i.e. toward the source behaviour.

    # one run
    python compare_head_maps.py results/Qwen1.5-14B-Chat/from_female-long_to_male-long/atp

    # two runs, e.g. bd = prompt+response  vs  bd = prompt only
    python compare_head_maps.py \\
        results_old/Qwen1.5-14B-Chat/from_female-long_to_male-long/atp \\
        results/Qwen1.5-14B-Chat/from_female-long_to_male-long/atp \\
        --labels "prompt+response" "prompt only" --out bd_comparison.png

Always reassembles from the heads_*.pt shards. It deliberately ignores any
numerator_1_heads.pt cache, which in pre-fix runs was built from range(LEN)=50
of the 100 shards -- comparing a 50-example cache against a 100-example run
would confound the change under test with the example count.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


def discover_shards(run_dir, stem="heads"):
    rx = re.compile(rf"^{re.escape(stem)}_(\d+)\.pt$")
    found = []
    for p in glob.glob(os.path.join(run_dir, f"{stem}_*.pt")):
        m = rx.match(os.path.basename(p))
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])          # numeric, so heads_2 precedes heads_10
    return found


def load_per_example(run_dir, num_heads, stem="heads"):
    """-> array [n_examples, n_layers, num_heads] of per-head attribution.

    Each shard is [layers, num_heads*head_dim] after squeeze; the head dimension
    is folded by summing within each head's contiguous block, matching
    logits_handler's einops.reduce('l (n m) b -> l n b', 'sum')."""
    shards = discover_shards(run_dir, stem)
    if not shards:
        raise FileNotFoundError(f"No {stem}_*.pt shards in {run_dir}")
    per_example = []
    flat_dim = None
    for idx, path in shards:
        t = torch.load(path, map_location="cpu")
        t = t.squeeze().to(torch.float32)
        if t.ndim != 2:
            raise ValueError(f"{path}: expected [layers, num_heads*head_dim] after "
                             f"squeeze, got {tuple(t.shape)}")
        if flat_dim is None:
            flat_dim = t.shape[1]
            if flat_dim % num_heads:
                raise ValueError(f"{path}: last dim {flat_dim} is not divisible by "
                                 f"--num_heads {num_heads}. Pass the model's real head count.")
        elif t.shape[1] != flat_dim:
            raise ValueError(f"{path}: last dim {t.shape[1]} != {flat_dim} from earlier shards")
        n_layers = t.shape[0]
        head_dim = flat_dim // num_heads
        per_example.append(t.reshape(n_layers, num_heads, head_dim).sum(-1).numpy())

    arr = np.stack(per_example)
    cache = os.path.join(run_dir, "numerator_1_heads.pt")
    note = ""
    if os.path.exists(cache):
        try:
            n_cached = torch.load(cache, map_location="cpu").shape[-1]
            if n_cached != len(shards):
                note = (f"   NOTE: numerator_1_heads.pt here holds {n_cached} examples but "
                        f"{len(shards)} shards exist -- that cache is stale/partial (ignored).")
        except Exception:
            pass
    print(f"{run_dir}\n   {len(shards)} shards, indices {shards[0][0]}..{shards[-1][0]}, "
          f"{arr.shape[1]} layers x {num_heads} heads x {flat_dim // num_heads} head_dim")
    if note:
        print(note)
    return arr


def topk_set(mean_map, frac):
    k = max(1, int(round(frac * mean_map.size)))
    flat = mean_map.ravel()
    idx = np.argpartition(-flat, k - 1)[:k]
    return set(map(int, idx)), k


def describe(mean_map, per_ex, label, frac):
    sel, k = topk_set(mean_map, frac)
    L, H = mean_map.shape
    print(f"\n{label}: top {k} heads ({frac:.0%})")
    order = sorted(sel, key=lambda i: -mean_map.ravel()[i])
    for i in order[:10]:
        l, h = divmod(i, H)
        col = per_ex[:, l, h]
        same_sign = float(np.mean(np.sign(col) == np.sign(col.mean())))
        sem = col.std(ddof=1) / np.sqrt(len(col))
        print(f"   L{l:>2} H{h:>2}  effect {mean_map[l, h]:+.4g} +-{sem:.2g} (sem), "
              f"sign consistent in {same_sign:.0%} of examples")
    layers = sorted({i // H for i in sel})
    print(f"   spread over {len(layers)} layers, range L{layers[0]}-L{layers[-1]}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="1 or 2 atp/ run directories")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--num_heads", type=int, default=40,
                    help="num_attention_heads for this model (Qwen1.5-14B/32B: 40, Gemma-3-12B: 16)")
    ap.add_argument("--topk", type=float, default=0.05, help="fraction, to match the eval sweep")
    ap.add_argument("--out", default="head_maps.png")
    args = ap.parse_args()

    if len(args.runs) > 2:
        print("At most two runs.", file=sys.stderr)
        return 1
    labels = args.labels or [os.path.basename(r.rstrip("/")) or r for r in args.runs]
    if len(labels) != len(args.runs):
        print("--labels must match the number of runs.", file=sys.stderr)
        return 1

    per_ex = [load_per_example(r, args.num_heads) for r in args.runs]
    means = [p.mean(axis=0) for p in per_ex]
    sels = [describe(m, p, lab, args.topk) for m, p, lab in zip(means, per_ex, labels)]

    if len(means) == 2:
        if means[0].shape != means[1].shape:
            print(f"\nShape mismatch {means[0].shape} vs {means[1].shape}; cannot compare.",
                  file=sys.stderr)
            return 1
        a, b = means[0].ravel(), means[1].ravel()
        rho = spearmanr(a, b).correlation
        r = pearsonr(a, b)[0]
        inter = len(sels[0] & sels[1])
        union = len(sels[0] | sels[1])
        k = len(sels[0])
        chance = k * k / a.size
        print(f"\n--- {labels[0]}  vs  {labels[1]} ---")
        print(f"   Spearman rho {rho:+.3f}   Pearson r {r:+.3f}")
        print(f"   top-{args.topk:.0%} overlap: {inter}/{k} heads shared "
              f"(Jaccard {inter/union:.3f}); chance overlap ~{chance:.1f} heads")
        if inter <= chance * 1.5:
            print("   -> selections are near-independent: the change reshapes which heads "
                  "are chosen, not just their magnitudes.")
        elif inter >= 0.8 * k:
            print("   -> selections are largely the same; the change mostly rescales effects.")

    # ---- figure -----------------------------------------------------------
    n_panels = 1 if len(means) == 1 else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)
    axes = axes[0]
    # diverging map centred at 0: attribution is signed, and a sequential map
    # would hide the sign structure that decides what gets selected
    vmax = max(np.abs(m).max() for m in means)
    for ax, m, lab in zip(axes, means, labels):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{lab}\n(mean over {per_ex[0].shape[0]} examples)")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, label="indirect effect")
    if len(means) == 2:
        d = means[1] - means[0]
        dmax = np.abs(d).max()
        im = axes[2].imshow(d, cmap="PuOr_r", vmin=-dmax, vmax=dmax, aspect="auto")
        axes[2].set_title(f"difference\n({labels[1]} - {labels[0]})")
        axes[2].set_xlabel("Head"); axes[2].set_ylabel("Layer")
        fig.colorbar(im, ax=axes[2], label="delta")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
