#!/usr/bin/env python3
"""Compare two ATP head maps -- e.g. privilege vs position.

    python compare_head_maps.py \\
        --a results/gpt-oss-20b/from_user-single_to_dev-single/atp/numerator_1_heads.pt \\
        --b results/gpt-oss-20b/from_first-single_to_second-single/atp/numerator_1_heads.pt \\
        --label_a privilege --label_b position \\
        --out results_head_overlap

WHAT IT REPORTS
---------------
  top-k overlap        how many of each map's top k heads are shared, against the
                       hypergeometric expectation and a permutation p-value
  Spearman (all heads) rank agreement over the whole 1536-head map, which does not
                       depend on a threshold
  layer-level Spearman the same after collapsing heads to layers, since two maps
                       can disagree head-by-head while agreeing on depth
  per-layer table      each map's aggregated score by layer, for a figure

READ THE RESULT IN THE RIGHT DIRECTION
--------------------------------------
LOW overlap is the strong result. It says the privilege heads are not the
position heads, so the privilege localization is not explained by positional
deference. That is a dissociation, and dissociations are hard to explain away.

HIGH overlap is AMBIGUOUS, not a refutation. Both corpora share an eight-demo ICL
preamble, a one-word answer, and the same instruction frames, so heads doing
"attend to the final question and copy a demonstrated answer" will rank high in
both regardless of what distinguishes the contrasts. Before concluding that
privilege IS position, run a third contrast that shares the surface form but
neither construct, and check how much overlap that produces. That number is the
floor against which these two should be read -- without it, "70% overlap" has no
scale.

The signed-vs-absolute choice matters and is exposed as --score. Signed scores
ask whether the two contrasts push the same heads in the same direction; absolute
scores ask whether the same heads are involved at all. They answer different
questions; report which you used.
"""

import argparse
import json
import os
import sys

import numpy as np


def load_map(path, score="abs"):
    """-> [n_layers, n_heads] float array of per-head scores."""
    # Existence check BEFORE importing torch, so a missing map reports the
    # missing map rather than an unrelated ImportError.
    if not os.path.exists(path):
        # The reduced map is written LAZILY -- experiment.py writes one shard per
        # item during --patch_model, and logits_handler.load_logits only reduces
        # them into numerator_1_heads.pt when something consumes it. So this file
        # being absent right after a localization run is the normal state, not a
        # failed run. Say which it is rather than just "missing".
        d = os.path.dirname(path)
        import glob as _glob
        shards = _glob.glob(os.path.join(d, "heads_*.pt")) + \
            _glob.glob(os.path.join(d, "heads", "*.pt"))
        if shards:
            sys.exit(
                f"missing head map: {path}\n"
                f"  but {len(shards)} attribution shard(s) ARE present in {d}.\n"
                f"  The map is built lazily by logits_handler.load_logits. Run the\n"
                f"  eval stage for this contrast to materialize it -- do NOT reduce\n"
                f"  the shards by hand, or this map and the one you are comparing it\n"
                f"  against will have been produced by different reducers.")
        sys.exit(f"missing head map: {path}\n  (no shards in {d} either -- has the "
                 f"localization run?)")
    import torch
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        for k in ("numerator_1", "effects", "heads"):
            if k in t:
                t = t[k]
                break
        else:
            sys.exit(f"{path}: dict without a recognised key; got {sorted(t)}")
    a = t.detach().float().numpy() if hasattr(t, "detach") else np.asarray(t, float)
    # Shards carry a trailing item axis; the map is the mean over items.
    while a.ndim > 2:
        a = a.mean(axis=-1)
    if a.ndim != 2:
        sys.exit(f"{path}: expected [layers, heads] after reduction, got {a.shape}")
    return np.abs(a) if score == "abs" else a


def topk_overlap(a, b, k):
    """Shared members of each map's top k, with a chance baseline."""
    n = a.size
    ia = set(np.argsort(-a.ravel())[:k].tolist())
    ib = set(np.argsort(-b.ravel())[:k].tolist())
    shared = len(ia & ib)
    expected = k * k / n           # hypergeometric mean
    jaccard = shared / len(ia | ib) if (ia | ib) else 0.0
    return shared, expected, jaccard, ia, ib


def permutation_p(a, b, k, n_perm=10000, seed=0):
    """P(overlap >= observed) when one map's head labels are shuffled.

    Shuffling breaks any real correspondence while preserving both score
    distributions, so this is the right null for "these two maps agree more than
    two arbitrary maps of the same shape would".
    """
    rng = np.random.default_rng(seed)
    flat_a, flat_b = a.ravel(), b.ravel()
    obs = len(set(np.argsort(-flat_a)[:k].tolist())
              & set(np.argsort(-flat_b)[:k].tolist()))
    idx_b = np.argsort(-flat_b)[:k]
    n = flat_a.size
    top_a = set(np.argsort(-flat_a)[:k].tolist())
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)[:k]
        if len(top_a & set(perm.tolist())) >= obs:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1), len(idx_b)


def spearman(x, y):
    """Rank correlation without scipy."""
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first numerator_1_heads.pt")
    ap.add_argument("--b", required=True, help="second numerator_1_heads.pt")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--score", choices=["abs", "signed"], default="abs",
                    help="abs: are the same heads involved? "
                         "signed: are they pushed the same way? Default: abs")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100],
                    help="top-k sizes to report overlap at")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--out", default=None, help="directory for the CSV/JSON output")
    args = ap.parse_args()

    A = load_map(args.a, args.score)
    B = load_map(args.b, args.score)
    if A.shape != B.shape:
        sys.exit(f"shape mismatch: {args.label_a} {A.shape} vs {args.label_b} {B.shape}")
    n_layers, n_heads = A.shape
    print(f"{args.label_a}: {args.a}")
    print(f"{args.label_b}: {args.b}")
    print(f"shape: {n_layers} layers x {n_heads} heads = {A.size} heads  "
          f"(score={args.score})\n")

    rows = []
    print(f"{'k':>5} {'shared':>7} {'chance':>7} {'jaccard':>8} {'p':>8}")
    for k in args.ks:
        if k > A.size:
            continue
        shared, expected, jac, _, _ = topk_overlap(A, B, k)
        _, p, _ = permutation_p(A, B, k, args.n_perm)
        print(f"{k:>5} {shared:>7} {expected:>7.1f} {jac:>8.3f} {p:>8.4f}")
        rows.append({"k": k, "shared": shared, "chance": expected,
                     "jaccard": jac, "p": p})

    rho_head = spearman(A.ravel(), B.ravel())
    la, lb = A.sum(axis=1), B.sum(axis=1)
    rho_layer = spearman(la, lb)
    print(f"\nSpearman over all {A.size} heads : {rho_head:+.3f}")
    print(f"Spearman over {n_layers} layers     : {rho_layer:+.3f}")

    print(f"\n{'layer':>5} {args.label_a:>12} {args.label_b:>12}")
    for l in range(n_layers):
        print(f"{l:>5} {la[l]:>12.4g} {lb[l]:>12.4g}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "overlap.json"), "w") as f:
            json.dump({"a": args.a, "b": args.b, "score": args.score,
                       "shape": [n_layers, n_heads],
                       "topk": rows,
                       "spearman_head": rho_head,
                       "spearman_layer": rho_layer}, f, indent=2)
        import csv
        with open(os.path.join(args.out, "per_layer.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", args.label_a, args.label_b])
            for l in range(n_layers):
                w.writerow([l, float(la[l]), float(lb[l])])
        print(f"\nwrote {args.out}/overlap.json and per_layer.csv")

    print("\nLow overlap is the strong result (dissociation). High overlap is\n"
          "ambiguous until you have a same-surface-form control contrast to\n"
          "calibrate against -- see the module docstring.")


if __name__ == "__main__":
    sys.exit(main())
