#!/usr/bin/env python3
"""Bootstrap CIs on the headline rates, from the per-item merged CSVs.

    python bootstrap_cis.py --cells "*devuser*targeted_steer*" --n 10000
    python bootstrap_cis.py --ablations   # the necessity cells only

CPU only, seconds per cell.

WHY THIS IS NOT OPTIONAL
------------------------
Every rate in the results is over ~100 items, where the 95% interval is roughly
+-9 points. Several comparisons already in hand are smaller than that:

  * cross-arm steering, 0.97 vs 0.89 on the diagonal -- almost certainly not a
    real difference, and reading it as one would support a claim about
    arm-specific mechanisms that the data does not carry;
  * the ablation gaps, +0.26 to +0.47 -- comfortably outside, and saying so
    with an interval is stronger than saying it with a bare number;
  * the naive-transfer non-effects, 0.59 vs 0.57 -- an interval makes "no
    detectable difference" a statement rather than an absence.

BOOTSTRAP OVER ITEMS, NOT OVER ROWS
-----------------------------------
Resampling is by PAIR where a pair key is present. The four counterbalanced
variants of one colour pair share a colour and are not independent draws, so
resampling rows treats 100 correlated observations as 100 independent ones and
returns an interval that is too narrow. Pair-level resampling is the honest unit.
"""

import os
import csv
import fnmatch
import glob
import json
import random
import argparse
import statistics
from collections import defaultdict


def load_cell(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rates(rows, dev_key="post_choice"):
    n = len(rows)
    if not n:
        return {}
    dev = sum(r[dev_key] == "dev" for r in rows) / n
    user = sum(r[dev_key] == "user" for r in rows) / n
    other = sum(r[dev_key] not in ("dev", "user") for r in rows) / n
    return {"dev_post": dev, "user_post": user, "broken_post": other,
            "user_net": user - other, "n": n}


def bootstrap(rows, n_boot, seed, key):
    """Percentile CIs, resampling whole pairs."""
    rng = random.Random(seed)
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r.get("pair_key") or id(r)].append(r)
    pairs = list(by_pair.values())

    out = defaultdict(list)
    for _ in range(n_boot):
        draw = []
        for _ in range(len(pairs)):
            draw.extend(rng.choice(pairs))
        for k, v in rates(draw, key).items():
            if k != "n":
                out[k].append(v)
    ci = {}
    for k, vs in out.items():
        vs.sort()
        ci[k] = (vs[int(0.025 * len(vs))], vs[int(0.975 * len(vs))])
    return ci, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_pipeline_conflict_single")
    ap.add_argument("--cells", default="*",
                    help="glob against the cell path, e.g. '*targeted_mean*'")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", default="post_choice")
    ap.add_argument("--metrics", nargs="+",
                    default=["dev_post", "user_net", "broken_post"])
    ap.add_argument("--out", default="bootstrap_cis.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(
        args.root, "**", "merged_eval_outputs.csv"), recursive=True))
    if args.cells != "*":
        paths = [p for p in paths if fnmatch.fnmatch(p, f"*{args.cells}*")]
    if not paths:
        raise SystemExit(f"no merged_eval_outputs.csv under {args.root} "
                         f"matching {args.cells!r}")

    print(f"{len(paths)} cell(s), {args.n} bootstrap draws, resampled by pair\n")
    results = {}
    for p in paths:
        rows = load_cell(p)
        if not rows:
            continue
        # The merged CSV holds every (N, topk) grid point; CIs are per point.
        by_point = defaultdict(list)
        for r in rows:
            by_point[(r.get("N"), r.get("topk"))].append(r)

        cell = os.path.relpath(os.path.dirname(p), args.root)
        print(f"== {cell}")
        for (n_val, topk), sub in sorted(by_point.items()):
            pt = rates(sub, args.key)
            ci, n_pairs = bootstrap(sub, args.n, args.seed, args.key)
            bits = "  ".join(
                f"{m}={pt[m]:.2f} [{ci[m][0]:.2f},{ci[m][1]:.2f}]"
                for m in args.metrics if m in pt)
            print(f"   N={n_val:<4} topk={topk:<6} n={pt['n']:>4} "
                  f"pairs={n_pairs:>3}  {bits}")
            results[f"{cell}|N{n_val}|topk{topk}"] = {
                "point": pt, "ci": {k: list(v) for k, v in ci.items()},
                "n_pairs": n_pairs}
        print()

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"wrote {args.out}")
    print("\nTwo differences are only real if their intervals do not overlap --\n"
          "and overlapping intervals do not prove equality either, they just\n"
          "mean this sample cannot tell. Say which one you mean.")


if __name__ == "__main__":
    main()
