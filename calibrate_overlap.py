#!/usr/bin/env python3
"""Calibrate the privilege/induction overlap against null distributions.

    python calibrate_overlap.py --induction induction_heads.json

WHY A RAW OVERLAP NUMBER MEANS NOTHING
--------------------------------------
"2 of 15 shared" and "median induction rank 294 of 1536" both sound
interpretable and neither is, because the privilege heads are not a random
sample of head space. They sit in layers 11-23, and induction heads cluster in
mid-to-late layers too. Any set drawn from those layers looks elevated on
induction whether or not it has anything to do with copying.

So the null is not chance. It is a LAYER-MATCHED draw: the same number of heads
from each layer the privilege set used. The gap between the layer-matched null
and the uniform null is itself the depth contribution -- how much of the apparent
induction association is explained by where these heads are rather than which
they are.

Reported for each arm:

  observed     median induction rank of the privilege heads, and top-k overlap
  uniform      null from heads drawn anywhere
  matched      null from heads drawn from the privilege layer histogram
  p            fraction of matched draws at least as extreme as observed

A LOW p AGAINST THE MATCHED NULL is the only result that supports "these heads
are induction heads". A low p against uniform but not against matched supports
only "these heads are at induction-head depths", which is a claim about the
architecture, not about this circuit.

THE CEILING IS STILL MISSING
----------------------------
This gives the floor. The other half is how much two maps of the SAME thing
agree -- split-half the privilege attribution map and measure its overlap with
itself. Without it, a middling overlap is unreadable: it could mean the two
constructs are half-shared, or that ATP maps simply do not reproduce well at
this sample size. See --help for how to produce it.
"""

import json
import glob
import random
import argparse
import statistics

import pandas as pd


def load_privilege(arm, results_dir, topk):
    pat = (f"{results_dir}/gpt-oss-20b/{arm}__*/atp/{arm}-dev-single-test_eval/"
           f"{arm}_steer/eval/numerator_1_targeted_{topk}.csv")
    fs = glob.glob(pat)
    if not fs:
        raise SystemExit(f"no privilege head set at {pat}")
    df = pd.read_csv(fs[0])
    heads = [(int(r.layer), int(r.neuron)) for r in df.itertuples()]
    hist = df.groupby("layer").size().to_dict()
    return heads, {int(k): int(v) for k, v in hist.items()}


def stats_for(heads, rank, top_set):
    ranks = [rank[h] for h in heads if h in rank]
    return {
        "median_rank": statistics.median(ranks) if ranks else float("nan"),
        "overlap": len(set(heads) & top_set),
    }


def draw_uniform(n, n_layers, n_heads, rng):
    return rng.sample([(l, h) for l in range(n_layers) for h in range(n_heads)], n)


def draw_matched(hist, n_heads, rng):
    out = []
    for layer, k in hist.items():
        out.extend((layer, h) for h in rng.sample(range(n_heads), min(k, n_heads)))
    return out


def main():
    ap = argparse.ArgumentParser(
        epilog="Ceiling: build two half-corpora from an arm's -all files (items "
               "0..n/2 and n/2..n), run ATP on each with --data_dir pointing at "
               "them, and compare the two head sets the same way. That is the "
               "most agreement two maps of the same construct can show.")
    ap.add_argument("--induction", default="induction_heads.json")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--arms", nargs="+",
                    default=["devuser", "sysuser", "sysdev"])
    ap.add_argument("--topk", default="0.01")
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ind = json.load(open(args.induction))
    ranked = ind["ranked"]
    rank = {(r["layer"], r["head"]): i for i, r in enumerate(ranked)}
    n_layers = len(ind["scores"])
    n_heads = len(next(iter(ind["scores"].values())))
    rng = random.Random(args.seed)

    print(f"induction map: {n_layers} layers x {n_heads} heads = {len(rank)} heads")
    print(f"nulls: {args.n_draws} draws each\n")

    hdr = (f"{'arm':<9}{'k':>4}{'obs med':>9}{'unif med':>10}{'match med':>11}"
           f"{'p(med)':>8}{'obs ovl':>9}{'match ovl':>11}{'p(ovl)':>8}")
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for arm in args.arms:
        heads, hist = load_privilege(arm, args.results_dir, args.topk)
        k = len(heads)
        top_set = {(r["layer"], r["head"]) for r in ranked[:k]}
        obs = stats_for(heads, rank, top_set)

        uni, mat = [], []
        for _ in range(args.n_draws):
            uni.append(stats_for(draw_uniform(k, n_layers, n_heads, rng),
                                 rank, top_set))
            mat.append(stats_for(draw_matched(hist, n_heads, rng),
                                 rank, top_set))

        um = statistics.median(d["median_rank"] for d in uni)
        mm = statistics.median(d["median_rank"] for d in mat)
        mo = statistics.mean(d["overlap"] for d in mat)
        # One-sided: how often does a matched draw look AT LEAST as
        # induction-like as the real set (lower rank / higher overlap)?
        p_med = sum(d["median_rank"] <= obs["median_rank"] for d in mat) / len(mat)
        p_ovl = sum(d["overlap"] >= obs["overlap"] for d in mat) / len(mat)

        print(f"{arm:<9}{k:>4}{obs['median_rank']:>9.0f}{um:>10.0f}{mm:>11.0f}"
              f"{p_med:>8.3f}{obs['overlap']:>9}{mo:>11.1f}{p_ovl:>8.3f}")
        out[arm] = {"k": k, "layer_hist": hist, "observed": obs,
                    "uniform_median": um, "matched_median": mm,
                    "matched_overlap_mean": mo,
                    "p_median_rank": p_med, "p_overlap": p_ovl}

    print("\nHOW TO READ\n"
          "  p is against the MATCHED null, the one that controls for depth.\n"
          "  p < 0.05 on both columns: the privilege heads really are unusually\n"
          "  induction-like for their layers. p large while the uniform median\n"
          "  is far above the matched median: the apparent association is depth,\n"
          "  and the honest sentence is that the two sets sit at similar depths\n"
          "  rather than that they overlap.\n"
          "  Neither case is readable without the ceiling -- see --help.")

    with open("overlap_calibration.json", "w") as f:
        json.dump({"args": vars(args), "results": out}, f, indent=2)
    print("\nwrote overlap_calibration.json")


if __name__ == "__main__":
    main()
