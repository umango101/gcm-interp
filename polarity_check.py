#!/usr/bin/env python3
"""Polarity check: is a negative signed correlation opposite polarity or independence?

    python polarity_check.py --results results --model gpt-oss-20b

The sign of an ATP map depends on which condition was labelled `source` and which
`base` -- an arbitrary per-corpus choice. So a signed comparison between two maps
built from different corpora is only meaningful once the polarities are aligned,
and a STRONG NEGATIVE correlation means the maps are anti-aligned, NOT that they
are unrelated. Independence looks like rho ~ 0, not rho = -0.29.

This reports every signed comparison under both alignments, plus abs (which is
polarity-invariant, and is printed as the anchor). The number to read is the
FLOOR: whatever the induction control produces under its best alignment is the
overlap two contrasts of matched surface form generate for generic reasons. A
position result is only elevated if it exceeds that.

Imports compare_head_maps rather than reimplementing the statistics, so the
numbers here and in the figure cannot diverge.
"""

import argparse
import importlib.util
import os
import sys


def load_module(path="compare_head_maps.py"):
    if not os.path.exists(path):
        sys.exit(f"{path} not found; run from the repo root")
    spec = importlib.util.spec_from_file_location("chm", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--algo", default="atp")
    ap.add_argument("--privilege", default="from_user-single_to_dev-single")
    ap.add_argument("--controls", nargs="+",
                    default=["position=from_first-single_to_second-single",
                             "induction=from_induction-single_to_noinduction-single"],
                    help="label=dirname pairs")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()

    chm = load_module()

    def path_for(contrast):
        return os.path.join(args.results, args.model, contrast, args.algo,
                            "numerator_1_heads.pt")

    A_abs = chm.load_map(path_for(args.privilege), "abs")
    A_sgn = chm.load_map(path_for(args.privilege), "signed")
    n = A_abs.size
    print(f"privilege: {path_for(args.privilege)}")
    print(f"shape: {A_abs.shape[0]} layers x {A_abs.shape[1]} heads = {n} heads\n")

    hdr = (f"{'control':<12} {'score':<18} "
           + " ".join(f"k={k:<5}" for k in args.ks)
           + f" {'rho_head':>9} {'rho_layer':>10}")
    print(hdr)
    print("-" * len(hdr))

    floor = {}
    for spec in args.controls:
        label, contrast = spec.split("=", 1)
        p = path_for(contrast)
        if not os.path.exists(p):
            print(f"{label:<12} MISSING {p}")
            continue

        B_abs = chm.load_map(p, "abs")
        B_sgn = chm.load_map(p, "signed")

        rows = [
            ("abs (invariant)", A_abs, B_abs),
            ("signed as-built", A_sgn, B_sgn),
            ("signed flipped", A_sgn, -B_sgn),
        ]
        best = None
        for name, A, B in rows:
            shared = [chm.topk_overlap(A, B, k)[0] for k in args.ks]
            rh = chm.spearman(A.ravel(), B.ravel())
            rl = chm.spearman(A.sum(axis=1), B.sum(axis=1))
            print(f"{label:<12} {name:<18} "
                  + " ".join(f"{s:<7}" for s in shared)
                  + f" {rh:>+9.3f} {rl:>+10.3f}")
            if name.startswith("signed"):
                cand = (shared[args.ks.index(args.k)], name, rh)
                if best is None or cand[0] > best[0]:
                    best = cand
        chance = args.k * args.k / n
        print(f"{'':<12} {'-> best signed':<18} {best[1]} "
              f"({best[0]}/{args.k}, chance {chance:.1f})\n")
        floor[label] = best[0]

    if "induction" in floor:
        f = floor["induction"]
        print(f"FLOOR (induction, best alignment): {f}/{args.k}")
        for label, v in floor.items():
            if label == "induction":
                continue
            verdict = ("NOT elevated above the floor" if v <= f * 1.25
                       else "ELEVATED above the floor")
            print(f"  {label}: {v}/{args.k} -> {verdict}")
        print("\nThe 1.25x threshold is a reporting convention, not a test. State the\n"
              "two numbers and let the reader judge; do not report the verdict alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
