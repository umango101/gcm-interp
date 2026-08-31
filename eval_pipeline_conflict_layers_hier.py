#!/usr/bin/env python3
"""Score the residual-stream (layer) sweep with the single-token conflict scorer.

    python eval_pipeline_conflict_layers_hier.py
    python eval_pipeline_conflict_layers_hier.py --list
    python eval_pipeline_conflict_layers_hier.py --arms sysdev --eval_tests dev-single-test

WHY THIS IMPORTS eval_pipeline_conflict_single_hier
---------------------------------------------------
Same reason it imported the single scorer before: one definition of each metric
in the tree, and the head arm and layer arm scored by identical code on
identical labels. That is the precondition for reading one against the other,
and two scorers that agree today drift tomorrow.

WHAT CHANGED FOR THE HIERARCHY CORPORA
--------------------------------------
The old wrapper set eps.METHOD, a single string, because there was one method
directory and one cell. Cells are now DISCOVERED, and `method` is one of the
axes discovery walks -- so the override became a FILTER (eps.METHODS) rather
than a constant. Everything else about the redirect is unchanged.

The layer tree carries the same four axes as the head tree: arm, method, eval
test file, and steering arm. It has no random baseline and no ablation, so those
two axes are singletons here, but the filters exist and behave identically.

WHAT IS OVERRIDDEN
------------------
  METHODS     -> ['atp-per-layer']  (LayerConfig.method_dir under per_layer)
  RESULTS_DIR -> ./results_layers
  EVAL_ROOT / OUT_ROOT -> *_layers, so layer outputs cannot overwrite the head
                          arm's under the identical relative path. The method
                          segment differs, but relying on that to keep two arms
                          apart is one renamed constant away from silent
                          clobbering.

NOT overridden: NS and TOP_KS stay None, i.e. discovered from the filenames. In
per-layer mode the third numeric field is a LAYER INDEX rather than a top-k
fraction, and discovery reads it either way. Pinning TOP_KS to range(n_layers)
would only create a way to be wrong about the layer count.

READING THE OUTPUT. The `topk` axis of every layer-arm plot is a layer index.
The heatmaps are therefore layer x coefficient, not top-k x coefficient, and the
head-arm and layer-arm figures share metrics and styling but not that axis.
"""

import argparse
import os
import sys

import eval_pipeline_conflict_single_hier as eps


def main():
    ap = argparse.ArgumentParser(
        description="Score the layer sweep via the single-token scorer's stages.")
    ap.add_argument("--stages", nargs="+",
                    choices=list(eps.PER_CELL_STAGES) + ["judge"],
                    default=["merge", "accuracies", "plots"],
                    help="Which stages to run, in order. build_prompts and judge "
                         "are no-ops in the single-token scorer.")
    ap.add_argument("--results_dir", default=None,
                    help="Root holding {model}/{arm}__from_.../ for the layer gen "
                         "files. Default: <repo>/results_layers")
    ap.add_argument("--method", default="atp-per-layer",
                    help="Method directory segment. Default: atp-per-layer")
    ap.add_argument("--cells", nargs="+", type=int, default=None,
                    help="0-based cell indices to run. Default: all.")
    # Same filters as the head scorer, so one habit covers both arms.
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--eval_tests", nargs="+", default=None,
                    help="dev-single-test devNaive-single-test")
    ap.add_argument("--steer_arms", nargs="+", default=None)
    ap.add_argument("--reps", nargs="+", default=None)
    ap.add_argument("--ablations", nargs="+", default=None,
                    choices=["steer", "mean"])
    ap.add_argument("--list", action="store_true",
                    help="print the discovered cells and exit")
    args = ap.parse_args()

    # Order matters: Cell freezes every path at construction from these globals.
    # METHODS is a filter, not a constant -- method is an axis discovery walks.
    eps.METHODS = [args.method]
    eps.ARMS = args.arms or []
    eps.EVAL_TESTS = args.eval_tests or []
    eps.STEER_ARMS = args.steer_arms or []
    eps.REPS_FILTER = args.reps or []
    eps.STEER_METHODS = args.ablations or []
    eps.RESULTS_DIR = args.results_dir or os.path.join(eps.BASE_DIR, "results_layers")
    eps.EVAL_ROOT = os.path.join(eps.BASE_DIR, "eval_pipeline_conflict_single_layers")
    eps.OUT_ROOT = os.path.join(eps.BASE_DIR, "results_pipeline_conflict_single_layers")

    cells = eps.all_cells()
    if args.cells is not None:
        bad = [i for i in args.cells if i < 0 or i >= len(cells)]
        if bad:
            raise IndexError(f"--cells {bad} out of range (have {len(cells)})")
        cells = [cells[i] for i in args.cells]

    if args.list:
        for i, c in enumerate(cells):
            n = len([f for f in os.listdir(c.gen_dir) if c.gen_re.match(f)])
            print(f"  [{i:>3}] {c}   ({n} gen files)")
        print(f"\n{len(cells)} cells")
        return 0

    print(f"Layer arm: instruction privilege  model={eps.MODEL_ID}  "
          f"method={args.method}")
    print(f"  gen input : {eps.RESULTS_DIR}")
    print(f"  test data : {eps.DATA_DIR}")
    print(f"  intermed. : {eps.EVAL_ROOT}")
    print(f"  outputs   : {eps.OUT_ROOT}")
    print(f"  metrics   : {eps.PLOT_METRICS}")
    for c in cells:
        print(f"  - {c}")

    for stage in args.stages:
        print("=" * 70)
        print(f"STAGE: {stage}")
        if stage == "judge":
            eps.stage_judge_all(cells)
        else:
            fn = eps.PER_CELL_STAGES[stage]
            for cell in cells:
                print(f"  cell: {cell}")
                fn(cell)

    print("=" * 70)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
