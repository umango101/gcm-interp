#!/usr/bin/env python3
"""Score the layer-matched RANDOM arm with the same scorer as the targeted arm.

    python eval_pipeline_conflict_single_random.py --stages merge accuracies plots

Requires apply_reps_param.py to have been applied first.

Three module globals differ from the targeted run, and nothing else does:

  REPS        'targeted' -> 'random'   (eval_runner's reps_type, in every filename)
  METHOD      'atp'      -> 'random'   (config.set_output_prefix uses patch_algo,
                                        so the random arm writes under random/)
  EVAL_ROOT / OUT_ROOT -> *_random     so the two arms' intermediates and results
                                       cannot overwrite each other

Every metric, label and plotting decision is inherited. That is the point: a
baseline scored by even slightly different code is not a baseline, and a later
change to user_net or broken_post would otherwise have to be mirrored by hand.
"""

import argparse
import os
import sys

import eval_pipeline_conflict_single as eps


def main():
    ap = argparse.ArgumentParser(
        description="Score the random-baseline arm (same scorer, REPS='random').")
    ap.add_argument("--stages", nargs="+",
                    choices=list(eps.PER_CELL_STAGES) + ["judge"],
                    default=["merge", "accuracies", "plots"])
    ap.add_argument("--results_dir", default=None,
                    help="default: <repo>/results (where the random/ tree lives)")
    ap.add_argument("--method", default="random",
                    help="method directory segment; matches --patch_algo random")
    args = ap.parse_args()

    if not hasattr(eps, "REPS"):
        sys.exit("eval_pipeline_conflict_single.py has no REPS constant -- "
                 "run apply_reps_param.py first.")

    eps.REPS = "random"
    eps.METHOD = args.method
    if args.results_dir:
        eps.RESULTS_DIR = args.results_dir
    eps.EVAL_ROOT = os.path.join(eps.BASE_DIR, "eval_pipeline_conflict_single_random")
    eps.OUT_ROOT = os.path.join(eps.BASE_DIR, "results_pipeline_conflict_single_random")

    cells = eps.all_cells()
    print(f"Random baseline arm  model={eps.MODEL_ID}  method={eps.METHOD}  "
          f"reps={eps.REPS}")
    print(f"  gen input : {eps.RESULTS_DIR}")
    print(f"  outputs   : {eps.OUT_ROOT}")
    for c in cells:
        print(f"  - {c}")
        print(f"    gen dir : {c.gen_dir}")

    for stage in args.stages:
        print("=" * 70)
        print(f"STAGE: {stage}")
        fn = eps.PER_CELL_STAGES[stage]
        for cell in cells:
            fn(cell)

    print("=" * 70)
    print("Done. Compare against the targeted arm at MATCHED (N, top_k) cells -- "
          "the grids are identical by construction, so a cell-by-cell difference "
          "is the effect of ATP's ranking with depth held fixed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
