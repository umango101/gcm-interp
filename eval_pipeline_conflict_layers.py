#!/usr/bin/env python3
"""Score the residual-stream (layer) sweep with the single-token conflict scorer.

    python eval_pipeline_conflict_layers.py --stages merge accuracies plots

WHY THIS NOW IMPORTS eval_pipeline_conflict_single
--------------------------------------------------
It used to reconfigure `eval_pipeline_conflict` (the judge-based scorer). That
was the right call when the layer arm's answers needed an LLM judge, but it is
now the reason the layer figures are stale: every metric fix -- `user_net`, the
flip-direction correction -- landed in `eval_pipeline_conflict_single`, so the
layer arm kept plotting the old metric set from a different module.

Importing the single scorer instead fixes three things at once:

  * The layer figures pick up `user_net` and the corrected flip denominator
    automatically, and will keep picking up future metric changes. There is one
    definition of each metric in the tree, not two.
  * The head arm and the layer arm are now scored by the SAME code on the same
    labels, which is the precondition for reading one against the other. Two
    scorers that agree today drift tomorrow.
  * No judge, so no GPU and no vLLM. The layer arm's scoring is now a few seconds
    of pandas on the login node. `scripts/judge_layers_user.sh` is obsolete --
    it existed only to give the judge stage an allocation.

WHAT IS OVERRIDDEN
------------------
Four module globals, all read by Cell.__init__, so every override lands before
all_cells():

  METHOD      'atp' -> 'atp-per-layer'   (LayerConfig.method_dir under per_layer)
  RESULTS_DIR ./results -> ./results_layers
  EVAL_ROOT / OUT_ROOT  -> *_layers, so the layer outputs cannot overwrite the
                           head arm's under the identical relative path
                           ({model}/{localization}/{method}/...). The method
                           segment differs, but relying on that to keep two arms
                           apart is one renamed constant away from silent
                           clobbering.

NOT overridden: NS and TOP_KS stay None, i.e. discovered from the gen filenames
on disk. In per-layer mode the third numeric field of the filename is a layer
index rather than a top-k fraction, and discovery handles that without being told
-- the regex reads it as `\\d+(\\.\\d+)?` either way. Pinning TOP_KS to
range(n_layers) would only create a way to be wrong about the layer count.
"""

import argparse
import os
import sys

import eval_pipeline_conflict_single as eps


def main():
    ap = argparse.ArgumentParser(
        description="Score the layer sweep via eval_pipeline_conflict_single's stages.")
    ap.add_argument("--stages", nargs="+",
                    choices=list(eps.PER_CELL_STAGES) + ["judge"],
                    default=["merge", "accuracies", "plots"],
                    help="Which stages to run, in order. build_prompts and judge "
                         "are no-ops in the single-token scorer.")
    ap.add_argument("--results_dir", default=None,
                    help="Root holding {model}/from_.../ for the layer gen files. "
                         "Default: <repo>/results_layers")
    ap.add_argument("--method", default="atp-per-layer",
                    help="Method directory segment. Default: atp-per-layer")
    ap.add_argument("--cells", nargs="+", type=int, default=None,
                    help="0-based cell indices to run. Default: all.")
    args = ap.parse_args()

    # Order matters: Cell freezes every path at construction from these globals.
    eps.METHOD = args.method
    eps.RESULTS_DIR = args.results_dir or os.path.join(eps.BASE_DIR, "results_layers")
    eps.EVAL_ROOT = os.path.join(eps.BASE_DIR, "eval_pipeline_conflict_single_layers")
    eps.OUT_ROOT = os.path.join(eps.BASE_DIR, "results_pipeline_conflict_single_layers")

    cells = eps.all_cells()
    if args.cells is not None:
        bad = [i for i in args.cells if i < 0 or i >= len(cells)]
        if bad:
            raise IndexError(f"--cells {bad} out of range (have {len(cells)})")
        cells = [cells[i] for i in args.cells]

    print(f"Layer arm: user->dev instruction privilege  "
          f"model={eps.MODEL_ID}  method={eps.METHOD}")
    print(f"  gen input : {eps.RESULTS_DIR}")
    print(f"  test data : {eps.DATA_DIR}")
    print(f"  intermed. : {eps.EVAL_ROOT}")
    print(f"  outputs   : {eps.OUT_ROOT}")
    print(f"  metrics   : {eps.PLOT_METRICS}")
    for c in cells:
        print(f"  - {c}")
        print(f"    gen dir : {c.gen_dir}")

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


if __name__ == "__main__":
    sys.exit(main())
