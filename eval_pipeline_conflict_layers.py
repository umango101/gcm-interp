#!/usr/bin/env python3
"""Score the user-single per-layer sweep with the existing conflict scorer.

    python eval_pipeline_conflict_layers.py --results_dir ./results_layers

``eval_pipeline_conflict.py`` is imported and reconfigured rather than edited or
copied. Edited, and the head arm's scoring changes underneath it; copied, and the
two drift -- the judge prompt, the Harmony ``final``-channel extraction, and the
three-way developer/user/neither labelling are exactly the parts that must NOT
diverge from the head arm, because the point is to read one against the other.

What differs, and nothing here is in the scoring logic:

1. PAIRS gains a ``user`` entry (user-single -> dev-single). The stock table only
   knows the role/within/inverted corpora.
2. EXPERIMENT_SPECS becomes a single cell, named "" so that
   ``os.path.join(name, MODEL_ID, ...)`` collapses to ``{MODEL_ID}/...``. The
   stock nine conditions each own an ``{experiment}/`` directory level because
   they would otherwise overwrite each other; with one condition there is
   nothing to keep apart, so the layer run writes straight into ./results_layers
   and no ``--localization_root`` split is needed.
3. METHOD -> 'atp-per-layer' (LayerConfig.method_dir under --sweep_mode per_layer).
4. TOP_KS -> layer indices; NS -> whatever --n_vals the sweep used.

The gen filename regex needs no change. It parses
``{N}_targeted_{steer|mean}_{topk}_{eval_source}_gen.json`` with ``topk`` as
``\\d+(\\.\\d+)?``; in per-layer mode that third field holds a layer index
instead of a count, an integer either way.

The scorer already handles the extended-preamble corpus without modification:
``load_test_prompts`` accepts a ``system`` turn as the privileged instruction (the
new rows use ``system`` where the role/within rows used ``developer``) and takes
the LAST user turn, so it picks the final question rather than an ICL demo.
"""

import argparse
import sys

import eval_pipeline_conflict as epc


# (name, localization key, steering key, eval key). One cell: localize
# user-single -> dev-single and steer on the same corpus. name="" keeps the
# tree flat -- see the module docstring.
LAYER_EXPERIMENT_SPECS = [("", "user", "user", "user")]


def main():
    ap = argparse.ArgumentParser(
        description="Score the user-single layer sweep via eval_pipeline_conflict's stages.")
    ap.add_argument("--stages", nargs="+",
                    default=["merge", "prompts", "judge", "accuracies", "plots"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--results_dir", default="./results_layers",
                    help="Root of the layer results tree (holds {model}/from_.../).")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n_layers", type=int, default=24,
                    help="Layer count for the model. Read it off the row count of "
                         "layer_effects.csv rather than trusting this default -- a wrong "
                         "value only produces a grid warning, not wrong numbers.")
    ap.add_argument("--n_vals", type=str, default="2,5,8,10",
                    help="Must match --n_vals from the sweep, or _check_grid reports "
                         "the missing cells as gaps.")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    # Order matters: Cell freezes its paths at construction and reads these
    # module globals, so every override has to land before all_cells().
    epc.set_roots(args.repo_root or epc.BASE_DIR, args.results_dir, args.data_dir)
    epc.PAIRS = dict(epc.PAIRS, user=("user-single", "dev-single"))
    epc.PRIMARY_OF = dict(epc.PRIMARY_OF, user="user")
    epc.METHOD = "atp-per-layer"
    epc.TOP_KS = list(range(args.n_layers))
    epc.NS = [int(x) for x in args.n_vals.replace(",", " ").split()]
    epc.EXPERIMENT_SPECS = LAYER_EXPERIMENT_SPECS

    cells = epc.all_cells()

    print(f"Layer arm: user-single -> dev-single  model={epc.MODEL_ID} method={epc.METHOD}")
    print(f"  gen input : {epc.RESULTS_DIR}")
    print(f"  test data : {epc.DATA_DIR}")
    print(f"  grid      : NS={epc.NS} layers=0..{args.n_layers - 1}")
    for c in cells:
        print(f"  - {c}")
        print(f"    gen dir : {c.gen_dir}")

    for stage in args.stages:
        print("=" * 70)
        print(f"STAGE: {stage}")
        if stage == "judge":
            epc.stage_judge_all(cells, batch_size=args.batch_size,
                                resume=not args.no_resume)
        else:
            fn = epc.PER_CELL_STAGES[stage]
            for cell in cells:
                fn(cell)

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
