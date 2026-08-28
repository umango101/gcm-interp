#!/usr/bin/env python3
"""Make the reps token ('targeted' / 'random') configurable in the single scorer.

    python apply_reps_param.py --check
    python apply_reps_param.py

WHY
---
eval_runner names gen files by `reps_type`, which is 'targeted' for an ATP arm
and 'random' for a random-baseline arm:

    10_targeted_steer_0.05_user-single_gen.json
    10_random_steer_0.05_user-single_gen.json

eval_pipeline_conflict_single.py hardcodes `targeted` in three places -- the gen
filename regex, the accuracy filename, and the REPS column -- so a random arm is
invisible to it and stage_merge raises "No gen files matching ...". Since the
whole point of the baseline is to score the two arms IDENTICALLY, forking the
scorer would defeat it: any later fix to the metrics would have to be applied
twice, and a divergence between the two copies would look like a result.

This lifts the token to a module-level REPS constant read at Cell construction,
so eval_pipeline_conflict_single_random.py can set REPS='random' and reuse every
line of the scoring logic unchanged.
"""

import argparse
import sys
from pathlib import Path

PATH = "eval_pipeline_conflict_single.py"

EDITS = [
    ("REPS constant beside STEER_METHOD",
     'STEER_METHOD = "steer"  # part of the gen filename and of the accuracy filename',
     'STEER_METHOD = "steer"  # part of the gen filename and of the accuracy filename\n'
     '# eval_runner writes "targeted" for an ATP arm and "random" for a random\n'
     '# baseline arm. Overridden by the random wrapper; everything downstream --\n'
     '# metrics, labels, plots -- is shared so the two arms are scored identically.\n'
     'REPS = "targeted"'),

    ("gen filename regex uses REPS",
     '            r"^(?P<N>\\d+)_targeted_(?P<STEERING_METHOD>steer|mean)_"',
     '            rf"^(?P<N>\\d+)_{REPS}_(?P<STEERING_METHOD>steer|mean)_"'),

    ("REPS column follows the constant",
     '                "REPS": "targeted",',
     '                "REPS": REPS,'),

    ("accuracy filename uses REPS",
     '    return f"{n}_targeted_{STEER_METHOD}_topk_{top_k}_gen_accuracy_{name}.json.accuracy.json"',
     '    return f"{n}_{REPS}_{STEER_METHOD}_topk_{top_k}_gen_accuracy_{name}.json.accuracy.json"'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    p = Path(args.root).resolve() / PATH
    if not p.exists():
        print(f"MISSING  {PATH} not found under {args.root}")
        return 1

    t = p.read_text()
    pending = already = failed = 0
    for desc, old, new in EDITS:
        if new in t:
            print(f"SKIP     {desc} (already applied)")
            already += 1
            continue
        n = t.count(old)
        if n != 1:
            print(f"FAIL     {desc} -- anchor matched {n} times, expected 1")
            failed += 1
            continue
        t = t.replace(old, new, 1)
        print(f"{'WOULD' if args.check else 'APPLY':<8} {desc}")
        pending += 1

    if failed:
        print(f"\n{failed} hunk(s) failed -- nothing written.")
        return 1
    if args.check:
        print(f"\ncheck: {pending} to apply, {already} already applied")
        return 0
    p.write_text(t)
    print(f"\ndone: {pending} applied, {already} already applied")
    print("\nThe targeted arm is unaffected (REPS defaults to 'targeted').")
    print("Score the random arm with eval_pipeline_conflict_single_random.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
