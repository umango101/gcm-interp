#!/usr/bin/env python3
"""Add a `user_net` = user_post - broken_post panel to eval_pipeline_conflict_single.py.

    python apply_user_net.py --check
    python apply_user_net.py

Idempotent. After applying, rerun the two cheap stages (no GPU, no regeneration):

    python eval_pipeline_conflict_single.py --stages accuracies plots

WHAT IT ADDS
------------
`user_net` = user_post - broken_post, i.e. (n_user - n_other) / n_rows. It answers
the question the raw user_post panel cannot: at the dark band around layers 13-18,
did steering induce the user's answer, or did it break generation into emitting a
word that happens to be in the prompt? A cell that is dark in user_post and dark in
broken_post nets out near zero.

WHY NOT JUST REDEFINE `flip`
----------------------------
Two reasons, and both are about not losing information you already have:

  * `flip` is a RATE over a conditional denominator (items that could move).
    user_net is a SIGNED score over all items. Reusing the name would make the
    layer arm's flip and the head arm's flip mean different things while both
    are called flip, which defeats the cross-arm comparability this file was
    written for.
  * `flip` stays in ACCURACY_METRICS and in summary.json, so it costs nothing to
    keep. It is only dropped from PLOT_METRICS, where it is currently degenerate:
    with an unsteered baseline of dev=50 / user=0 / other=0 it is identically
    equal to user_post, which is why the two heatmaps came out pixel-identical.
    That identity is a property of the baseline, not a bug, and it will come back
    the moment the baseline stops being saturated.

READ IT WITH THE COMPONENT PANELS, NOT INSTEAD OF THEM
------------------------------------------------------
user_net is lossy: 0.0 means "nothing happened" AND "0.3 induced, 0.3 broken".
PLOT_METRICS keeps user_post and broken_post alongside it so any interesting cell
can be decomposed. Do not report user_net on its own.

ALSO INCLUDED (say --no-flip-direction to skip)
-----------------------------------------------
`failed = orig != FLIP_TARGET` becomes `failed = orig == LABEL_DEVELOPER`, so
`flip` means "of the items the unsteered model answered with the DEVELOPER's word,
what fraction moved to the user's word" -- the direction you described. The old
form also counted an unsteered off-task response as a flip opportunity. On the
current data the two sets are identical (other=0), so this changes no number
today; it stops the denominator inflating on any future run whose baseline is not
clean.
"""

import argparse
import sys
from pathlib import Path

PATH = "eval_pipeline_conflict_single.py"

METRIC_OLD = """        "broken_post": float((post == LABEL_NEITHER).sum() / total),
        # headline: intervention efficacy"""

METRIC_NEW = """        "broken_post": float((post == LABEL_NEITHER).sum() / total),
        # Net effect: user answers induced, minus generations broken. Signed, in
        # [-1, 1], over ALL items rather than a conditional denominator. Separates
        # "steering produced the user's word" from "steering broke the model and a
        # prompt word fell out", which user_post alone cannot distinguish.
        # Lossy on purpose -- 0.0 covers both "nothing happened" and "as much
        # breakage as effect" -- so it is plotted next to its two components, never
        # instead of them.
        "user_net": float(((post == LABEL_USER).sum()
                           - (post == LABEL_NEITHER).sum()) / total),
        # headline: intervention efficacy"""

ACC_OLD = ('ACCURACY_METRICS = ["dev_post", "dev_orig", "user_post", "broken_post", '
           '"flip", "flip_broken"]')
ACC_NEW = ('ACCURACY_METRICS = ["dev_post", "dev_orig", "user_post", "broken_post", '
           '"user_net", "flip", "flip_broken"]')

PLOT_OLD = 'PLOT_METRICS = ["flip", "dev_post", "user_post", "broken_post"]'
PLOT_NEW = ('# flip is omitted here, not removed: it stays in ACCURACY_METRICS and\n'
            '# summary.json. With a saturated baseline (dev=50/user=0/other=0) it is\n'
            '# identically equal to user_post, so plotting both produced two identical\n'
            '# figures. user_net is the panel that adds information.\n'
            'PLOT_METRICS = ["user_net", "dev_post", "user_post", "broken_post"]')

HEAT_OLD = """    cmap = "Blues" if metric == "broken_post" else "Reds"
    ax = sns.heatmap(df, annot=True, vmin=0, vmax=1, cmap=cmap, fmt=".2f")"""

HEAT_NEW = """    if metric == "user_net":
        # Signed metric: a diverging map centred on zero, or negative cells clip to
        # white on a 0-1 Reds scale and become indistinguishable from "no effect" --
        # exactly the cells where breakage outweighed the induced answer.
        cmap, vmin, vmax, center = "RdBu_r", -1.0, 1.0, 0.0
    else:
        # Blues for broken_post so a failure mode never reads as a success at a
        # glance -- the layer arm's convention, kept so the two sets of figures can
        # sit side by side in the writeup.
        cmap = "Blues" if metric == "broken_post" else "Reds"
        vmin, vmax, center = 0.0, 1.0, None
    ax = sns.heatmap(df, annot=True, vmin=vmin, vmax=vmax, center=center,
                     cmap=cmap, fmt=".2f")"""

FLIP_OLD = """    failed = orig != FLIP_TARGET
    n_failed = int(failed.sum())"""

FLIP_NEW = """    # `== LABEL_DEVELOPER`, not `!= FLIP_TARGET`: flip means "moved from the
    # developer's word to the user's word". The looser form also counted an
    # unsteered off-task response as an opportunity. Identical on a clean
    # baseline; keeps the denominator honest when the baseline is not clean.
    failed = orig == LABEL_DEVELOPER
    n_failed = int(failed.sum())"""

CORE = [
    ("_metrics: add user_net", METRIC_OLD, METRIC_NEW),
    ("ACCURACY_METRICS: include user_net", ACC_OLD, ACC_NEW),
    ("PLOT_METRICS: user_net replaces the degenerate flip panel", PLOT_OLD, PLOT_NEW),
    ("_heatmap: diverging scale for the signed metric", HEAT_OLD, HEAT_NEW),
]
FLIP_EDIT = ("_metrics: flip denominator = items that followed the developer",
             FLIP_OLD, FLIP_NEW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--no-flip-direction", action="store_true",
                    help="skip the flip-denominator change; add user_net only")
    args = ap.parse_args()

    path = Path(args.root).resolve() / PATH
    if not path.exists():
        print(f"MISSING  {PATH} not found under {args.root}")
        return 1

    edits = list(CORE) + ([] if args.no_flip_direction else [FLIP_EDIT])
    text = path.read_text()
    pending = already = failed = 0

    for desc, old, new in edits:
        if new in text:
            print(f"SKIP     {desc} (already applied)")
            already += 1
            continue
        n = text.count(old)
        if n != 1:
            print(f"FAIL     {desc} -- anchor matched {n} times, expected 1")
            failed += 1
            continue
        text = text.replace(old, new, 1)
        print(f"{'WOULD' if args.check else 'APPLY':<8} {desc}")
        pending += 1

    if failed:
        print(f"\n{failed} hunk(s) failed -- nothing written.")
        return 1
    if args.check:
        print(f"\ncheck: {pending} to apply, {already} already applied")
        return 0

    path.write_text(text)
    print(f"\ndone: {pending} applied, {already} already applied")
    if pending:
        print("\nRerun the cheap stages (no GPU, no regeneration):")
        print("  python eval_pipeline_conflict_single.py --stages accuracies plots")
        print("Existing accuracy JSONs predate user_net, so `accuracies` must rerun")
        print("before `plots`, or _load_acc_cell raises on the missing key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
