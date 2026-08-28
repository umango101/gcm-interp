#!/usr/bin/env python3
"""Generate the instruction-hierarchy conflict datasets for one arm.

Three arms -- developer/user, system/user, system/developer -- built from one
module (hierarchy_common.py) so that the color pool, the QC gate, token-length
matching, the demo selection, the counterbalancing and the held-out split are
identical and only the Harmony role rendering differs. This is one script with
--arm rather than three near-identical files on purpose: three copies drift, and
the whole point of the comparison is that nothing except the role rendering
varies. Run it three times.

Same three-stage flow as the original generator:

  1. --mode candidates   emit candidate files over a large pool of color pairs
  2. qc_hierarchy_datasets.py   test every pair against the model
  3. --mode final --qc ...      emit the five real files from surviving pairs

Output filenames match the existing corpus exactly, so every downstream
consumer works per arm with no changes:

  dev-single-desired-all.jsonl     privileged preamble, final answer = privileged word
  dev-single-undesired-all.jsonl   privileged preamble, final answer = subordinate word
  user-single-desired-all.jsonl    subordinate preamble, final answer = subordinate word
  user-single-undesired-all.jsonl  subordinate preamble, final answer = privileged word
  dev-single-test.jsonl            privileged preamble, no final assistant turn

Read "dev" as "privileged" and "user" as "subordinate". Each record also carries
arm / privileged_role / subordinate_role explicitly.

    # stage 1, once per arm
    for ARM in devuser sysuser sysdev; do
      python make_hierarchy_datasets.py --mode candidates --arm $ARM \
        --out_dir data/gpt-oss-20b/hier-$ARM/candidates \
        --tokenizer openai/gpt-oss-20b
    done

    # stage 2
    python qc_hierarchy_datasets.py \
      --data_dir data/gpt-oss-20b/hier-devuser/candidates \
      --out data/gpt-oss-20b/hier-devuser/pair_qc.json

    # stage 3
    python make_hierarchy_datasets.py --mode final --arm devuser \
      --out_dir data/gpt-oss-20b/hier-devuser \
      --meta data/gpt-oss-20b/hier-devuser/candidates/candidate_meta.json \
      --qc data/gpt-oss-20b/hier-devuser/pair_qc.json \
      --tokenizer openai/gpt-oss-20b

QC IS PER ARM AND MUST STAY THAT WAY
------------------------------------
Do not reuse one arm's pair_qc.json for another. A pair survives only if the
model answers as expected under BOTH preambles, and that is exactly the quantity
the arm is manipulating -- sharing a gate would silently import one arm's
difficulty into another and make the arms non-comparable in the one respect
they are supposed to differ.

CROSS-ARM COMPARISONS
---------------------
The devuser arm has two leading messages (system block + developer instruction)
where the sys* arms have one, so prompts differ in length across arms. Within an
arm every contrast is a length-matched minimal pair, which is what the patching
code requires. Across arms, absolute token positions and any position-indexed
figure are not directly comparable; layer indices and head identities are.
"""

import os
import json
import argparse

import determinism
from harmony_canonical import (
    canonical_system_block, DEFAULT_DATE, DEFAULT_REASONING,
)
# NOTE: import the MODULE for ACTIVE_CONFLICT_CATEGORIES, never the name. A
# from-import binds the list object at import time, so set_active_categories()
# would rebind the module global and this file would still write the old one.
import hierarchy_common as H
from hierarchy_common import (
    ARMS, FORMS, COLOR_POOL, ALL_FILES, TEST_FILE, set_active_categories,
    N_CANDIDATE_PAIRS, N_LOC, N_TEST, N_CONFLICT_DEMOS, N_AGREE_DEMOS,
    build_pairs, select_demos, select_demos_rule, emit, pair_key, verify,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["candidates", "final"], required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--form", choices=FORMS, default="rule",
                    help="'rule': both sides state a standing rule and the user "
                         "asks neutral questions (primary; no copy cue at the "
                         "answer position). 'request': the privileged side "
                         "states a rule and the subordinate side makes a "
                         "per-item request (the original corpus).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reasoning", default=DEFAULT_REASONING,
                    choices=["low", "medium", "high"])
    ap.add_argument("--date", default=DEFAULT_DATE,
                    help="pinned 'Current date:' in the canonical system block. "
                         "Pinned, not today's date, so token counts are stable "
                         "across days. Use the same value for all three arms.")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--qc", default=None, help="pair_qc.json (required for final)")
    ap.add_argument("--meta", default=None,
                    help="candidate_meta.json from stage 1 (demos + pairs)")
    ap.add_argument("--n_candidate_pairs", type=int, default=N_CANDIDATE_PAIRS)
    ap.add_argument("--n_loc", type=int, default=N_LOC)
    ap.add_argument("--n_test", type=int, default=N_TEST)
    ap.add_argument("--n_conflict_demos", type=int, default=N_CONFLICT_DEMOS)
    ap.add_argument("--n_agree_demos", type=int, default=N_AGREE_DEMOS)
    args = ap.parse_args()

    arm = ARMS[args.arm]
    system_block = canonical_system_block(args.reasoning, args.date)
    if arm.neutral_user and args.form == "rule":
        raise SystemExit(
            f"arm {arm.key!r} exists to add a user turn to a conversation that "
            "has none, which is a request-form problem. The rule form already "
            "has user turns throughout. Use --arm sysdev --form rule.")
    print(f"arm '{arm.key}' / form '{args.form}': {arm.privileged} "
          f"(privileged) vs {arm.subordinate} (subordinate)")
    print(f"  {arm.describe(args.form)}")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = args.meta or os.path.join(args.out_dir, "candidate_meta.json")

    # ---- stage 1 -----------------------------------------------------------
    if args.mode == "candidates":
        demos = (select_demos(tok, args.n_conflict_demos, args.n_agree_demos)
                 if args.form == "request"
                 else select_demos_rule(tok, args.n_conflict_demos,
                                        args.n_agree_demos))
        pairs = build_pairs(tok, COLOR_POOL, args.n_candidate_pairs)
        need = args.n_loc + args.n_test
        print(f"{len(pairs)} candidate pairs (need {need} to survive QC)")
        if len(pairs) < need:
            raise SystemExit(f"only {len(pairs)} pairs buildable; enlarge COLOR_POOL")
        # Only the two -desired files are needed for QC: the checks ask whether
        # the model produces the expected answer under each preamble.
        for fname, cond, which, inc in (ALL_FILES[0], ALL_FILES[2]):
            n = emit(os.path.join(args.out_dir, fname), arm, args.form,
                     system_block, pairs, cond, which, inc, demos)
            print(f"  wrote {n:>4} candidate lines -> {fname}")
        with open(meta_path, "w") as f:
            json.dump({"arm": arm.key,
                       "form": args.form,
                       # Restored in stage 3: the rules and the questions must
                       # be built from the same category list QC ran against.
                       "categories": H.ACTIVE_CONFLICT_CATEGORIES,
                       "system_block": system_block,
                       "reasoning": args.reasoning,
                       "date": args.date,
                       "pairs": [pair_key(a, b) for a, b in pairs],
                       "demos": demos}, f, indent=2)
        print(f"  wrote {meta_path}")
        return

    # ---- stage 3 -----------------------------------------------------------
    if not args.qc:
        raise SystemExit("--mode final requires --qc pair_qc.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing {meta_path}; --meta must point at stage 1's output")

    with open(meta_path) as f:
        meta = json.load(f)
    # The demos and the block must be the ones QC ran against, or the pass/fail
    # results do not transfer to these files.
    if meta.get("arm") != arm.key or meta.get("form") != args.form:
        raise SystemExit(
            f"{meta_path} was built for arm {meta.get('arm')!r} / form "
            f"{meta.get('form')!r}, not {arm.key!r} / {args.form!r}. Each "
            "arm-form cell needs its own stage 1 and its own QC.")
    if meta.get("system_block") != system_block:
        raise SystemExit(
            f"{meta_path} was built with a different system block "
            f"(reasoning={meta.get('reasoning')!r}, date={meta.get('date')!r}) "
            f"than this run (reasoning={args.reasoning!r}, date={args.date!r}). "
            "Rerun stage 1, or pass the original values.")
    demos = [tuple(d) for d in meta["demos"]]
    if meta.get("categories"):
        set_active_categories(meta["categories"])
        print(f"  restored {len(meta['categories'])} categories from stage 1")
    print(f"  reusing the {len(demos)} demos from stage 1")

    with open(args.qc) as f:
        qc = json.load(f)
    if qc.get("arm") not in (None, arm.key) or \
            qc.get("form") not in (None, args.form):
        raise SystemExit(
            f"{args.qc} is QC for arm {qc.get('arm')!r} / form "
            f"{qc.get('form')!r}, not {arm.key!r} / {args.form!r}. Sharing a QC "
            "gate across cells imports one cell's difficulty into another.")
    survivors = [tuple(k.split("|")) for k in qc["passing_pairs"]]

    need = args.n_loc + args.n_test
    print(f"{len(survivors)} pairs passed QC, "
          f"{len(qc.get('failing_pairs', {}))} dropped; need {need}")
    if len(survivors) < need:
        raise SystemExit(
            f"only {len(survivors)} pairs survived QC but {need} are needed.\n"
            "Pair survival is roughly (per-line rate)^4, so a modest line "
            "failure rate wipes out most pairs. Raise the per-line rate rather "
            "than the pool: increase --n_conflict_demos, or lower --n_loc / "
            "--n_test. If one arm survives far less than the others, that gap "
            "is itself a result -- report it, do not tune it away.")

    loc_pairs = survivors[:args.n_loc]
    test_pairs = survivors[args.n_loc:args.n_loc + args.n_test]

    for fname, cond, which, inc in ALL_FILES:
        n = emit(os.path.join(args.out_dir, fname), arm, args.form,
                 system_block, loc_pairs, cond, which, inc, demos)
        print(f"  wrote {n:>4} lines -> {fname}")
    fname, cond, which, inc = TEST_FILE
    n = emit(os.path.join(args.out_dir, fname), arm, args.form,
             system_block, test_pairs, cond, which, inc, demos)
    print(f"  wrote {n:>4} lines -> {fname}   (held-out colors)")

    with open(os.path.join(args.out_dir, "arm_manifest.json"), "w") as f:
        json.dump({"arm": arm.key,
                   "form": args.form,
                   "privileged_role": arm.privileged,
                   "subordinate_role": arm.subordinate,
                   "reasoning": args.reasoning,
                   "date": args.date,
                   "note": arm.note,
                   "n_loc_pairs": len(loc_pairs),
                   "n_test_pairs": len(test_pairs)}, f, indent=2)

    # The builder is deterministic, but the QC gate feeding it is not, so a
    # later rebuild can only be shown to match rather than assumed to.
    man_path = os.path.join(args.out_dir, "arm_manifest.json")
    with open(man_path) as f:
        _man = json.load(f)
    _man["sha256"] = determinism.hash_dir(args.out_dir)
    with open(man_path, "w") as f:
        json.dump(_man, f, indent=2)

    print("\nchecks:")
    verify(args.out_dir, arm, args.form, system_block, demos)


if __name__ == "__main__":
    main()
