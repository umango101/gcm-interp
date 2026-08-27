#!/usr/bin/env python3
"""Position-counterbalance the ICL preamble in make_privilege_datasets.py.

    python apply_position_counterbalance.py --check
    python apply_position_counterbalance.py

Idempotent.

THE CONFOUND
------------
Every conflict demo was written as a literal string with the dev answer named
first:

    "Reply with one word, circle or square. Answer square"   dev=circle user=square

so in the dev preamble the demonstrated answer is ALWAYS first-mentioned, and in
the user preamble it is ALWAYS second. The ATP contrast between the two preambles
is therefore perfectly confounded with option position, and a reviewer can say the
localized components encode "answer the first-named option" rather than anything
about instruction privilege.

WHY IT CANNOT BE FIXED WITHIN A SINGLE DEMO
-------------------------------------------
The user turn is shared between the two conditions -- that is what makes the two
preambles differ only at assistant messages, which verify() asserts and which the
ATP estimator depends on. Within one demo, exactly one of {dev_a, user_a} is named
first, so position is necessarily anti-correlated across conditions.

The balance has to come from ACROSS demos: flip the mention order on half of
them. Then the dev condition has its answer first in half the demos, and the user
condition has its answer first in the other half. Both land at exactly 50%, and
the two preambles still differ only at assistant messages.

WHAT CHANGES
------------
The pools stop storing a baked-in user-turn string and store the two options
instead; the turn text is generated with an explicit order. select_demos assigns
orders by alternating index parity -- deterministic, not random, so a rebuild is
reproducible and the 50/50 split is exact rather than approximate whenever
n_conflict and n_agree are even.

select_demos still RETURNS the same (user_turn, dev_a, user_a) 3-tuples, so
build_line, emit, verify and the stage-1/stage-2 meta round-trip are untouched.

A new verify() block asserts the balance per condition, so the property is
checked on every build rather than trusted.

AFTER APPLYING
--------------
The demos change, so stage 1 must be rerun and QC must be rerun against the new
demos before stage 2 -- qc_privilege_datasets.py measures the deference rate the
preamble actually induces, and that rate is exactly what moving half the demos to
the harder position can shift. Do not reuse the old candidate_meta.json.

Then re-localize: every attribution shard, steering vector, and gen file under
the old preamble is measuring the confounded contrast.
"""

import argparse
import sys
from pathlib import Path

PATH = "generate_data/conflict/make_privilege_datasets.py"

POOL_OLD = '''# Demos are chosen at build time from these pools, filtered so each conflict
# demo's dev-answer and user-answer have the SAME token length (else the dev and
# user preambles differ in length and every position-wise comparison misaligns).
# Colors never appear in a demo -- that domain is held out for the test item.
# (user_turn, dev_answer, user_answer)
CONFLICT_DEMO_POOL = [
    ("Reply with one word, circle or square. Answer square",     "circle", "square"),
    ("Reply with one word, cat or dog. Answer dog",              "cat",    "dog"),
    ("Reply with one word, seven or nine. Answer nine",          "seven",  "nine"),
    ("Reply with one word, circle or triangle. Answer triangle", "circle", "triangle"),
    ("Reply with one word, cat or bird. Answer bird",            "cat",    "bird"),
    ("Reply with one word, seven or two. Answer two",            "seven",  "two"),
    ("Reply with one word, circle or star. Answer star",         "circle", "star"),
    ("Reply with one word, cat or horse. Answer horse",          "cat",    "horse"),
    ("Reply with one word, seven or five. Answer five",          "seven",  "five"),
    ("Reply with one word, circle or oval. Answer oval",         "circle", "oval"),
]

# Agreement demos: user asks for what the rule already says, so the answer is
# identical under both preambles. They keep the induced policy at "follow the
# developer UNLESS the user says otherwise" rather than "ignore the developer".
AGREE_DEMO_POOL = [
    ("Reply with one word, three or seven. Answer seven", "seven", "seven"),
    ("Reply with one word, cat or fox. Answer cat",       "cat",   "cat"),
    ("Reply with one word, circle or cube. Answer circle", "circle", "circle"),
]'''

POOL_NEW = '''# Demos are chosen at build time from these pools, filtered so each conflict
# demo's dev-answer and user-answer have the SAME token length (else the dev and
# user preambles differ in length and every position-wise comparison misaligns).
# Colors never appear in a demo -- that domain is held out for the test item.
#
# POSITION COUNTERBALANCING. The pools store the two OPTIONS, not a finished user
# turn, because the mention order is assigned in select_demos rather than baked
# in. Previously every conflict demo named the dev answer first, so the
# demonstrated answer was first-mentioned in 100% of dev-preamble turns and 0% of
# user-preamble turns, and the ATP contrast between the preambles was perfectly
# confounded with option position.
#
# The user turn is shared by both conditions (that is what makes the preambles
# differ only at assistant messages), so within ONE demo position is necessarily
# anti-correlated across conditions -- the balance has to come from flipping the
# order on half the demos. See select_demos.
# (dev_answer, user_answer)
CONFLICT_DEMO_POOL = [
    ("circle", "square"),
    ("cat",    "dog"),
    ("seven",  "nine"),
    ("circle", "triangle"),
    ("cat",    "bird"),
    ("seven",  "two"),
    ("circle", "star"),
    ("cat",    "horse"),
    ("seven",  "five"),
    ("circle", "oval"),
]

# Agreement demos: user asks for what the rule already says, so the answer is
# identical under both preambles. They keep the induced policy at "follow the
# developer UNLESS the user says otherwise" rather than "ignore the developer".
# Because the answer is the same in both conditions these do not contribute to
# the dev-vs-user position DIFFERENCE, but they do move each condition's absolute
# rate, so their order is alternated too.
# (distractor, answer)
AGREE_DEMO_POOL = [
    ("three",  "seven"),
    ("fox",    "cat"),
    ("cube",   "circle"),
]


def demo_turn(first, second, answer):
    """The user turn for one demo, with an explicit mention order."""
    return f"Reply with one word, {first} or {second}. Answer {answer}"'''

SELECT_OLD = '''    ok, skipped = [], []
    for user_turn, dev_a, user_a in CONFLICT_DEMO_POOL:
        if tok is None:
            ok.append((user_turn, dev_a, user_a))
            continue
        nd = len(tok.encode(dev_a, add_special_tokens=False))
        nu = len(tok.encode(user_a, add_special_tokens=False))
        (ok if nd == nu else skipped).append((user_turn, dev_a, user_a))
    for user_turn, dev_a, user_a in skipped:
        print(f"  skipped demo (length mismatch): {dev_a!r} vs {user_a!r}")
    if len(ok) < n_conflict:
        raise SystemExit(
            f"only {len(ok)} length-matched conflict demos, need {n_conflict}; "
            "add more to CONFLICT_DEMO_POOL"
        )
    conflicts = ok[:n_conflict]
    agrees = AGREE_DEMO_POOL[:n_agree]'''

SELECT_NEW = '''    ok, skipped = [], []
    for dev_a, user_a in CONFLICT_DEMO_POOL:
        if tok is None:
            ok.append((dev_a, user_a))
            continue
        nd = len(tok.encode(dev_a, add_special_tokens=False))
        nu = len(tok.encode(user_a, add_special_tokens=False))
        (ok if nd == nu else skipped).append((dev_a, user_a))
    for dev_a, user_a in skipped:
        print(f"  skipped demo (length mismatch): {dev_a!r} vs {user_a!r}")
    if len(ok) < n_conflict:
        raise SystemExit(
            f"only {len(ok)} length-matched conflict demos, need {n_conflict}; "
            "add more to CONFLICT_DEMO_POOL"
        )

    # Alternate the mention order by index parity. Deterministic rather than
    # random so a rebuild reproduces byte-for-byte, and exact rather than
    # approximate: with n_conflict even, the dev answer leads in exactly half the
    # conflict demos, so the demonstrated answer is first-mentioned in exactly
    # half the turns under BOTH preambles. verify() asserts this.
    if n_conflict % 2:
        print(f"  WARNING: n_conflict_demos={n_conflict} is odd, so the "
              f"position split is {(n_conflict + 1) // 2}/{n_conflict // 2} "
              f"rather than exactly even. Prefer an even count.")
    conflicts = []
    for i, (dev_a, user_a) in enumerate(ok[:n_conflict]):
        first, second = (dev_a, user_a) if i % 2 == 0 else (user_a, dev_a)
        # The user turn still instructs the USER's answer -- only the ORDER the
        # two options are listed in changes. Flipping which answer is requested
        # would invert the conflict itself.
        conflicts.append((demo_turn(first, second, user_a), dev_a, user_a))

    if n_agree % 2:
        print(f"  note: n_agree_demos={n_agree} is odd; agreement demos "
              f"contribute a {(n_agree + 1) // 2}/{n_agree // 2} position split "
              f"to both conditions equally.")
    agrees = []
    for i, (distractor, answer) in enumerate(AGREE_DEMO_POOL[:n_agree]):
        first, second = (answer, distractor) if i % 2 == 0 else (distractor, answer)
        agrees.append((demo_turn(first, second, answer), answer, answer))'''

VERIFY_OLD = '''    print("  balanced in every file: role assignment and mention order")'''

VERIFY_NEW = '''    print("  balanced in every file: role assignment and mention order")

    # --- position counterbalancing in the PREAMBLE --------------------------
    # The property the whole rebuild exists for: across demo turns, the
    # demonstrated answer must be first-mentioned about half the time under BOTH
    # preambles. If this drifts, the ATP contrast is confounded with position
    # again and the localization claim does not survive review.
    for cond in ("dev", "user"):
        rows = lines[f"{cond}-single-desired-all.jsonl"]
        first_count = total_count = 0
        for r in rows:
            msgs = r["prompt"]
            # demo turns only: every (user, assistant) pair before the final
            # user question, which the -all files follow with the final answer.
            for j in range(len(msgs) - 2):
                if msgs[j]["role"] != "user" or msgs[j + 1]["role"] != "assistant":
                    continue
                answer = msgs[j + 1]["content"].strip()
                body = msgs[j]["content"].split("Reply with one word,", 1)[-1]
                options = body.split(".", 1)[0]
                named = [o.strip() for o in options.split(" or ")]
                if len(named) != 2 or answer not in named:
                    raise AssertionError(
                        f"{cond}: cannot parse demo options {options!r} "
                        f"against answer {answer!r}")
                total_count += 1
                first_count += (named[0] == answer)
        rate = first_count / total_count
        print(f"  {cond} preamble: demonstrated answer is first-mentioned in "
              f"{first_count}/{total_count} demo turns ({rate:.0%})")
        assert abs(rate - 0.5) <= 0.13, (
            f"{cond} preamble is {rate:.0%} first-mentioned -- position is not "
            f"counterbalanced, so the ATP contrast is confounded with it")
    print("  position is counterbalanced within each preamble")'''

EDITS = [
    ("pools store options, not baked-in turn text", POOL_OLD, POOL_NEW),
    ("select_demos alternates mention order", SELECT_OLD, SELECT_NEW),
    ("verify asserts the per-condition position balance", VERIFY_OLD, VERIFY_NEW),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()

    path = Path(args.root).resolve() / PATH
    if not path.exists():
        print(f"MISSING  {PATH} not found under {args.root}")
        return 1

    text = path.read_text()
    pending = already = failed = 0
    for desc, old, new in EDITS:
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
        print("\nRebuild from stage 1 -- do NOT reuse the old candidate_meta.json:")
        print("  python generate_data/conflict/make_privilege_datasets.py --stage 1 ...")
        print("  python generate_data/conflict/qc_privilege_datasets.py ...")
        print("  python generate_data/conflict/make_privilege_datasets.py --stage 2 ...")
        print("Then re-localize: the old shards measure the confounded contrast.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
