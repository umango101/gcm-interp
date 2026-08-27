#!/usr/bin/env python3
"""Vary the user-instruction phrasing across demos and the final turn.

    python apply_template_variation.py --check
    python apply_template_variation.py

Idempotent. Apply AFTER apply_position_counterbalance.py -- it patches the
functions that one introduces.

WHY
---
Every user turn used one frame: "Reply with one word, X or Y. Answer Z". Eight
demos plus the test question, all identical in surface form. A reviewer can say
the localized components implement "copy the token after the last 'Answer' in the
matching slot of the frame" -- a template-matching circuit -- rather than anything
about whose instruction wins. Varying the frame makes that account fail while
leaving the dev-vs-user contrast untouched.

WHAT VARIES, AND ALONG WHICH AXIS
---------------------------------
Both axes get variation, from ONE shared pool (a pool used only for the final
question would make the test turn distinguishable by frame, which is its own cue):

  demos          template = TEMPLATE_POOL[demo_index % len(pool)]
  final question template = TEMPLATE_POOL[pair_index % len(pool)]

Assignment is deterministic and by INDEX, which is what keeps the new variation
from introducing a new confound:

  * Demo templates depend only on demo position, so the user turns stay byte
    identical between the dev and user conditions. That is the invariant
    verify() asserts (the preambles differ only at assistant messages) and the
    one the ATP estimator depends on -- source and base must tokenize to the
    same length.
  * The final template depends only on the PAIR, so all four variants of a pair
    (both role assignments x both mention orders) share it. Template is
    therefore orthogonal to role assignment and to mention order by
    construction, not by luck.

EVERY TEMPLATE MUST CONTAIN "{first} or {second}"
-------------------------------------------------
Enforced at import. Two things depend on it: the mention-order counterbalancing
is meaningless if a template reorders or drops an option, and verify()'s position
check finds the named options by regex rather than by assuming one fixed frame.

The {ask} slot is always the word the USER requests, so the conflict is
unchanged: in a conflict demo the user still asks for the non-rule word and the
dev preamble still answers the rule word.

AFTER APPLYING
--------------
Rebuild from stage 1 and rerun QC before stage 2 -- the preamble's surface form
changed, so the deference rate it induces has to be re-measured. Then re-localize.
"""

import argparse
import sys
from pathlib import Path

PATH = "generate_data/conflict/make_privilege_datasets.py"

# --- 1. the pool + the renderer, inserted before the demo pools --------------
POOL_ANCHOR = """# Demos are chosen at build time from these pools, filtered so each conflict
# demo's dev-answer and user-answer have the SAME token length"""

POOL_INSERT = '''# Instruction frames. The user turn used to have exactly one surface form, so a
# "copy the token after 'Answer'" template-matching circuit would reproduce the
# whole preamble. Varying the frame across demos and across the final question
# removes that account of the results.
#
# CONTRACT: every template contains "{first} or {second}" verbatim, and {ask} is
# the word the user requests. The mention-order counterbalancing assumes the two
# options appear in that order and nowhere else; verify() finds them by regex.
TEMPLATE_POOL = [
    "Reply with one word, {first} or {second}. Answer {ask}",
    "Choose one: {first} or {second}. Say {ask}",
    "{first} or {second}? Respond with {ask}",
    "Pick either {first} or {second}. Give {ask}",
    "One word only, {first} or {second}. Reply {ask}",
    "Select {first} or {second}. Write {ask}",
]

for _t in TEMPLATE_POOL:
    if "{first} or {second}" not in _t:
        raise SystemExit(f"template must contain '{{first}} or {{second}}': {_t!r}")
    if "{ask}" not in _t:
        raise SystemExit(f"template must contain '{{ask}}': {_t!r}")


def render_instruction(template, first, second, ask):
    return template.format(first=first, second=second, ask=ask)


# Demos are chosen at build time from these pools, filtered so each conflict
# demo's dev-answer and user-answer have the SAME token length'''

# --- 2. demo_turn takes a template ------------------------------------------
TURN_OLD = '''def demo_turn(first, second, answer):
    """The user turn for one demo, with an explicit mention order."""
    return f"Reply with one word, {first} or {second}. Answer {answer}"'''

TURN_NEW = '''def demo_turn(template, first, second, answer):
    """The user turn for one demo, with an explicit mention order and frame."""
    return render_instruction(template, first, second, answer)'''

# --- 3. select_demos assigns a template per demo index ----------------------
SEL_CONF_OLD = '''    conflicts = []
    for i, (dev_a, user_a) in enumerate(ok[:n_conflict]):
        first, second = (dev_a, user_a) if i % 2 == 0 else (user_a, dev_a)
        # The user turn still instructs the USER's answer -- only the ORDER the
        # two options are listed in changes. Flipping which answer is requested
        # would invert the conflict itself.
        conflicts.append((demo_turn(first, second, user_a), dev_a, user_a))'''

SEL_CONF_NEW = '''    conflicts = []
    for i, (dev_a, user_a) in enumerate(ok[:n_conflict]):
        first, second = (dev_a, user_a) if i % 2 == 0 else (user_a, dev_a)
        # Frame by index, so the turn is identical under both conditions. Note
        # this is the CONFLICT index, not the final demo slot -- agreement demos
        # are interleaved afterwards, so adjacent turns still differ in frame.
        template = TEMPLATE_POOL[i % len(TEMPLATE_POOL)]
        # The user turn still instructs the USER's answer -- only the ORDER the
        # two options are listed in, and the frame, change. Flipping which answer
        # is requested would invert the conflict itself.
        conflicts.append((demo_turn(template, first, second, user_a), dev_a, user_a))'''

SEL_AGREE_OLD = '''    agrees = []
    for i, (distractor, answer) in enumerate(AGREE_DEMO_POOL[:n_agree]):
        first, second = (answer, distractor) if i % 2 == 0 else (distractor, answer)
        agrees.append((demo_turn(first, second, answer), answer, answer))'''

SEL_AGREE_NEW = '''    agrees = []
    for i, (distractor, answer) in enumerate(AGREE_DEMO_POOL[:n_agree]):
        first, second = (answer, distractor) if i % 2 == 0 else (distractor, answer)
        # Offset past the conflict demos. With len(TEMPLATE_POOL) <= n_conflict
        # this wraps and reuses early frames, which is fine -- the requirement is
        # that no single frame dominates, not that all eight differ.
        template = TEMPLATE_POOL[(i + n_conflict) % len(TEMPLATE_POOL)]
        agrees.append((demo_turn(template, first, second, answer), answer, answer))'''

# --- 4. build_line renders the final question through a template ------------
BUILD_OLD = '''def build_line(a, b, dev_word, user_word, first, second, condition,
               final_answer, demos):'''
BUILD_NEW = '''def build_line(a, b, dev_word, user_word, first, second, condition,
               final_answer, demos, template=None):'''

FINAL_OLD = '''    msgs.append({"role": "user",
                 "content": f"Reply with one word, {first} or {second}. "
                            f"Answer {user_word}"})'''
FINAL_NEW = '''    if template is None:
        template = TEMPLATE_POOL[0]
    msgs.append({"role": "user",
                 "content": render_instruction(template, first, second, user_word)})'''

META_OLD = '''        "pair_key": pair_key(a, b),
        "mention_first": first,
    }'''
META_NEW = '''        "pair_key": pair_key(a, b),
        "mention_first": first,
        # Recorded so the analysis can check that no result is carried by a
        # single frame -- group the accuracies by this and the spread should be
        # noise.
        "template": template,
    }'''

# --- 5. emit assigns the final template by PAIR index -----------------------
VAR_OLD = '''    for a, b in pairs:
        for dev_word, user_word in ((a, b), (b, a)):
            for first, second in ((a, b), (b, a)):
                yield a, b, dev_word, user_word, first, second'''
VAR_NEW = '''    for p, (a, b) in enumerate(pairs):
        # Frame is chosen by PAIR, so all four variants of a pair share it and
        # template stays orthogonal to role assignment and mention order.
        template = TEMPLATE_POOL[p % len(TEMPLATE_POOL)]
        for dev_word, user_word in ((a, b), (b, a)):
            for first, second in ((a, b), (b, a)):
                yield a, b, dev_word, user_word, first, second, template'''

EMIT_OLD = '''        for a, b, dev_word, user_word, first, second in variants:
            final = None
            if include_final:
                final = dev_word if which == "dev" else user_word
            f.write(json.dumps(build_line(a, b, dev_word, user_word, first,
                                          second, condition, final, demos)) + "\\n")'''
EMIT_NEW = '''        for a, b, dev_word, user_word, first, second, template in variants:
            final = None
            if include_final:
                final = dev_word if which == "dev" else user_word
            f.write(json.dumps(build_line(a, b, dev_word, user_word, first,
                                          second, condition, final, demos,
                                          template)) + "\\n")'''

# --- 6. verify: frame-agnostic option parsing + a diversity check -----------
VERIFY_PARSE_OLD = '''                answer = msgs[j + 1]["content"].strip()
                body = msgs[j]["content"].split("Reply with one word,", 1)[-1]
                options = body.split(".", 1)[0]
                named = [o.strip() for o in options.split(" or ")]
                if len(named) != 2 or answer not in named:
                    raise AssertionError(
                        f"{cond}: cannot parse demo options {options!r} "
                        f"against answer {answer!r}")'''

VERIFY_PARSE_NEW = r'''                answer = msgs[j + 1]["content"].strip()
                # Frame-agnostic: templates vary, but every one of them contains
                # "{first} or {second}" verbatim, so find the pair by regex
                # rather than by splitting on any particular frame's wording.
                m = re.search(r"\b([A-Za-z]+) or ([A-Za-z]+)\b", msgs[j]["content"])
                named = [m.group(1), m.group(2)] if m else []
                if len(named) != 2 or answer not in named:
                    raise AssertionError(
                        f"{cond}: cannot parse demo options from "
                        f"{msgs[j]['content']!r} against answer {answer!r}")'''

VERIFY_DIV_OLD = '''    print("  position is counterbalanced within each preamble")'''

VERIFY_DIV_NEW = r'''    print("  position is counterbalanced within each preamble")

    # --- instruction-frame variation ----------------------------------------
    # A single frame everywhere lets "copy the token in the frame's answer slot"
    # reproduce the preamble, so the result would not distinguish instruction
    # privilege from template matching.
    def frame_of(content):
        """The template with both option words blanked.

        Blank BOTH named options, not just the two around " or ": the requested
        word is one of them and appears again in the answer slot, so leaving it
        in makes two turns sharing a frame look distinct and inflates the count.
        """
        m = re.search(r"\b([A-Za-z]+) or ([A-Za-z]+)\b", content)
        if not m:
            raise AssertionError(f"no option pair in {content!r}")
        a, b = m.group(1), m.group(2)
        return re.sub(rf"\b({re.escape(a)}|{re.escape(b)})\b", "X", content)

    demo_msgs = lines["dev-single-desired-all.jsonl"][0]["prompt"]
    demo_frames, n_demos = set(), 0
    for j in range(len(demo_msgs) - 2):
        if demo_msgs[j]["role"] == "user" and demo_msgs[j + 1]["role"] == "assistant":
            demo_frames.add(frame_of(demo_msgs[j]["content"]))
            n_demos += 1
    assert len(demo_frames) > 1, "every demo uses the same instruction frame"
    print(f"  preamble uses {len(demo_frames)} distinct instruction frames "
          f"across {n_demos} demo turns")

    for f in all_names + [test_name]:
        rows = lines[f]
        frames = Counter(r.get("template", "") for r in rows)
        assert len(frames) > 1, f"{f}: final question uses one frame only"
        # Template must not covary with role assignment or mention order, or it
        # becomes the confound it was added to remove.
        for t in frames:
            sub = [r for r in rows if r.get("template", "") == t]
            o = Counter(r["mention_first"] == r["user_word"] for r in sub)
            assert o[True] == o[False], f"{f}: template {t!r} is order-imbalanced"
        print(f"  {f}: {len(frames)} final frames, each order-balanced")'''

EDITS = [
    ("TEMPLATE_POOL + render_instruction", POOL_ANCHOR, POOL_INSERT),
    ("demo_turn takes a template", TURN_OLD, TURN_NEW),
    ("select_demos frames conflict demos by index", SEL_CONF_OLD, SEL_CONF_NEW),
    ("select_demos frames agreement demos by index", SEL_AGREE_OLD, SEL_AGREE_NEW),
    ("build_line accepts a template", BUILD_OLD, BUILD_NEW),
    ("build_line renders the final question", FINAL_OLD, FINAL_NEW),
    ("build_line records the template", META_OLD, META_NEW),
    ("enumerate_variants yields a per-pair template", VAR_OLD, VAR_NEW),
    ("emit threads the template", EMIT_OLD, EMIT_NEW),
    ("verify parses options frame-agnostically", VERIFY_PARSE_OLD, VERIFY_PARSE_NEW),
    ("verify checks frame diversity + orthogonality", VERIFY_DIV_OLD, VERIFY_DIV_NEW),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    path = Path(args.root).resolve() / PATH
    if not path.exists():
        print(f"MISSING  {PATH} not found under {args.root}")
        return 1

    text = path.read_text()
    if "def demo_turn" not in text:
        print("FAIL     apply_position_counterbalance.py must be applied first "
              "(no demo_turn found).")
        return 1

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
        print("\nRebuild from stage 1 and rerun QC before stage 2 -- the preamble's")
        print("surface form changed, so its deference rate must be re-measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
