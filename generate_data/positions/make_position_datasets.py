#!/usr/bin/env python3
"""Build the POSITION control corpus: first-named vs second-named, no privilege.

Two stages, mirroring make_privilege_datasets.py:

    python generate_data/conflict/make_position_datasets.py --mode candidates \\
        --out_dir data/gpt-oss-20b/position/candidates --tokenizer openai/gpt-oss-20b
    # QC the candidates, then:
    python generate_data/conflict/make_position_datasets.py --mode final \\
        --out_dir data/gpt-oss-20b/first-single --qc pair_qc.json \\
        --meta data/gpt-oss-20b/position/candidates/candidate_meta.json \\
        --tokenizer openai/gpt-oss-20b

WHAT THIS IS FOR
----------------
The privilege corpus contrasts "follow the developer" against "follow the user".
Because the demonstrated answer used to sit in a fixed position, a reviewer could
say the localized heads implement "answer the first-named option" rather than
anything about instruction privilege. Counterbalancing the preamble removes that
confound from the privilege contrast, but it does not tell you whether the heads
ATP finds are positional heads.

This corpus isolates position with the privilege axis deleted entirely:

  * NO rule anywhere. The system message states the output format and nothing
    else, and is byte-identical in both conditions.
  * The user turns NEVER request an answer. "Reply with one word, X or Y." is
    genuinely ambiguous; only the demos disambiguate it.
  * The two conditions differ ONLY in which POSITION the demos answer in --
    first-named throughout, or second-named throughout.

So ATP on first-single -> second-single localizes whatever encodes "which of two
named options to emit, by position". Compare that map against the privilege map
(compare_head_maps.py) to see how much the privilege result is positional.

DESIGN CONSTRAINTS INHERITED FROM THE PRIVILEGE CORPUS
------------------------------------------------------
1. The user turns are shared by both conditions, so the two preambles differ only
   at assistant messages and tokenize to the same length. align_toks and the ATP
   estimator both require this.
2. Each demo's two options are token-length matched, or the preambles differ in
   length.
3. Demo options are shapes/animals/numbers; the final item uses colors, held out
   from the demos entirely, so the test item is not a memorized demo.
4. Position is NOT counterbalanced here -- it is the manipulation. What IS
   balanced is which lexical item sits in position one, so "answer the first
   option" cannot be satisfied by "always answer 'circle'".

FIELD NAMES
-----------
Rows carry first_word/second_word. They ALSO carry dev_word/user_word as aliases
(dev_word = first_word, user_word = second_word) purely so qc_privilege_datasets.py
runs unmodified apart from its FILES list -- its "dev" check becomes "answers the
first-named option" and its "user" check becomes "answers the second-named
option". The aliases mean nothing about roles; there are no roles in this corpus.
"""

import os
import re
import json
import random
import argparse
from collections import Counter

# =============================================================================
# STIMULI
# =============================================================================

# No rule, no mention of either option, identical in both conditions. Present so
# the message structure matches the privilege corpus (system + demos + question)
# rather than falling through to the harmony default system block.
NEUTRAL_SYSTEM = "Answer with exactly one word. Do not explain."

# The privilege frames with the instruction clause removed. Keeping the frames
# otherwise identical means a head that fires on the shared surface form shows up
# in BOTH maps, so overlap is attributable to the position-vs-privilege
# distinction rather than to the corpora looking different.
TEMPLATE_POOL = [
    "Reply with one word, {first} or {second}.",
    "Choose one: {first} or {second}.",
    "{first} or {second}?",
    "Pick either {first} or {second}.",
    "One word only, {first} or {second}.",
    "Select {first} or {second}.",
]

for _t in TEMPLATE_POOL:
    if "{first} or {second}" not in _t:
        raise SystemExit(f"template must contain '{{first}} or {{second}}': {_t!r}")
    if "{ask}" in _t:
        raise SystemExit(f"position templates must NOT request an answer: {_t!r}")

# Demo vocabulary. Colors are excluded -- that domain is held out for the final
# item, and verify() asserts the two are disjoint. Kept deliberately broad
# (shapes, animals, numbers, objects) because demos are paired by TOKEN LENGTH at
# build time, and a hand-written pair list runs out fast: the original ten pairs
# yielded nine length-matched ones, which capped --n_demos at nine.
DEMO_WORD_POOL = [
    # shapes
    "circle", "square", "triangle", "hexagon", "oval", "star", "cube", "cone",
    "sphere", "prism", "arch", "spiral", "wedge", "ring", "cross", "diamond",
    "octagon", "pentagon", "rhombus", "cylinder",
    # animals
    "cat", "dog", "bird", "fish", "horse", "mouse", "frog", "bear", "wolf",
    "deer", "goat", "sheep", "duck", "goose", "eagle", "shark", "whale",
    "tiger", "lion", "zebra", "walrus", "rabbit", "turtle", "spider", "beetle",
    "falcon", "otter", "badger", "weasel", "ferret", "lizard", "python",
    # numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
    # objects
    "chair", "table", "lamp", "clock", "spoon", "fork", "knife", "plate",
    "bottle", "basket", "ladder", "hammer", "wrench", "anchor", "candle",
    "mirror", "pillow", "blanket", "curtain", "window", "hinge", "button",
    "pencil", "eraser", "folder", "stapler", "magnet", "compass", "kettle",
    "teapot", "drum", "flute", "piano", "guitar", "violin", "trumpet",
    "bicycle", "wagon", "rocket", "engine", "piston", "gear", "lever",
    "helmet", "glove", "boot", "jacket", "scarf", "ribbon", "buckle",
]


N_DEMOS = 32

COLOR_POOL = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "cyan", "magenta", "teal", "maroon", "navy",
    "olive", "lime", "aqua", "silver", "gold", "beige", "ivory", "coral",
    "crimson", "scarlet", "violet", "indigo", "turquoise", "lavender",
    "salmon", "khaki", "tan", "plum", "peach", "mint", "ruby", "emerald",
    "sapphire", "jade", "bronze", "copper", "charcoal", "cream", "azure",
    "mustard", "burgundy", "lilac", "sand", "amber", "rose", "chestnut",
    "cobalt", "fuchsia", "ochre", "slate", "russet", "mauve", "sepia",
    "cherry", "denim", "apricot", "blush", "bone", "brass", "butter",
    "camel", "canary", "caramel", "cedar", "chalk", "chocolate", "cinnamon",
    "clay", "cocoa", "coffee", "cotton", "ebony", "eggplant", "fern",
    "flame", "forest", "frost", "ginger", "glacier", "granite", "grape",
    "hazel", "honey", "iron", "jasmine", "lemon", "linen", "mahogany",
    "mango", "maple", "marigold", "midnight", "mist", "moss", "mulberry",
    "nickel", "oat", "ocean", "onyx", "opal", "papaya", "pearl", "pebble",
    "pepper", "pewter", "pine", "pistachio", "poppy", "pumpkin", "quartz",
    "raisin", "raven", "rust", "saffron", "sage", "sky", "smoke", "snow",
    "steel", "stone", "storm", "straw", "sunset", "taupe", "thistle",
    "tomato", "topaz", "walnut", "wheat", "wine", "zinc", "almond", "ash",
    "basil", "berry", "birch", "blossom", "cloud", "cactus", "dusk", "fog",
    "ice", "ink", "kelp", "lagoon", "lilac", "moon", "nectar", "oak",
    "pumice", "reef", "sable", "shell", "silt", "spruce", "tide", "umber",
    "vanilla", "wisteria", "amethyst", "auburn", "biscuit", "cerulean",
]

# The demo and final-item vocabularies must not overlap, or the test item is a
# memorized demo. Checked at import so a careless pool edit fails immediately
# rather than at verify() after a GPU QC run.
_dupes = sorted(set(DEMO_WORD_POOL) & set(COLOR_POOL))
if _dupes:
    raise SystemExit(f"DEMO_WORD_POOL overlaps COLOR_POOL: {_dupes}")

# Two variants per pair (both mention orders), not the privilege corpus's four --
# there is no role axis here -- so a 100-line -all file needs 50 pairs, and the
# loc and test pair sets must be disjoint.
N_CANDIDATE_PAIRS = 95
N_LOC = 50           # 50 pairs x 2 orders -> 100 lines per -all file
N_TEST = 25          # 25 pairs x 2 ->  50 test lines

CONDITIONS = ("first", "second")


def pair_key(a, b):
    return f"{a}|{b}"


def build_pairs(tok, pool, n_wanted):
    """Token-length-matched pairs, so the two options are interchangeable."""
    seen, by_len = set(), {}
    for w in pool:
        if w in seen:
            continue
        seen.add(w)
        n = len(tok.encode(w, add_special_tokens=False)) if tok else 1
        by_len.setdefault(n, []).append(w)
    if tok is None:
        print("  WARNING: no tokenizer -- pairing blindly, lengths NOT verified.")
    print("  words by token length: "
          + ", ".join(f"{n}tok:{len(v)}" for n, v in sorted(by_len.items())))
    pairs = []
    for n in sorted(by_len):
        bucket = by_len[n]
        for i in range(0, len(bucket) - 1, 2):
            pairs.append((bucket[i], bucket[i + 1]))
    return pairs[:n_wanted]


def select_demos(tok, n_demos):
    """Length-matched demo option pairs, with position-one alternated lexically.

    Returns (option_a, option_b) in listing order. The CONDITION decides which of
    the two the assistant answers, so the order here is a lexical control only:
    alternating it stops "answer the first option" from coinciding with "always
    answer the left-hand member of the pool entry".
    """
    # Bucket by token length and pair within a bucket, so both options of a demo
    # are interchangeable. Pairing from a word pool rather than filtering a fixed
    # pair list is what lets --n_demos scale: gpt-oss has a strong primacy bias,
    # and raising the demo count is the first lever for overriding it.
    buckets = {}
    for w in DEMO_WORD_POOL:
        n = len(tok.encode(w, add_special_tokens=False)) if tok else 1
        buckets.setdefault(n, []).append(w)
    if tok is None:
        print("  WARNING: no tokenizer -- demo pairing blind, lengths NOT verified.")
    print("  demo words by token length: "
          + ", ".join(f"{n}tok:{len(v)}" for n, v in sorted(buckets.items())))
    ok = []
    for n in sorted(buckets):
        bucket = buckets[n]
        for i in range(0, len(bucket) - 1, 2):
            ok.append((bucket[i], bucket[i + 1]))
    if len(ok) < n_demos:
        raise SystemExit(
            f"only {len(ok)} length-matched demo pairs, need {n_demos}. "
            f"Add words to DEMO_WORD_POOL -- preferably to whichever token-length "
            f"bucket above has an odd count, since one word there is currently unpaired.")
    demos = []
    for i, (a, b) in enumerate(ok[:n_demos]):
        first, second = (a, b) if i % 2 == 0 else (b, a)
        template = TEMPLATE_POOL[i % len(TEMPLATE_POOL)]
        demos.append((template.format(first=first, second=second), first, second))
    print(f"  demos: {len(demos)}")
    return demos


def build_line(a, b, first, second, condition, final_answer, demos, template):
    """One line. `condition` picks the POSITION the demos answer in."""
    msgs = [{"role": "system", "content": NEUTRAL_SYSTEM}]
    for user_turn, d_first, d_second in demos:
        msgs.append({"role": "user", "content": user_turn})
        msgs.append({"role": "assistant",
                     "content": d_first if condition == "first" else d_second})
    msgs.append({"role": "user",
                 "content": template.format(first=first, second=second)})
    if final_answer is not None:
        msgs.append({"role": "assistant", "content": final_answer})
    return {
        "prompt": msgs,
        "first_word": first,
        "second_word": second,
        # Aliases for qc_privilege_datasets.py only -- see the module docstring.
        "dev_word": first,
        "user_word": second,
        "target": final_answer,
        "condition": condition,
        "pair_key": pair_key(a, b),
        "mention_first": first,
        "template": template,
    }


def enumerate_variants(pairs):
    """2 lines per pair: both mention orders. There is no role axis here."""
    for p, (a, b) in enumerate(pairs):
        template = TEMPLATE_POOL[p % len(TEMPLATE_POOL)]
        for first, second in ((a, b), (b, a)):
            yield a, b, first, second, template


ALL_FILES = [
    ("first-single-desired-all.jsonl",   "first",  "first"),
    ("first-single-undesired-all.jsonl", "first",  "second"),
    ("second-single-desired-all.jsonl",  "second", "second"),
    ("second-single-undesired-all.jsonl", "second", "first"),
]
TEST_FILE = ("second-single-test.jsonl", "second", None)


def emit(path, pairs, condition, which, demos):
    variants = list(enumerate_variants(pairs))
    with open(path, "w") as f:
        for a, b, first, second, template in variants:
            final = None
            if which is not None:
                final = first if which == "first" else second
            f.write(json.dumps(build_line(a, b, first, second, condition,
                                          final, demos, template)) + "\n")
    return len(variants)


# =============================================================================
# CHECKS
# =============================================================================

def verify(out_dir):
    lines = {}
    for fname, _, _ in ALL_FILES + [TEST_FILE]:
        with open(os.path.join(out_dir, fname)) as f:
            lines[fname] = [json.loads(l) for l in f]

    all_names = [f for f, _, _ in ALL_FILES]
    test_name = TEST_FILE[0]

    n = len(lines[all_names[0]])
    assert all(len(lines[f]) == n for f in all_names), "-all line counts differ"
    print(f"  four -all files: {n} lines each; test: {len(lines[test_name])}")

    for f, rows in lines.items():
        assert all(r["first_word"] != r["second_word"] for r in rows), f
    print("  every line names two distinct options")

    # No rule, and no user turn that requests an answer -- the whole point.
    for f, rows in lines.items():
        for r in rows:
            sys_msg = r["prompt"][0]
            assert sys_msg["role"] == "system" and sys_msg["content"] == NEUTRAL_SYSTEM, f
            for m in r["prompt"]:
                if m["role"] == "user":
                    assert not re.search(r"\b(answer|say|respond with|give|reply|write|use)\s+\w+\s*$",
                                         m["content"].strip(), re.I), \
                        f"{f}: user turn requests an answer: {m['content']!r}"
    print("  no rule in any system message; no user turn requests an answer")

    # Lexical balance: each option appears in position one equally often.
    for f in all_names + [test_name]:
        o = Counter(r["mention_first"] == r["first_word"] for r in lines[f])
        assert o[False] == 0, f  # mention_first IS first_word by construction
        seen = Counter()
        for r in lines[f]:
            seen[r["first_word"]] += 1
        pairs_seen = {r["pair_key"] for r in lines[f]}
        for pk in pairs_seen:
            a, b = pk.split("|")
            rows = [r for r in lines[f] if r["pair_key"] == pk]
            assert Counter(r["first_word"] for r in rows) == Counter({a: 1, b: 1}), \
                f"{f}: pair {pk} is not order-balanced"
    print("  every pair appears in both mention orders")

    # THE manipulation: demos answer position one in `first`, position two in
    # `second`. This is asserted at 100%, not counterbalanced -- unlike the
    # privilege corpus, position here IS the contrast.
    for cond in CONDITIONS:
        rows = lines[f"{cond}-single-desired-all.jsonl"]
        msgs = rows[0]["prompt"]
        first_count = total = 0
        for j in range(len(msgs) - 2):
            if msgs[j]["role"] != "user" or msgs[j + 1]["role"] != "assistant":
                continue
            answer = msgs[j + 1]["content"].strip()
            m = re.search(r"\b([A-Za-z]+) or ([A-Za-z]+)\b", msgs[j]["content"])
            assert m, f"no option pair in {msgs[j]['content']!r}"
            named = [m.group(1), m.group(2)]
            assert answer in named, f"{cond}: answer {answer!r} not in {named}"
            total += 1
            first_count += (named[0] == answer)
        want = total if cond == "first" else 0
        assert first_count == want, (
            f"{cond} preamble answers position one in {first_count}/{total} demos, "
            f"expected {want} -- the position manipulation is not clean")
        print(f"  {cond} preamble: demos answer position one in "
              f"{first_count}/{total} turns")

    # Same invariant the ATP estimator needs.
    for cond in CONDITIONS:
        d = lines[f"{cond}-single-desired-all.jsonl"]
        u = lines[f"{cond}-single-undesired-all.jsonl"]
        for i, (x, y) in enumerate(zip(d, u)):
            assert x["prompt"][:-1] == y["prompt"][:-1], f"{cond} line {i}"
            assert x["prompt"][-1]["content"] != y["prompt"][-1]["content"]
    print("  desired/undesired share a prefix, differ only in the final token")

    d = lines["first-single-desired-all.jsonl"]
    u = lines["second-single-desired-all.jsonl"]
    diffs = set()
    for x, y in zip(d, u):
        assert len(x["prompt"]) == len(y["prompt"])
        for j, (mx, my) in enumerate(zip(x["prompt"], y["prompt"])):
            if mx != my:
                diffs.add((j, mx["role"]))
    assert all(role == "assistant" for _, role in diffs), f"non-assistant diff {diffs}"
    print(f"  first vs second preambles differ only at assistant messages "
          f"{sorted(j for j, _ in diffs)}")

    # Demo vocabulary must not leak into the final item.
    demo_words = set()
    for m in lines[all_names[0]][0]["prompt"]:
        if m["role"] == "user" and m is not lines[all_names[0]][0]["prompt"][-2]:
            found = re.search(r"\b([A-Za-z]+) or ([A-Za-z]+)\b", m["content"])
            if found:
                demo_words.update(found.groups())
    final_words = set()
    for f in all_names + [test_name]:
        for r in lines[f]:
            final_words.update((r["first_word"], r["second_word"]))
    overlap = demo_words & final_words
    assert not overlap, f"demo vocabulary leaks into the final item: {sorted(overlap)}"
    print(f"  demo vocabulary ({len(demo_words)} words) is disjoint from the "
          f"final items ({len(final_words)} words)")

    train_pairs = {r["pair_key"] for f in all_names for r in lines[f]}
    test_pairs = {r["pair_key"] for r in lines[test_name]}
    train_colors = {w for f in all_names for r in lines[f]
                    for w in (r["first_word"], r["second_word"])}
    test_colors = {w for r in lines[test_name]
                   for w in (r["first_word"], r["second_word"])}
    assert not (train_pairs & test_pairs), f"pair overlap: {train_pairs & test_pairs}"
    assert not (train_colors & test_colors), \
        f"word overlap: {sorted(train_colors & test_colors)}"
    print(f"  test set is held out: {len(test_pairs)} pairs / {len(test_colors)} words")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["candidates", "final"], required=True)
    ap.add_argument("--out_dir", default="data/gpt-oss-20b/position/candidates")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--qc", default=None, help="pair_qc.json (required for final)")
    ap.add_argument("--n_candidate_pairs", type=int, default=N_CANDIDATE_PAIRS)
    ap.add_argument("--n_loc", type=int, default=N_LOC)
    ap.add_argument("--n_test", type=int, default=N_TEST)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--n_demos", type=int, default=N_DEMOS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = args.meta or os.path.join(args.out_dir, "candidate_meta.json")

    if args.mode == "candidates":
        print("=== stage 1: candidates ===")
        demos = select_demos(tok, args.n_demos)
        pairs = build_pairs(tok, COLOR_POOL, args.n_candidate_pairs)
        need = args.n_loc + args.n_test
        print(f"{len(pairs)} candidate pairs (need {need} to survive QC)")
        if len(pairs) < need:
            raise SystemExit(f"only {len(pairs)} pairs buildable; enlarge COLOR_POOL")
        for fname, cond, which in (ALL_FILES[0], ALL_FILES[2]):
            n = emit(os.path.join(args.out_dir, fname), pairs, cond, which, demos)
            print(f"  wrote {n:>4} candidate lines -> {fname}")
        with open(meta_path, "w") as f:
            json.dump({"pairs": [pair_key(a, b) for a, b in pairs],
                       "demos": demos}, f, indent=2)
        print(f"  wrote {meta_path}")
        return

    print("=== stage 2: final datasets ===")
    if not args.qc:
        raise SystemExit("--mode final requires --qc pair_qc.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing {meta_path}; --meta must point at stage 1's output")
    with open(meta_path) as f:
        meta = json.load(f)
    demos = [tuple(d) for d in meta["demos"]]
    print(f"  reusing the {len(demos)} demos from stage 1")

    with open(args.qc) as f:
        qc = json.load(f)
    survivors = [tuple(k.split("|")) for k in qc["passing_pairs"]]
    need = args.n_loc + args.n_test
    print(f"{len(survivors)} pairs passed QC; need {need}")
    if len(survivors) < need:
        raise SystemExit(f"only {len(survivors)} pairs survived QC but {need} needed")

    loc_pairs = survivors[:args.n_loc]
    test_pairs = survivors[args.n_loc:args.n_loc + args.n_test]

    for fname, cond, which in ALL_FILES:
        n = emit(os.path.join(args.out_dir, fname), loc_pairs, cond, which, demos)
        print(f"  wrote {n:>4} lines -> {fname}")
    fname, cond, which = TEST_FILE
    n = emit(os.path.join(args.out_dir, fname), test_pairs, cond, which, demos)
    print(f"  wrote {n:>4} lines -> {fname}   (held-out words)")

    print("\nchecks:")
    verify(args.out_dir)


if __name__ == "__main__":
    main()
