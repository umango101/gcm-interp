#!/usr/bin/env python3
"""Build the INDUCTION control corpus: match-and-copy available vs not.

    python generate_data/induction/make_induction_datasets.py --mode candidates \\
        --out_dir data/gpt-oss-20b/induction/candidates --tokenizer openai/gpt-oss-20b
    # QC, then:
    python generate_data/induction/make_induction_datasets.py --mode final \\
        --out_dir data/gpt-oss-20b/induction-single --qc pair_qc_induction.json \\
        --meta data/gpt-oss-20b/induction/candidates/candidate_meta.json \\
        --tokenizer openai/gpt-oss-20b

WHAT THIS ISOLATES
------------------
An induction head does match-and-copy: it finds an earlier occurrence of the
current token and copies what followed it. Both the privilege and position
corpora present an ICL preamble and ask for a one-word answer, so induction heads
plausibly score highly in BOTH of those ATP maps for reasons that have nothing to
do with either construct. This corpus measures that directly, so the shared
component can be identified rather than assumed away.

THE CONTRAST
------------
Every row is a list of arbitrary key -> value pairs, then a query:

    zolt   -> brivan          zolt   -> brivan
    quenn  -> drapel          quenn  -> drapel
    ...    (8 pairs)          ...    (SAME 8 pairs)
    vosk   -> plendor         vosk   -> plendor
    ------------------------  ------------------------
    query: quenn              query: yarnu          <- never seen
    answer: drapel            answer: drapel

  induction-single    the query REPEATS an earlier key, so the answer is
                      recoverable by matching it and copying the value
  noinduction-single  the query is a fresh key, so nothing matches

The preambles are BYTE-IDENTICAL between the two conditions. They differ in one
token: the query. That is what makes this contrast narrow -- narrower than the
privilege or position corpora, which differ across every assistant turn.

TWO CONFOUNDS DELIBERATELY CONTROLLED
-------------------------------------
1. The answer token is in context in BOTH conditions. `noinduction` keeps the
   whole preamble, including the (key, value) pair whose value is the desired
   answer -- only the query changes. So the contrast is "can the model MATCH",
   not "is the answer token present at all". Dropping the pair instead would
   confound induction with mere token presence.
2. The matched key is not always the most recent. Its position is spread across
   the preamble and balanced by the builder, so the contrast is not "attend to
   the last demo".

The values are nonce strings, filtered to a fixed token length. Real words let
semantic association substitute for match-and-copy, which is the mechanism this
corpus exists to isolate.

FIELD NAMES
-----------
Rows carry query/answer/distractor plus dev_word/user_word aliases (dev_word =
answer, user_word = distractor) so qc_privilege_datasets.py scores them
unmodified: its "dev" check becomes "does the model retrieve the matched value".
The aliases mean nothing about roles.
"""

import os
import json
import random
import argparse
from collections import Counter

# =============================================================================
# STIMULI
# =============================================================================

NEUTRAL_SYSTEM = "Answer with exactly one word. Do not explain."

# The frame is deliberately minimal and identical on every turn. The privilege
# and position corpora vary their frames to defeat template-matching; here a
# constant frame is correct -- varying it would add a second thing the ATP
# contrast could be picking up, and the query token is already the only
# difference between conditions.
DEMO_FRAME = "{key}"
QUERY_FRAME = "{key}"

N_PAIRS = 8            # matches the other corpora's 8-turn preamble

# Nonce syllables. Combined into pronounceable strings, then filtered to a single
# token length at build time so keys are interchangeable and values are
# interchangeable.
_ONSETS = ["b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s",
           "t", "v", "z", "br", "dr", "fl", "gl", "kr", "pl", "sk", "sn",
           "st", "sw", "tr", "vr", "zl"]
_NUCLEI = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "ia"]
_CODAS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v",
          "z", "lk", "mp", "nd", "nt", "rk", "rn", "sk", "st"]


def nonce_pool(rng, n_wanted, syllables=2):
    """Pronounceable nonce strings, deduplicated, in a stable order."""
    seen, out = set(), []
    guard = 0
    while len(out) < n_wanted and guard < n_wanted * 400:
        guard += 1
        w = ""
        for _ in range(syllables):
            w += rng.choice(_ONSETS) + rng.choice(_NUCLEI)
        w += rng.choice(_CODAS)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def by_token_length(tok, words):
    buckets = {}
    for w in words:
        n = len(tok.encode(w, add_special_tokens=False)) if tok else 1
        buckets.setdefault(n, []).append(w)
    return buckets


# VALUES are real words, not nonces, and KEYS stay nonces. The asymmetry is
# deliberate:
#
#   * The value is the token being PREDICTED, so it must be a single token. QC
#     scores by first-token id, and gpt-oss encodes a bare word and a
#     space-prefixed word with DIFFERENT first tokens ('kujaf' -> [182098, 1553]
#     vs ' kujaf' -> [33430, 1553]), so any multi-token answer makes that
#     comparison unreliable in both directions -- correct retrievals get scored
#     off-task. Nonces are essentially never single tokens in o200k (the pool
#     yields ~1 in 4000), so the value has to come from real vocabulary.
#   * Nothing is guessable from this, because the key -> value assignment is
#     random. Semantic association could only help if the KEY carried a hint,
#     and keys remain nonces.
#   * It also brings this corpus closer to the privilege and position corpora,
#     whose answers are single-token real words -- which is what makes the
#     head-overlap comparison interpretable.
VALUE_WORD_POOL = [
    "apple", "anchor", "arrow", "bridge", "basket", "candle", "castle",
    "cactus", "carpet", "cabin", "dagger", "desert", "dragon", "engine",
    "feather", "forest", "garden", "guitar", "hammer", "harbor", "helmet",
    "island", "jacket", "jungle", "kettle", "ladder", "lantern", "marble",
    "meadow", "mirror", "needle", "orchid", "palace", "pencil", "pillow",
    "planet", "prison", "puzzle", "rabbit", "ribbon", "rocket", "saddle",
    "shadow", "shovel", "signal", "silver", "sponge", "statue", "temple",
    "thunder", "ticket", "tunnel", "turtle", "valley", "velvet", "window",
    "wizard", "yogurt", "zipper", "acorn", "beacon", "bucket", "button",
    "camera", "canvas", "cellar", "circus", "collar", "cotton", "crayon",
    "crystal", "curtain", "diamond", "dolphin", "drawer", "eagle", "engine",
    "fabric", "falcon", "fossil", "funnel", "gadget", "glacier", "granite",
    "gravel", "hanger", "hazard", "hollow", "hunter", "jigsaw", "kernel",
    "lagoon", "lantern", "lizard", "locker", "magnet", "mantle", "meteor",
    "mosaic", "muffin", "napkin", "nectar", "nozzle", "orbit", "otter",
    "outlet", "pallet", "parcel", "pastry", "pebble", "pepper", "picnic",
    "pigeon", "pirate", "piston", "plague", "plasma", "pocket", "poison",
    "pollen", "portal", "prairie", "pretzel", "python", "quiver", "racket",
    "rattle", "ravine", "relic", "rhythm", "rooster", "rubble", "runner",
    "saloon", "sandal", "sapling", "sausage", "scepter", "scroll", "sculpt",
    "server", "shrine", "shuttle", "silence", "siren", "socket", "spiral",
    "sprout", "squire", "stable", "stencil", "stitch", "stream", "stripe",
    "summit", "sundial", "syrup", "tablet", "talon", "tavern", "teapot",
    "tender", "thimble", "thistle", "throne", "timber", "tinder", "toaster",
    "token", "tomato", "torrent", "tractor", "trellis", "tribute", "trolley",
    "trumpet", "tunic", "turban", "vessel", "vinegar", "violin", "voyage",
    "waffle", "wagon", "walnut", "warden", "whistle", "willow", "wisdom",
    "wonder", "wreath", "yonder", "zenith",
]

N_CANDIDATE_ROWS = 160
N_LOC = 100
N_TEST = 50

CONDITIONS = ("induction", "noinduction")


def build_row(rng, keys, values, fresh_keys, n_pairs, row_idx):
    """One row's stimulus: the preamble, the matched index, and the two answers.

    match_idx cycles rather than being sampled so its distribution over preamble
    positions is exactly uniform across the corpus -- otherwise "attend to a
    recent demo" would partly explain the contrast.
    """
    ks = rng.sample(keys, n_pairs)
    vs = rng.sample(values, n_pairs)
    match_idx = row_idx % n_pairs
    distractor_idx = (match_idx + 1 + rng.randrange(n_pairs - 1)) % n_pairs
    return {
        "keys": ks,
        "values": vs,
        "match_idx": match_idx,
        "answer": vs[match_idx],
        "distractor": vs[distractor_idx],
        "fresh_key": rng.choice(fresh_keys),
    }


def build_line(row, condition, which):
    """`condition` decides whether the query matches; `which` picks the answer."""
    msgs = [{"role": "system", "content": NEUTRAL_SYSTEM}]
    for k, v in zip(row["keys"], row["values"]):
        msgs.append({"role": "user", "content": DEMO_FRAME.format(key=k)})
        msgs.append({"role": "assistant", "content": v})

    query = (row["keys"][row["match_idx"]] if condition == "induction"
             else row["fresh_key"])
    msgs.append({"role": "user", "content": QUERY_FRAME.format(key=query)})

    final = None
    if which is not None:
        final = row["answer"] if which == "answer" else row["distractor"]
        msgs.append({"role": "assistant", "content": final})

    return {
        "prompt": msgs,
        "query": query,
        "answer": row["answer"],
        "distractor": row["distractor"],
        "match_idx": row["match_idx"],
        # Aliases for qc_privilege_datasets.py only -- see the module docstring.
        "dev_word": row["answer"],
        "user_word": row["distractor"],
        "target": final,
        "condition": condition,
        "pair_key": f"{row['keys'][row['match_idx']]}|{row['answer']}",
    }


ALL_FILES = [
    ("induction-single-desired-all.jsonl",     "induction",   "answer"),
    ("induction-single-undesired-all.jsonl",   "induction",   "distractor"),
    ("noinduction-single-desired-all.jsonl",   "noinduction", "answer"),
    ("noinduction-single-undesired-all.jsonl", "noinduction", "distractor"),
]
TEST_FILE = ("noinduction-single-test.jsonl", "induction", None)


def emit(path, rows, condition, which):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(build_line(row, condition, which)) + "\n")
    return len(rows)


# =============================================================================
# CHECKS
# =============================================================================

def verify(out_dir, tok=None):
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
        assert all(r["answer"] != r["distractor"] for r in rows), f
    print("  answer and distractor are always distinct")

    # The narrow contrast: preambles identical, one query token apart.
    ind = lines["induction-single-desired-all.jsonl"]
    noi = lines["noinduction-single-desired-all.jsonl"]
    diffs = set()
    for x, y in zip(ind, noi):
        assert len(x["prompt"]) == len(y["prompt"])
        for j, (mx, my) in enumerate(zip(x["prompt"], y["prompt"])):
            if mx != my:
                diffs.add((j, mx["role"]))
    assert diffs, "the two conditions are identical -- no contrast at all"
    assert all(role == "user" for _, role in diffs), \
        f"conditions differ somewhere other than the query: {diffs}"
    assert len({j for j, _ in diffs}) == 1, \
        f"conditions differ at more than one turn: {sorted(diffs)}"
    print(f"  induction vs noinduction differ at exactly one user turn "
          f"{sorted(j for j, _ in diffs)} -- the query")

    # The query really does / does not match.
    for cond, want_match in (("induction", True), ("noinduction", False)):
        rows = lines[f"{cond}-single-desired-all.jsonl"]
        matched = 0
        for r in rows:
            keys = [m["content"] for m in r["prompt"][1:-2:2]]
            query = r["prompt"][-2]["content"]
            matched += (query in keys)
        want = len(rows) if want_match else 0
        assert matched == want, \
            f"{cond}: query matches an earlier key in {matched}/{len(rows)}, want {want}"
        print(f"  {cond}: query matches an earlier key in {matched}/{len(rows)} rows")

    # The answer token is in context in BOTH conditions -- so the contrast is
    # about matching, not about token presence.
    for cond in CONDITIONS:
        for r in lines[f"{cond}-single-desired-all.jsonl"]:
            vals = [m["content"] for m in r["prompt"][2:-2:2]]
            assert r["answer"] in vals, f"{cond}: answer not present in the preamble"
            assert r["distractor"] in vals, f"{cond}: distractor not in the preamble"
    print("  answer AND distractor appear in the preamble under both conditions")

    # Recency control.
    for cond in CONDITIONS:
        c = Counter(r["match_idx"] for r in lines[f"{cond}-single-desired-all.jsonl"])
        spread = max(c.values()) - min(c.values())
        # Stage 2 stratifies by position, so spread is normally 0 or 1. The slack
        # to 2 covers a slot that ran short of QC survivors. Beyond that the
        # recency control is not doing its job and the contrast is partly "attend
        # to a recent demo", so fail rather than warn.
        assert min(c.values()) > 0, \
            f"{cond}: position slot(s) empty: {dict(sorted(c.items()))}"
        assert spread <= 2, (
            f"{cond}: match position is too uneven: {dict(sorted(c.items()))}. "
            f"Raise --n_candidate_rows so every position slot has enough QC survivors.")
        print(f"  {cond}: matched key position spans {len(c)} slots, "
              f"{min(c.values())}-{max(c.values())} rows each (spread {spread})")

    for cond in CONDITIONS:
        d = lines[f"{cond}-single-desired-all.jsonl"]
        u = lines[f"{cond}-single-undesired-all.jsonl"]
        for i, (x, y) in enumerate(zip(d, u)):
            assert x["prompt"][:-1] == y["prompt"][:-1], f"{cond} line {i}"
            assert x["prompt"][-1]["content"] != y["prompt"][-1]["content"]
    print("  desired/undesired share a prefix, differ only in the final token")

    if tok is not None:
        def ntok(s):
            return len(tok.encode(s, add_special_tokens=False))
        bad = []
        for x, y in zip(ind, noi):
            if ntok(x["prompt"][-2]["content"]) != ntok(y["prompt"][-2]["content"]):
                bad.append((x["prompt"][-2]["content"], y["prompt"][-2]["content"]))
        assert not bad, f"query token-length mismatch (align_toks will fail): {bad[:3]}"
        for r in ind:
            assert ntok(r["answer"]) == ntok(r["distractor"]), \
                f"answer/distractor length mismatch: {r['answer']} vs {r['distractor']}"
        print("  queries are token-length matched; so are answer and distractor")
    else:
        print("  (no tokenizer: token-length matching NOT verified)")

    train_keys = {k for f in all_names for r in lines[f]
                  for k in [m["content"] for m in r["prompt"][1:-2:2]]}
    test_keys = {k for r in lines[test_name]
                 for k in [m["content"] for m in r["prompt"][1:-2:2]]}
    overlap = train_keys & test_keys
    assert not overlap, f"key vocabulary overlaps train/test: {sorted(overlap)[:5]}"
    print(f"  test set is held out: {len(test_keys)} keys, zero overlap with "
          f"the {len(train_keys)} training keys")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["candidates", "final"], required=True)
    ap.add_argument("--out_dir", default="data/gpt-oss-20b/induction/candidates")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--qc", default=None, help="pair_qc_induction.json (final only)")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--n_candidate_rows", type=int, default=N_CANDIDATE_ROWS)
    ap.add_argument("--n_loc", type=int, default=N_LOC)
    ap.add_argument("--n_test", type=int, default=N_TEST)
    ap.add_argument("--n_pairs", type=int, default=N_PAIRS)
    ap.add_argument("--key_tokens", type=int, default=2,
                    help="token length to filter keys to")
    ap.add_argument("--value_tokens", type=int, default=1,
                    help="token length to filter values to. Keep at 1: the value is "
                         "the predicted token and QC scores it by first-token id.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = args.meta or os.path.join(args.out_dir, "candidate_meta.json")

    if args.mode == "candidates":
        print("=== stage 1: candidates ===")
        # KEYS: nonces. They are matched, never predicted, so multi-token is fine
        # and nonsense is desirable -- a real-word key could be associated with a
        # real-word value by something other than match-and-copy.
        raw = nonce_pool(rng, 4000)
        key_buckets = by_token_length(tok, raw)
        print("  nonce keys by token length: "
              + ", ".join(f"{n}tok:{len(v)}" for n, v in sorted(key_buckets.items())))

        # VALUES: real words filtered to a SINGLE token. See VALUE_WORD_POOL.
        val_buckets = by_token_length(tok, sorted(set(VALUE_WORD_POOL)))
        print("  word values by token length: "
              + ", ".join(f"{n}tok:{len(v)}" for n, v in sorted(val_buckets.items())))

        if tok is None:
            print("  WARNING: no tokenizer -- lengths NOT verified.")
            keys_all = raw[:2000]
            vals_all = sorted(set(VALUE_WORD_POOL))
        else:
            keys_all = key_buckets.get(args.key_tokens, [])
            vals_all = val_buckets.get(args.value_tokens, [])

        need_keys = (args.n_pairs + 1) * 4
        need_vals = args.n_pairs * 3
        if len(keys_all) < need_keys:
            raise SystemExit(
                f"only {len(keys_all)} nonce keys at {args.key_tokens} tokens, need "
                f"{need_keys}. Pick a --key_tokens with more entries in the histogram "
                f"above, or raise nonce_pool().")
        if len(vals_all) < need_vals:
            raise SystemExit(
                f"only {len(vals_all)} word values at {args.value_tokens} token(s), need "
                f"{need_vals}. --value_tokens 1 is strongly preferred (QC scores by "
                f"first-token id, and multi-token answers make that unreliable); add "
                f"single-token words to VALUE_WORD_POOL rather than raising it.")
        if args.value_tokens != 1:
            print(f"  WARNING: --value_tokens {args.value_tokens}. QC compares FIRST "
                  f"token ids, and a bare vs space-prefixed word can differ there, so "
                  f"multi-token answers get scored off-task even when retrieved "
                  f"correctly. Prefer 1.")

        # Keys and values are split train/test up front so the held-out set shares
        # no vocabulary with the localization set.
        k_cut, v_cut = int(len(keys_all) * 0.7), int(len(vals_all) * 0.7)
        pools = {"train": (keys_all[:k_cut], vals_all[:v_cut]),
                 "test": (keys_all[k_cut:], vals_all[v_cut:])}
        # Fresh keys must never appear as a preamble key in the same split.
        fresh = {s: pools[s][0][-max(20, args.n_pairs):] for s in pools}
        usable = {s: (pools[s][0][:-len(fresh[s])], pools[s][1]) for s in pools}

        rows = [build_row(rng, usable["train"][0], usable["train"][1],
                          fresh["train"], args.n_pairs, i)
                for i in range(args.n_candidate_rows)]
        print(f"{len(rows)} candidate rows")
        for fname, cond, which in (ALL_FILES[0], ALL_FILES[2]):
            n = emit(os.path.join(args.out_dir, fname), rows, cond, which)
            print(f"  wrote {n:>4} candidate lines -> {fname}")
        with open(meta_path, "w") as f:
            json.dump({"rows": rows,
                       "test_pools": {"keys": usable["test"][0],
                                      "values": usable["test"][1],
                                      "fresh": fresh["test"]},
                       "n_pairs": args.n_pairs, "seed": args.seed}, f, indent=2)
        print(f"  wrote {meta_path}")
        return

    print("=== stage 2: final datasets ===")
    if not args.qc:
        raise SystemExit("--mode final requires --qc pair_qc_induction.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing {meta_path}; --meta must point at stage 1's output")
    with open(meta_path) as f:
        meta = json.load(f)

    with open(args.qc) as f:
        qc = json.load(f)
    passing = set(qc["passing_pairs"])
    survivors = [r for r in meta["rows"]
                 if f"{r['keys'][r['match_idx']]}|{r['answer']}" in passing]
    print(f"{len(survivors)}/{len(meta['rows'])} rows passed QC; need {args.n_loc}")
    if len(survivors) < args.n_loc:
        raise SystemExit(
            f"only {len(survivors)} rows survived QC but {args.n_loc} are needed. "
            f"Raise --n_candidate_rows, or the model is not doing the task and the "
            f"contrast would be measuring nothing.")

    # Stratified by match position, not survivors[:n_loc]. match_idx is uniform
    # across the CANDIDATES (row_idx % n_pairs), but QC drops rows unevenly, so
    # taking a prefix of the survivors inherits whatever imbalance QC introduced --
    # which reintroduces exactly the recency confound the cycling was there to
    # prevent. Round-robin over the position buckets instead.
    from collections import defaultdict
    by_pos = defaultdict(list)
    for r in survivors:
        by_pos[r["match_idx"]].append(r)
    loc_rows, exhausted = [], set()
    while len(loc_rows) < args.n_loc:
        progressed = False
        for pos in sorted(by_pos):
            if len(loc_rows) >= args.n_loc:
                break
            if by_pos[pos]:
                loc_rows.append(by_pos[pos].pop(0))
                progressed = True
            else:
                exhausted.add(pos)
        if not progressed:
            break
    from collections import Counter as _C
    dist = _C(r["match_idx"] for r in loc_rows)
    spread = max(dist.values()) - min(dist.values()) if dist else 0
    print(f"  match position across the selected rows: "
          f"{dict(sorted(dist.items()))} (spread {spread})")
    if exhausted:
        print(f"  NOTE: position slot(s) {sorted(exhausted)} ran out of QC survivors, "
              f"so the balance is imperfect. Raise --n_candidate_rows for a clean split.")
    tp = meta["test_pools"]
    test_rows = [build_row(rng, tp["keys"], tp["values"], tp["fresh"],
                           meta["n_pairs"], i)
                 for i in range(args.n_test)]

    for fname, cond, which in ALL_FILES:
        n = emit(os.path.join(args.out_dir, fname), loc_rows, cond, which)
        print(f"  wrote {n:>4} lines -> {fname}")
    fname, cond, which = TEST_FILE
    n = emit(os.path.join(args.out_dir, fname), test_rows, cond, which)
    print(f"  wrote {n:>4} lines -> {fname}   (held-out nonces)")

    print("\nchecks:")
    verify(args.out_dir, tok)


if __name__ == "__main__":
    main()
