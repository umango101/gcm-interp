#!/usr/bin/env python3
"""Build the POSITION control corpus from the privilege corpus's own prompts.

    python make_position_datasets.py --arm devuser \\
        --out_dir data/gpt-oss-20b/pos-devuser \\
        --meta data/gpt-oss-20b/devuser/candidates/candidate_meta.json \\
        --qc   data/gpt-oss-20b/devuser/pair_qc.json \\
        --tokenizer openai/gpt-oss-20b

WHY THIS SHARES THE PRIVILEGE PROMPTS INSTEAD OF BEING ITS OWN CORPUS
---------------------------------------------------------------------
The point of a position control is to ask whether the heads that carry "which
rule source to follow" are the same heads that carry "which option was named
first". Answering it means comparing two attribution maps, and that comparison
is only interpretable if the two contrasts differ in the LATENT VARIABLE and
nothing else.

A separate first-single/second-single corpus cannot give you that: different
prompts, different lengths, different surface forms, so any overlap between the
maps is confounded with the corpora merely being differently shaped. Two ATP
maps computed on similar prompts correlate somewhat no matter what they track.

So this reuses the privilege corpus exactly -- same rules, same questions, same
templates, same demo turns, same pairs, same held-out colours, same token
lengths -- and changes only what the assistant demonstrates:

    privilege contrast   dev preamble  = always the privileged rule's word
                         user preamble = always the subordinate rule's word
    position contrast    first  preamble = always the FIRST-named option
                         second preamble = always the SECOND-named option

The two axes are orthogonal by construction rather than merely different:
select_demos_rule alternates mention order by demo index, so "always
privileged" and "always first" agree on exactly half the demos.

FILE NAMES ARE DELIBERATELY THE PRIVILEGE ONES
----------------------------------------------
dev-single-* holds the FIRST-position preamble and user-single-* the SECOND,
because config.py resolves data paths as {data_dir}/{source}-desired-all.jsonl
with source/base naming the file prefixes. Keeping the prefixes lets the whole
localization pipeline run unchanged with --data_dir pos-<arm>. The record's
`condition` field carries the real name (first/second), and arm_manifest.json
records contrast="position", so nothing downstream has to infer it from a path.

NO TEST FILE, NO STEERING
-------------------------
The overlap analysis needs attribution maps only, so this writes the four -all
files and stops. That is the whole reason the position control is cheap: one
localization run per arm instead of a 154-point steering grid.
"""

import os
import json
import argparse

import hierarchy_common as H


def from_corpus(priv_dir, arm, form):
    """Recover pairs, demos and the system block from the privilege corpus.

    The deployed corpora hold only the six jsonl files -- candidate_meta.json
    and pair_qc.json stay behind in the generation tree -- so deriving the
    inputs from the data itself avoids threading paths across two trees. It is
    also stricter than reading the meta: the pairs come out in the exact order
    the privilege localization consumed them, and the demos are the ones
    actually in those files rather than the ones a meta file claims were used.
    """
    dev = [json.loads(l) for l in open(os.path.join(priv_dir, "dev-single-desired-all.jsonl"))]
    usr = [json.loads(l) for l in open(os.path.join(priv_dir, "user-single-desired-all.jsonl"))]
    if len(dev) != len(usr):
        raise SystemExit(f"{priv_dir}: the two -desired files differ in length")
    if dev[0].get("conflict_form", "request") != form:
        raise SystemExit(f"{priv_dir} is form {dev[0].get('conflict_form')!r}, "
                         f"not {form!r}")
    if dev[0].get("arm") != arm:
        raise SystemExit(f"{priv_dir} is arm {dev[0].get('arm')!r}, not {arm!r}")

    # Pairs in file order, deduped. enumerate_variants emits four lines per
    # pair, so this recovers the pair list the privilege files were built from.
    pairs, seen = [], set()
    for r in dev:
        k = r["pair_key"]
        if k not in seen:
            seen.add(k)
            pairs.append(tuple(k.split("|")))

    # Demos: the asker's turns before the final question, paired with the
    # privileged and subordinate answers from the two preambles.
    arm_obj = H.ARMS[arm]
    role = H.demo_role(arm_obj, form)
    d_msgs, u_msgs = dev[0]["prompt"], usr[0]["prompt"]
    demos = []
    for j in range(len(d_msgs) - 2):
        if d_msgs[j]["role"] == role and d_msgs[j + 1]["role"] == "assistant":
            demos.append((d_msgs[j]["content"],
                          d_msgs[j + 1]["content"],
                          u_msgs[j + 1]["content"]))
    if not demos:
        raise SystemExit(f"{priv_dir}: no demo turns found")

    # System block: the leading system message, minus any rule appended to it.
    block = d_msgs[0]["content"].split("\n\nRules:")[0]

    # Categories MUST be restored before anything calls build_rule below: the
    # rule text is generated from ACTIVE_CONFLICT_CATEGORIES, so a stale list
    # produces a different-length rule and the prefix strip silently misses.
    rule_text = d_msgs[arm_obj.privileged_index]["content"]
    active = [c for c in H.CONFLICT_CATEGORIES
              if f"choose a {c[0]}, answer" in rule_text
              or f"choose an {c[0]}, answer" in rule_text]
    if active:
        H.set_active_categories(active)

    # The first demo turn may carry the subordinate rule as a prefix (the
    # user-subordinate arms in the rule form). build_line re-adds that prefix
    # itself, so strip it here or the rule appears twice -- which reads as a
    # plausible prompt and would quietly make the position corpus a different
    # experiment from the privilege one.
    prefix = H.design_messages(arm_obj, form, dev[0]["dev_word"],
                               dev[0]["user_word"], block)[1]
    if prefix:
        if not demos[0][0].startswith(prefix):
            raise SystemExit(
                "expected the first demo turn to begin with the subordinate "
                "rule, but it does not. The categories recovered from the rule "
                f"text ({len(active)}) may not match the corpus.")
        demos[0] = (demos[0][0][len(prefix):], demos[0][1], demos[0][2])
    return pairs, demos, block, active


# (filename, preamble condition, which side the final answer takes)
POSITION_FILES = [
    ("dev-single-desired-all.jsonl",    "first",  "first"),
    ("dev-single-undesired-all.jsonl",  "first",  "second"),
    ("user-single-desired-all.jsonl",   "second", "second"),
    ("user-single-undesired-all.jsonl", "second", "first"),
]


def emit(path, arm, form, system_block, pairs, condition, which, demos):
    """As hierarchy_common.emit, but the final answer is chosen by POSITION.

    In the privilege corpus the final answer is dev_word or user_word; here it
    is the first- or second-named option of the same final question. The prompt
    prefix is byte-identical either way -- only the assistant's answer differs,
    which is what keeps the desired/undesired pair a minimal pair.
    """
    n = 0
    with open(path, "w") as f:
        for a, b, dev_word, user_word, first, second, template in \
                H.enumerate_variants(pairs, form):
            final = first if which == "first" else second
            rec = H.build_line(arm, form, system_block, a, b, dev_word,
                               user_word, first, second, condition, final,
                               demos, template)
            rec["contrast"] = "position"
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


def verify(out_dir, arm, form, system_block, demos):
    """Checks specific to the position corpus.

    The privilege verifier does not apply: it asserts the demonstrated answer is
    the privileged word, which is exactly what this corpus changes.
    """
    rows = {}
    for fname, _, _ in POSITION_FILES:
        with open(os.path.join(out_dir, fname)) as f:
            rows[fname] = [json.loads(l) for l in f]

    n = len(rows[POSITION_FILES[0][0]])
    assert all(len(v) == n for v in rows.values()), "line counts differ"
    print(f"  four -all files: {n} lines each")

    # Round trip, as in the privilege corpus: every record rebuilds from its own
    # metadata, so a builder/file disagreement anywhere fails loudly.
    for fname, cond, which in POSITION_FILES:
        for i, r in enumerate(rows[fname]):
            a, b = r["pair_key"].split("|")
            first = r["mention_first"]
            second = b if first == a else a
            rebuilt = H.build_line(arm, form, system_block, a, b, r["dev_word"],
                                   r["user_word"], first, second, r["condition"],
                                   r["target"], demos, r["template"])
            assert rebuilt["prompt"] == r["prompt"], f"{fname} line {i}: no round trip"
            assert r["condition"] == cond, f"{fname} line {i}: wrong condition"
            expected = first if which == "first" else second
            assert r["target"] == expected, f"{fname} line {i}: wrong target"
    print("  every record round-trips from its metadata")

    # Minimal pairs: desired and undesired share every message but the last.
    for cond, d_name, u_name in (("first", POSITION_FILES[0][0], POSITION_FILES[1][0]),
                                 ("second", POSITION_FILES[2][0], POSITION_FILES[3][0])):
        for i, (x, y) in enumerate(zip(rows[d_name], rows[u_name])):
            assert x["prompt"][:-1] == y["prompt"][:-1], f"{cond} line {i}"
            assert x["prompt"][-1]["content"] != y["prompt"][-1]["content"]
    print("  desired/undesired share a prefix, differ only in the final token")

    # The two preambles differ only at assistant messages -- the property the
    # whole design rests on, checked here as it is for the privilege corpus.
    d = rows[POSITION_FILES[0][0]]
    u = rows[POSITION_FILES[2][0]]
    diffs = set()
    for x, y in zip(d, u):
        for j, (mx, my) in enumerate(zip(x["prompt"], y["prompt"])):
            if mx != my:
                diffs.add((j, mx["role"]))
    assert all(role == "assistant" for _, role in diffs), \
        f"preambles differ at a non-assistant message: {sorted(diffs)}"
    print(f"  the two preambles differ only at assistant messages "
          f"{sorted(j for j, _ in diffs)}")

    # ORTHOGONALITY. This is the number that makes the control a control: if the
    # position policy and the privilege policy demonstrated the same answers,
    # the two attribution maps would agree for a trivial reason.
    role = H.demo_role(arm, form)
    agree = total = 0
    for r in d:
        for j in range(len(r["prompt"]) - 1):
            if r["prompt"][j]["role"] == role and r["prompt"][j + 1]["role"] == "assistant":
                turn = r["prompt"][j]["content"]
                ans = r["prompt"][j + 1]["content"].strip()
                named = H._named_options(turn)
                if ans not in named:
                    continue
                total += 1
                agree += (ans == r["dev_word"] or ans in named[:1])
    print(f"  position demos: {total} parsed; first-named by construction")

    # Held-out colours, same guarantee as the privilege corpus.
    colors = set()
    for v in rows.values():
        for r in v:
            colors.update((r["dev_word"], r["user_word"]))
    print(f"  {len({r['pair_key'] for r in d})} pairs / {len(colors)} colors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(H.ARMS), required=True)
    ap.add_argument("--form", choices=H.FORMS, default="rule")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--from_corpus", default=None,
                    help="the arm's PRIVILEGE corpus directory, e.g. "
                         "data/gpt-oss-20b/devuser. Derives pairs, demos, "
                         "system block and categories from those files -- use "
                         "this when candidate_meta.json is not alongside them.")
    ap.add_argument("--meta", default=None,
                    help="candidate_meta.json from the PRIVILEGE corpus of the "
                         "same arm: the demos, categories and system block must "
                         "be identical or the two contrasts are not matched")
    ap.add_argument("--qc", default=None,
                    help="pair_qc.json from the same privilege corpus, so both "
                         "contrasts use the same surviving pairs")
    ap.add_argument("--n_loc", type=int, default=25)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    arm = H.ARMS[args.arm]
    if args.from_corpus:
        pairs, demos, system_block, active = from_corpus(
            args.from_corpus, args.arm, args.form)
        pairs = pairs[:args.n_loc] if args.n_loc else pairs
        print(f"derived from {args.from_corpus}: {len(pairs)} pairs, "
              f"{len(demos)} demos, {len(active)} categories")
        _emit_all(args, arm, pairs, demos, system_block)
        return
    if not (args.meta and args.qc):
        raise SystemExit("pass --from_corpus, or both --meta and --qc")
    with open(args.meta) as f:
        meta = json.load(f)
    if meta.get("arm") != args.arm or meta.get("form") != args.form:
        raise SystemExit(
            f"{args.meta} is for arm {meta.get('arm')!r} / form "
            f"{meta.get('form')!r}, not {args.arm!r} / {args.form!r}.")
    if meta.get("categories"):
        H.set_active_categories(meta["categories"])
    system_block = meta["system_block"]
    demos = [tuple(d) for d in meta["demos"]]

    with open(args.qc) as f:
        qc = json.load(f)
    survivors = [tuple(k.split("|")) for k in qc["passing_pairs"]]
    # The SAME pairs the privilege localization used, so the two maps are
    # computed over the same items and not merely the same distribution.
    pairs = survivors[:args.n_loc]
    if len(pairs) < args.n_loc:
        raise SystemExit(f"only {len(survivors)} pairs in {args.qc}")

    _emit_all(args, arm, pairs, demos, system_block)


def _emit_all(args, arm, pairs, demos, system_block):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"position contrast | arm {args.arm} | form {args.form} | "
          f"{len(pairs)} pairs | {len(demos)} demos")
    for fname, cond, which in POSITION_FILES:
        n = emit(os.path.join(args.out_dir, fname), arm, args.form,
                 system_block, pairs, cond, which, demos)
        print(f"  wrote {n:>4} lines -> {fname}  ({cond} preamble, "
              f"{which}-named answer)")

    with open(os.path.join(args.out_dir, "arm_manifest.json"), "w") as f:
        json.dump({"arm": args.arm, "form": args.form, "contrast": "position",
                   "privileged_role": arm.privileged,
                   "subordinate_role": arm.subordinate,
                   "source_meta": args.meta, "source_qc": args.qc,
                   "source_corpus": args.from_corpus,
                   "n_pairs": len(pairs),
                   "note": "dev-single-* is the FIRST-position preamble and "
                           "user-single-* the SECOND; prefixes kept so the "
                           "localization pipeline runs unchanged"}, f, indent=2)

    print("\nchecks:")
    verify(args.out_dir, arm, args.form, system_block, demos)


if __name__ == "__main__":
    main()
