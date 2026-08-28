#!/usr/bin/env python3
"""Build the no-preamble transfer set: devNaive-single-test.jsonl.

Same conflict, same items, same counterbalancing as dev-single-test.jsonl, with
the ICL demonstration preamble removed. The remaining conflict has to be settled
by whatever prior the model was trained with rather than by anything shown in
context, which makes this the transfer target for heads localized on the
in-context contrast: steer the same sites here and see whether they still move
the answer.

    python make_devnaive_test.py \
        --in data/gpt-oss-20b/hier-devuser/dev-single-test.jsonl

--meta and --out default to the layout the generator writes, beside --in.

WHY THIS REBUILDS INSTEAD OF STRIPPING TURNS
--------------------------------------------
An earlier version deleted the demo turns from each record. That is wrong in the
rule form: for arms whose subordinate is the user, the subordinate's RULE rides
on the first question turn, so deleting the preamble deletes one side of the
conflict and leaves an item with a single rule in it. The failure is silent --
the file still parses, the counts still look right, and the localization runs
against a corpus that is not testing a conflict at all.

Rebuilding each item through build_line with an empty demo list cannot make that
mistake: the builder places the subordinate rule on the first message the asker
sends, whichever message that turns out to be. Every field is taken from the
record's own metadata, so the pairs, mention order, role assignment and template
are identical to the preamble version item for item.
"""

import json
import argparse
from collections import Counter

import hierarchy_common as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="dev-single-test.jsonl for one arm")
    # Both default to the layout make_hierarchy_datasets.py writes, so the
    # common case is just --in. The meta carries the system block and the
    # active category list: the rules here must be built from the same
    # categories the preamble version used.
    ap.add_argument("--meta", default=None,
                    help="candidate_meta.json for the SAME arm "
                         "(default: <dir of --in>/candidates/candidate_meta.json)")
    ap.add_argument("--out", default=None,
                    help="default: <dir of --in>/devNaive-single-test.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import os
    d = os.path.dirname(args.inp) or "."
    if args.meta is None:
        args.meta = os.path.join(d, "candidates", "candidate_meta.json")
    if args.out is None:
        args.out = os.path.join(d, "devNaive-single-test.jsonl")
    if not os.path.exists(args.meta):
        raise SystemExit(
            f"no meta at {args.meta}. Pass --meta explicitly; it must be the "
            "candidate_meta.json from the same arm's stage 1.")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    with open(args.meta) as f:
        meta = json.load(f)
    if meta.get("categories"):
        H.set_active_categories(meta["categories"])
    system_block = meta["system_block"]

    with open(args.inp) as f:
        rows = [json.loads(l) for l in f]
    if not rows:
        raise SystemExit(f"{args.inp} is empty")

    arm_key = rows[0]["arm"]
    form = rows[0].get("conflict_form", "request")
    if meta.get("arm") != arm_key or meta.get("form") != form:
        raise SystemExit(
            f"{args.meta} is for arm {meta.get('arm')!r} / form "
            f"{meta.get('form')!r}, but the test file is {arm_key!r} / {form!r}. "
            "Each arm needs its own meta.")
    arm = H.ARMS[arm_key]

    out, checks = [], Counter()
    for i, r in enumerate(rows):
        if r["arm"] != arm_key or r.get("conflict_form", "request") != form:
            raise SystemExit(f"line {i}: mixed arms or forms in one file")
        a, b = r["pair_key"].split("|")
        first = r["mention_first"]
        second = b if first == a else a

        # Sanity: the preamble version has to round-trip from this metadata, or
        # the naive version built from the same metadata is not the same item.
        full = H.build_line(arm, form, system_block, a, b, r["dev_word"],
                            r["user_word"], first, second, r["condition"],
                            r["target"], [tuple(d) for d in meta["demos"]],
                            r["template"])
        if full["prompt"] != r["prompt"]:
            raise SystemExit(
                f"line {i}: the test record does not round-trip from its own "
                "metadata. The meta file and the test file are out of step -- "
                "regenerate the arm rather than trusting this output.")

        rec = H.build_line(arm, form, system_block, a, b, r["dev_word"],
                           r["user_word"], first, second, r["condition"],
                           None, [], r["template"])
        rec["preamble"] = "none"
        rec["n_demos_removed"] = len(meta["demos"])

        # Both sides of the conflict must survive. For a user-subordinate arm in
        # the rule form this is the whole point of rebuilding.
        text = " ".join(m["content"] for m in rec["prompt"])
        if r["dev_word"] not in text:
            raise SystemExit(f"line {i}: privileged word missing after rebuild")
        if r["user_word"] not in text:
            raise SystemExit(f"line {i}: subordinate word missing after rebuild")
        checks[tuple(m["role"] for m in rec["prompt"])] += 1
        out.append(rec)

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, args.out)

    print(f"wrote {args.out}  ({len(out)} records)")
    print(f"  arm {arm_key} / form {form}: privileged={arm.privileged}, "
          f"subordinate={arm.subordinate}")
    for roles, n in checks.items():
        print(f"  {n:>4} records with roles {list(roles)}")
    print(f"  demos removed per record: {len(meta['demos'])}")
    print(f"  mention_first == dev_word: "
          f"{sum(r['mention_first'] == r['dev_word'] for r in out)}/{len(out)}")
    print(f"  distinct pair_key: {len({r['pair_key'] for r in out})}")


if __name__ == "__main__":
    main()
