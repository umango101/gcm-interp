#!/usr/bin/env python3
"""Check CONFLICT_CATEGORIES against the real tokenizer before you rely on them.

    python check_categories.py

A category is usable only if its two answer words tokenize to the same length --
otherwise the privileged and subordinate rule strings differ in length, the two
preambles misalign, and every position-wise comparison in the patching code is
wrong. select_demos_rule drops unusable categories silently at build time; this
tells you which ones those are, so a demo-count sweep does not quietly run with
fewer categories than you asked for.

It also re-checks the COLOR_POOL collision guard against the tokenizer, and
reports how many demos you can actually reach.
"""

import argparse

from transformers import AutoTokenizer

import hierarchy_common as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    def n(w):
        return len(tok.encode(w, add_special_tokens=False))

    print(f"{'category':<14}{'privileged':<12}{'tok':>4}"
          f"{'subordinate':>14}{'tok':>4}   status")
    print("-" * 60)
    usable = 0
    for cat, p, s in H.CONFLICT_CATEGORIES:
        np_, ns = n(p), n(s)
        ok = np_ == ns
        usable += ok
        print(f"{cat:<14}{p:<12}{np_:>4}{s:>14}{ns:>4}   "
              f"{'ok' if ok else 'DROPPED (length mismatch)'}")

    print(f"\n{usable}/{len(H.CONFLICT_CATEGORIES)} categories usable "
          f"-> max n_conflict_demos = {usable}")

    print(f"\nagreement categories")
    for cat, w in H.AGREE_CATEGORIES:
        print(f"  {cat:<12}{w:<12}{n(w):>4}")

    clash = ({w for _, w, _ in H.CONFLICT_CATEGORIES}
             | {w for _, _, w in H.CONFLICT_CATEGORIES}
             | {w for _, w in H.AGREE_CATEGORIES}) & set(H.COLOR_POOL)
    print(f"\nCOLOR_POOL collisions: {sorted(clash) if clash else 'none'}")


if __name__ == "__main__":
    main()
