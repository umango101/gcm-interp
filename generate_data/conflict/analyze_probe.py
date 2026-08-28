#!/usr/bin/env python3
"""Read probe_levels.json and say what the off-task argmax tokens actually are.

No GPU, no model: this reads the per-row records the probe already wrote.

    python analyze_probe.py probe_levels.json

Two questions it answers:

  1. WHAT ARE THE OFF-TASK TOKENS? If they are surface variants of the rule word
     -- capitalized, leading-space, quoted -- then the compliance numbers are a
     scoring artifact and the model was complying all along.

  2. WHAT IS THE FORCED-CHOICE ACCURACY? The fraction of items where the rule
     word outscores the alternative, i.e. margin > 0. This is the quantity the
     ATP objective actually differentiates -- a logit difference between two
     candidate answers -- so it is the compliance metric that matches the method,
     and it is unaffected by whichever surface form wins the argmax.
"""

import sys
import json
from collections import Counter


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "probe_levels.json"
    with open(path) as f:
        blob = json.load(f)
    results = blob["results"]

    print(f"{'variant':<16} {'argmax':>7} {'forced':>7} {'fc_first':>9} "
          f"{'fc_second':>10} {'mean_margin':>12}")
    for name, r in results.items():
        rows = r["rows"]
        n = len(rows)
        forced = [x["margin"] > 0 for x in rows]
        first = [x for x in rows if x["mention_first_is_rule_word"]]
        second = [x for x in rows if not x["mention_first_is_rule_word"]]
        fc_first = sum(x["margin"] > 0 for x in first) / len(first) if first else float("nan")
        fc_second = sum(x["margin"] > 0 for x in second) / len(second) if second else float("nan")
        print(f"{name:<16} {sum(x['complied'] for x in rows) / n:>7.0%} "
              f"{sum(forced) / n:>7.0%} {fc_first:>9.0%} {fc_second:>10.0%} "
              f"{sum(x['margin'] for x in rows) / n:>+12.2f}")

    print("\noff-task argmax tokens, most common first:")
    for name, r in results.items():
        off = [x for x in r["rows"] if x["offtask"]]
        if not off:
            print(f"  {name:<16} none")
            continue
        # Is the off-task token a surface variant of the word the rule asked
        # for? That is the difference between a scoring bug and a real refusal
        # to comply.
        variant = sum(1 for x in off
                      if x["argmax_token"].strip().lower() == x["rule_word"].lower())
        common = Counter(repr(x["argmax_token"]) for x in off).most_common(8)
        print(f"  {name:<16} {len(off):>3} off-task, {variant} of them a "
              f"surface variant of the rule word")
        print(f"       {', '.join(f'{t} x{c}' for t, c in common)}")


if __name__ == "__main__":
    main()
