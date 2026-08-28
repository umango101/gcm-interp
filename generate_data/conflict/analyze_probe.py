#!/usr/bin/env python3
"""Summarize a probe_levels*.json run.

    python analyze_probe.py probe_levels_v2.json

Stdlib only, no GPU: everything comes from the per-row records the probe wrote.
Runs from the older probe are handled too -- fields it did not write show as "-"
instead of crashing.

WHAT EACH COLUMN IS
  forced    the rule word outscores the alternative. This is the logit
            difference ATP differentiates, so it is the gate.
  95% CI    Wilson interval on forced. At n=100 the half-width is several
            points even at ceiling, so treat smaller differences as noise
            rather than as a hierarchy effect.
  1st/2nd   forced, split by whether the rule word was named first. The gap is
            position bias showing through; read it against norule's gap, which
            is that bias with no rule present at all.
  argmax    the top token is the rule word in some surface form. A large
            forced-argmax gap means the preference is right but the emitted
            token is not, so generation-scored evals will be noisy on that
            variant even though the logit readout is clean.
  nll_pre   mean surprisal of the PREFIX (system block, rule, any developer
            message). This is the off-distribution measure. nll_final is a
            different quantity -- how predictable the question is given that
            prefix -- and a stripped-down prefix makes a color question MORE
            expected, so a malformed variant can score lower on it.
"""

import sys
import json
import math
from collections import Counter

CEILING = 0.95          # forced CI lower bound needed to call a placement clean
MAX_ORDER_GAP = 0.05    # |1st - 2nd| above this is position bias showing through
MAX_SURFACE_GAP = 0.10  # forced - argmax above this means surface-form trouble

NAN = float("nan")


def wilson(k, n, z=1.96):
    """95% CI for a proportion. Beats the normal approximation at ceiling,
    which is where every interesting number in this table sits."""
    if n == 0:
        return (NAN, NAN)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else NAN


def _rate(rows, key):
    vals = [bool(r[key]) for r in rows if key in r]
    return (sum(vals), len(vals))


def _frac(rows, key):
    k, n = _rate(rows, key)
    return k / n if n else NAN


def pct(x):
    return "-" if x != x else f"{x:.0%}"


def num(x, digits=2):
    return "-" if x != x else f"{x:+.{digits}f}"


def summarize(name, rows):
    k, n = _rate(rows, "forced_choice")
    first = [r for r in rows if r.get("mention_first_is_rule_word")]
    second = [r for r in rows if not r.get("mention_first_is_rule_word")]
    f1, f2 = _frac(first, "forced_choice"), _frac(second, "forced_choice")
    return {
        "name": name,
        "n": len(rows),
        "forced": k / n if n else NAN,
        "ci": wilson(k, n) if n else (NAN, NAN),
        "first": f1,
        "second": f2,
        "gap": f1 - f2 if f1 == f1 and f2 == f2 else NAN,
        "argmax": _frac(rows, "complied"),
        "offtask": _frac(rows, "offtask"),
        "margin": _mean(rows, "margin"),
        "nll_prefix": _mean(rows, "nll_prefix"),
        "nll_final": _mean(rows, "nll_final_turn"),
    }


def verdict(s, floor_gap):
    if s["name"] == "norule":
        return "FLOOR -- no rule present; anything above 50% here is pure bias"
    if s["ci"][0] != s["ci"][0]:
        return "no forced_choice field in this run; rerun the probe"
    notes = []
    if s["ci"][0] >= CEILING and abs(s["gap"]) <= MAX_ORDER_GAP:
        notes.append("USABLE")
    elif s["forced"] >= 0.90:
        notes.append(f"MARGINAL (forced CI lower {s['ci'][0]:.1%} vs the "
                     f"{CEILING:.0%} threshold)")
    else:
        notes.append(f"NOT USABLE (forced CI lower {s['ci'][0]:.1%})")
    if s["gap"] == s["gap"] and abs(s["gap"]) > MAX_ORDER_GAP:
        share = ""
        if floor_gap == floor_gap and floor_gap:
            share = f" ({abs(s['gap']) / abs(floor_gap):.0%} of the norule gap)"
        notes.append(f"order gap {s['gap']:+.0%}{share}")
    if s["forced"] == s["forced"] and s["argmax"] == s["argmax"] \
            and (s["forced"] - s["argmax"]) > MAX_SURFACE_GAP:
        notes.append(f"surface-form gap {s['forced'] - s['argmax']:+.0%}: "
                     "generation-scored evals will be noisy")
    return "; ".join(notes)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "probe_levels.json"
    with open(path) as f:
        blob = json.load(f)
    results = blob["results"]
    stats = {name: summarize(name, r["rows"]) for name, r in results.items()}
    floor_gap = stats.get("norule", {}).get("gap", NAN)

    args = blob.get("args", {})
    bits = [f"{k}={v}" for k, v in args.items()
            if k in ("model", "reasoning", "date", "generation_prompt", "n_pairs")]
    print(f"{path}   {'  '.join(bits)}\n")

    cols = [("variant", 18), ("n", 5), ("forced", 8), ("95% CI", 14),
            ("1st", 6), ("2nd", 6), ("gap", 7), ("argmax", 8),
            ("offtask", 9), ("margin", 8), ("nll_pre", 9), ("nll_fin", 9)]
    print("".join(h.rjust(w) if i else h.ljust(w)
                  for i, (h, w) in enumerate(cols)))
    print("-" * sum(w for _, w in cols))
    for s in stats.values():
        ci = "-" if s["ci"][0] != s["ci"][0] else \
            f"[{s['ci'][0]:.0%},{s['ci'][1]:.0%}]"
        row = [s["name"].ljust(18), str(s["n"]).rjust(5),
               pct(s["forced"]).rjust(8), ci.rjust(14),
               pct(s["first"]).rjust(6), pct(s["second"]).rjust(6),
               pct(s["gap"]).rjust(7), pct(s["argmax"]).rjust(8),
               pct(s["offtask"]).rjust(9), num(s["margin"]).rjust(8),
               num(s["nll_prefix"], 3).rjust(9), num(s["nll_final"], 3).rjust(9)]
        print("".join(row))

    print("\nverdicts")
    for s in stats.values():
        print(f"  {s['name']:<18} {verdict(s, floor_gap)}")

    # --- surprisal ----------------------------------------------------------
    # Only prefixes that actually carry a rule are comparable: norule's prefix
    # is different text (canonical block, nothing appended), so its lower
    # surprisal says "boilerplate is predictable", not "this placement is more
    # natural". It is a scale check, not a floor.
    base = stats.get("devuser") or next(iter(stats.values()))
    ruled = [s for s in stats.values()
             if s["name"] != "norule" and s["nll_prefix"] == s["nll_prefix"]]
    if not ruled or base["nll_prefix"] != base["nll_prefix"]:
        print("\nprefix surprisal: not recorded in this run "
              "(nll_prefix was added after it)")
        ruled = []
    else:
        print(f"\nprefix surprisal, vs {base['name']} "
              f"({base['nll_prefix']:.3f} nats/token)")
    for s2 in ruled:
        d = s2["nll_prefix"] - base["nll_prefix"]
        print(f"  {s2['name']:<18} {s2['nll_prefix']:.3f}  "
              f"{'' if s2 is base else f'{d:+.3f}'}")
    if ruled and "norule" in stats \
            and stats["norule"]["nll_prefix"] == stats["norule"]["nll_prefix"]:
        print(f"  {'norule':<18} {stats['norule']['nll_prefix']:.3f}  "
              f"{stats['norule']['nll_prefix'] - base['nll_prefix']:+.3f}  "
              f"(different text -- no rule appended; a scale check, not a floor)")
    spread = (max(s2["nll_prefix"] for s2 in ruled)
              - min(s2["nll_prefix"] for s2 in ruled)) if ruled else NAN
    if spread == spread:
        print(f"  spread across rule-bearing prefixes: {spread:.3f} nats/token")
        if "norule" in stats and \
                stats["norule"]["nll_prefix"] == stats["norule"]["nll_prefix"]:
            rule_cost = base["nll_prefix"] - stats["norule"]["nll_prefix"]
            print(f"  for scale, appending a rule at all costs "
                  f"{rule_cost:+.3f} -- if the spread is well under that, the\n"
                  f"  placements are not measurably different in how "
                  f"off-distribution they are.")
    if ruled:
        print("  nll_fin is NOT a second opinion on this: it measures how "
              "predictable the\n  question is given the prefix, so a prefix "
              "with less around the rule makes a\n  color question MORE "
              "expected and scores lower. Do not read it as naturalness.")

    print("\noff-task argmax tokens")
    for name, r in results.items():
        off = [x for x in r["rows"] if x.get("offtask")]
        if not off:
            print(f"  {name:<18} none")
            continue
        # A surface variant of the rule word means the model complied and the
        # scorer disagreed about spelling. Anything else is a real miss, and
        # what those tokens are is usually the whole diagnosis.
        variant = sum(1 for x in off
                      if x.get("argmax_token", "").strip().lower()
                      == x.get("rule_word", "").lower())
        common = Counter(repr(x.get("argmax_token", "")) for x in off).most_common(6)
        print(f"  {name:<18} {len(off):>3} off-task; {variant} are a surface "
              f"variant of the rule word")
        print(f"  {'':<18} {', '.join(f'{t} x{c}' for t, c in common)}")

    clean = [s["name"] for s in stats.values()
             if s["name"] != "norule" and s["ci"][0] == s["ci"][0]
             and s["ci"][0] >= CEILING and abs(s["gap"]) <= MAX_ORDER_GAP]
    print(f"\nclean placements: {', '.join(clean) if clean else 'none'}")


if __name__ == "__main__":
    main()
