#!/usr/bin/env python3
"""Does attribution patching predict which layers causally matter?

    python atp_vs_measured.py
    python atp_vs_measured.py --arms devuser --tests dev-single-test

Attribution patching is a first-order approximation to activation patching, and
its linearization is worst exactly where effects are large -- which is the
regime any localization claim lives in. The per-layer sweep makes this testable
in a way the head arm cannot: it steers all 24 layers INDIVIDUALLY and measures
each, so nothing is selected and there is a causal ground truth for every layer.
The ATP score is then a prediction and the measured flip rate is the outcome.

The head arm has no equivalent. It steers only the top-k ATP chose, so a head
the map ranks low is never tested and the map cannot be scored against anything.
Sweeping all 1536 heads individually is infeasible, which is why the layer arm
carries this claim -- worth stating rather than leaving as an apparent
inconsistency between the two arms.

WHAT IS REPORTED, PER (ARM x TEST FILE x COEFFICIENT)

  spearman / pearson  ATP marginal score vs measured user_net, over 24 layers.
                      Spearman is primary: ATP scores and flip rates are not on
                      a common scale and only the ORDERING is claimed.
  top-k hit rate      how many of ATP's top-k layers are in the measured top-k.
  best layer          ATP's argmax vs the measured argmax. A localization claim
                      that misses the single most effective layer is in trouble
                      however well the ranks correlate.
  damage-adjusted     the same correlation after dropping layers whose
                      broken_post exceeds a threshold. A layer that destroys
                      generation scores user_net near zero for a reason that has
                      nothing to do with carrying the policy, and ATP has no way
                      to represent "this layer is load-bearing but fragile".
                      Without this the correlation is penalised for a failure
                      that is not ATP's.

Coefficient matters and is not averaged over. At small N nothing moves and the
correlation is noise; at large N everything breaks and it is noise again. The
informative comparison is at the N where the sweep is actually discriminating,
so every N is reported and the choice is visible rather than buried.
"""

import os
import re
import csv
import json
import glob
import math
import argparse
import statistics


def spearman(xs, ys):
    """Rank correlation, average ranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def load_atp(path):
    """{layer: marginal}. `marginal` is the per-layer contribution; `cumulative`
    is a running sum and would correlate with layer index rather than effect."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(float(row["layer"]))] = float(row["marginal"])
    return out


def load_measured(path):
    """{N: {layer: {...}}} from the per-layer sweep summary.

    In per-layer mode the `topk` field of each key is a LAYER INDEX, not a top-k
    fraction -- the scorer's axis names are shared with the head arm.
    """
    key_re = re.compile(r"^N(?P<n>\d+)_topk(?P<layer>\d+(?:\.\d+)?)$")
    out = {}
    for k, v in json.load(open(path)).items():
        m = key_re.match(k)
        if m:
            out.setdefault(int(m.group("n")), {})[int(float(m.group("layer")))] = v
    return out


def analyse(atp, measured, n, metric, k, max_broken):
    layers = sorted(set(atp) & set(measured))
    if len(layers) < 3:
        return None
    # abs(): ATP's sign says which way a layer pushes; the magnitude is the
    # claim about how much it matters, and the sweep measures a single steering
    # direction so only magnitude is comparable.
    pred = [abs(atp[l]) for l in layers]
    obs = [measured[l][metric] for l in layers]

    top_pred = {l for l in sorted(layers, key=lambda l: -abs(atp[l]))[:k]}
    top_obs = {l for l in sorted(layers, key=lambda l: -measured[l][metric])[:k]}

    keep = [l for l in layers if measured[l]["broken_post"] <= max_broken]
    if len(keep) >= 3:
        adj = spearman([abs(atp[l]) for l in keep],
                       [measured[l][metric] for l in keep])
    else:
        adj = float("nan")

    return {
        "n_layers": len(layers),
        "spearman": spearman(pred, obs),
        "pearson": pearson(pred, obs),
        "spearman_damage_adjusted": adj,
        "n_kept_after_damage_filter": len(keep),
        "topk_hits": len(top_pred & top_obs),
        "k": k,
        "atp_best": max(layers, key=lambda l: abs(atp[l])),
        "measured_best": max(layers, key=lambda l: measured[l][metric]),
        "measured_best_value": max(measured[l][metric] for l in layers),
        "atp_best_measured_value": measured[max(layers, key=lambda l: abs(atp[l]))][metric],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers_results", default="results_layers")
    ap.add_argument("--scored", default="results_pipeline_conflict_single_layers")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--arms", nargs="+", default=["devuser", "sysuser", "sysdev"])
    ap.add_argument("--tests", nargs="+",
                    default=["dev-single-test", "devNaive-single-test"])
    ap.add_argument("--metric", default="user_net")
    ap.add_argument("--k", type=int, default=5, help="top-k for the hit rate")
    ap.add_argument("--max_broken", type=float, default=0.25,
                    help="drop layers whose broken_post exceeds this from the "
                         "damage-adjusted correlation")
    ap.add_argument("--out", default="atp_vs_measured.json")
    args = ap.parse_args()

    hdr = (f"{'arm':<9}{'test':<22}{'N':>4}{'rho':>7}{'r':>7}{'rho|dmg':>9}"
           f"{'kept':>6}{'hit':>5}{'ATP*':>6}{'meas*':>7}{'ATP*val':>9}{'best':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for arm in args.arms:
        atp_path = os.path.join(args.layers_results, args.model,
                                f"{arm}__from_user-single_to_dev-single",
                                "atp", "eval", "layer_effects.csv")
        if not os.path.exists(atp_path):
            print(f"  {arm}: no layer_effects.csv at {atp_path}")
            continue
        atp = load_atp(atp_path)

        for test in args.tests:
            pat = os.path.join(
                args.scored, args.model, f"{arm}__from_user-single_to_dev-single",
                "atp-per-layer", f"{arm}-{test}_eval", "*_steer",
                "targeted_steer", "accuracy", "summary.json")
            fs = glob.glob(pat)
            if not fs:
                print(f"  {arm}/{test}: no scored summary; run "
                      f"eval_pipeline_conflict_layers_hier.py first")
                continue
            measured = load_measured(fs[0])

            for n in sorted(measured):
                a = analyse(atp, measured[n], n, args.metric, args.k,
                            args.max_broken)
                if not a:
                    continue
                results[f"{arm}:{test}:N{n}"] = a
                print(f"{arm:<9}{test:<22}{n:>4}{a['spearman']:>7.2f}"
                      f"{a['pearson']:>7.2f}{a['spearman_damage_adjusted']:>9.2f}"
                      f"{a['n_kept_after_damage_filter']:>6}"
                      f"{a['topk_hits']:>3}/{a['k']}{a['atp_best']:>6}"
                      f"{a['measured_best']:>7}"
                      f"{a['atp_best_measured_value']:>9.2f}"
                      f"{a['measured_best_value']:>7.2f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nHOW TO READ\n"
          "  rho is the headline: ATP's predicted ordering against the measured\n"
          "  one, over all 24 layers. Compare ATP* and meas* -- if ATP's top\n"
          "  layer achieves much less than the measured best, the map misses the\n"
          "  most effective layer even when rho looks acceptable, and that is\n"
          "  the failure a top-k localization would actually suffer.\n"
          "  Read rho at the N where the sweep discriminates. At tiny N nothing\n"
          "  moves and at large N everything breaks; both give noise.\n"
          "  rho|dmg drops layers that mostly break generation, so ATP is not\n"
          "  penalised for lacking a way to say 'load-bearing but fragile'.")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""Does attribution patching predict which layers causally matter?

    python atp_vs_measured.py
    python atp_vs_measured.py --arms devuser --tests dev-single-test

Attribution patching is a first-order approximation to activation patching, and
its linearization is worst exactly where effects are large -- which is the
regime any localization claim lives in. The per-layer sweep makes this testable
in a way the head arm cannot: it steers all 24 layers INDIVIDUALLY and measures
each, so nothing is selected and there is a causal ground truth for every layer.
The ATP score is then a prediction and the measured flip rate is the outcome.

The head arm has no equivalent. It steers only the top-k ATP chose, so a head
the map ranks low is never tested and the map cannot be scored against anything.
Sweeping all 1536 heads individually is infeasible, which is why the layer arm
carries this claim -- worth stating rather than leaving as an apparent
inconsistency between the two arms.

WHAT IS REPORTED, PER (ARM x TEST FILE x COEFFICIENT)

  spearman / pearson  ATP marginal score vs measured user_net, over 24 layers.
                      Spearman is primary: ATP scores and flip rates are not on
                      a common scale and only the ORDERING is claimed.
  top-k hit rate      how many of ATP's top-k layers are in the measured top-k.
  best layer          ATP's argmax vs the measured argmax. A localization claim
                      that misses the single most effective layer is in trouble
                      however well the ranks correlate.
  damage-adjusted     the same correlation after dropping layers whose
                      broken_post exceeds a threshold. A layer that destroys
                      generation scores user_net near zero for a reason that has
                      nothing to do with carrying the policy, and ATP has no way
                      to represent "this layer is load-bearing but fragile".
                      Without this the correlation is penalised for a failure
                      that is not ATP's.

Coefficient matters and is not averaged over. At small N nothing moves and the
correlation is noise; at large N everything breaks and it is noise again. The
informative comparison is at the N where the sweep is actually discriminating,
so every N is reported and the choice is visible rather than buried.
"""

import os
import re
import csv
import json
import glob
import math
import argparse
import statistics


def spearman(xs, ys):
    """Rank correlation, average ranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def load_atp(path):
    """{layer: marginal}. `marginal` is the per-layer contribution; `cumulative`
    is a running sum and would correlate with layer index rather than effect."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(float(row["layer"]))] = float(row["marginal"])
    return out


def load_measured(path):
    """{N: {layer: {...}}} from the per-layer sweep summary.

    In per-layer mode the `topk` field of each key is a LAYER INDEX, not a top-k
    fraction -- the scorer's axis names are shared with the head arm.
    """
    key_re = re.compile(r"^N(?P<n>\d+)_topk(?P<layer>\d+(?:\.\d+)?)$")
    out = {}
    for k, v in json.load(open(path)).items():
        m = key_re.match(k)
        if m:
            out.setdefault(int(m.group("n")), {})[int(float(m.group("layer")))] = v
    return out


def analyse(atp, measured, n, metric, k, max_broken):
    layers = sorted(set(atp) & set(measured))
    if len(layers) < 3:
        return None
    # abs(): ATP's sign says which way a layer pushes; the magnitude is the
    # claim about how much it matters, and the sweep measures a single steering
    # direction so only magnitude is comparable.
    pred = [abs(atp[l]) for l in layers]
    obs = [measured[l][metric] for l in layers]

    top_pred = {l for l in sorted(layers, key=lambda l: -abs(atp[l]))[:k]}
    top_obs = {l for l in sorted(layers, key=lambda l: -measured[l][metric])[:k]}

    keep = [l for l in layers if measured[l]["broken_post"] <= max_broken]
    if len(keep) >= 3:
        adj = spearman([abs(atp[l]) for l in keep],
                       [measured[l][metric] for l in keep])
    else:
        adj = float("nan")

    return {
        "n_layers": len(layers),
        "spearman": spearman(pred, obs),
        "pearson": pearson(pred, obs),
        "spearman_damage_adjusted": adj,
        "n_kept_after_damage_filter": len(keep),
        "topk_hits": len(top_pred & top_obs),
        "k": k,
        "atp_best": max(layers, key=lambda l: abs(atp[l])),
        "measured_best": max(layers, key=lambda l: measured[l][metric]),
        "measured_best_value": max(measured[l][metric] for l in layers),
        "atp_best_measured_value": measured[max(layers, key=lambda l: abs(atp[l]))][metric],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers_results", default="results_layers")
    ap.add_argument("--scored", default="results_pipeline_conflict_single_layers")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--arms", nargs="+", default=["devuser", "sysuser", "sysdev"])
    ap.add_argument("--tests", nargs="+",
                    default=["dev-single-test", "devNaive-single-test"])
    ap.add_argument("--metric", default="user_net")
    ap.add_argument("--k", type=int, default=5, help="top-k for the hit rate")
    ap.add_argument("--max_broken", type=float, default=0.25,
                    help="drop layers whose broken_post exceeds this from the "
                         "damage-adjusted correlation")
    ap.add_argument("--out", default="atp_vs_measured.json")
    args = ap.parse_args()

    hdr = (f"{'arm':<9}{'test':<22}{'N':>4}{'rho':>7}{'r':>7}{'rho|dmg':>9}"
           f"{'kept':>6}{'hit':>5}{'ATP*':>6}{'meas*':>7}{'ATP*val':>9}{'best':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for arm in args.arms:
        atp_path = os.path.join(args.layers_results, args.model,
                                f"{arm}__from_user-single_to_dev-single",
                                "atp", "eval", "layer_effects.csv")
        if not os.path.exists(atp_path):
            print(f"  {arm}: no layer_effects.csv at {atp_path}")
            continue
        atp = load_atp(atp_path)

        for test in args.tests:
            pat = os.path.join(
                args.scored, args.model, f"{arm}__from_user-single_to_dev-single",
                "atp-per-layer", f"{arm}-{test}_eval", "*_steer",
                "targeted_steer", "accuracy", "summary.json")
            fs = glob.glob(pat)
            if not fs:
                print(f"  {arm}/{test}: no scored summary; run "
                      f"eval_pipeline_conflict_layers_hier.py first")
                continue
            measured = load_measured(fs[0])

            for n in sorted(measured):
                a = analyse(atp, measured[n], n, args.metric, args.k,
                            args.max_broken)
                if not a:
                    continue
                results[f"{arm}:{test}:N{n}"] = a
                print(f"{arm:<9}{test:<22}{n:>4}{a['spearman']:>7.2f}"
                      f"{a['pearson']:>7.2f}{a['spearman_damage_adjusted']:>9.2f}"
                      f"{a['n_kept_after_damage_filter']:>6}"
                      f"{a['topk_hits']:>3}/{a['k']}{a['atp_best']:>6}"
                      f"{a['measured_best']:>7}"
                      f"{a['atp_best_measured_value']:>9.2f}"
                      f"{a['measured_best_value']:>7.2f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nHOW TO READ\n"
          "  rho is the headline: ATP's predicted ordering against the measured\n"
          "  one, over all 24 layers. Compare ATP* and meas* -- if ATP's top\n"
          "  layer achieves much less than the measured best, the map misses the\n"
          "  most effective layer even when rho looks acceptable, and that is\n"
          "  the failure a top-k localization would actually suffer.\n"
          "  Read rho at the N where the sweep discriminates. At tiny N nothing\n"
          "  moves and at large N everything breaks; both give noise.\n"
          "  rho|dmg drops layers that mostly break generation, so ATP is not\n"
          "  penalised for lacking a way to say 'load-bearing but fragile'.")


if __name__ == "__main__":
    main()
