#!/usr/bin/env python3
"""Unsteered deference on every test file, before any intervention.

Scores dev-single-test.jsonl (with ICL preamble) and devNaive-single-test.jsonl
(no preamble) for each arm: six files, ~600 forward passes, no generation.

    python score_baselines.py --data_root data/gpt-oss-20b --out baselines.json

WHY BOTH FILES
--------------
dev-single-test should sit near ceiling. It is held-out colors under the same
preamble QC gated on, so anything much below the QC line rate means the held-out
colors behave differently from the localization set, and every steering number
measured on it inherits that.

devNaive is the baseline a transfer result moves away from. A steering effect
there is only interpretable against the model's prior deference with no
demonstrations -- and in sysdev, where two rule lists sit adjacent with nothing
to arbitrate them, that prior may be weak or order-driven. If naive deference is
near 50% with a large order split, the arm has no stable prior to transfer to,
and that has to be known before the steering run rather than after.

WHAT IS REPORTED
----------------
  deference   the privileged word outscores the subordinate one. This is the
              forced-choice logit difference ATP differentiates, so it is the
              quantity that connects to the localization.
  95% CI      Wilson. At n=100 the half-width is 4-8 points; differences smaller
              than that between arms or files are not differences.
  1st/2nd     deference split by whether the privileged word is named first.
              The gap is position bias showing through. The probe measured a
              36-point gap with no rule present at all, so a gap here is read
              against that, not against zero.
  argmax      the top token is the privileged word in some surface form. A large
              deference-argmax gap means the preference is right but the emitted
              token is not, which matters if anything downstream is scored by
              generation rather than by logits.
"""

import os
import json
import math
import argparse
import statistics

import determinism
import harmony_canonical as hc
from answer_scoring import score, collision

# torch/transformers are imported inside main(), so --dry_run works on a login
# node without the GPU stack loaded.

FILES = [("dev-single-test.jsonl", "preamble"),
         ("devNaive-single-test.jsonl", "naive")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def render(tok, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    if not text.rstrip().endswith("<|message|>"):
        text = text + "<|channel|>final<|message|>"
    return text


def score_file(model, tok, rows):
    out, skipped = [], 0
    for r in rows:
        msgs = r["prompt"]
        if msgs[-1]["role"] == "assistant":       # test files should not have one
            msgs = msgs[:-1]
        # The privileged word is the target: these files all carry the
        # privileged-following condition, so deference means it wins.
        target, distractor = r["dev_word"], r["user_word"]
        if collision(tok, target, distractor):
            skipped += 1
            continue
        import torch
        with torch.no_grad():
            ids = tok(render(tok, msgs), return_tensors="pt").to(model.device)
            sc = score(model(**ids).logits[0, -1], tok, target, distractor)
        sc["privileged_first"] = r["mention_first"] == target
        out.append(sc)
    return out, skipped


def summarize(rows):
    n = len(rows)
    k = sum(r["forced_choice"] for r in rows)
    first = [r for r in rows if r["privileged_first"]]
    second = [r for r in rows if not r["privileged_first"]]
    f1 = sum(r["forced_choice"] for r in first) / len(first) if first else float("nan")
    f2 = sum(r["forced_choice"] for r in second) / len(second) if second else float("nan")
    return {
        "n": n,
        "deference": k / n if n else float("nan"),
        "ci": wilson(k, n),
        "first": f1,
        "second": f2,
        "gap": f1 - f2,
        "argmax": sum(r["complied"] for r in rows) / n if n else float("nan"),
        "offtask": sum(r["offtask"] for r in rows) / n if n else float("nan"),
        "mean_margin": statistics.mean(r["margin"] for r in rows) if n else float("nan"),
        "sd_margin": statistics.stdev(r["margin"] for r in rows) if n > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/gpt-oss-20b",
                    help="directory holding the hier-<arm> folders")
    ap.add_argument("--arms", nargs="+",
                    default=["hier-devuser", "hier-sysuser", "hier-sysdev"])
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--generation_prompt", choices=["final", "bare"], default="final",
                    help="must match what localization and steering use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_nondeterministic", action="store_true")
    ap.add_argument("--out", default="baselines.json")
    ap.add_argument("--dry_run", action="store_true",
                    help="load and check the files, skip the model")
    args = ap.parse_args()

    todo = []
    for arm in args.arms:
        for fname, kind in FILES:
            path = os.path.join(args.data_root, arm, fname)
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}")
            with open(path) as f:
                rows = [json.loads(l) for l in f]
            todo.append((arm, kind, path, rows))
            print(f"  {arm:<14} {kind:<9} {len(rows):>4} records  {path}")

    if args.dry_run:
        print("\ndry run: files load and parse; skipping the model")
        return

    fingerprint = determinism.enforce(
        args.seed, allow_nondeterministic=args.allow_nondeterministic)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = hc.install(AutoTokenizer.from_pretrained(args.model),
                     generation_prompt=args.generation_prompt)
    print(f"\nloading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto",
                                                 device_map="auto")
    model.eval()

    results = {}
    hdr = (f"\n{'arm':<14}{'file':<10}{'n':>5}{'defer':>8}{'95% CI':>14}"
           f"{'1st':>6}{'2nd':>6}{'gap':>7}{'argmax':>8}{'offtask':>9}{'margin':>9}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for arm, kind, path, rows in todo:
        scored, skipped = score_file(model, tok, rows)
        s = summarize(scored)
        s["skipped_collisions"] = skipped
        results[f"{arm}:{kind}"] = s
        ci = f"[{s['ci'][0]:.0%},{s['ci'][1]:.0%}]"
        print(f"{arm:<14}{kind:<10}{s['n']:>5}{s['deference']:>8.0%}{ci:>14}"
              f"{s['first']:>6.0%}{s['second']:>6.0%}{s['gap']:>+7.0%}"
              f"{s['argmax']:>8.0%}{s['offtask']:>9.0%}"
              f"{s['mean_margin']:>+9.2f}", flush=True)

    print("\npreamble -> naive, per arm")
    for arm in args.arms:
        p = results.get(f"{arm}:preamble")
        nv = results.get(f"{arm}:naive")
        if not p or not nv:
            continue
        print(f"  {arm:<14} deference {p['deference']:.0%} -> {nv['deference']:.0%}"
              f"   margin {p['mean_margin']:+.2f} -> {nv['mean_margin']:+.2f}"
              f"   order gap {p['gap']:+.0%} -> {nv['gap']:+.0%}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "fingerprint": fingerprint, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nREAD THIS BEFORE STEERING\n"
          "  preamble deference well below the QC line rate means the held-out\n"
          "  colors differ from the localization set. Naive deference near 50%\n"
          "  with a large order gap means the arm has no stable prior for a\n"
          "  transfer result to move away from -- report that rather than\n"
          "  steering into it.")


if __name__ == "__main__":
    main()
