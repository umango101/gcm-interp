#!/usr/bin/env python3
"""Stage 2: pair-level QC for one arm's candidate files.

  CHECK 1  dev-single-desired-all.jsonl   -> model should answer the PRIVILEGED word
  CHECK 2  user-single-desired-all.jsonl  -> model should answer the SUBORDINATE word

A color pair passes only if all of its lines pass both checks. One bad line
condemns the pair, and since both role assignments live under one pair key,
dropping a pair drops its inverse too.

    python qc_hierarchy_datasets.py \
        --data_dir data/gpt-oss-20b/hier-devuser/candidates \
        --out data/gpt-oss-20b/hier-devuser/pair_qc.json

RENDERING
---------
This installs harmony_canonical before rendering. That is not optional: the
stock gpt-oss template rewrites role="system" into a developer message and
injects a volatile date line, so QC run against it would gate pairs under a
different prompt than the experiments use -- and would collapse the three arms
into one. The records carry their own system block, so the template's default
block is never emitted and the arm is determined entirely by the data.

--generation_prompt must match what localization and eval use, or QC gates pairs
at a position the experiment never reads.
"""

import os
import json
import argparse
import statistics
from collections import defaultdict

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import harmony_canonical as hc
from answer_scoring import score, collision

MIN_TRANSFORMERS = (4, 55)

CHECKS = [
    ("dev-single-desired-all.jsonl", "dev"),      # privileged
    ("user-single-desired-all.jsonl", "user"),    # subordinate
]


def preflight():
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    print(f"transformers {transformers.__version__} | torch {torch.__version__}")
    if ver < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers {transformers.__version__} too old for gpt-oss "
            f"(need >= {'.'.join(map(str, MIN_TRANSFORMERS))})")


def render(tok, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    if not text.rstrip().endswith("<|message|>"):
        text = text + "<|channel|>final<|message|>"
    return text


@torch.no_grad()
def evaluate(model, tok, rows, target_role, gate):
    """gate: 'forced' (target outscores distractor) or 'argmax'.

    Default is 'forced'. Gating on argmax throws away every item where the model
    prefers the right word but capitalizes it, and since a pair survives only if
    all four of its lines pass, a 40% per-line surface-form failure wipes out
    almost every pair -- for a reason that has nothing to do with the conflict.
    """
    out = []
    for r in rows:
        msgs = r["prompt"]
        if msgs[-1]["role"] == "assistant":       # strip the answer under test
            msgs = msgs[:-1]
        target = r["dev_word"] if target_role == "dev" else r["user_word"]
        distractor = r["user_word"] if target_role == "dev" else r["dev_word"]

        shared = collision(tok, target, distractor)
        if shared:
            out.append({"pair_key": r["pair_key"], "target": target,
                        "argmax_token": "", "correct": False, "margin": 0.0,
                        "offtask": True, "collision": True})
            continue
        ids = tok(render(tok, msgs), return_tensors="pt").to(model.device)
        sc = score(model(**ids).logits[0, -1], tok, target, distractor)
        out.append({
            "pair_key": r["pair_key"],
            "target": target,
            "correct": sc["forced_choice"] if gate == "forced" else sc["complied"],
            "collision": False,
            **sc,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--gate", choices=["forced", "argmax"], default="forced",
                    help="'forced' passes a line when the expected word "
                         "outscores the other, which is what ATP "
                         "differentiates. 'argmax' additionally requires the "
                         "model's top token to be that word in some surface "
                         "form -- much stricter, and it drops pairs for "
                         "capitalization.")
    ap.add_argument("--generation_prompt", choices=["final", "bare"],
                    default="final",
                    help="'final' forces the answer position (needed for a "
                         "single-token logit readout); 'bare' is the canonical "
                         "generation prompt and lets the model reason first.")
    args = ap.parse_args()

    preflight()
    tok = hc.install(AutoTokenizer.from_pretrained(args.model),
                     generation_prompt=args.generation_prompt)
    print(f"loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map="auto")
    model.eval()

    per_check, fail_reasons, pair_order = {}, defaultdict(list), []
    arm = form = None

    for fname, role in CHECKS:
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run stage 1 first")
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        arm = arm or rows[0].get("arm")
        form = rows[0].get("conflict_form")
        if rows[0].get("arm") != arm:
            raise SystemExit(f"{fname} is arm {rows[0].get('arm')!r}, expected {arm!r}")
        if not pair_order:
            seen = set()
            for r in rows:
                if r["pair_key"] not in seen:
                    seen.add(r["pair_key"])
                    pair_order.append(r["pair_key"])

        res = evaluate(model, tok, rows, role, args.gate)
        margins = [x["margin"] for x in res]
        per_check[fname] = {
            "target_role": role,
            "n": len(res),
            "line_pass_rate": sum(x["correct"] for x in res) / len(res),
            "mean_margin": statistics.mean(margins),
            "sd_margin": statistics.stdev(margins) if len(margins) > 1 else 0.0,
            "offtask_rate": sum(x["offtask"] for x in res) / len(res),
            "argmax_rate": sum(x.get("complied", False) for x in res) / len(res),
            "forced_rate": sum(x.get("forced_choice", False) for x in res) / len(res),
            "collisions": sum(x["collision"] for x in res),
        }
        print(f"  {fname}: pass {per_check[fname]['line_pass_rate']:.0%} "
              f"(gate={args.gate}), forced {per_check[fname]['forced_rate']:.0%}, "
              f"argmax {per_check[fname]['argmax_rate']:.0%}, "
              f"margin {per_check[fname]['mean_margin']:+.2f}, "
              f"offtask {per_check[fname]['offtask_rate']:.0%}")
        for x in res:
            if not x["correct"]:
                fail_reasons[x["pair_key"]].append(
                    f"{fname}: wanted {x['target']!r}, got {x['argmax_token']!r} "
                    f"(margin {x['margin']:+.2f})")

    passing = [k for k in pair_order if k not in fail_reasons]
    print(f"\n{len(passing)}/{len(pair_order)} pairs passed both checks")

    with open(args.out, "w") as f:
        json.dump({"arm": arm,
                   "form": form,
                   "args": vars(args),
                   "per_check": per_check,
                   "passing_pairs": passing,
                   "failing_pairs": dict(fail_reasons)}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
