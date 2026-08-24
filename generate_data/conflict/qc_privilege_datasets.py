"""
Quality check: does gpt-oss-20b produce the expected final answer?

Runs over the CANDIDATE files and gates at the level of the color pair.

  CHECK 1  dev-single-desired-all.jsonl   -> model should answer the DEV's word
  CHECK 2  user-single-desired-all.jsonl  -> model should answer the USER's word

A color pair PASSES only if all of its lines pass BOTH checks. One bad line
condemns the pair, and since both role assignments (a-as-developer and
b-as-developer) live under one pair key, dropping a pair drops its inverse too.

Writes pair_qc.json with the surviving pair keys, which stage 3 of
make_privilege_datasets.py consumes to emit the real files.

For each line the final assistant turn is stripped, the prompt is rendered with
the final channel opened, and the answer position is read two ways: the greedy
argmax (the behavioural check) and logit(target) - logit(distractor) (how
decisively, and the quantity attribution patching differentiates).

    python qc_privilege_datasets.py --data_dir <candidates dir>
"""

import os
import json
import argparse
import statistics
from collections import defaultdict

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS = (4, 55)

CHECKS = [
    ("dev-single-desired-all.jsonl", "dev"),
    ("user-single-desired-all.jsonl", "user"),
]


def preflight():
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    print(f"transformers {transformers.__version__} | torch {torch.__version__}")
    if ver < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers {transformers.__version__} too old for gpt-oss "
            f"(need >= {'.'.join(map(str, MIN_TRANSFORMERS))}); use conflict-syc."
        )


def render(tok, msgs, reasoning):
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       reasoning_effort=reasoning)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    if not text.rstrip().endswith("<|message|>"):
        text = text + "<|channel|>final<|message|>"
    return text


def first_token_id(tok, word):
    ids = tok.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{word!r} encodes to nothing")
    return ids[0]


@torch.no_grad()
def evaluate(model, tok, rows, target_role, reasoning):
    out = []
    for r in rows:
        msgs = r["prompt"]
        if msgs[-1]["role"] == "assistant":       # strip the answer we're testing
            msgs = msgs[:-1]
        target = r["dev_word"] if target_role == "dev" else r["user_word"]
        distractor = r["user_word"] if target_role == "dev" else r["dev_word"]

        ids = tok(render(tok, msgs, reasoning), return_tensors="pt").to(model.device)
        logits = model(**ids).logits[0, -1]
        t, d = first_token_id(tok, target), first_token_id(tok, distractor)
        top = int(torch.argmax(logits))

        out.append({
            "pair_key": r["pair_key"],
            "dev_word": r["dev_word"],
            "user_word": r["user_word"],
            "target": target,
            "argmax_token": tok.decode([top]),
            "correct": top == t,
            "margin": float(logits[t] - logits[d]),
            "offtask": top not in (t, d),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/gpt-oss-20b/privilege/candidates")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--out", default="pair_qc.json")
    args = ap.parse_args()

    preflight()
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map="auto")
    model.eval()

    per_check = {}
    fail_reasons = defaultdict(list)
    pair_order = []

    for fname, role in CHECKS:
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run stage 1 first")
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        if not pair_order:
            seen = set()
            for r in rows:                        # preserve generation order
                if r["pair_key"] not in seen:
                    seen.add(r["pair_key"])
                    pair_order.append(r["pair_key"])

        res = evaluate(model, tok, rows, role, args.reasoning)
        margins = [x["margin"] for x in res]
        per_check[fname] = {
            "target_role": role,
            "n": len(res),
            "line_pass_rate": sum(x["correct"] for x in res) / len(res),
            "mean_margin": statistics.mean(margins),
            "sd_margin": statistics.stdev(margins) if len(margins) > 1 else 0.0,
            "offtask_rate": sum(x["offtask"] for x in res) / len(res),
            "results": res,
        }
        for x in res:
            if not x["correct"]:
                fail_reasons[x["pair_key"]].append(
                    f"{fname.split('-')[0]}: {x['dev_word']}/{x['user_word']} "
                    f"said {x['argmax_token']!r} want {x['target']!r}"
                )

        c = per_check[fname]
        print(f"\n{fname}  (expect the {role}'s word)")
        print(f"  line pass rate {c['line_pass_rate']:.2f}  "
              f"mean margin {c['mean_margin']:+.2f} (sd {c['sd_margin']:.2f})  "
              f"off-task {c['offtask_rate']:.2f}")

    passing = [k for k in pair_order if k not in fail_reasons]
    failing = {k: v for k, v in fail_reasons.items()}

    print(f"\n{'=' * 64}")
    # Pair survival is roughly (per-line rate)^lines_per_pair, so a modest line
    # failure rate collapses the pair count. Show the arithmetic.
    lines_per_pair = per_check[CHECKS[0][0]]["n"] / max(1, len(pair_order))
    worst = min(c["line_pass_rate"] for c in per_check.values())
    print(f"expected pair survival ~ {worst:.2f}^{lines_per_pair:.0f} = "
          f"{worst ** lines_per_pair:.2f}  (binding check: line rate {worst:.2f})")
    print(f"pairs tested   {len(pair_order)}")
    print(f"pairs passing  {len(passing)}   (all lines correct on BOTH checks)")
    print(f"pairs dropped  {len(failing)}")
    for k, reasons in list(failing.items())[:15]:
        print(f"   {k:<24} {len(reasons)} bad line(s) | {reasons[0]}")
    if len(failing) > 15:
        print(f"   ... and {len(failing) - 15} more")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "passing_pairs": passing,
                   "failing_pairs": failing, "per_check": per_check}, f, indent=2)
    print(f"\nwrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
