#!/usr/bin/env python3
"""How many ICL demos does it take to induce subordinate-following, per arm?

QC gates a pair only if all four of its lines pass under BOTH preambles, so pair
survival is roughly the per-line rate to the fourth power. At a 23% user-condition
line rate that is ~0.3% -- one pair in seventy. Before changing the demo count,
measure the curve rather than guessing at it.

    python sweep_demos.py --arms devuser sysuser sysdev \\
        --demos 4 6 8 10 --out sweep_demos.json

For each (arm, n_conflict_demos) it builds items in memory -- no files, no
generation -- and reports, per preamble condition, the fraction of lines where
the demonstrated policy's answer outscores the other. That is the same
forced-choice quantity QC gates on and ATP differentiates.

WHY THIS IS A RESULT, NOT JUST TUNING
-------------------------------------
The number of demos needed to override a level is a measure of how strongly that
boundary is held. Reporting the curve -- and the demo count you settled on --
turns "we used 10 demos" into a calibrated statement about the manipulation.
Picking the first count that passes QC and not showing the curve is the version
a reviewer should object to.

Read the MARGIN column alongside the rate. A condition sitting just below 50%
with a margin near zero is one demo away; a condition at 20% with a margin of
-1.0 is not going to be fixed by two more demos, and that itself is the finding.
"""

import json
import argparse
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import harmony_canonical as hc
from harmony_canonical import canonical_system_block
from answer_scoring import score, collision
import hierarchy_common as H


def render(tok, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    if not text.rstrip().endswith("<|message|>"):
        text = text + "<|channel|>final<|message|>"
    return text


def build_items(arm, system_block, pairs, demos, n_pairs):
    """Both conditions for each of the four counterbalanced variants."""
    items = []
    for a, b, dev_w, user_w, first, second, template in \
            H.enumerate_variants(pairs[:n_pairs], "rule"):
        for condition, target, distractor in (
                ("dev", dev_w, user_w), ("user", user_w, dev_w)):
            rec = H.build_line(arm, "rule", system_block, a, b, dev_w, user_w,
                               first, second, condition, None, demos, template)
            items.append({"condition": condition, "target": target,
                          "distractor": distractor, "prompt": rec["prompt"]})
    return items


@torch.no_grad()
def run(model, tok, items):
    out = []
    for it in items:
        if collision(tok, it["target"], it["distractor"]):
            continue
        ids = tok(render(tok, it["prompt"]), return_tensors="pt").to(model.device)
        sc = score(model(**ids).logits[0, -1], tok, it["target"], it["distractor"])
        out.append({"condition": it["condition"], **sc})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["devuser", "sysuser", "sysdev"])
    ap.add_argument("--demos", nargs="+", type=int, default=[4, 6, 8, 10])
    ap.add_argument("--n_agree_demos", type=int, default=2)
    ap.add_argument("--n_pairs", type=int, default=10,
                    help="pairs per cell; 4 variants x 2 conditions each")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--reasoning", default=hc.DEFAULT_REASONING)
    ap.add_argument("--date", default=hc.DEFAULT_DATE)
    ap.add_argument("--out", default="sweep_demos.json")
    args = ap.parse_args()

    tok = hc.install(AutoTokenizer.from_pretrained(args.model))
    system_block = canonical_system_block(args.reasoning, args.date)
    pairs = H.build_pairs(tok, H.COLOR_POOL, args.n_pairs)

    print(f"loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto",
                                                 device_map="auto")
    model.eval()

    results = {}
    print(f"\n{'arm':<10}{'demos':>7}{'dev':>8}{'user':>8}"
          f"{'dev_marg':>10}{'user_marg':>11}{'pair_est':>10}")
    print("-" * 64)
    for arm_key in args.arms:
        arm = H.ARMS[arm_key]
        for n in args.demos:
            # select_demos_rule sets the active category list, so the rules and
            # the questions stay in step at every sweep point.
            demos = H.select_demos_rule(tok, n, args.n_agree_demos)
            rows = run(model, tok, build_items(arm, system_block, pairs,
                                               demos, args.n_pairs))
            cell = {}
            for cond in ("dev", "user"):
                sub = [r for r in rows if r["condition"] == cond]
                cell[cond] = sum(r["forced_choice"] for r in sub) / len(sub)
                cell[cond + "_margin"] = statistics.mean(r["margin"] for r in sub)
            # A pair passes QC only if all four of its lines pass under both
            # preambles, so this is the quantity that decides whether a corpus
            # is buildable at all.
            cell["pair_survival_est"] = (cell["dev"] * cell["user"]) ** 4
            cell["n_rows"] = len(rows)
            results[f"{arm_key}:{n}"] = cell
            print(f"{arm_key:<10}{n:>7}{cell['dev']:>8.0%}{cell['user']:>8.0%}"
                  f"{cell['dev_margin']:>+10.2f}{cell['user_margin']:>+11.2f}"
                  f"{cell['pair_survival_est']:>10.1%}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\npair_est is (dev_rate * user_rate)^4 -- the fraction of pairs "
          "expected to\nsurvive QC. You need roughly 50/70 = 71%, so aim for "
          "both per-line rates\nabove ~93%. If the user column plateaus well "
          "below that, more demos are not\nthe answer and the plateau is the "
          "result to report.")


if __name__ == "__main__":
    main()
