#!/usr/bin/env python3
"""Stage 2: pair-level QC for one arm's candidate files.

  CHECK 1  dev-single-desired-all.jsonl   -> model should answer the PRIVILEGED word
  CHECK 2  user-single-desired-all.jsonl  -> model should answer the SUBORDINATE word

A color pair passes only if all of its lines pass, under whichever gate is set.

GATES
-----
  forced    (default) both checks must succeed: the model follows the
            demonstrated policy under both preambles. Right for arms where the
            ICL manipulation actually flips behaviour.

  privileged  only the privileged check must succeed. Nothing is selected on
            the strength of the manipulation, so the corpus carries the full
            swing distribution and you report it rather than filter on it.
            This is the right default for the user-boundary arms: the swing is
            a continuous measure of how much the preamble moved the model, not
            a correctness criterion, and thresholding it to hit a pair budget
            is selection on the outcome. Items where the preamble did little
            dilute the ATP gradient; they do not corrupt it.

  contrast  the privileged check must succeed, AND the dev-vs-user logit
            difference must swing by at least --min_swing between the two
            preambles. Use this where the demos cannot reverse the model's
            preference, only neutralise it -- at the user boundary the swing
            asymptotes to indifference around zero rather than crossing it, so
            the "forced" gate rejects nearly every pair for a reason that is
            itself the finding.

            This is legitimate because attribution patching differentiates a
            LOGIT DIFFERENCE, not a generated token: it needs the two conditions
            to separate, not the model to emit the labelled answer. But it
            changes what the contrast means -- preference versus indifference,
            not preference-A versus preference-B -- and the paper has to say so.
            The steering evaluation then asks whether intervening pushes the
            boundary across zero, which in-context evidence alone cannot do. One bad line
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
    """Score every line of one file.

    `correct` is the per-line verdict for the 'forced' and 'argmax' gates. The
    'contrast' gate cannot be decided from one file alone -- it needs both
    preambles for the same item -- so it is resolved in main().

    Gating on argmax throws away every item where the model prefers the right
    word but capitalizes it, and since a pair survives only if all its lines
    pass, a 40% per-line surface-form failure wipes out almost every pair for a
    reason unrelated to the conflict.
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
            "correct": sc["complied"] if gate == "argmax" else sc["forced_choice"],
            "collision": False,
            **sc,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--min_swing", type=float, default=0.0,
                    help="for --gate contrast: minimum swing, in logits, of "
                         "the dev-vs-user difference between the two "
                         "preambles. Set it from the printed distribution, not "
                         "from the pair count you need. 0.0 excludes only "
                         "items the preamble pushed the WRONG way.")
    ap.add_argument("--gate",
                    choices=["forced", "argmax", "contrast", "privileged"],
                    default="forced",
                    help="'forced' passes a line when the expected word "
                         "outscores the other, which is what ATP "
                         "differentiates. 'argmax' additionally requires the "
                         "model's top token to be that word in some surface "
                         "form -- much stricter, and it drops pairs for "
                         "capitalization. 'privileged' requires only the "
                         "privileged check and selects nothing on the "
                         "manipulation. 'contrast' adds a swing threshold. "
                         "See the module docstring.")
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
    per_file_rows = {}
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
        per_file_rows[fname] = res
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
        # Only the single-file gates condemn pairs here. 'contrast' and
        # 'privileged' both need the two files together and are resolved below;
        # letting either fall through means the SUBORDINATE file's failures
        # still kill pairs, which is exactly the thing those gates exist to
        # avoid.
        if args.gate in ("forced", "argmax"):
            for x in res:
                if not x["correct"]:
                    fail_reasons[x["pair_key"]].append(
                        f"{fname}: wanted {x['target']!r}, got "
                        f"{x['argmax_token']!r} (margin {x['margin']:+.2f})")

    if args.gate in ("contrast", "privileged"):
        dev_rows = per_file_rows[CHECKS[0][0]]
        user_rows = per_file_rows[CHECKS[1][0]]
        if len(dev_rows) != len(user_rows):
            raise SystemExit("the two files have different line counts; they "
                             "are built from the same variant enumeration and "
                             "must align positionally")
        swings = []
        for d, u in zip(dev_rows, user_rows):
            # Both margins are (target - distractor) for their own file, so the
            # user file's margin is the NEGATIVE of the dev-vs-user difference.
            # Put both on the same axis before subtracting.
            d_dev = d["margin"]
            d_user = -u["margin"]
            swing = d_dev - d_user
            swings.append(swing)
            if not d["correct"]:
                fail_reasons[d["pair_key"]].append(
                    f"privileged check failed (margin {d_dev:+.2f})")
            elif args.gate == "contrast" and swing < args.min_swing:
                fail_reasons[d["pair_key"]].append(
                    f"swing {swing:+.2f} < {args.min_swing}")
        ordered = sorted(swings)
        def pctile(q):
            return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        per_check["contrast"] = {
            "gate": args.gate,
            "min_swing": args.min_swing if args.gate == "contrast" else None,
            "mean_swing": statistics.mean(swings),
            "sd_swing": statistics.stdev(swings) if len(swings) > 1 else 0.0,
            "swing_percentiles": {str(q): pctile(q)
                                  for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.95)},
            "frac_swing_negative": sum(1 for x in swings if x < 0) / len(swings),
            "line_pass_rate": sum(
                1 for d, sw in zip(dev_rows, swings)
                if d["correct"] and (args.gate == "privileged"
                                     or sw >= args.min_swing)) / len(swings),
        }
        c = per_check["contrast"]
        print(f"  swing: mean {c['mean_swing']:+.2f} (sd {c['sd_swing']:.2f}), "
              f"{c['frac_swing_negative']:.1%} negative")
        print("         percentiles " + "  ".join(
            f"p{int(float(q) * 100)}={v:+.2f}"
            for q, v in c["swing_percentiles"].items()))
        print(f"  gate={args.gate}: line pass {c['line_pass_rate']:.0%}")

    passing = [k for k in pair_order if k not in fail_reasons]
    line_rate = None
    if args.gate in ("contrast", "privileged"):
        line_rate = per_check["contrast"]["line_pass_rate"]
    elif per_check:
        line_rate = min(v["line_pass_rate"] for v in per_check.values()
                        if "line_pass_rate" in v)
    obs = len(passing) / len(pair_order) if pair_order else 0.0
    msg = f"\n{len(passing)}/{len(pair_order)} pairs passed (gate={args.gate})"
    if line_rate is not None:
        est = line_rate ** 4
        msg += f"; line pass {line_rate:.0%}, independence estimate {est:.0%}"
        # Failures cluster within a pair, so observed survival normally runs
        # ABOVE the independence estimate. Coming in far below it means pairs
        # are being condemned by something the gate is not supposed to consider.
        if est > 0.05 and obs < 0.5 * est:
            msg += ("\n  WARNING: survival is far below the estimate. Check "
                    "that the gate is\n  actually being applied -- a gate that "
                    "silently falls through to the\n  per-file loop will "
                    "reproduce a stricter gate's numbers exactly.")
    print(msg)

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
