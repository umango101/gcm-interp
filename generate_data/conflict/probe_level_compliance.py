#!/usr/bin/env python3
"""Gate check: is each hierarchy level functional, and how off-distribution is it?

Run this BEFORE generating full corpora. It asks two questions about each way of
placing an instruction, using items with NO conflict -- the rule says a color,
the question names that color and one other, and nothing contradicts the rule:

  1. COMPLIANCE. Does the model actually follow a rule placed at this level?
     If a developer-level rule is obeyed 100% of the time and a system-level
     rule 60%, then every deference rate in the system arms is partly a
     format-familiarity effect rather than a hierarchy effect, and no conflict
     result from those arms can be read cleanly.

  2. SURPRISAL. What does the rendered prompt cost in mean per-token NLL,
     relative to the canonical placement? This is the quantitative version of
     "is this off-distribution", and reporting the number is much better than
     asserting a belief either way.

Variants probed:

  devuser        canonical system block + developer "# Instructions" rule, user question
  devuser_nohdr  same, without the "# Instructions" header
  sysuser        rule appended to the canonical system block, user question
  sysdev         rule appended to the canonical system block, developer question
                 -- note this conversation has no user turn in it at all
  sysdev_ruleform
                 system-level rule + a neutral developer message + user
                 question. This is the shape the RULE form's sysdev arm uses;
                 the sysdev variants above are the REQUEST form's shape, where a
                 developer message has to carry the question itself.
  sysdev_user_late
                 as sysdev, neutral user turn AFTER the question. Preserves
                 canonical role order but moves the answer position, so read it
                 against sysdev_user rather than against sysdev.
  sysdev_user    as sysdev, plus one neutral user turn after the system block.
                 sysdev_user vs sysdev separates "no user present" from
                 "repeated developer turns" as the explanation for any
                 compliance gap. This is the cheapest place to run that
                 comparison: it costs one more pass here, versus a whole QC
                 stage and a localization run if you find out later.
  sysbare        rule as the ENTIRE system message, user question
                 -- this is what the current corpus renders as: because a system
                 message is present, the template suppresses its default block,
                 so the model sees no identity line, no reasoning level and no
                 channel declaration. Comparing it against sysuser tells you how
                 much the existing results were affected.
  norule         canonical block, no rule at all, user question
                 -- the floor. Which of the two words wins by default, and how
                 much of any "compliance" is really position or lexical bias.

    python probe_level_compliance.py --repo_root . --out probe_levels.json

Every variant sees the same color pairs, both role assignments and both mention
orders, so nothing here is confounded with position or with the words.
"""

import json
import argparse
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import harmony_canonical as hc
from answer_scoring import score, collision
from harmony_canonical import canonical_system_block, developer_message
from hierarchy_common import (
    RULE, COLOR_POOL, NEUTRAL_TEMPLATE_POOL, NEUTRAL_USER_TURN,
    NEUTRAL_USER_TURN_LATE, build_pairs, render_instruction,
)


# name -> (leading messages builder, subordinate role, trailing messages)
def _variants(system_block):
    def dev(rule, header=True):
        body = developer_message(rule) if header else rule
        return [{"role": "system", "content": system_block},
                {"role": "developer", "content": body}]

    def sysblock(r):
        return [{"role": "system", "content": system_block + "\n\n" + r}]

    return {
        "devuser":       (lambda r: dev(r, True), "user", []),
        "devuser_nohdr": (lambda r: dev(r, False), "user", []),
        "sysuser":       (sysblock, "user", []),
        "sysdev":        (sysblock, "developer", []),
        "sysdev_user":   (lambda r: sysblock(r) + [{"role": "user",
                                                    "content": NEUTRAL_USER_TURN}],
                          "developer", []),
        "sysdev_user_late": (sysblock, "developer",
                             [{"role": "user", "content": NEUTRAL_USER_TURN_LATE}]),
        # The rule form's sysdev composition, minus the conflict: a system-level
        # rule, a developer message that says nothing about the choice, and a
        # user question. If this is at ceiling, a developer message's mere
        # presence does not disturb system-level compliance, and the rule-form
        # sysdev arm is built entirely from shapes that probe clean.
        "sysdev_ruleform": (lambda r: sysblock(r) + [
            {"role": "developer",
             "content": developer_message("Answer concisely.")}], "user", []),
        "sysbare":       (lambda r: [{"role": "system", "content": r}], "user", []),
        "norule":        (lambda r: [{"role": "system", "content": system_block}], "user", []),
    }


def render(tok, msgs, add_generation_prompt=True):
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)
    if add_generation_prompt and not text.rstrip().endswith("<|message|>"):
        text = text + "<|channel|>final<|message|>"
    return text


def build_items(pairs, n_pairs):
    """Uncontested items: 4 per pair (2 role assignments x 2 mention orders)."""
    items = []
    for p, (a, b) in enumerate(pairs[:n_pairs]):
        template = NEUTRAL_TEMPLATE_POOL[p % len(NEUTRAL_TEMPLATE_POOL)]
        for rule_word, other in ((a, b), (b, a)):
            for first, second in ((a, b), (b, a)):
                items.append({
                    "pair_key": f"{a}|{b}",
                    "rule_word": rule_word,
                    "other_word": other,
                    "question": render_instruction(template, first, second),
                    "mention_first": first,
                    "template": template,
                })
    return items


@torch.no_grad()
def run_variant(model, tok, items, leading, subordinate_role, trailing=()):
    rows = []
    for it in items:
        msgs = (leading(RULE.format(color=it["rule_word"]))
                + [{"role": subordinate_role, "content": it["question"]}]
                + list(trailing))

        full = render(tok, msgs, add_generation_prompt=True)
        ids = tok(full, return_tensors="pt").to(model.device)
        out = model(**ids)

        sc = score(out.logits[0, -1], tok, it["rule_word"], it["other_word"])

        # Mean per-token NLL of the rendered prompt, and of the final
        # subordinate turn alone. The boundary comes from re-rendering without
        # the last message; if the template is not prefix-stable we report only
        # the whole-prompt number rather than a wrong segment.
        lp = torch.log_softmax(out.logits[0].float(), dim=-1)
        tgt = ids["input_ids"][0][1:]
        nll = -lp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)
        whole = float(nll.mean())

        prefix = render(tok, msgs[:-1], add_generation_prompt=False)
        if full.startswith(prefix):
            start = len(tok(prefix)["input_ids"])
            seg = nll[max(start - 1, 0):]
            final_turn = float(seg.mean()) if seg.numel() else float("nan")
            # The prefix is the part whose naturalness is actually in question:
            # the system block, the rule, any developer message. nll_final_turn
            # measures how predictable the QUESTION is given that prefix, which
            # conflates "this prefix is natural" with "this prefix makes a color
            # question likely" -- strip the boilerplate and a color question
            # becomes the obvious continuation, so a malformed prefix can score
            # LOWER. Read nll_prefix for off-distribution-ness.
            head = nll[:max(start - 1, 0)]
            prefix_nll = float(head.mean()) if head.numel() else float("nan")
        else:
            final_turn = prefix_nll = float("nan")

        rows.append({
            "pair_key": it["pair_key"],
            "rule_word": it["rule_word"],
            "mention_first_is_rule_word": it["mention_first"] == it["rule_word"],
            "nll_prompt": whole,
            "nll_prefix": prefix_nll,
            "nll_final_turn": final_turn,
            **sc,
        })
    return rows


def summarize(rows):
    def mean(key, sub=None):
        vals = [r[key] for r in (sub if sub is not None else rows)]
        vals = [v for v in vals if v == v]                 # drop NaN
        return statistics.mean(vals) if vals else float("nan")

    first = [r for r in rows if r["mention_first_is_rule_word"]]
    second = [r for r in rows if not r["mention_first_is_rule_word"]]
    return {
        "n": len(rows),
        "compliance": sum(r["complied"] for r in rows) / len(rows),
        # The metric that matches the ATP objective: does the rule word outscore
        # the alternative, regardless of which surface form tops the argmax.
        "forced_choice": sum(r["forced_choice"] for r in rows) / len(rows),
        "forced_choice_rule_first":
            (sum(r["forced_choice"] for r in first) / len(first)
             if first else float("nan")),
        "forced_choice_rule_second":
            (sum(r["forced_choice"] for r in second) / len(second)
             if second else float("nan")),
        # Split by mention order: a variant that only "complies" when the rule
        # word is named first is showing position bias, not rule-following.
        "compliance_rule_first": (sum(r["complied"] for r in first) / len(first)
                                  if first else float("nan")),
        "compliance_rule_second": (sum(r["complied"] for r in second) / len(second)
                                   if second else float("nan")),
        "mean_margin": mean("margin"),
        "offtask_rate": sum(r["offtask"] for r in rows) / len(rows),
        "mean_nll_prompt": mean("nll_prompt"),
        "mean_nll_prefix": mean("nll_prefix"),
        "mean_nll_final_turn": mean("nll_final_turn"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--reasoning", default=hc.DEFAULT_REASONING,
                    choices=["low", "medium", "high"])
    ap.add_argument("--date", default=hc.DEFAULT_DATE)
    ap.add_argument("--generation_prompt", choices=["final", "bare"],
                    default="final")
    ap.add_argument("--n_pairs", type=int, default=25, help="4 items per pair")
    ap.add_argument("--out", default="probe_levels.json")
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset; default is all")
    args = ap.parse_args()

    system_block = canonical_system_block(args.reasoning, args.date)
    tok = hc.install(AutoTokenizer.from_pretrained(args.model),
                     system_block, args.generation_prompt)
    variants = _variants(system_block)
    if args.variants:
        keep = [v.strip() for v in args.variants.split(",")]
        unknown = [v for v in keep if v not in variants]
        if unknown:
            raise SystemExit(f"unknown variant(s): {unknown}")
        variants = {k: variants[k] for k in keep}

    pairs = build_pairs(tok, COLOR_POOL, args.n_pairs)
    # Two colors sharing a first token make the margin identically zero, so the
    # item contributes nothing. Drop them loudly.
    clashes = [(a, b) for a, b in pairs[:args.n_pairs] if collision(tok, a, b)]
    if clashes:
        print(f"  dropping {len(clashes)} pair(s) whose colors share a first "
              f"token: {clashes}")
        pairs = [p for p in pairs if p not in clashes]
    items = build_items(pairs, args.n_pairs)
    print(f"{len(items)} uncontested items over {min(args.n_pairs, len(pairs))} pairs")

    print(f"loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map="auto")
    model.eval()

    results = {}
    for name, (leading, role, trailing) in variants.items():
        rows = run_variant(model, tok, items, leading, role, trailing)
        results[name] = {"subordinate_role": role,
                         "summary": summarize(rows),
                         "rows": rows}
        s = results[name]["summary"]
        print(f"  {name:<16} forced {s['forced_choice']:.0%} "
              f"(first {s['forced_choice_rule_first']:.0%} / "
              f"second {s['forced_choice_rule_second']:.0%})  "
              f"argmax {s['compliance']:.0%}  "
              f"margin {s['mean_margin']:+.2f}  "
              f"offtask {s['offtask_rate']:.0%}  "
              f"nll_prefix {s['mean_nll_prefix']:.3f}", flush=True)

    base = results.get("devuser", {}).get("summary")
    if base:
        print("\nrelative to devuser (the on-distribution placement):")
        for name, r in results.items():
            s = r["summary"]
            print(f"  {name:<16} Dforced {s['forced_choice'] - base['forced_choice']:+.2f}  "
                  f"Dargmax {s['compliance'] - base['compliance']:+.2f}  "
                  f"Dnll_prefix "
                  f"{s['mean_nll_prefix'] - base['mean_nll_prefix']:+.3f}  "
                  f"Dnll_final_turn "
                  f"{s['mean_nll_final_turn'] - base['mean_nll_final_turn']:+.3f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args),
                   "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\nHOW TO READ THIS\n"
          "  FORCED is the gate: it is the logit difference ATP differentiates,\n"
          "  so a level is usable if forced-choice is at ceiling and roughly\n"
          "  equal across mention orders. ARGMAX is the behavioural question --\n"
          "  would a generation emit it -- and a large forced/argmax gap means\n"
          "  the model prefers the right word but writes it in some other\n"
          "  surface form. sysbare vs sysuser measures what the suppressed\n"
          "  system block cost the existing runs; norule is the floor, where\n"
          "  any forced-choice above 50%% is position or lexical bias.\n"
          "  For off-distribution-ness read nll_prefix: the surprisal of the\n"
          "  system block, rule and any developer message -- the part whose\n"
          "  naturalness is in question. nll_final_turn measures how predictable\n"
          "  the QUESTION is given that prefix, which is a different thing: a\n"
          "  stripped-down prefix makes a color question MORE expected, so a\n"
          "  malformed variant can score lower on it.")


if __name__ == "__main__":
    main()
