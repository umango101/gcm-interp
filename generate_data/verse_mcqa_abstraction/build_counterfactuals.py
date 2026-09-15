#!/usr/bin/env python
"""
Build the clean / corrupted pairs for the verse-MCQA causal abstraction experiment.

Hypothesised mechanism:  content -> option index -> option letter -> copy letter to answer.

Every base row (verse-single-desired-all.jsonl) has four options in a 2x2 design:
  verse & answers the question   (the clean answer)
  prose & answers the question   (label taken from prose-desired-all.jsonl, same ids)
  two distractors                (verse/prose responses to a different question)

Three corrupted datasets are produced, each paired row-by-row with its clean row:

  step 1  task      "written in verse" -> "written in prose"; options and letters untouched.
                    Corrupted answer = prose letter.   Metric: logit(prose) - logit(verse).
  step 2  position  same question, options permuted so the verse response sits at a new
                    index.  Corrupted answer = std letter at the new index.
  step 3  letter    step-2 permutation (reused per row) + labels A-D -> P-S (the instruction's
                    letter list is changed to match).  Corrupted answer = alt letter at the new
                    index; the std letter at that index is kept as an auxiliary "index" metric.

Step-2/3 permutations are chosen so that (i) no slot keeps its content (derangement), (ii) the
verse response moves onto a slot that held a distractor in the clean prompt, and (iii) the clean
answer slot is filled by a distractor in the corrupted prompt.  Your toy example
(V,d1,P,d2) -> (d1,P,d2,V) is one member of this set.  Candidates are tried in a
row-seeded order and the first one for which BOTH the step-2 and step-3 prompts pass QC is used
for both steps, so steps 2 and 3 differ only in the letter labels.

QC: the model's greedy next token after the chat-templated prompt (same templating as
data_handler.get_templated_prompts: user turn only, add_generation_prompt=True) must be exactly
the answer-letter token, for the clean prompt and for the corrupted prompt.

Outputs (default: <repo>/data/<model_name>/verse-mcqa-abstraction/):
  clean.jsonl                      repo format {id, prompt:[user, assistant]}
  step{1,2,3}-corrupted.jsonl      repo format
  step{1,2,3}-pairs.jsonl          everything the patching script needs (letters, roles, QC)
  qc_report.json
"""

import argparse
import itertools
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

STD_LETTERS = "ABCD"
ANSWER_SUFFIX = "\nAnswer: ("
INSTR_RE = re.compile(
    r'^Which of the following responses is written in (?P<fmt>\w+) and correctly answers the '
    r'question\? Please respond with only "A", "B", "C", or "D"\.$'
)
INSTR_TMPL = ("Which of the following responses is written in {fmt} and correctly answers the "
              "question? Please respond with only {letter_list}.")


# ----------------------------------------------------------------------------- prompt parsing
def letter_list(letters):
    return ", ".join(f'"{x}"' for x in letters[:-1]) + f', or "{letters[-1]}"'


def render(question_line, fmt, letters, options):
    head = f"{question_line}\n{INSTR_TMPL.format(fmt=fmt, letter_list=letter_list(letters))}\n"
    body = "".join(f"({L}) {o}\n" for L, o in zip(letters, options))
    return head + body + "Answer: ("


def parse(content):
    """Split a base prompt into (question line, fmt, [4 options]) and verify it round-trips."""
    if not (content.startswith("Question: ") and content.endswith(ANSWER_SUFFIX)):
        raise ValueError("unexpected prompt frame")
    head, rest = content.split("\n(A) ", 1)
    question_line, instr = head.split("\n", 1)
    m = INSTR_RE.match(instr)
    if m is None:
        raise ValueError(f"unexpected instruction line: {instr!r}")
    body = rest[: -len(ANSWER_SUFFIX)]
    options, cur = [], body
    for L in STD_LETTERS[1:]:
        if f"\n({L}) " not in cur:
            raise ValueError(f"missing option ({L})")
        opt, cur = cur.split(f"\n({L}) ", 1)
        options.append(opt)
    options.append(cur)
    parsed = dict(question_line=question_line, fmt=m.group("fmt"), options=options)
    if render(question_line, parsed["fmt"], STD_LETTERS, options) != content:
        raise ValueError("prompt does not round-trip through the parser")
    return parsed


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def answer_of(row):
    msgs = row["prompt"]
    if msgs[-1]["role"] != "assistant":
        raise ValueError(f"row {row['id']} has no assistant answer")
    return msgs[-1]["content"].strip()


# ----------------------------------------------------------------------------- permutations
def candidate_orders(v, p, rng):
    """order[j] = clean slot whose content goes to corrupted slot j."""
    distractors = {i for i in range(4) if i not in (v, p)}
    out = []
    for order in itertools.permutations(range(4)):
        if any(order[j] == j for j in range(4)):
            continue
        if order.index(v) not in distractors:      # verse lands on a clean-distractor slot
            continue
        if order[v] not in distractors:            # clean answer slot now holds a distractor
            continue
        out.append(list(order))
    rng.shuffle(out)
    return out


def clean_roles(v, p):
    roles, k = [], 1
    for i in range(4):
        if i == v:
            roles.append("verse")
        elif i == p:
            roles.append("prose")
        else:
            roles.append(f"distractor_{k}")
            k += 1
    return roles


# ----------------------------------------------------------------------------- QC scorer
class NextTokenScorer:
    def __init__(self, model_name, dtype, batch_size):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map="auto"
        ).eval()
        self.batch_size = batch_size

    def letter_id(self, letter):
        ids = self.tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"letter {letter!r} is not a single token: {ids}")
        return ids[0]

    def template(self, content):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )

    def score(self, contents, answers, report_letters):
        """Greedy next token for each prompt + margin of the expected answer token."""
        torch = self.torch
        texts = [self.template(c) for c in contents]
        lengths = [len(self.tok(t)["input_ids"]) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        letter_ids = {L: self.letter_id(L) for L in report_letters}
        results = [None] * len(texts)
        base = self.model.base_model
        head = self.model.get_output_embeddings()
        for s in range(0, len(order), self.batch_size):
            idx = order[s: s + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True)
            dev = self.model.device
            ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                h = base(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1]
                logits = head(h.to(head.weight.device)).float()
            for b, i in enumerate(idx):
                row = logits[b]
                ans_id = letter_ids[answers[i]]
                top_id = int(row.argmax())
                others = row.clone()
                others[ans_id] = float("-inf")
                results[i] = dict(
                    top_token=self.tok.decode([top_id]),
                    correct=top_id == ans_id,
                    margin=float(row[ans_id] - others.max()),
                    letter_logits={L: round(float(row[t]), 3) for L, t in letter_ids.items()},
                    n_tokens=lengths[i],
                )
            print(f"  QC scored {min(s + self.batch_size, len(order))}/{len(order)}", flush=True)
        return results


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--base_path", default=None,
                    help="default: <repo>/data/<model_name>/verse-single/verse-single-desired-all.jsonl")
    ap.add_argument("--prose_path", default=None,
                    help="same rows asked 'in prose'; supplies the prose-answer letter. "
                         "default: sibling prose-desired-all.jsonl")
    ap.add_argument("--out_dir", default=None,
                    help="default: <repo>/data/<model_name>/verse-mcqa-abstraction")
    ap.add_argument("--step3_letters", default="PQRS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--no_qc", action="store_true",
                    help="build prompts without a model (first candidate permutation, qc=null)")
    args = ap.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    data_dir = REPO_ROOT / "data" / model_name / "verse-single"
    base_path = Path(args.base_path) if args.base_path else data_dir / "verse-single-desired-all.jsonl"
    prose_path = Path(args.prose_path) if args.prose_path else base_path.parent / "prose-desired-all.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / model_name / "verse-mcqa-abstraction"
    alt = args.step3_letters
    if len(alt) != 4 or len(set(alt)) != 4 or set(alt) & set(STD_LETTERS):
        raise ValueError("--step3_letters must be 4 distinct letters disjoint from ABCD")

    if not args.no_qc:
        try:
            from determinism import enable_determinism
            enable_determinism()
        except ImportError:
            pass

    base = load_jsonl(base_path)
    prose_by_id = {r["id"]: r for r in load_jsonl(prose_path)}
    if args.max_rows:
        base = base[: args.max_rows]
    print(f"base rows: {len(base)}  ({base_path})")

    # ---- build every prompt we may need
    items, jobs = [], []          # jobs: (item_idx, key, content, answer)
    for row in base:
        rid = row["id"]
        user_msgs = [m for m in row["prompt"] if m["role"] != "assistant"]
        if len(user_msgs) != 1:
            raise ValueError(f"row {rid}: expected exactly one non-assistant message")
        parsed = parse(user_msgs[0]["content"])
        if parsed["fmt"] != "verse":
            raise ValueError(f"row {rid}: base prompt is not the verse condition")
        if rid not in prose_by_id:
            raise ValueError(f"row {rid}: missing from {prose_path}")
        prose_parsed = parse(prose_by_id[rid]["prompt"][0]["content"])
        if prose_parsed["options"] != parsed["options"] or prose_parsed["fmt"] != "prose":
            raise ValueError(f"row {rid}: prose file does not match base options")
        v = STD_LETTERS.index(answer_of(row))
        p = STD_LETTERS.index(answer_of(prose_by_id[rid]))
        if v == p:
            raise ValueError(f"row {rid}: verse and prose answers coincide")

        rng = random.Random(f"verse-mcqa-abstraction|{args.seed}|{rid}")
        orders = candidate_orders(v, p, rng)
        if not orders:
            raise RuntimeError(f"row {rid}: no valid permutation")
        it = dict(id=rid, parsed=parsed, v=v, p=p, orders=orders, roles=clean_roles(v, p))
        k = len(items)
        items.append(it)
        opts, q = parsed["options"], parsed["question_line"]
        jobs.append((k, "clean", render(q, "verse", STD_LETTERS, opts), STD_LETTERS[v]))
        jobs.append((k, "step1", render(q, "prose", STD_LETTERS, opts), STD_LETTERS[p]))
        for ci, order in enumerate(orders if not args.no_qc else orders[:1]):
            perm_opts = [opts[j] for j in order]
            new_v = order.index(v)
            jobs.append((k, f"step2|{ci}", render(q, "verse", STD_LETTERS, perm_opts), STD_LETTERS[new_v]))
            jobs.append((k, f"step3|{ci}", render(q, "verse", alt, perm_opts), alt[new_v]))

    # ---- QC
    if args.no_qc:
        qc = [None] * len(jobs)
    else:
        print(f"QC: scoring {len(jobs)} prompts with {args.model}")
        scorer = NextTokenScorer(args.model, args.dtype, args.batch_size)
        qc = scorer.score([j[2] for j in jobs], [j[3] for j in jobs], STD_LETTERS + alt)
    table = {}
    for (k, key, content, ans), res in zip(jobs, qc):
        table[(k, key)] = dict(content=content, answer=ans, qc=res)

    def ok(k, key):
        res = table[(k, key)]["qc"]
        return True if res is None else res["correct"]

    # ---- select + write
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {name: open(out_dir / f"{name}.jsonl", "w") for name in
               ["clean", "step1-corrupted", "step2-corrupted", "step3-corrupted",
                "step1-pairs", "step2-pairs", "step3-pairs"]}
    stats = Counter()
    target_dist = {s: Counter() for s in (1, 2, 3)}

    def repo_row(rid, content, ans):
        return {"id": rid, "prompt": [{"role": "user", "content": content},
                                      {"role": "assistant", "content": ans}]}

    for k, it in enumerate(items):
        rid, v, p = it["id"], it["v"], it["p"]
        clean = table[(k, "clean")]
        stats["rows"] += 1
        if not ok(k, "clean"):
            stats["clean_fail"] += 1
            continue
        stats["clean_pass"] += 1
        writers["clean"].write(json.dumps(repo_row(rid, clean["content"], clean["answer"])) + "\n")
        common = dict(id=rid, clean_prompt=[{"role": "user", "content": clean["content"]}],
                      clean_answer=clean["answer"], clean_letters=STD_LETTERS,
                      clean_roles=it["roles"], clean_qc=clean["qc"])

        s1 = table[(k, "step1")]
        if ok(k, "step1"):
            stats["step1_pass"] += 1
            target_dist[1][s1["answer"]] += 1
            writers["step1-corrupted"].write(json.dumps(repo_row(rid, s1["content"], s1["answer"])) + "\n")
            writers["step1-pairs"].write(json.dumps(dict(
                common, step=1, corrupted_prompt=[{"role": "user", "content": s1["content"]}],
                corrupted_answer=s1["answer"], corrupted_letters=STD_LETTERS,
                corrupted_roles=it["roles"], order=[0, 1, 2, 3], aux_answers={},
                corrupted_qc=s1["qc"])) + "\n")

        n_cand = len(it["orders"]) if not args.no_qc else 1
        s2_alone = any(ok(k, f"step2|{ci}") for ci in range(n_cand))
        stats["step2_pass_any_candidate"] += int(s2_alone)
        chosen = next((ci for ci in range(n_cand) if ok(k, f"step2|{ci}") and ok(k, f"step3|{ci}")), None)
        if chosen is None:
            continue
        stats["step23_pass"] += 1
        stats[f"step23_candidate_index_{chosen}"] += 1
        order = it["orders"][chosen]
        new_v = order.index(v)
        corr_roles = [it["roles"][j] for j in order]
        for step, letters, key in ((2, STD_LETTERS, f"step2|{chosen}"), (3, alt, f"step3|{chosen}")):
            rec = table[(k, key)]
            target_dist[step][rec["answer"]] += 1
            writers[f"step{step}-corrupted"].write(json.dumps(repo_row(rid, rec["content"], rec["answer"])) + "\n")
            writers[f"step{step}-pairs"].write(json.dumps(dict(
                common, step=step, corrupted_prompt=[{"role": "user", "content": rec["content"]}],
                corrupted_answer=rec["answer"], corrupted_letters=letters, corrupted_roles=corr_roles,
                order=order,
                aux_answers={"index": STD_LETTERS[new_v]} if step == 3 else {},
                corrupted_qc=rec["qc"])) + "\n")

    for w in writers.values():
        w.close()
    report = dict(model=args.model, base_path=str(base_path), prose_path=str(prose_path),
                  seed=args.seed, step3_letters=alt, qc_enabled=not args.no_qc,
                  counts=dict(stats), corrupted_answer_distribution={
                      f"step{s}": dict(sorted(c.items())) for s, c in target_dist.items()})
    with open(out_dir / "qc_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()