"""
Shared pieces for the verse-long causal abstraction experiment.

Prompt anatomy, question pairing, and the four-reference teacher-forced scorer used by both
generate_data/verse_long_abstraction/build_pairs.py and verse_long_abstraction.py.

The verse-long user turn is  "Respond in {verse|prose} and concisely. {question}"  and the two
format conditions differ by a single token, so a (verse, prose) x (Q1, Q2) design gives four
prompts and four reference answers per pair:

    q1_verse  data/{model}/verse-long/verse-long-desired-all.jsonl   row i
    q1_prose  data/{model}/verse-long/prose-desired-all.jsonl        row i
    q2_verse  ...                                                    row j
    q2_prose  ...                                                    row j

Every measurement is the per-token log-probability of each of those four references under some
(possibly patched) prompt, from which two contrasts are derived:

    topic  = mean logp(q2_*) - mean logp(q1_*)
    format = mean logp(*_verse) - mean logp(*_prose)
"""

import json
import re
import unicodedata
from pathlib import Path

REF_KEYS = ("q1_verse", "q1_prose", "q2_verse", "q2_prose")
PROMPT_RE = re.compile(r"^Respond in (?P<fmt>verse|prose) and concisely\. (?P<question>.+)$", re.S)
PROMPT_TMPL = "Respond in {fmt} and concisely. {question}"
BLANK_TMPL = "Respond in {fmt} and concisely."
POSITION_SETS = ("fmt", "question", "suffix", "last", "prompt", "concisely")


# --------------------------------------------------------------------------- io
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def resolve_data_paths(repo_root, model_name, data_dir=None):
    """verse-long dir differs between trees: prose file is 'prose-desired-all' in some,
    'prose-long-desired-all' in others. Accept either."""
    base = Path(data_dir) if data_dir else Path(repo_root) / "data" / model_name / "verse-long"
    verse = _first_existing(base, ["verse-long-desired-all.jsonl", "verse-desired-all.jsonl"])
    prose = _first_existing(base, ["prose-long-desired-all.jsonl", "prose-desired-all.jsonl"])
    return verse, prose


def _first_existing(base, names):
    for n in names:
        if (base / n).exists():
            return base / n
    raise FileNotFoundError(f"none of {names} found in {base}")


def split_prompt(content):
    m = PROMPT_RE.match(content.strip())
    if m is None:
        raise ValueError(f"unexpected verse-long user turn: {content[:120]!r}")
    return m.group("fmt"), m.group("question")


def render_prompt(fmt, question):
    return PROMPT_TMPL.format(fmt=fmt, question=question)


def user_msg(fmt, question=None):
    text = BLANK_TMPL.format(fmt=fmt) if question is None else render_prompt(fmt, question)
    return [{"role": "user", "content": text}]


def load_rows(verse_path, prose_path):
    """-> {id: {question, verse_answer, prose_answer}} with the single-token-difference check."""
    verse, prose = load_jsonl(verse_path), load_jsonl(prose_path)
    prose_by_id = {r["id"]: r for r in prose}
    out = {}
    for r in verse:
        rid = r["id"]
        if rid not in prose_by_id:
            raise ValueError(f"id {rid} missing from {prose_path}")
        vf, vq = split_prompt(r["prompt"][0]["content"])
        pf, pq = split_prompt(prose_by_id[rid]["prompt"][0]["content"])
        if vf != "verse" or pf != "prose" or vq != pq:
            raise ValueError(f"id {rid}: verse/prose prompts are not a single-word minimal pair")
        answers = {}
        for key, row in (("verse_answer", r), ("prose_answer", prose_by_id[rid])):
            if row["prompt"][-1]["role"] != "assistant":
                raise ValueError(f"id {rid}: {key} row has no assistant turn")
            answers[key] = row["prompt"][-1]["content"]
        out[rid] = dict(question=vq, **answers)
    return out


# --------------------------------------------------------------------------- pairing
_WORD_RE = re.compile(r"[a-z]+")
_STOP = set("""a an the of to in on for with and or is are do does did what how why when which who
whom whose can could may might will would shall should our their its his her we you they it this
that these those be been being as at by from into over under about more most ways way role play""".split())


def content_words(text):
    words = _WORD_RE.findall(unicodedata.normalize("NFKD", text).lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_by_length(rows, length_key, max_similarity=0.2):
    """Greedy minimum-similarity matching inside equal-length buckets.

    length_key must be the length of the *templated* verse prompt in tokens, not of the question
    measured on its own: the question is preceded by a space in the template, and the merge
    changes the token count. Equal templated length plus a fixed prefix and suffix is what makes
    every position set line up index-for-index between the two prompts.
    """
    buckets = {}
    for rid, n in length_key.items():
        buckets.setdefault(n, []).append(rid)
    words = {rid: content_words(rows[rid]["question"]) for rid in rows}
    pairs, rejected = [], []
    for n, ids in sorted(buckets.items()):
        cands = sorted(
            ((jaccard(words[a], words[b]), a, b) for i, a in enumerate(ids) for b in ids[i + 1:]),
            key=lambda t: (t[0], t[1], t[2]),
        )
        used = set()
        for sim, a, b in cands:
            if a in used or b in used:
                continue
            if sim > max_similarity:
                rejected.append(dict(a=a, b=b, similarity=round(sim, 3), question_tokens=n))
                continue
            used.update((a, b))
            pairs.append(dict(q1=a, q2=b, similarity=round(sim, 3), question_tokens=n))
    return pairs, rejected


# --------------------------------------------------------------------------- anatomy
class Anatomy:
    """Token ids for one templated prompt plus the index lists the experiment patches."""

    def __init__(self, tokenizer, fmt, question=None, extra_control="concisely"):
        self.fmt = fmt
        self.question = question
        msgs = user_msg(fmt, question)
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tokenizer(text, return_offsets_mapping=True)
        self.text = text
        self.ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        def span(lo, hi):
            idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < hi and e > lo]
            if not idx:
                raise ValueError(f"empty token span for chars [{lo},{hi}) in {text[lo:hi]!r}")
            return idx

        content = msgs[0]["content"]
        u0 = text.rfind(content)
        if u0 < 0:
            raise ValueError("templated text does not contain the user content verbatim")
        u1 = u0 + len(content)
        f0 = text.index(fmt, u0)
        c0 = text.index("concisely", u0)

        self.positions = {
            "fmt": span(f0, f0 + len(fmt)),
            "suffix": span(u1, len(text)),
            "last": [len(self.ids) - 1],
            "prompt": list(range(len(self.ids))),
            extra_control: span(c0, c0 + len("concisely")),
        }
        if question is not None:
            q0 = u1 - len(question)
            self.positions["question"] = span(q0, u1)
        if self.positions["suffix"][-1] != len(self.ids) - 1:
            raise ValueError("suffix span does not reach the final token")

    def check_alignment(self, other, position_set):
        """Patching source -> base at these indices is only meaningful if the two prompts agree
        on length (and, outside the patched span, on tokens)."""
        if position_set not in self.positions or position_set not in other.positions:
            raise KeyError(f"position set {position_set!r} missing")
        if len(self.ids) != len(other.ids):
            return f"token lengths differ ({len(self.ids)} vs {len(other.ids)})"
        if self.positions[position_set] != other.positions[position_set]:
            return f"{position_set} indices differ"
        return None


# --------------------------------------------------------------------------- scoring
class ReferenceBatch:
    """One prompt x R references, right-padded.

    Right padding keeps the prompt at indices 0..T-1 in every row, so a patch at prompt
    positions applies identically across references and needs no index remapping. Causal
    attention makes trailing pads inert, and default position_ids are then correct.
    """

    def __init__(self, tokenizer, prompt_ids, references, max_ref_tokens=None, device=None):
        import torch

        self.keys = list(references)
        self.n_prompt = len(prompt_ids)
        ref_ids = []
        for k in self.keys:
            r = tokenizer(references[k], add_special_tokens=False)["input_ids"]
            if max_ref_tokens:
                r = r[:max_ref_tokens]
            if not r:
                raise ValueError(f"reference {k!r} is empty after tokenization")
            ref_ids.append(r)
        self.ref_lens = [len(r) for r in ref_ids]
        total = self.n_prompt + max(self.ref_lens)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((len(self.keys), total), pad, dtype=torch.long)
        mask = torch.zeros((len(self.keys), total), dtype=torch.long)
        tgt = torch.zeros((len(self.keys), total), dtype=torch.long)
        score = torch.zeros((len(self.keys), total), dtype=torch.bool)
        for r, rid in enumerate(ref_ids):
            row = list(prompt_ids) + rid
            ids[r, : len(row)] = torch.tensor(row)
            mask[r, : len(row)] = 1
            # predicted at t-1, so the gather index sits one position to the left
            tgt[r, self.n_prompt - 1: len(row) - 1] = torch.tensor(rid)
            score[r, self.n_prompt - 1: len(row) - 1] = True
        if device is not None:
            ids, mask, tgt, score = (x.to(device) for x in (ids, mask, tgt, score))
        self.input_ids, self.attention_mask, self.target, self.score_mask = ids, mask, tgt, score

    def tile(self, n):
        """Repeat the batch n times along dim 0 (one block per patched layer)."""
        import torch

        return (self.input_ids.repeat(n, 1), self.attention_mask.repeat(n, 1),
                self.target.repeat(n, 1), self.score_mask.repeat(n, 1))

    def reduce(self, gathered, score_mask, n_blocks=1):
        """gathered: [n_blocks*R, L] log-probs of the target token at each position."""
        lp = (gathered * score_mask).sum(-1)
        n = score_mask.sum(-1).clamp(min=1)
        mean = (lp / n).view(n_blocks, len(self.keys)).tolist()
        total = lp.view(n_blocks, len(self.keys)).tolist()
        return [{k: dict(logp_mean=m[i], logp_sum=t[i], n_tokens=self.ref_lens[i])
                 for i, k in enumerate(self.keys)} for m, t in zip(mean, total)]


def contrasts(scores):
    """topic > 0 means the run favours Q2; format > 0 means it favours verse."""
    g = lambda k: scores[k]["logp_mean"]
    return dict(
        topic=0.5 * (g("q2_verse") + g("q2_prose")) - 0.5 * (g("q1_verse") + g("q1_prose")),
        format=0.5 * (g("q1_verse") + g("q2_verse")) - 0.5 * (g("q1_prose") + g("q2_prose")),
    )


def gather_logprobs(logits, target):
    """log P(target[t] | ...) at every position, in float32."""
    import torch

    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- preflight
def _preflight():
    """Tokenizer-only check of the verse-long tree. No model weights, no GPU, a few seconds on
    a login node. Run this before queueing build_pairs.py: it answers whether the prompts parse,
    how many pairs the length constraint leaves, and whether the format arm is viable at all
    under this tokenizer."""
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description=_preflight.__doc__)
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--max_similarity", type=float, default=0.2)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model_name = args.model.rstrip("/").split("/")[-1]
    verse_path, prose_path = resolve_data_paths(args.repo_root, model_name, args.data_dir)
    rows = load_rows(verse_path, prose_path)
    print(f"verse file : {verse_path}")
    print(f"prose file : {prose_path}")
    print(f"rows parsed: {len(rows)} (prompts all match the "
          f"'Respond in {{verse|prose}} and concisely. <q>' frame)")

    tok = AutoTokenizer.from_pretrained(args.model)
    anat = {rid: {"verse": Anatomy(tok, "verse", r["question"]),
                  "prose": Anatomy(tok, "prose", r["question"])} for rid, r in rows.items()}
    aligned = {rid: len(a["verse"].ids) == len(a["prose"].ids) for rid, a in anat.items()}
    v_tok = tok.encode(" verse", add_special_tokens=False)
    p_tok = tok.encode(" prose", add_special_tokens=False)
    print(f"\n' verse' -> {len(v_tok)} token(s), ' prose' -> {len(p_tok)} token(s)")
    print(f"format-aligned rows (verse/prose prompts equal length): "
          f"{sum(aligned.values())}/{len(rows)}")
    if not any(aligned.values()):
        print("  !! the format condition cannot run under this tokenizer; topic, compete and")
        print("     transplant are unaffected")

    lengths = {rid: len(a["verse"].ids) for rid, a in anat.items()}
    buckets = Counter(lengths.values())
    print(f"\ntemplated verse-prompt lengths: {min(lengths.values())}-{max(lengths.values())} "
          f"tokens across {len(buckets)} buckets")
    pairs, rejected = pair_by_length(rows, lengths, args.max_similarity)
    print(f"length-matched pairs: {len(pairs)}  "
          f"({len(rejected)} candidates rejected as too lexically similar)")
    print(f"  of those, format-aligned on both sides: "
          f"{sum(aligned[p['q1']] and aligned[p['q2']] for p in pairs)}")

    bad = []
    for p in pairs:
        a, b = anat[p["q1"]]["verse"], anat[p["q2"]]["verse"]
        for ps in POSITION_SETS:
            if ps in a.positions and a.check_alignment(b, ps):
                bad.append((p["q1"], p["q2"], ps, a.check_alignment(b, ps)))
    print(f"position-set alignment failures among those pairs: {len(bad)}")
    for entry in bad[:5]:
        print("   ", entry)

    if pairs:
        p = pairs[0]
        a = anat[p["q1"]]["verse"]
        print(f"\nexample pair ({p['q1']}, {p['q2']}), similarity {p['similarity']}, "
              f"{len(a.ids)} prompt tokens")
        print(f"  Q1: {rows[p['q1']]['question']}")
        print(f"  Q2: {rows[p['q2']]['question']}")
        for ps in POSITION_SETS:
            if ps in a.positions:
                idx = a.positions[ps]
                shown = idx if len(idx) <= 6 else idx[:3] + ["..."] + idx[-2:]
                print(f"  {ps:<10} {len(idx):>3} tok  {shown}  "
                      f"{tok.decode([a.ids[i] for i in idx])[:60]!r}")
        ref_lens = {k: len(tok(v, add_special_tokens=False)["input_ids"])
                    for k, v in (("q1_verse", rows[p["q1"]]["verse_answer"]),
                                 ("q1_prose", rows[p["q1"]]["prose_answer"]),
                                 ("q2_verse", rows[p["q2"]]["verse_answer"]),
                                 ("q2_prose", rows[p["q2"]]["prose_answer"]))}
        print(f"  reference answer lengths (tokens): {ref_lens}")


if __name__ == "__main__":
    _preflight()