#!/usr/bin/env python
"""
build_summary_dataset.py
========================

QC a JSON/JSONL dataset of books with Qwen1.5-14B-Chat and emit 14 contrastive
JSONL files for length-steering experiments.

Pipeline (all stages resumable on SLURM requeue):

  1. tokcheck  (fail-fast global precondition)
        Confirm the tokenizer splits "brief" and "detailed" into the same number
        of tokens. This is a hard precondition -> the run aborts if it fails
        (unless ALLOW_TOKEN_LENGTH_MISMATCH=1).

  2. generate
        For every book, greedily generate a brief and a detailed summary.
        Checkpointed to checkpoint/summaries.jsonl (append-only, fsync per chunk).

  3. prune   (per-book QC -> filters, does NOT hard-fail by default)
        (a) LENGTH:  detailed must be >= LENGTH_RATIO x brief   (word count)
        (b) MCQA:    build the 4-way (2x2) MC question from a distracter book,
                     ask Qwen for the correct brief answer and the correct
                     detailed answer, and require BOTH to be right.
        Books failing any check are dropped and logged to output/prune_report.json.
        Set STRICT=1 to hard-fail the run instead of pruning.

  4. build
        Assign ids over the surviving books (input order). The first CAP (=100)
        books populate the 12 aligned files; the remainder populate the two
        *-test files. All aligned files share the same id->book mapping; the two
        test files share the same (higher) ids with each other.

The distracter/permutation design matches the earlier pipeline: for each book we
pick one random OTHER (length-passing) book and use its brief + detailed
summaries as the two distracters, giving a clean 2x2 {correct-brief,
correct-detailed, distracter-brief, distracter-detailed}. The A/B/C/D
permutation is drawn once per book (seeded by title) and shared across every
file so the desired / undesired / steering variants stay aligned.

Everything is greedy (temperature=0) and deterministic, so requeue is safe.
"""

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# CONFIG (every value overridable via environment variable of the same name)
# ---------------------------------------------------------------------------
def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default

CONFIG = {
    "MODEL":            _env("MODEL", "allenai/OLMo-2-1124-13B-DPO"),
    "INPUT":            _env("INPUT", "books.jsonl"),
    "BOOK_FIELD":       _env("BOOK_FIELD", ""),        # "" => auto-detect
    "OUTPUT_DIR":       _env("OUTPUT_DIR", "output/OLMo-2-1124-13B-DPO"),
    "CHECKPOINT_DIR":   _env("CHECKPOINT_DIR", "checkpoint/OLMo-2-1124-13B-DPO"),

    "STAGES":           _env("STAGES", "tokcheck,generate,prune,build"),

    # generation
    "TENSOR_PARALLEL":  _env("TENSOR_PARALLEL", 1, int),
    "GPU_MEM_UTIL":     _env("GPU_MEM_UTIL", 0.90, float),
    "MAX_MODEL_LEN":    _env("MAX_MODEL_LEN", 4096, int),
    "CHUNK_SIZE":       _env("CHUNK_SIZE", 64, int),   # books per checkpoint flush
    "BRIEF_MAX_TOKENS": _env("BRIEF_MAX_TOKENS", 900, int),
    "DETAILED_MAX_TOKENS": _env("DETAILED_MAX_TOKENS", 900, int),
    "MC_MAX_TOKENS":    _env("MC_MAX_TOKENS", 4, int),

    # prompt wording (must contain {book}); flows to BOTH generation and the
    # emitted files so the recorded prompts always match what was asked. Kept
    # parallel ("a one X summary") so the contrastive word sits at a matched
    # token position for activation work.
    "BRIEF_PROMPT":     _env("BRIEF_PROMPT", "Please give a one sentence summary of {book}"),
    "DETAILED_PROMPT":  _env("DETAILED_PROMPT", "Please give a one paragraph summary of {book}"),
    # the contrastive pair whose token lengths must match (tokcheck gate)
    "CONTRAST_A":       _env("CONTRAST_A", "sentence"),
    "CONTRAST_B":       _env("CONTRAST_B", "paragraph"),
    # descriptor used inside the multiple-choice prompts ("...the correct one
    # sentence summary of the book X") — used by both the -single- files and the
    # QC checks, so they stay in sync with the -long- generation prompts.
    "BRIEF_LABEL":      _env("BRIEF_LABEL", "one sentence"),
    "DETAILED_LABEL":   _env("DETAILED_LABEL", "one paragraph"),

    # QC
    "LENGTH_RATIO":     _env("LENGTH_RATIO", 2.0, float),
    "LENGTH_METRIC":    _env("LENGTH_METRIC", "word"),   # "word" | "char" | "token"
    "VERIFY_MODE":      _env("VERIFY_MODE", "filter"),   # "filter" (drop) | "report" (keep, log accuracy)
    "STRICT":           _env("STRICT", 0, int),          # 1 => hard-fail on any QC drop
    "ALLOW_TOKEN_LENGTH_MISMATCH": _env("ALLOW_TOKEN_LENGTH_MISMATCH", 0, int),

    # output
    "CAP":              _env("CAP", 100, int),
    "SEED":             _env("SEED", 0, int),
}

LETTERS = ["A", "B", "C", "D"]

_CHOICE_RE = None
def parse_choice(text: str) -> str:
    """Extract the model's A/B/C/D choice, ignoring letters embedded in words.

    Matches a standalone A-D not flanked by other letters, so 'Answer: C' -> 'C',
    '(A)' -> 'A', 'The answer is B.' -> 'B'. Returns '' if none found.
    """
    global _CHOICE_RE
    if _CHOICE_RE is None:
        import re
        _CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
    m = _CHOICE_RE.search((text or "").upper())
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Preemption handling: finish the current chunk, checkpoint, exit for requeue.
# ---------------------------------------------------------------------------
_PREEMPTED = False

def _on_signal(signum, frame):
    global _PREEMPTED
    _PREEMPTED = True
    print(f"[signal] caught {signal.Signals(signum).name}; will checkpoint and "
          f"exit at the next chunk boundary", flush=True)

for _sig in (signal.SIGUSR1, signal.SIGTERM):
    try:
        signal.signal(_sig, _on_signal)
    except (ValueError, OSError):
        pass


def _requeue_and_exit():
    """Flush is already durable (fsync per chunk); ask SLURM to requeue us."""
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        os.system(f"scontrol requeue {jid}")
    print("[signal] checkpoint durable; exiting for requeue", flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Durable append-only JSONL checkpoint helpers
# ---------------------------------------------------------------------------
def append_jsonl(path: str, records: List[dict]):
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[dict]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_books(path: str, book_field: str) -> List[str]:
    if not os.path.exists(path):
        sys.exit(f"[fatal] input not found: {path}")

    raw: List = []
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        sys.exit(f"[fatal] input is empty: {path}")

    # Try JSONL first, then a single JSON array/object.
    try:
        raw = [json.loads(l) for l in text.splitlines() if l.strip()]
    except json.JSONDecodeError:
        obj = json.loads(text)
        raw = obj if isinstance(obj, list) else [obj]

    def extract(rec) -> str:
        if isinstance(rec, str):
            return rec
        if isinstance(rec, dict):
            if book_field:
                if book_field not in rec:
                    sys.exit(f"[fatal] BOOK_FIELD='{book_field}' missing in a record")
                return str(rec[book_field])
            for cand in ("book", "title", "name", "text"):
                if cand in rec:
                    return str(rec[cand])
            sys.exit(f"[fatal] could not auto-detect book field; set BOOK_FIELD. "
                     f"keys seen: {list(rec.keys())}")
        return str(rec)

    books = [extract(r).strip() for r in raw]
    books = [b for b in books if b]
    if not books:
        sys.exit("[fatal] no books parsed from input")

    # De-dup while preserving order (title is our stable key downstream).
    seen, uniq = set(), []
    for b in books:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    if len(uniq) != len(books):
        print(f"[warn] dropped {len(books) - len(uniq)} duplicate titles", flush=True)
    return uniq


# ---------------------------------------------------------------------------
# Prompt construction (reproduces the user's templates verbatim)
# ---------------------------------------------------------------------------
def brief_gen_prompt(book: str) -> str:
    return CONFIG["BRIEF_PROMPT"].format(book=book)

def detailed_gen_prompt(book: str) -> str:
    return CONFIG["DETAILED_PROMPT"].format(book=book)

def option_block(opts: List[str]) -> str:
    # Note the deliberate trailing space after option (A), matching the spec.
    return f"(A) {opts[0]} \n(B) {opts[1]}\n(C) {opts[2]}\n(D) {opts[3]}"

def mc_brief_prompt(book: str, opts: List[str]) -> str:
    return (f'Please identify the correct {CONFIG["BRIEF_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D".\n'
            + option_block(opts))

def mc_detailed_prompt(book: str, opts: List[str]) -> str:
    return (f'Please identify the correct {CONFIG["DETAILED_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D".\n'
            + option_block(opts))

def mc_brief_steer_prompt(book: str) -> str:
    return (f'Please identify the correct {CONFIG["BRIEF_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D"')

def mc_detailed_steer_prompt(book: str) -> str:
    return (f'Please identify the correct {CONFIG["DETAILED_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D"')


# ---------------------------------------------------------------------------
# Per-book record
# ---------------------------------------------------------------------------
@dataclass
class Book:
    title: str
    brief: Optional[str] = None
    detailed: Optional[str] = None
    # assigned during prune:
    distr_brief: Optional[str] = None
    distr_detailed: Optional[str] = None
    brief_idx: int = -1
    detailed_idx: int = -1
    distr_brief_idx: int = -1
    distr_detailed_idx: int = -1
    mc_brief_answer: Optional[str] = None
    mc_detailed_answer: Optional[str] = None
    length_ok: bool = False
    mc_ok: bool = False

    def opts(self) -> List[str]:
        o = [None, None, None, None]
        o[self.brief_idx] = self.brief
        o[self.detailed_idx] = self.detailed
        o[self.distr_brief_idx] = self.distr_brief
        o[self.distr_detailed_idx] = self.distr_detailed
        return o

    @property
    def brief_letter(self) -> str:
        return LETTERS[self.brief_idx]

    @property
    def detailed_letter(self) -> str:
        return LETTERS[self.detailed_idx]


# ---------------------------------------------------------------------------
# Length metric
# ---------------------------------------------------------------------------
def measure(text: str, metric: str, tokenizer=None) -> int:
    if metric == "char":
        return len(text)
    if metric == "token" and tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())  # "word" (default)


# ---------------------------------------------------------------------------
# vLLM plumbing (imported lazily so non-GPU stages / --help stay light)
# ---------------------------------------------------------------------------
def load_model():
    from vllm import LLM
    from transformers import AutoTokenizer
    print(f"[model] loading {CONFIG['MODEL']} "
          f"(TP={CONFIG['TENSOR_PARALLEL']})", flush=True)
    tok = AutoTokenizer.from_pretrained(CONFIG["MODEL"], trust_remote_code=True)
    llm = LLM(
        model=CONFIG["MODEL"],
        tensor_parallel_size=CONFIG["TENSOR_PARALLEL"],
        gpu_memory_utilization=CONFIG["GPU_MEM_UTIL"],
        max_model_len=CONFIG["MAX_MODEL_LEN"],
        trust_remote_code=True,
        seed=CONFIG["SEED"],
    )
    return llm, tok


def chat_render(tok, user_content: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    )


def batched_generate(llm, tok, user_prompts: List[str], max_tokens: int) -> List[str]:
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    rendered = [chat_render(tok, p) for p in user_prompts]
    outs = llm.generate(rendered, sp)
    return [o.outputs[0].text.strip() for o in outs]


# ===========================================================================
# STAGE 1: tokenizer precondition
# ===========================================================================
def stage_tokcheck(tok):
    def n(s):
        return len(tok.encode(s, add_special_tokens=False))
    a, b = CONFIG["CONTRAST_A"], CONFIG["CONTRAST_B"]
    na, nb = n(a), n(b)                       # bare
    na_sp, nb_sp = n(" " + a), n(" " + b)     # in-context (leading space) — this is
                                              # the form that appears in the prompt
                                              # and the one position alignment needs
    print(f"[tokcheck] tokens: '{a}'={na} '{b}'={nb} | "
          f"' {a}'={na_sp} ' {b}'={nb_sp}", flush=True)
    # Gate on the in-context form: "a one {sentence|paragraph} summary" -> the
    # contrastive word always carries a leading space.
    if na_sp != nb_sp:
        msg = (f"[tokcheck] FAIL: ' {a}' -> {na_sp} tokens but ' {b}' -> {nb_sp} "
               f"tokens (in-context lengths must match for position alignment).")
        if CONFIG["ALLOW_TOKEN_LENGTH_MISMATCH"]:
            print(msg + " Continuing because ALLOW_TOKEN_LENGTH_MISMATCH=1.",
                  flush=True)
        else:
            sys.exit(msg + " Set ALLOW_TOKEN_LENGTH_MISMATCH=1 to override.")
    else:
        print(f"[tokcheck] PASS: ' {a}' and ' {b}' are both {na_sp} token(s)",
              flush=True)


# ===========================================================================
# STAGE 2: generate brief + detailed summaries (resumable, chunked)
# ===========================================================================
def stage_generate(llm, tok, titles: List[str], ckpt: str):
    done = {r["title"]: r for r in load_jsonl(ckpt)}
    todo = [t for t in titles if t not in done]
    print(f"[generate] {len(done)} cached / {len(todo)} to do", flush=True)

    cs = CONFIG["CHUNK_SIZE"]
    for i in range(0, len(todo), cs):
        chunk = todo[i:i + cs]
        briefs = batched_generate(llm, tok,
                                  [brief_gen_prompt(t) for t in chunk],
                                  CONFIG["BRIEF_MAX_TOKENS"])
        detaileds = batched_generate(llm, tok,
                                     [detailed_gen_prompt(t) for t in chunk],
                                     CONFIG["DETAILED_MAX_TOKENS"])
        recs = [{"title": t, "brief": b, "detailed": d}
                for t, b, d in zip(chunk, briefs, detaileds)]
        append_jsonl(ckpt, recs)
        print(f"[generate] chunk {i//cs + 1}: +{len(recs)} "
              f"(total {len(done)+i+len(recs)}/{len(titles)})", flush=True)
        if _PREEMPTED:
            _requeue_and_exit()


# ===========================================================================
# STAGE 3: prune (length QC + distracter assignment + MCQA QC)
# ===========================================================================
def _perm_for(title: str) -> List[int]:
    """Deterministic A/B/C/D permutation seeded by (global seed, title)."""
    h = hashlib.sha256(f"{CONFIG['SEED']}:{title}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    idxs = [0, 1, 2, 3]
    rng.shuffle(idxs)
    return idxs  # -> [brief_idx, detailed_idx, distr_brief_idx, distr_detailed_idx]


def _distracter_for(pos: int, n: int) -> int:
    """Deterministic distinct 'other book' index for the length-passing pool."""
    if n < 2:
        return -1
    h = hashlib.sha256(f"{CONFIG['SEED']}:distr:{pos}".encode()).hexdigest()
    offset = 1 + (int(h[:16], 16) % (n - 1))   # in [1, n-1] -> never self
    return (pos + offset) % n


def stage_prune(llm, tok, titles: List[str], gen_ckpt: str, mc_ckpt: str,
                out_dir: str) -> List[Book]:
    gen = {r["title"]: r for r in load_jsonl(gen_ckpt)}
    missing = [t for t in titles if t not in gen]
    if missing:
        sys.exit(f"[fatal] prune: {len(missing)} books missing summaries "
                 f"(run the generate stage first). e.g. {missing[:3]}")

    books = [Book(title=t, brief=gen[t]["brief"], detailed=gen[t]["detailed"])
             for t in titles]

    # (a) length check
    metric = CONFIG["LENGTH_METRIC"]
    ratio = CONFIG["LENGTH_RATIO"]
    length_fail = []
    for b in books:
        lb = measure(b.brief, metric, tok)
        ld = measure(b.detailed, metric, tok)
        b.length_ok = (lb > 0 and ld >= ratio * lb)
        if not b.length_ok:
            length_fail.append({"title": b.title, "brief_len": lb, "detailed_len": ld})

    pool = [b for b in books if b.length_ok]   # only length-passers can be options
    print(f"[prune] length: {len(pool)}/{len(books)} pass "
          f"(detailed >= {ratio}x brief, metric={metric})", flush=True)
    if not pool:
        sys.exit("[fatal] no book passed the length check")

    # (b) assign distracters + permutation over the length-passing pool
    n = len(pool)
    for pos, b in enumerate(pool):
        dpos = _distracter_for(pos, n)
        d = pool[dpos]
        b.distr_brief = d.brief
        b.distr_detailed = d.detailed
        p = _perm_for(b.title)
        b.brief_idx, b.detailed_idx, b.distr_brief_idx, b.distr_detailed_idx = p

    # (c) MCQA verification (resumable, chunked)
    mc_done = {r["title"]: r for r in load_jsonl(mc_ckpt)}
    todo = [b for b in pool if b.title not in mc_done]
    print(f"[prune] mcqa: {len(mc_done)} cached / {len(todo)} to verify", flush=True)

    cs = CONFIG["CHUNK_SIZE"]
    for i in range(0, len(todo), cs):
        chunk = todo[i:i + cs]
        br_ans = batched_generate(llm, tok,
                                  [mc_brief_prompt(b.title, b.opts()) for b in chunk],
                                  CONFIG["MC_MAX_TOKENS"])
        de_ans = batched_generate(llm, tok,
                                  [mc_detailed_prompt(b.title, b.opts()) for b in chunk],
                                  CONFIG["MC_MAX_TOKENS"])
        recs = [{"title": b.title, "brief_ans": ba, "detailed_ans": da}
                for b, ba, da in zip(chunk, br_ans, de_ans)]
        append_jsonl(mc_ckpt, recs)
        print(f"[prune] mcqa chunk {i//cs + 1}: +{len(recs)}", flush=True)
        if _PREEMPTED:
            _requeue_and_exit()

    mc_done = {r["title"]: r for r in load_jsonl(mc_ckpt)}

    mc_fail = []
    n_brief_ok = n_detailed_ok = 0
    for b in pool:
        rec = mc_done[b.title]
        b.mc_brief_answer = parse_choice(rec["brief_ans"])
        b.mc_detailed_answer = parse_choice(rec["detailed_ans"])
        ok_b = b.mc_brief_answer == b.brief_letter
        ok_d = b.mc_detailed_answer == b.detailed_letter
        n_brief_ok += ok_b
        n_detailed_ok += ok_d
        b.mc_ok = ok_b and ok_d
        if not b.mc_ok:
            mc_fail.append({"title": b.title,
                            "brief_expected": b.brief_letter, "brief_got": b.mc_brief_answer,
                            "detailed_expected": b.detailed_letter,
                            "detailed_got": b.mc_detailed_answer})

    mode = CONFIG["VERIFY_MODE"]
    n_pass = sum(b.mc_ok for b in pool)
    n_same = sum(1 for b in pool
                 if b.mc_brief_answer and b.mc_brief_answer == b.mc_detailed_answer)
    print(f"[prune] mcqa accuracy: brief {n_brief_ok}/{len(pool)}, "
          f"detailed {n_detailed_ok}/{len(pool)}, both {n_pass}/{len(pool)} | "
          f"same answer to both Qs: {n_same}/{len(pool)} (mode={mode})", flush=True)

    if mode == "report":
        valid = pool                      # keep all length-passers
    elif mode == "filter":
        valid = [b for b in pool if b.mc_ok]
    else:
        sys.exit(f"[fatal] VERIFY_MODE must be 'filter' or 'report', got '{mode}'")

    report = {
        "n_input": len(books),
        "n_length_pass": len(pool),
        "n_valid": len(valid),
        "verify_mode": mode,
        "mcqa_brief_correct": n_brief_ok,
        "mcqa_detailed_correct": n_detailed_ok,
        "mcqa_both_correct": n_pass,
        "mcqa_same_answer_both_questions": n_same,
        "length_ratio": ratio, "length_metric": metric,
        "length_failures": length_fail,
        "mcqa_failures": mc_fail,
    }
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "prune_report.jsonl"), [report])

    n_dropped = len(books) - len(valid)
    if n_dropped and CONFIG["STRICT"]:
        sys.exit(f"[fatal] STRICT=1: {n_dropped} book(s) failed QC "
                 f"(see {out_dir}/prune_report.jsonl)")
    if not valid:
        sys.exit("[fatal] no book passed all QC checks")

    return valid


# ===========================================================================
# STAGE 4: build the 14 files
# ===========================================================================
def _row(idx: int, user: str, assistant: Optional[str] = None) -> dict:
    prompt = [{"role": "user", "content": user}]
    if assistant is not None:
        prompt.append({"role": "assistant", "content": assistant})
    return {"id": idx, "prompt": prompt}


def stage_build(valid: List[Book], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap = CONFIG["CAP"]
    # ids follow input order among survivors
    for i, b in enumerate(valid):
        b._id = i  # type: ignore[attr-defined]

    head = valid[:cap]          # -> 12 aligned files
    tail = valid[cap:]          # -> 2 test files

    files: Dict[str, List[dict]] = {}

    def emit(name, row):
        files.setdefault(name, []).append(row)

    for b in head:
        i = b._id  # type: ignore[attr-defined]
        opts = b.opts()
        mc_b = mc_brief_prompt(b.title, opts)
        mc_d = mc_detailed_prompt(b.title, opts)

        # ---- single (multiple-choice) ----
        emit("sentence-single-desired-all.jsonl",   _row(i, mc_b, b.brief_letter))
        emit("paragraph-single-desired-all.jsonl",  _row(i, mc_d, b.detailed_letter))
        emit("sentence-single-undesired-all.jsonl", _row(i, mc_b, b.detailed_letter))
        emit("paragraph-single-undesired-all.jsonl",_row(i, mc_d, b.brief_letter))
        emit("sentence-single-steering.jsonl",
             _row(i, mc_brief_steer_prompt(b.title), b.brief_letter))
        # NOTE: spec labelled this 6th file "paragraph-single-desired-all" (a
        # duplicate name) but its content is the paragraph steering variant;
        # written as paragraph-single-steering.jsonl.
        emit("paragraph-single-steering.jsonl",
             _row(i, mc_detailed_steer_prompt(b.title), b.detailed_letter))

        # ---- long (free-form) ----
        gb, gd = brief_gen_prompt(b.title), detailed_gen_prompt(b.title)
        emit("sentence-long-desired-all.jsonl",    _row(i, gb, b.brief))
        emit("paragraph-long-desired-all.jsonl",   _row(i, gd, b.detailed))
        emit("sentence-long-undesired-all.jsonl",  _row(i, gb, b.detailed))
        emit("paragraph-long-undesired-all.jsonl", _row(i, gd, b.brief))
        emit("sentence-long-steering.jsonl",       _row(i, gb, b.brief))
        emit("paragraph-long-steering.jsonl",      _row(i, gd, b.detailed))

    for b in tail:
        i = b._id  # type: ignore[attr-defined]
        emit("sentence-long-test.jsonl",   _row(i, brief_gen_prompt(b.title)))
        emit("sentence-single-test.jsonl", _row(i, mc_brief_prompt(b.title, b.opts())))

    expected = [
        "sentence-single-desired-all.jsonl", "paragraph-single-desired-all.jsonl",
        "sentence-single-undesired-all.jsonl", "paragraph-single-undesired-all.jsonl",
        "sentence-single-steering.jsonl", "paragraph-single-steering.jsonl",
        "sentence-long-desired-all.jsonl", "paragraph-long-desired-all.jsonl",
        "sentence-long-undesired-all.jsonl", "paragraph-long-undesired-all.jsonl",
        "sentence-long-steering.jsonl", "paragraph-long-steering.jsonl",
        "sentence-long-test.jsonl", "sentence-single-test.jsonl",
    ]
    for name in expected:
        rows = files.get(name, [])
        write_jsonl(os.path.join(out_dir, name), rows)
        print(f"[build] {name}: {len(rows)} rows", flush=True)

    assert len(expected) == 14
    print(f"[build] wrote 14 files to {out_dir}/ "
          f"(head={len(head)}, tail={len(tail)})", flush=True)


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CONFIG["INPUT"])
    ap.add_argument("--stages", default=CONFIG["STAGES"])
    args = ap.parse_args()
    CONFIG["INPUT"] = args.input
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
    gen_ckpt = os.path.join(CONFIG["CHECKPOINT_DIR"], "summaries.jsonl")
    mc_ckpt = os.path.join(CONFIG["CHECKPOINT_DIR"], "mcqa.jsonl")

    t0 = time.time()
    titles = load_books(CONFIG["INPUT"], CONFIG["BOOK_FIELD"])
    print(f"[main] {len(titles)} books | stages={stages}", flush=True)

    need_model = any(s in stages for s in ("tokcheck", "generate", "prune"))
    llm = tok = None
    if need_model:
        llm, tok = load_model()

    if "tokcheck" in stages:
        stage_tokcheck(tok)

    if "generate" in stages:
        stage_generate(llm, tok, titles, gen_ckpt)

    valid = None
    if "prune" in stages:
        valid = stage_prune(llm, tok, titles, gen_ckpt, mc_ckpt, CONFIG["OUTPUT_DIR"])

    if "build" in stages:
        if valid is None:
            # build alone: recompute QC deterministically from checkpoints
            valid = stage_prune(llm, tok, titles, gen_ckpt, mc_ckpt, CONFIG["OUTPUT_DIR"]) \
                if tok is not None else _rebuild_valid_from_ckpt(titles, gen_ckpt, mc_ckpt)
        stage_build(valid, CONFIG["OUTPUT_DIR"])

    print(f"[main] done in {time.time()-t0:.1f}s", flush=True)


def _rebuild_valid_from_ckpt(titles, gen_ckpt, mc_ckpt) -> List[Book]:
    """Reconstruct the valid-book list from checkpoints without a GPU.

    Used only when 'build' runs standalone (no tokenizer loaded). Length metric
    'token' is unavailable here; falls back to 'word'.
    """
    gen = {r["title"]: r for r in load_jsonl(gen_ckpt)}
    mc = {r["title"]: r for r in load_jsonl(mc_ckpt)}
    metric = CONFIG["LENGTH_METRIC"]
    if metric == "token":
        print("[build-standalone] token metric needs a tokenizer; using 'word'",
              flush=True)
        metric = "word"
    ratio = CONFIG["LENGTH_RATIO"]

    books = [Book(title=t, brief=gen[t]["brief"], detailed=gen[t]["detailed"])
             for t in titles if t in gen]
    for b in books:
        lb, ld = measure(b.brief, metric), measure(b.detailed, metric)
        b.length_ok = lb > 0 and ld >= ratio * lb
    pool = [b for b in books if b.length_ok]
    n = len(pool)
    for pos, b in enumerate(pool):
        d = pool[_distracter_for(pos, n)]
        b.distr_brief, b.distr_detailed = d.brief, d.detailed
        b.brief_idx, b.detailed_idx, b.distr_brief_idx, b.distr_detailed_idx = \
            _perm_for(b.title)

    for b in pool:
        r = mc.get(b.title)
        if r is None:
            continue
        b.mc_ok = (parse_choice(r["brief_ans"]) == b.brief_letter and
                   parse_choice(r["detailed_ans"]) == b.detailed_letter)

    if CONFIG["VERIFY_MODE"] == "report":
        return pool
    return [b for b in pool if b.mc_ok]


if __name__ == "__main__":
    main()
