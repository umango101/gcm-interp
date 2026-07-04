#!/usr/bin/env python
"""
Post-build verification for the MCQA datasets.

Asks Qwen1.5-14B-Chat each multiple-choice question in
  sentence-single-desired-all.jsonl   (correct answer = the brief summary)
  paragraph-single-desired-all.jsonl  (correct answer = the detailed summary)
and DELETES rows the model does not answer as expected.

These two files share an `id` per book and the same A/B/C/D option block, so by
default a book is kept only if Qwen answers BOTH versions correctly (REQUIRE_BOTH),
keeping the two files aligned. Original `id`s are preserved (gaps are expected),
so rows still line up across files. Originals are backed up to <file>.bak.

Run this AFTER build_summary_dataset.py has written output/*.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

# Route sampling through the PyTorch-native path so vLLM doesn't JIT-compile the
# FlashInfer sampler (which needs nvcc). Must precede any vllm import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def _env(key, default):
    v = os.environ.get(key)
    return v if v is not None else default


CONFIG = {
    "OUTPUT_DIR":      _env("OUTPUT_DIR", "output"),
    "DESIRED_FILES":   ["sentence-single-desired-all.jsonl", "paragraph-single-desired-all.jsonl"],

    # A book is kept only if Qwen answers correctly in BOTH judged files (True), or
    # in EITHER of them (False). The single surviving id set is applied to ALL_FILES,
    # so any deleted line is removed from every one of the output files.
    "REQUIRE_BOTH":    _env("REQUIRE_BOTH", "1") == "1",
    "DRY_RUN":         _env("DRY_RUN", "0") == "1",   # report only, don't rewrite

    # Judged on DESIRED_FILES; pruned on ALL of these (all 10 files).
    "ALL_FILES": [
        "sentence-single-desired-all.jsonl", "paragraph-single-desired-all.jsonl",
        "sentence-single-undesired-all.jsonl", "paragraph-single-undesired-all.jsonl",
        "sentence-long-desired-all.jsonl", "paragraph-long-desired-all.jsonl",
        "sentence-long-undesired-all.jsonl", "paragraph-long-undesired-all.jsonl",
        "sentence-single-steering.jsonl", "paragraph-single-steering.jsonl",
    ],

    # --- Model / vLLM ---
    "MODEL":            _env("MODEL", "Qwen/Qwen1.5-14B-Chat"),
    "TENSOR_PARALLEL":  int(_env("TENSOR_PARALLEL", "1")),
    "GPU_MEM_UTIL":     float(_env("GPU_MEM_UTIL", "0.90")),
    "MAX_MODEL_LEN":    int(_env("MAX_MODEL_LEN", "4096")),
    "DTYPE":            _env("DTYPE", "bfloat16"),
    "SYSTEM_PROMPT":    _env("SYSTEM_PROMPT", "You are a helpful assistant."),
    "SEED":             int(_env("SEED", "1234")),
    "ANSWER_MAX_TOKENS": int(_env("ANSWER_MAX_TOKENS", "8")),
    "DEBUG_SAMPLES":     int(_env("DEBUG_SAMPLES", "8")),   # print this many sample judgments (and up to 2x failures)
}

LETTER_SET = {"A", "B", "C", "D"}

# Two of the four options summarize the SAME book (one brief, one detailed), so the
# only thing distinguishing the intended answer is LENGTH. Make that the explicit
# selection key, or the model just picks the richer (detailed) option for both
# questions and the brief file fails almost everywhere.
JUDGE_INSTRUCTIONS = {
    "brief": (
        "\n\nTwo of the four options are summaries of {book} (one longer, one shorter); "
        "the other two summarize a DIFFERENT book. Among the two options about {book}, "
        "choose the SHORTER one."
        "\n\nRespond with ONLY the single letter of the correct option (A, B, C, or D). "
        "Do not output any other text."
    ),
    "detailed": (
        "\n\nTwo of the four options are summaries of {book} (one longer, one shorter); "
        "the other two summarize a DIFFERENT book. Among the two options about {book}, "
        "choose the LONGER one."
        "\n\nRespond with ONLY the single letter of the correct option (A, B, C, or D). "
        "Do not output any other text."
    ),
}


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def save_jsonl(path, rows):
    with Path(path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _extract_book(raw_user_content):
    """Pull the book title out of '...summary of the book <TITLE>\n(A) ...'."""
    m = re.search(r"summary of the book (.+?)\n", raw_user_content)
    return m.group(1).strip() if m else "this book"


def build_judge_user_text(raw_user_content, kind):
    """Strip the stored 'Answer: (' cue and append a length-aware bare-letter instruction.

    kind is 'brief' (sentence file) or 'detailed' (paragraph file).
    """
    t = raw_user_content.rstrip()
    if t.endswith("Answer: ("):
        t = t[: -len("Answer: (")].rstrip()
    book = _extract_book(raw_user_content)
    return t + JUDGE_INSTRUCTIONS[kind].format(book=book)


def _letter_from_logprobs(first_pos):
    """Highest-logprob A/B/C/D token at the first generated position."""
    best, best_lp = None, float("-inf")
    for lp in (first_pos or {}).values():
        tok = (getattr(lp, "decoded_token", "") or "").strip().upper()
        if tok in LETTER_SET and lp.logprob > best_lp:
            best, best_lp = tok, lp.logprob
    return best


def _letter_from_text(text):
    """Parse a letter from text WITHOUT matching letters inside words."""
    if not text:
        return None
    s = text.strip()
    m = re.fullmatch(r"\(?\s*([ABCD])\s*[\).:]?\s*", s)              # bare "A" / "(A)" / "A."
    if m:
        return m.group(1)
    m = re.search(r"answer\b[^A-Da-d]{0,12}([ABCD])\b", s, re.IGNORECASE)  # "answer is (B)"
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([ABCD])\)", s)                                # "(C)" anywhere
    if m:
        return m.group(1)
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", s)           # standalone uppercase letter
    return m.group(1) if m else None


def predict_letters(judge_texts):
    """Greedy-decode Qwen's answer letter for each judge prompt.

    Primary signal is the first-token logprob over {A,B,C,D}; text parsing is the
    fallback. Both ignore letters embedded in words, unlike the old parser.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(CONFIG["MODEL"], trust_remote_code=True)
    llm = LLM(
        model=CONFIG["MODEL"],
        tensor_parallel_size=CONFIG["TENSOR_PARALLEL"],
        gpu_memory_utilization=CONFIG["GPU_MEM_UTIL"],
        max_model_len=CONFIG["MAX_MODEL_LEN"],
        dtype=CONFIG["DTYPE"],
        seed=CONFIG["SEED"],
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=CONFIG["ANSWER_MAX_TOKENS"],
                        seed=CONFIG["SEED"], logprobs=20)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": CONFIG["SYSTEM_PROMPT"]},
             {"role": "user", "content": jt}],
            tokenize=False, add_generation_prompt=True)
        for jt in judge_texts
    ]
    outs = llm.generate(prompts, sp)
    results = []
    for o in outs:
        out = o.outputs[0]
        first_pos = out.logprobs[0] if out.logprobs else None
        letter = _letter_from_logprobs(first_pos) or _letter_from_text(out.text)
        results.append((letter, out.text))
    return results


def main():
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    files = CONFIG["DESIRED_FILES"]

    data = {}
    for fn in files:
        p = out_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Expected {p}; run build_summary_dataset.py first.")
        data[fn] = load_jsonl(p)
    print(f"[verify-mcqa] loaded {', '.join(f'{fn}={len(data[fn])}' for fn in files)}", flush=True)

    # One combined batch across both files. kind drives the length-aware instruction.
    contents, index = [], []
    for fn in files:
        kind = "brief" if fn.startswith("sentence") else "detailed"
        for i, rec in enumerate(data[fn]):
            contents.append(build_judge_user_text(rec["prompt"][0]["content"], kind))
            index.append((fn, i))

    preds = predict_letters(contents)
    if len(preds) != len(contents):
        raise RuntimeError(f"prediction count {len(preds)} != prompt count {len(contents)}")

    # correctness[id][file] = bool
    correctness, per_file_correct = {}, {fn: 0 for fn in files}
    debug_rows = []
    for (fn, i), (pred, text) in zip(index, preds):
        rec = data[fn][i]
        expected = rec["prompt"][1]["content"].strip().upper()
        ok = (pred == expected)
        correctness.setdefault(rec["id"], {})[fn] = ok
        per_file_correct[fn] += int(ok)
        if len(debug_rows) < CONFIG["DEBUG_SAMPLES"] or (not ok and len(debug_rows) < 2 * CONFIG["DEBUG_SAMPLES"]):
            debug_rows.append((fn, rec["id"], expected, pred, ok, (text or "").strip().replace("\n", " ")[:60]))

    if CONFIG["DEBUG_SAMPLES"]:
        print("[verify-mcqa] sample judgments (file | id | expected | predicted | ok | model_text):", flush=True)
        for fn, bid, exp, pred, ok, txt in debug_rows:
            print(f"    {fn[:24]:24} id={bid:<4} exp={exp} pred={pred} {'OK ' if ok else 'XX '} {txt!r}", flush=True)

    # Single global keep set, applied to every file so a dropped id is gone everywhere.
    if CONFIG["REQUIRE_BOTH"]:
        keep_ids = {bid for bid, d in correctness.items() if all(d.get(fn, False) for fn in files)}
    else:
        keep_ids = {bid for bid, d in correctness.items() if any(d.get(fn, False) for fn in files)}
    dropped_ids = sorted(bid for bid in correctness if bid not in keep_ids)

    # Report.
    report = {
        "n_books": len(correctness),
        "require_both": CONFIG["REQUIRE_BOTH"],
        "per_file_judged_accuracy": {
            fn: round(per_file_correct[fn] / max(len(data[fn]), 1), 4) for fn in files
        },
        "rows_before": len(correctness),
        "rows_kept": len(keep_ids),
        "rows_dropped": len(dropped_ids),
        "files_pruned": CONFIG["ALL_FILES"],
        "dropped_ids": dropped_ids,
    }
    for fn in files:
        print(f"[verify-mcqa] {fn}: judged acc={report['per_file_judged_accuracy'][fn]:.3f}", flush=True)
    print(f"[verify-mcqa] keeping {len(keep_ids)}/{len(correctness)} ids; "
          f"dropping {len(dropped_ids)} from all {len(CONFIG['ALL_FILES'])} files", flush=True)

    (out_dir / "mcqa_verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if CONFIG["DRY_RUN"]:
        print("[verify-mcqa] DRY_RUN: no files modified.", flush=True)
        return

    # Prune EVERY file to the single keep set (back up originals, preserve order/ids).
    for fn in CONFIG["ALL_FILES"]:
        p = out_dir / fn
        if not p.exists():
            print(f"[verify-mcqa] skip missing {fn}", flush=True)
            continue
        rows = data[fn] if fn in data else load_jsonl(p)
        kept = [r for r in rows if r["id"] in keep_ids]
        shutil.copyfile(p, p.with_suffix(p.suffix + ".bak"))
        save_jsonl(p, kept)
        print(f"[verify-mcqa] {fn}: wrote {len(kept)} rows (backup -> {fn}.bak)", flush=True)

    print("[verify-mcqa] done.", flush=True)


if __name__ == "__main__":
    main()
