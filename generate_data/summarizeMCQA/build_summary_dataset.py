#!/usr/bin/env python
"""
Generate short + long book summaries with Qwen1.5-14B-Chat (vLLM) and build
8 contrastive JSONL datasets.

Stages (run in order, each is independently resumable / re-runnable):
  1. GENERATE  -- vLLM inference, chunked, checkpointed, preemption-safe.
  2. VERIFY    -- confirm long >> short by length; write length_report.json.
  3. BUILD     -- assemble the 8 output .jsonl files (CPU-only, deterministic).

Design notes:
  * Single CONFIG block below. No CLI args needed; override via env if desired.
  * Fails fast on infra problems (missing model, OOM, bad checkpoint).
  * Generation checkpoint is append-only JSONL: one complete line per book, so a
    SIGKILL mid-write loses at most one partial (skipped) line, never corrupts
    earlier work. On requeue, finished books are skipped.
  * Heavy imports (vllm / transformers) live inside the generate stage so the
    BUILD stage can run on a CPU node without a GPU env.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from pathlib import Path

# FlashInfer JIT-compiles its top-k/top-p sampler kernel at runtime, which needs
# the CUDA toolkit (nvcc) — absent on most compute nodes. We decode greedily, so
# route sampling through the PyTorch-native path instead (no nvcc needed). Must be
# set before vllm is imported. Override by exporting VLLM_USE_FLASHINFER_SAMPLER=1
# only if you have a matching CUDA toolkit on PATH.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# --------------------------------------------------------------------------- #
# CONFIG                                                                       #
# --------------------------------------------------------------------------- #
def _env(key, default):
    v = os.environ.get(key)
    return v if v is not None else default


CONFIG = {
    # --- IO ---
    "BOOKS_JSON":     _env("BOOKS_JSON", "books.json"),   # full ordered list: defines ids; used by verify/build
    "OUTPUT_DIR":     _env("OUTPUT_DIR", "output"),
    "CHECKPOINT_DIR": _env("CHECKPOINT_DIR", "checkpoint"),

    # --- Sharding / stages ---
    # SHARD selects which subset GENERATE handles: "" => full books.json,
    # "part1"/"part2"/... => books_<shard>.json -> checkpoint/summaries_<shard>.jsonl.
    # Run the two shards as separate (concurrent) jobs, then a final job with
    # STAGES=verify,build that unions every checkpoint/summaries*.jsonl.
    "SHARD":  _env("SHARD", ""),
    "STAGES": _env("STAGES", "generate,verify,build"),

    # --- Model / vLLM ---
    "MODEL":               _env("MODEL", "Qwen/Qwen1.5-14B-Chat"),
    "TENSOR_PARALLEL":     int(_env("TENSOR_PARALLEL", "1")),   # 1 GPU per shard (run 2 shards concurrently); set 2 for single-job TP
    "GPU_MEM_UTIL":        float(_env("GPU_MEM_UTIL", "0.90")),
    "MAX_MODEL_LEN":       int(_env("MAX_MODEL_LEN", "4096")),
    "DTYPE":               _env("DTYPE", "bfloat16"),
    "SYSTEM_PROMPT":       _env("SYSTEM_PROMPT", "You are a helpful assistant."),

    # --- Sampling ---
    "TEMPERATURE":         float(_env("TEMPERATURE", "0.0")),   # greedy => reproducible/resume-safe
    "BRIEF_MAX_TOKENS":    int(_env("BRIEF_MAX_TOKENS", "200")),
    "DETAILED_MAX_TOKENS": int(_env("DETAILED_MAX_TOKENS", "1024")),

    # --- Pipeline ---
    "CHUNK_SIZE":          int(_env("CHUNK_SIZE", "16")),       # books per checkpoint flush
    "SEED":                int(_env("SEED", "1234")),

    # --- Length verification ---
    "LENGTH_RATIO_THRESHOLD": float(_env("LENGTH_RATIO_THRESHOLD", "1.5")),  # long_words >= 1.5 * short_words
    "STRICT_LENGTH":          _env("STRICT_LENGTH", "0") == "1",            # raise if any book fails the check
}

BRIEF_TMPL    = "Please give a short summary of {book}"
DETAILED_TMPL = "Please give a long summary of {book}"
LETTERS = "ABCD"

# Graceful-stop flag set by SLURM preemption signals.
_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"[signal] received {signal.Signals(signum).name}; will checkpoint and exit after current chunk.",
          flush=True)


for _sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
    signal.signal(_sig, _on_signal)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def load_books(path):
    books = json.loads(Path(path).read_text())
    if not isinstance(books, list) or not all(isinstance(b, str) for b in books):
        raise ValueError(f"{path} must be a JSON array of strings")
    if len(set(books)) != len(books):
        # Titles are used as checkpoint keys; duplicates would collide.
        dupes = [b for b in set(books) if books.count(b) > 1]
        raise ValueError(f"Duplicate book titles would break checkpoint keys: {dupes}")
    print(f"[load] {len(books)} books from {path}", flush=True)
    return books


def gen_books_path():
    """Book list that GENERATE handles for the current shard."""
    shard = CONFIG["SHARD"]
    return f"books_{shard}.json" if shard else CONFIG["BOOKS_JSON"]


def gen_checkpoint_path():
    """Checkpoint file GENERATE writes for the current shard."""
    suffix = f"_{CONFIG['SHARD']}" if CONFIG["SHARD"] else ""
    return str(Path(CONFIG["CHECKPOINT_DIR"]) / f"summaries{suffix}.jsonl")


def load_checkpoint_union():
    """Merge every checkpoint/summaries*.jsonl into one {book: {...}} map."""
    paths = sorted(Path(CONFIG["CHECKPOINT_DIR"]).glob("summaries*.jsonl"))
    merged = {}
    for p in paths:
        merged.update(load_checkpoint(str(p)))
    print(f"[ckpt] union of {len(paths)} file(s): {len(merged)} books total", flush=True)
    return merged


def load_checkpoint(path):
    """Return {book: {'short':..,'long':..}} from an append-only JSONL checkpoint."""
    done = {}
    p = Path(path)
    if not p.exists():
        return done
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Partial final line from a hard kill -- safe to ignore.
                continue
            if rec.get("short") and rec.get("long"):
                done[rec["book"]] = {"short": rec["short"], "long": rec["long"]}
    print(f"[ckpt] {len(done)} books already complete", flush=True)
    return done


def append_checkpoint(path, records):
    """Append complete records and fsync so they survive preemption."""
    with Path(path).open("a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------- #
# Stage 1: GENERATE                                                           #
# --------------------------------------------------------------------------- #
def stage_generate(books, ckpt):
    done = load_checkpoint(ckpt)
    todo = [b for b in books if b not in done]
    if not todo:
        print("[gen] nothing to generate; shard already complete.", flush=True)
        return

    # Heavy imports here so the BUILD stage can run without a GPU env.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[gen] loading {CONFIG['MODEL']} (tp={CONFIG['TENSOR_PARALLEL']}) ...", flush=True)
    t0 = time.time()
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
    print(f"[gen] model ready in {time.time() - t0:.1f}s", flush=True)

    short_sp = SamplingParams(temperature=CONFIG["TEMPERATURE"], max_tokens=CONFIG["BRIEF_MAX_TOKENS"], seed=CONFIG["SEED"])
    det_sp   = SamplingParams(temperature=CONFIG["TEMPERATURE"], max_tokens=CONFIG["DETAILED_MAX_TOKENS"], seed=CONFIG["SEED"])

    def render(prompt_text):
        msgs = [{"role": "system", "content": CONFIG["SYSTEM_PROMPT"]},
                {"role": "user", "content": prompt_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    n_done = len(done)
    for chunk in chunked(todo, CONFIG["CHUNK_SIZE"]):
        short_prompts = [render(BRIEF_TMPL.format(book=b)) for b in chunk]
        det_prompts   = [render(DETAILED_TMPL.format(book=b)) for b in chunk]

        short_out = llm.generate(short_prompts, short_sp)
        det_out   = llm.generate(det_prompts, det_sp)

        records = []
        for b, bo, do in zip(chunk, short_out, det_out):
            short = bo.outputs[0].text.strip()
            long = do.outputs[0].text.strip()
            if not short or not long:
                raise RuntimeError(f"Empty generation for {b!r} (short={len(short)}, long={len(long)})")
            records.append({"book": b, "short": short, "long": long})

        append_checkpoint(ckpt, records)
        n_done += len(records)
        print(f"[gen] {n_done}/{len(books)} books done (shard)", flush=True)

        if _STOP:
            print("[gen] stopping early due to preemption signal; progress checkpointed.", flush=True)
            sys.exit(0)


# --------------------------------------------------------------------------- #
# Stage 2: VERIFY                                                             #
# --------------------------------------------------------------------------- #
def stage_verify(books):
    done = load_checkpoint_union()
    missing = [b for b in books if b not in done]
    if missing:
        raise RuntimeError(f"VERIFY: {len(missing)} books missing from checkpoint, e.g. {missing[:5]}")

    thr = CONFIG["LENGTH_RATIO_THRESHOLD"]
    rows, failures = [], []
    for b in books:
        bw = len(done[b]["short"].split())
        dw = len(done[b]["long"].split())
        ratio = dw / max(bw, 1)
        ok = (dw > bw) and (ratio >= thr)
        rows.append({"book": b, "short_words": bw, "long_words": dw, "ratio": round(ratio, 2), "ok": ok})
        if not ok:
            failures.append(b)

    short_avg = sum(r["short_words"] for r in rows) / len(rows)
    det_avg   = sum(r["long_words"] for r in rows) / len(rows)
    report = {
        "n_books": len(rows),
        "threshold_ratio": thr,
        "avg_short_words": round(short_avg, 1),
        "avg_long_words": round(det_avg, 1),
        "avg_ratio": round(det_avg / max(short_avg, 1), 2),
        "n_failed": len(failures),
        "failed_books": failures,
        "per_book": rows,
    }
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    out = Path(CONFIG["OUTPUT_DIR"]) / "length_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"[verify] avg short={short_avg:.1f}w  long={det_avg:.1f}w  "
          f"ratio={det_avg / max(short_avg, 1):.2f}  failures={len(failures)}", flush=True)
    print(f"[verify] report -> {out}", flush=True)
    if failures:
        print(f"[verify] WARNING: {len(failures)} books below {thr}x: {failures[:10]}"
              f"{' ...' if len(failures) > 10 else ''}", flush=True)
        if CONFIG["STRICT_LENGTH"]:
            raise RuntimeError(f"STRICT_LENGTH set and {len(failures)} books failed the length check.")


# --------------------------------------------------------------------------- #
# Stage 3: BUILD                                                              #
# --------------------------------------------------------------------------- #
def stage_build(books):
    done = load_checkpoint_union()
    missing = [b for b in books if b not in done]
    if missing:
        raise RuntimeError(f"BUILD: {len(missing)} books missing from checkpoint, e.g. {missing[:5]}")

    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "sentence-single-desired-all.jsonl",
        "paragraph-single-desired-all.jsonl",
        "sentence-single-undesired-all.jsonl",
        "paragraph-single-undesired-all.jsonl",
        "sentence-long-desired-all.jsonl",
        "paragraph-long-desired-all.jsonl",
        "sentence-long-undesired-all.jsonl",
        "paragraph-long-undesired-all.jsonl",
        "sentence-single-steering.jsonl",
        "paragraph-single-steering.jsonl",
    ]
    handles = {n: (out_dir / n).open("w") for n in names}

    def emit(name, rec):
        handles[name].write(json.dumps(rec, ensure_ascii=False) + "\n")

    def single_option_block(correct_text, correct_idx):
        """Only the correct option at its lettered position; the other 3 lines blank."""
        return "\n".join(
            f"({LETTERS[i]}) {correct_text}" if i == correct_idx else ""
            for i in range(4)
        )

    rng = random.Random(CONFIG["SEED"])  # deterministic: permutations + distracter choice

    for idx, book in enumerate(books):
        short_response = done[book]["short"]
        long_response = done[book]["long"]

        # One distracter book (different from current); reuse for both distracters.
        distracter_book = rng.choice([b for b in books if b != book])
        short_distracter = done[distracter_book]["short"]
        long_distracter = done[distracter_book]["long"]

        # Assign the 4 items to 4 distinct option positions (A/B/C/D), shared
        # across the 4 "single" files so desired/undesired pairs stay aligned.
        positions = [0, 1, 2, 3]
        rng.shuffle(positions)
        short_idx, long_idx, short_distracter_idx, long_distracter_idx = positions

        options = [None, None, None, None]
        options[short_idx] = short_response
        options[long_idx] = long_response
        options[short_distracter_idx] = short_distracter
        options[long_distracter_idx] = long_distracter
        a, b_, c, d = options
        opt_block = f"(A) {a}\n(B) {b_}\n(C) {c}\n(D) {d}"

        # ---- File 1: sentence-single-desired (ask short -> answer short) ----
        emit("sentence-single-desired-all.jsonl", {
            "id": idx,
            "prompt": [
                {"role": "user",
                 "content": f"Please identify the correct short summary of the book {book}\n{opt_block}\nAnswer: ("},
                {"role": "assistant", "content": LETTERS[short_idx]},
            ],
        })

        # ---- File 2: paragraph-single-desired (ask long -> answer long) ----
        emit("paragraph-single-desired-all.jsonl", {
            "id": idx,
            "prompt": [
                {"role": "user",
                 "content": f"Please identify the correct long summary of the book {book}\n{opt_block}\nAnswer: ("},
                {"role": "assistant", "content": LETTERS[long_idx]},
            ],
        })

        # ---- File 3: sentence-single-undesired (same short prompt as File 1, answer = short distracter) ----
        emit("sentence-single-undesired-all.jsonl", {
            "id": idx,
            "prompt": [
                {"role": "user",
                 "content": f"Please identify the correct short summary of the book {book}\n{opt_block}\nAnswer: ("},
                {"role": "assistant", "content": LETTERS[short_distracter_idx]},
            ],
        })

        # ---- File 4: paragraph-single-undesired (same long prompt as File 2, answer = long distracter) ----
        emit("paragraph-single-undesired-all.jsonl", {
            "id": idx,
            "prompt": [
                {"role": "user",
                 "content": f"Please identify the correct long summary of the book {book}\n{opt_block}\nAnswer: ("},
                {"role": "assistant", "content": LETTERS[long_distracter_idx]},
            ],
        })

        # ---- Files 5-8: free-form (short/long prompt x desired/undesired length) ----
        short_q = BRIEF_TMPL.format(book=book)
        long_q = DETAILED_TMPL.format(book=book)

        emit("sentence-long-desired-all.jsonl", {
            "id": idx,
            "prompt": [{"role": "user", "content": short_q},
                       {"role": "assistant", "content": short_response}],
        })
        emit("paragraph-long-desired-all.jsonl", {
            "id": idx,
            "prompt": [{"role": "user", "content": long_q},
                       {"role": "assistant", "content": long_response}],
        })
        emit("sentence-long-undesired-all.jsonl", {
            "id": idx,
            "prompt": [{"role": "user", "content": short_q},
                       {"role": "assistant", "content": long_response}],
        })
        emit("paragraph-long-undesired-all.jsonl", {
            "id": idx,
            "prompt": [{"role": "user", "content": long_q},
                       {"role": "assistant", "content": short_response}],
        })

        # ---- Steering files: only the correct option is shown, at its position ----
        # NOTE: both prompts say "short"; sentence uses short_response, paragraph uses long_response.
        emit("sentence-single-steering.jsonl", {
            "id": idx,
            "prompt": [
                {"role": "user",
                 "content": f"Please identify the correct short summary of the book {book}\n"
                            f"{single_option_block(short_response, short_idx)}\nAnswer: ("},
                {"role": "assistant", "content": LETTERS[short_idx]},
            ],
        })
        emit("paragraph-single-steering.jsonl", {
            "id": idx,
            "prompt": [
                {"role": "user",
                 "content": f"Please identify the correct short summary of the book {book}\n"
                            f"{single_option_block(long_response, long_idx)}\nAnswer: ("},
                {"role": "assistant", "content": LETTERS[long_idx]},
            ],
        })

    for h in handles.values():
        h.close()
    print(f"[build] wrote {len(names)} files ({len(books)} lines each) -> {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #
def main():
    Path(CONFIG["CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in CONFIG["STAGES"].split(",") if s.strip()]

    if "generate" in stages:
        gen_books = load_books(gen_books_path())
        print(f"[gen] shard={CONFIG['SHARD'] or 'all'} -> {gen_checkpoint_path()} "
              f"({len(gen_books)} books)", flush=True)
        stage_generate(gen_books, gen_checkpoint_path())

    if "verify" in stages or "build" in stages:
        full_books = load_books(CONFIG["BOOKS_JSON"])
        if "verify" in stages:
            stage_verify(full_books)
        if "build" in stages:
            stage_build(full_books)

    print("[done] stages complete:", ",".join(stages), flush=True)


if __name__ == "__main__":
    main()
