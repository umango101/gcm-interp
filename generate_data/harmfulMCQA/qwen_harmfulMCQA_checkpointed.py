"""
qwen_harmfulMCQA.py

Builds a contrastive harmful/harmless expression dataset on Qwen1.5-14B-Chat for
steering / refusal-direction work. For each index it records, per task:
  answer_matching_behavior      -> the model's plain answer to that task
  answer_not_matching_behavior  -> the contrastive answer (jailbroken / cross-task)

Resumable under SLURM preemption + requeue:
  - Each finished index is appended (flush + fsync) to RESULTS_PATH as one JSON
    line. That file IS the checkpoint.
  - On startup we read RESULTS_PATH and skip indices already present.
  - SIGTERM / SIGUSR1 -> clean stop at the next index boundary.
  - Commit happens only after an index fully completes, so a mid-index kill just
    re-runs that one index (idempotent).
  - The two answered output files are regenerated from the original task lists
    overlaid with RESULTS_PATH at the end, so their format is unchanged.

Note: the jailbreak preamble is loaded at runtime from CONFIG.PREAMBLE_PATH
(plain text). Populate that file with your existing preamble text.
"""

import os
import re  # noqa: F401  (kept for parity / optional downstream parsing)
import sys
import json
import signal
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================
# CONFIG
# =============================================================
class CONFIG:
    SEED = 42
    MODEL_NAME = "Qwen/Qwen1.5-14B-Chat"

    DATA_DIR = "generate_data/harmfulMCQA"
    HARMLESS_IN = f"{DATA_DIR}/expression_harmless_tasks.jsonl"
    HARMFUL_IN = f"{DATA_DIR}/expression_harmful_tasks.jsonl"

    # Jailbreak preamble lives in its own file (data, not code). Populate it with
    # your existing preamble text before running.
    PREAMBLE_PATH = f"{DATA_DIR}/jailbreak_preamble.txt"

    # Durable, append-only source of truth + checkpoint.
    RESULTS_PATH = f"{DATA_DIR}/results.jsonl"
    # Final outputs (regenerated from RESULTS_PATH; same format as before).
    HARMLESS_OUT = f"{DATA_DIR}/expression_harmless_tasks_answered.jsonl"
    HARMFUL_OUT = f"{DATA_DIR}/expression_harmful_tasks_answered.jsonl"

    SYSTEM_PROMPT = "You are a helpful assistant."
    MAX_NEW_TOKENS = 256

    # Process at most this many *new* indices this run (None = all remaining).
    # Set to 1 to reproduce the old single-iteration smoke test.
    LIMIT = None


# =============================================================
# Determinism
# =============================================================
def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# =============================================================
# Preemption handling
# =============================================================
_STOP_REQUESTED = False


def _signal_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] received {signal.Signals(signum).name}; "
          f"will stop at next index boundary.", flush=True)


def install_signal_handlers() -> None:
    # SLURM sends SIGTERM on preemption; SIGUSR1 if you use --signal=B:USR1@<sec>.
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)


# =============================================================
# I/O
# =============================================================
def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_preamble(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preamble file not found: {path}\n"
            f"Create it and paste in the jailbreak preamble text from the "
            f"original script."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_completed(path: str) -> dict[int, dict]:
    """Read prior results. Tolerates a partial trailing line from a hard kill."""
    done: dict[int, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a hard kill; ignore
            done[rec["idx"]] = rec
    return done


def append_result(path: str, record: dict) -> None:
    """Append one durable JSON line (flush + fsync so a requeue sees it)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_outputs(harmless_tasks, harmful_tasks, completed) -> None:
    """Regenerate answered files: original tasks overlaid with completed fields."""
    harmless_out = [dict(t) for t in harmless_tasks]
    harmful_out = [dict(t) for t in harmful_tasks]
    for idx, rec in completed.items():
        if rec.get("status") != "done":
            continue
        if idx < len(harmless_out):
            harmless_out[idx].update(rec["harmless_fields"])
        if idx < len(harmful_out):
            harmful_out[idx].update(rec["harmful_fields"])

    with open(CONFIG.HARMLESS_OUT, "w", encoding="utf-8") as f:
        for d in harmless_out:
            f.write(json.dumps(d) + "\n")
    with open(CONFIG.HARMFUL_OUT, "w", encoding="utf-8") as f:
        for d in harmful_out:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(harmful_out)} harmful / {len(harmless_out)} harmless rows",
          flush=True)


# =============================================================
# Generation
# =============================================================
def build_inputs(system_prompt, user_content, tokenizer, device):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)


def generate(inputs, tokenizer, model) -> str:
    gen_kwargs = dict(
        max_new_tokens=CONFIG.MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def process_index(idx, harmful_line, harmless_line, preamble, tokenizer, model, device):
    """Returns a results record (status 'done' or 'skipped')."""
    harmful_user = harmful_line["task"] + "\nAnswer: "
    harmless_user = harmless_line["task"] + "\nAnswer: "

    inputs_harmful = build_inputs(CONFIG.SYSTEM_PROMPT, harmful_user, tokenizer, device)
    inputs_harmless = build_inputs(CONFIG.SYSTEM_PROMPT, harmless_user, tokenizer, device)
    inputs_jailbreak = build_inputs(preamble, harmful_user, tokenizer, device)

    # Length-match guard between the paired harmful/harmless prompts.
    if inputs_harmful["input_ids"].shape[1] != inputs_harmless["input_ids"].shape[1]:
        print(f"[{idx}] length mismatch; skipping.", flush=True)
        return {"idx": idx, "status": "skipped"}

    output_harmful = generate(inputs_harmful, tokenizer, model)
    output_harmless = generate(inputs_harmless, tokenizer, model)
    output_jailbreak = generate(inputs_jailbreak, tokenizer, model)

    return {
        "idx": idx,
        "status": "done",
        "harmful_fields": {
            "answer_matching_behavior": output_harmful,
            "answer_not_matching_behavior": output_jailbreak,
        },
        "harmless_fields": {
            "answer_matching_behavior": output_harmless,
            "answer_not_matching_behavior": output_harmful,
        },
    }


# =============================================================
# Main
# =============================================================
def main() -> None:
    set_determinism(CONFIG.SEED)
    install_signal_handlers()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {CONFIG.MODEL_NAME} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.MODEL_NAME, torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()

    harmless_tasks = load_jsonl(CONFIG.HARMLESS_IN)
    harmful_tasks = load_jsonl(CONFIG.HARMFUL_IN)
    preamble = load_preamble(CONFIG.PREAMBLE_PATH)
    print(f"harmless: {len(harmless_tasks)}  harmful: {len(harmful_tasks)}", flush=True)

    # Guard against unequal lengths (original indexed harmless by the harmful range).
    n = min(len(harmful_tasks), len(harmless_tasks))
    if len(harmful_tasks) != len(harmless_tasks):
        print(f"WARNING: length mismatch; processing first {n} indices.", flush=True)

    completed = load_completed(CONFIG.RESULTS_PATH)
    print(f"{len(completed)}/{n} indices already recorded; resuming.", flush=True)

    processed = 0
    for idx in range(n):
        if _STOP_REQUESTED:
            print("[main] stopping cleanly before next index (preemption).", flush=True)
            break
        if idx in completed:
            continue
        if CONFIG.LIMIT is not None and processed >= CONFIG.LIMIT:
            print(f"[main] reached LIMIT={CONFIG.LIMIT}; stopping.", flush=True)
            break

        record = process_index(
            idx, harmful_tasks[idx], harmless_tasks[idx], preamble,
            tokenizer, model, device,
        )
        append_result(CONFIG.RESULTS_PATH, record)  # commit point
        completed[idx] = record
        processed += 1
        print(f"[{idx}] {record['status']}", flush=True)

    write_outputs(harmless_tasks, harmful_tasks, completed)

    remaining = n - len(completed)
    if remaining > 0:
        print(f"[main] {remaining} indices remaining; requeue to finish.", flush=True)
    else:
        print("[main] all indices complete.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
