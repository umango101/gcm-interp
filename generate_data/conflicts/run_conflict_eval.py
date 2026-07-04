#!/usr/bin/env python
"""
run_conflict_eval.py

Sends the pre-rendered Harmony `instruction_1` / `instruction_2` prompts from
a system/user instruction-conflict JSONL to gpt-oss-20b via vLLM, and
checkpoints results so the job survives preemption on Engaging's preemptable
partition.

Design (matches the resilient architecture used elsewhere in gcm-interp):
  - Append-only JSONL checkpoint (results/responses.jsonl). Each line is a
    complete pair result. On restart, already-completed pair_ids are skipped.
  - Chunk-based processing: pairs are generated in small batches; after each
    chunk is flushed to disk, we check whether a stop signal arrived.
  - SIGTERM (sent by SLURM on preemption/eviction) and SIGUSR1 (sent early,
    via --signal=B:USR1@120 in the sbatch script) both set a stop flag rather
    than killing the process mid-write, so the checkpoint is never torn.
  - fsync after every write so a hard kill -9 can't lose a flushed line.

NOTE: this script assumes vLLM's offline LLM.generate() accepts
`prompt_token_ids` directly and that gpt-oss-20b's Harmony completions come
back with <|channel|>final<|message|>...<|end|> markers in the raw text. If
your existing run_precheck.py / gen_conflicts.py pipeline already has a
canonical model-loading or channel-parsing utility for gpt-oss-20b, swap that
in here (build_llm / extract_final_channel) instead, for consistency across
the repo.

Usage:
    python run_conflict_eval.py \
        --input pairs.jsonl \
        --output_dir results/ \
        --model openai/gpt-oss-20b \
        --chunk_size 16 \
        --max_new_tokens 256
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Preemption / requeue signal handling
# ---------------------------------------------------------------------------
_STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    global _STOP_REQUESTED
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    print(f"[signal] received {name}, will stop after current chunk is checkpointed", flush=True)
    _STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _handle_stop_signal)
signal.signal(signal.SIGUSR1, _handle_stop_signal)

# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def load_done_ids(checkpoint_path: Path) -> set:
    done = set()
    if not checkpoint_path.exists():
        return done
    with open(checkpoint_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["pair_id"])
            except (json.JSONDecodeError, KeyError):
                # Tolerate a torn last line from a mid-write kill; that pair
                # will simply be regenerated.
                continue
    return done


def append_result(checkpoint_path: Path, result: dict):
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Harmony channel parsing
# ---------------------------------------------------------------------------

FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
    re.DOTALL,
)


def extract_final_channel(generated_text: str) -> str:
    """Pull the 'final' channel content out of a Harmony completion.
    Falls back to the raw text if no channel markers are present."""
    match = FINAL_CHANNEL_RE.search(generated_text)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_llm(model_path: str, tp_size: int = 1, max_model_len: int = 4096):
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )


def generate_chunk(llm, sampling_params, chunk_records):
    from vllm import TokensPrompt

    prompts = []
    meta = []  # (pair_id, which_instruction)
    for rec in chunk_records:
        for key in ("instruction_1", "instruction_2"):
            prompts.append(TokensPrompt(prompt_token_ids=rec[key]["token_ids"]))
            meta.append((rec["pair_id"], key))

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    responses = {}
    for (pair_id, key), out in zip(meta, outputs):
        text = out.outputs[0].text
        responses.setdefault(pair_id, {})[key] = {
            "raw_text": text,
            "final_text": extract_final_channel(text),
        }
    return responses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to the conflict-pair JSONL")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--chunk_size", type=int, default=16, help="pairs per generation batch")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--tp_size", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "responses.jsonl"

    with open(args.input, "r") as f:
        records = [json.loads(l) for l in f if l.strip()]

    done_ids = load_done_ids(checkpoint_path)
    remaining = [r for r in records if r["pair_id"] not in done_ids]
    print(f"[main] {len(done_ids)} pairs already checkpointed, {len(remaining)} remaining", flush=True)

    if not remaining:
        print("[main] nothing to do, exiting cleanly")
        return

    from vllm import SamplingParams

    llm = build_llm(args.model, tp_size=args.tp_size)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["<|return|>", "<|end|>"],
    )

    n_chunks = (len(remaining) + args.chunk_size - 1) // args.chunk_size
    for chunk_idx in range(n_chunks):
        chunk = remaining[chunk_idx * args.chunk_size : (chunk_idx + 1) * args.chunk_size]
        t0 = time.time()
        responses = generate_chunk(llm, sampling_params, chunk)

        for rec in chunk:
            pid = rec["pair_id"]
            result = {
                "pair_id": pid,
                "category": rec.get("category"),
                "query": rec.get("query"),
                "conflict_1": rec["instruction_1"]["conflict"],
                "conflict_2": rec["instruction_2"]["conflict"],
                "response_1": responses[pid]["instruction_1"],
                "response_2": responses[pid]["instruction_2"],
            }
            append_result(checkpoint_path, result)

        print(
            f"[main] chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} pairs) "
            f"done in {time.time() - t0:.1f}s",
            flush=True,
        )

        if _STOP_REQUESTED:
            print("[main] stop signal seen, checkpoint is flushed, exiting for clean requeue", flush=True)
            sys.exit(0)

    print("[main] all pairs complete", flush=True)


if __name__ == "__main__":
    main()
