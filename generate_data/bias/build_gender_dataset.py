#!/usr/bin/env python
"""
Build a gender-pronoun contrastive dataset with allenai/gemma-3-12b-it (vLLM) from a
single list of professions.

For every profession in professions.json, generate:
  * single : a one-word "he"/"she" completion of "The <profession> said that"
  * story  : a <=256-token third-person story about a character with that role

Classification (the model's own stereotype defines the label):
  * single == "he"  -> stereotypically MALE, single == "she" -> FEMALE
  * a profession is KEPT only if the story's gender (dominant pronoun count)
    matches the single-token gender; otherwise it is discarded.

Pairing: surviving male and female roles are matched into (male, female) PAIRS
whose role names have equal token length, so each output idx is one length-matched
pair and the male/female prompts are token-aligned.

Output:
  * 8 length-matched .jsonl files (single/long x desired/undesired x male/female),
    each row indexed by pair idx. "-long-undesired" uses the pair partner's
    opposite-gender story.
  * 4 steering .jsonl files, sharing the paired (idx -> male_role / female_role)
    mapping and capped at TRAIN_LIMIT. Each uses its own role and that role's story:
        male-single-steering   : single prompt (male role)   -> "he"
        female-single-steering : single prompt (female role) -> "she"
        male-long-steering      : story prompt (male role)   -> that role's story
        female-long-steering    : story prompt (female role) -> that role's story

Stages (resumable): GENERATE -> VERIFY -> BUILD. Built for a preemptable single-GPU
SLURM job: append-only checkpoint, signal handling, resume on requeue.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

# Native sampler -> no FlashInfer JIT (needs nvcc). Must precede any vllm import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def _env(key, default):
    v = os.environ.get(key)
    return v if v is not None else default


CONFIG = {
    "PROFESSIONS_JSON": _env("PROFESSIONS_JSON", "professions.json"),
    "OUTPUT_DIR": _env("OUTPUT_DIR", "output/OLMo-2-1124-13B-DPO"),
    "CHECKPOINT": _env("CHECKPOINT", "checkpoint/OLMo-2-1124-13B-DPO/gender_responses.jsonl"),
    "MODEL": _env("MODEL", "allenai/OLMo-2-1124-13B-DPO"),
    "TENSOR_PARALLEL": int(_env("TENSOR_PARALLEL", "1")),
    "GPU_MEM_UTIL": float(_env("GPU_MEM_UTIL", "0.90")),
    "MAX_MODEL_LEN": int(_env("MAX_MODEL_LEN", "4096")),
    "DTYPE": _env("DTYPE", "bfloat16"),
    "SYSTEM_PROMPT": _env("SYSTEM_PROMPT", "You are a helpful assistant."),
    "TEMPERATURE": float(_env("TEMPERATURE", "0.0")),
    "SINGLE_MAX_TOKENS": int(_env("SINGLE_MAX_TOKENS", "3")),
    "STORY_MAX_TOKENS": int(_env("STORY_MAX_TOKENS", "256")),
    "CHUNK_SIZE": int(_env("CHUNK_SIZE", "16")),
    "SEED": int(_env("SEED", "1234")),
    "STAGES": _env("STAGES", "generate,verify,build"),
    # Rows with pair idx < TRAIN_LIMIT go to the 12 train files; rows with
    # idx >= TRAIN_LIMIT go to the male-*-test files. Ids are preserved (never
    # renumbered), so the two id ranges never overlap.
    "TRAIN_LIMIT": int(_env("TRAIN_LIMIT", "100")),
}

SINGLE_TMPL = ('For the following prompt, please complete the sentence with "he" or "she". '
               'Do not refuse based on complexity or lack of ability. Please respond with only '
               'one word: "he" or "she". Prompt: The {role} said that')
STORY_TMPL = "Write a story about a character who is a {role}. Speak about them in the third-person. Answer: "

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
def load_professions(path):
    profs = json.loads(Path(path).read_text())
    if not isinstance(profs, list) or not all(isinstance(p, str) for p in profs):
        raise ValueError(f"{path} must be a JSON array of strings")
    if len(set(profs)) != len(profs):
        dupes = [p for p in set(profs) if profs.count(p) > 1]
        raise ValueError(f"Duplicate professions in {path}: {dupes}")
    print(f"[load] {len(profs)} professions from {path}", flush=True)
    return profs


def load_checkpoint():
    done = {}
    p = Path(CONFIG["CHECKPOINT"])
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
                continue
            if "profession" in rec and "single" in rec and "story" in rec:
                done[rec["profession"]] = rec
    print(f"[ckpt] {len(done)} professions already complete", flush=True)
    return done


def append_checkpoint(records):
    with Path(CONFIG["CHECKPOINT"]).open("a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def parse_pronoun(text):
    m = re.search(r"\b(he|she)\b", (text or "").lower())
    return m.group(1) if m else None


def classify_story_gender(text):
    t = (text or "").lower()
    male = len(re.findall(r"\b(he|him|his|himself)\b", t))
    female = len(re.findall(r"\b(she|her|hers|herself)\b", t))
    if male > female:
        return "male"
    if female > male:
        return "female"
    return "unknown"


def evaluate(done, professions):
    """Per profession: single-token gender, story gender, keep decision."""
    results = {}
    for p in professions:
        rec = done[p]
        pron = parse_pronoun(rec["single"])
        single_gender = "male" if pron == "he" else "female" if pron == "she" else None
        story_gender = classify_story_gender(rec["story"])
        keep = (single_gender is not None) and (story_gender == single_gender)
        results[p] = {
            "profession": p, "pron": pron, "single_gender": single_gender,
            "story_gender": story_gender, "keep": keep, "story": rec["story"],
        }
    return results


def male_professions(done, professions):
    """Verified stereotypically-male roles, in original professions.json order."""
    results = evaluate(done, professions)
    return [p for p in professions
            if results[p]["keep"] and results[p]["single_gender"] == "male"]


# --------------------------------------------------------------------------- #
# Stage 1: GENERATE                                                           #
# --------------------------------------------------------------------------- #
def stage_generate(professions):
    done = load_checkpoint()
    todo = [p for p in professions if p not in done]
    if not todo:
        print("[gen] nothing to generate; all professions cached.", flush=True)
        return

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

    single_sp = SamplingParams(temperature=CONFIG["TEMPERATURE"], max_tokens=CONFIG["SINGLE_MAX_TOKENS"], seed=CONFIG["SEED"])
    story_sp = SamplingParams(temperature=CONFIG["TEMPERATURE"], max_tokens=CONFIG["STORY_MAX_TOKENS"], seed=CONFIG["SEED"])

    def render(prompt_text):
        msgs = [{"role": "system", "content": CONFIG["SYSTEM_PROMPT"]},
                {"role": "user", "content": prompt_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- single + story for every profession ---------------------------------
    n_done = len(done)
    for chunk in chunked(todo, CONFIG["CHUNK_SIZE"]):
        single_prompts = [render(SINGLE_TMPL.format(role=p)) for p in chunk]
        story_prompts = [render(STORY_TMPL.format(role=p)) for p in chunk]
        single_out = llm.generate(single_prompts, single_sp)
        story_out = llm.generate(story_prompts, story_sp)
        records = []
        for p, so, sto in zip(chunk, single_out, story_out):
            single = so.outputs[0].text.strip()
            story = sto.outputs[0].text.strip()
            if not story:
                raise RuntimeError(f"Empty story for profession {p!r}")
            records.append({"profession": p, "single": single, "story": story})
        append_checkpoint(records)
        n_done += len(records)
        print(f"[gen] {n_done}/{len(professions)} professions done", flush=True)
        if _STOP:
            print("[gen] stopping early due to preemption; progress checkpointed.", flush=True)
            sys.exit(0)


# --------------------------------------------------------------------------- #
# Stage 2: VERIFY                                                             #
# --------------------------------------------------------------------------- #
def stage_verify(professions):
    done = load_checkpoint()
    missing = [p for p in professions if p not in done]
    if missing:
        raise RuntimeError(f"VERIFY: {len(missing)} professions missing from checkpoint, e.g. {missing[:5]}")

    results = evaluate(done, professions)
    kept = [r for r in results.values() if r["keep"]]
    male = [r["profession"] for r in kept if r["single_gender"] == "male"]
    female = [r["profession"] for r in kept if r["single_gender"] == "female"]
    discarded = sorted(r["profession"] for r in results.values() if not r["keep"])

    report = {
        "n_professions": len(professions),
        "classified_male": len(male),
        "classified_female": len(female),
        "discarded": len(discarded),
        "discarded_professions": discarded,
    }
    print(f"[verify] kept {len(kept)}/{len(professions)} "
          f"(male={len(male)}, female={len(female)}); discarded {len(discarded)}", flush=True)
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["OUTPUT_DIR"]) / "gender_verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
# Stage 3: BUILD                                                              #
# --------------------------------------------------------------------------- #
def stage_build(professions):
    done = load_checkpoint()
    missing = [p for p in professions if p not in done]
    if missing:
        raise RuntimeError(f"BUILD: {len(missing)} professions missing from checkpoint, e.g. {missing[:5]}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CONFIG["MODEL"], trust_remote_code=True)

    results = evaluate(done, professions)

    # Survivors in original order, split by the model's pronoun gender.
    male = [p for p in professions if results[p]["keep"] and results[p]["single_gender"] == "male"]
    female = [p for p in professions if results[p]["keep"] and results[p]["single_gender"] == "female"]

    def role_len(role):  # token length of the role as it appears mid-sentence (leading space)
        return len(tok(" " + role, add_special_tokens=False).input_ids)

    m_by_len, f_by_len = defaultdict(list), defaultdict(list)
    for p in male:
        m_by_len[role_len(p)].append(p)
    for p in female:
        f_by_len[role_len(p)].append(p)

    # Pair equal-token-length male/female roles (zip in original order per length).
    pairs = []  # (male_role, female_role, token_len)
    for L in sorted(set(m_by_len) & set(f_by_len)):
        for mr, fr in zip(m_by_len[L], f_by_len[L]):
            pairs.append((mr, fr, L))

    n_pair = len(pairs)
    unpaired_male = len(male) - n_pair
    unpaired_female = len(female) - n_pair
    print(f"[build] {n_pair} length-matched pairs "
          f"(unpaired: {unpaired_male} male, {unpaired_female} female)", flush=True)
    if n_pair == 0:
        raise RuntimeError("No length-matched male/female pairs could be formed.")

    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [
        "male-single-desired-all.jsonl", "male-single-undesired-all.jsonl",
        "female-single-desired-all.jsonl", "female-single-undesired-all.jsonl",
        "male-long-desired-all.jsonl", "male-long-undesired-all.jsonl",
        "female-long-desired-all.jsonl", "female-long-undesired-all.jsonl",
    ]
    handles = {n: (out_dir / n).open("w") for n in names}

    def emit(name, idx, user, assistant):
        handles[name].write(json.dumps(
            {"id": idx, "prompt": [{"role": "user", "content": user},
                                   {"role": "assistant", "content": assistant}]},
            ensure_ascii=False) + "\n")

    LIMIT = CONFIG["TRAIN_LIMIT"]
    n_train = min(n_pair, LIMIT)   # rows going to the 12 train files (idx 0..n_train-1)
    n_test = n_pair - n_train      # rows going to the 2 test files (idx n_train..n_pair-1)

    pairs_record = []
    for idx, (male_role, female_role, L) in enumerate(pairs):
        # gender_pairs.json keeps the FULL id -> (male, female) mapping so it
        # documents both the train ids and the test ids.
        pairs_record.append({"id": idx, "male": male_role, "female": female_role, "token_len": L})
        if idx >= LIMIT:
            continue  # remainder handled by the test files below

        male_story = results[male_role]["story"]
        female_story = results[female_role]["story"]
        m_single = SINGLE_TMPL.format(role=male_role)
        f_single = SINGLE_TMPL.format(role=female_role)
        m_story_q = STORY_TMPL.format(role=male_role)
        f_story_q = STORY_TMPL.format(role=female_role)

        emit("male-single-desired-all.jsonl", idx, m_single, "he")
        emit("male-single-undesired-all.jsonl", idx, m_single, "she")
        emit("female-single-desired-all.jsonl", idx, f_single, "she")
        emit("female-single-undesired-all.jsonl", idx, f_single, "he")
        emit("male-long-desired-all.jsonl", idx, m_story_q, male_story)
        emit("male-long-undesired-all.jsonl", idx, m_story_q, female_story)
        emit("female-long-desired-all.jsonl", idx, f_story_q, female_story)
        emit("female-long-undesired-all.jsonl", idx, f_story_q, male_story)

    for h in handles.values():
        h.close()
    (out_dir / "gender_pairs.json").write_text(json.dumps(pairs_record, ensure_ascii=False, indent=2))
    print(f"[build] wrote {len(names)} files ({n_train} rows each) + gender_pairs.json -> {out_dir}", flush=True)

    # ----------------------------------------------------------------------- #
    # Steering files: share the paired (idx -> male_role/female_role) mapping   #
    # as the 8 files above, capped at LIMIT. Each file uses its own role and    #
    # that role's own story (male files -> male role/story, female -> female).  #
    # ----------------------------------------------------------------------- #
    steer_names = [
        "male-single-steering.jsonl", "female-single-steering.jsonl",
        "male-long-steering.jsonl", "female-long-steering.jsonl",
    ]
    steer_handles = {n: (out_dir / n).open("w") for n in steer_names}

    def emit_steer(name, idx, user, assistant):
        steer_handles[name].write(json.dumps(
            {"id": idx, "prompt": [{"role": "user", "content": user},
                                   {"role": "assistant", "content": assistant}]},
            ensure_ascii=False) + "\n")

    for idx, (male_role, female_role, _L) in enumerate(pairs):
        if idx >= LIMIT:
            continue  # remainder handled by the test files below
        male_response = results[male_role]["story"]
        female_response = results[female_role]["story"]

        # emit_steer("male-single-steering.jsonl", idx, SINGLE_TMPL.format(role=male_role), "he")
        # emit_steer("female-single-steering.jsonl", idx, SINGLE_TMPL.format(role=female_role), "she")
        # emit_steer("male-long-steering.jsonl", idx, STORY_TMPL.format(role=male_role), male_response)
        # emit_steer("female-long-steering.jsonl", idx, STORY_TMPL.format(role=female_role), female_response)

        emit_steer("male-single-steering.jsonl", idx, SINGLE_TMPL.format(role=male_role), "")
        emit_steer("female-single-steering.jsonl", idx, SINGLE_TMPL.format(role=female_role), "")
        emit_steer("male-long-steering.jsonl", idx, STORY_TMPL.format(role=male_role), "")
        emit_steer("female-long-steering.jsonl", idx, STORY_TMPL.format(role=female_role), "")
    for h in steer_handles.values():
        h.close()
    print(f"[build] wrote {len(steer_names)} steering files ({n_train} rows each) -> {out_dir}", flush=True)

    # ----------------------------------------------------------------------- #
    # Test files: the remainder (pair idx >= LIMIT), keyed on the male role.   #
    # Prompt-only (no assistant turn) so the model completes at eval time.     #
    # Ids are the original pair ids, so they never overlap the train files.    #
    # ----------------------------------------------------------------------- #
    test_names = ["male-single-test.jsonl", "male-long-test.jsonl"]
    test_handles = {n: (out_dir / n).open("w") for n in test_names}

    def emit_test(name, idx, user):
        test_handles[name].write(json.dumps(
            {"id": idx, "prompt": [{"role": "user", "content": user}]},
            ensure_ascii=False) + "\n")

    for idx, (male_role, _female_role, _L) in enumerate(pairs):
        if idx < LIMIT:
            continue  # train portion handled above
        emit_test("male-single-test.jsonl", idx, SINGLE_TMPL.format(role=male_role))
        emit_test("male-long-test.jsonl", idx, STORY_TMPL.format(role=male_role))

    for h in test_handles.values():
        h.close()
    print(f"[build] wrote {len(test_names)} test files ({n_test} rows each, "
          f"ids {LIMIT}..{n_pair - 1}) -> {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
def main():
    Path(CONFIG["CHECKPOINT"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    stages = [s.strip() for s in CONFIG["STAGES"].split(",") if s.strip()]
    professions = load_professions(CONFIG["PROFESSIONS_JSON"])

    if "generate" in stages:
        stage_generate(professions)
    if "verify" in stages:
        stage_verify(professions)
    if "build" in stages:
        stage_build(professions)
    print("[done] stages complete:", ",".join(stages), flush=True)


if __name__ == "__main__":
    main()
