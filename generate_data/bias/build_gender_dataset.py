#!/usr/bin/env python
"""
Build a gender-pronoun contrastive dataset with Qwen1.5-14B-Chat (vLLM) from a
single list of professions.

For every profession in professions.json, generate:
  * single : a one-word "he"/"she" completion of "The <profession> said that"
  * story  : a <=256-token third-person story about a character with that role

Classification (the model's own stereotype defines the label):
  * single == "he"  -> stereotypically MALE, single == "she" -> FEMALE
  * a profession is KEPT only if the story's gender (dominant pronoun count)
    matches the single-token gender; otherwise it is discarded.

Second generation pass (woman-in-a-male-role):
  * AFTER the male roles are known, ask Qwen for a third-person story about a
    WOMAN who holds each stereotypically-male profession. This counterfactual
    story is checkpointed separately and used for the *-long-steering files.

Pairing: surviving male and female roles are matched into (male, female) PAIRS
whose role names have equal token length, so each output idx is one length-matched
pair and the male/female prompts are token-aligned.

Output:
  * 8 length-matched .jsonl files (single/long x desired/undesired x male/female),
    each row indexed by pair idx. "-long-undesired" uses the pair partner's
    opposite-gender story.
  * 4 steering .jsonl files, each row indexed over ALL verified-male roles and
    keyed on the male role. The prompt is held fixed (the stereotypically-male
    profession); only the completion's gender changes:
        male-single-steering   : single prompt (male role) -> "he"
        female-single-steering : single prompt (male role) -> "she"
        male-long-steering      : story prompt (male role) -> original male-character story
        female-long-steering    : story prompt (male role) -> woman-character story

Stages (resumable): GENERATE -> VERIFY -> BUILD. Built for a preemptable single-GPU
SLURM job: append-only checkpoint, signal handling, resume on requeue. The woman
story generation is a second pass within GENERATE with its own checkpoint.
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
    "OUTPUT_DIR": _env("OUTPUT_DIR", "output"),
    "CHECKPOINT": _env("CHECKPOINT", "checkpoint/gender_responses.jsonl"),
    "WOMAN_CHECKPOINT": _env("WOMAN_CHECKPOINT", "checkpoint/gender_woman_responses.jsonl"),
    "MODEL": _env("MODEL", "Qwen/Qwen1.5-14B-Chat"),
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
# Generation-only prompt used to ELICIT the counterfactual woman-in-a-male-role
# story. This is intentionally NOT the prompt stored in the output files: the
# steering files store the neutral STORY_TMPL so that the male and female long
# examples share an identical prompt and differ only in the completion's gender.
WOMAN_STORY_TMPL = "Write a story about a woman who is a {role}. Speak about her in the third-person. Answer: "

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


def load_woman_checkpoint():
    done = {}
    p = Path(CONFIG["WOMAN_CHECKPOINT"])
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
            if "profession" in rec and "woman_story" in rec:
                done[rec["profession"]] = rec
    print(f"[ckpt] {len(done)} woman-stories already complete", flush=True)
    return done


def append_woman_checkpoint(records):
    with Path(CONFIG["WOMAN_CHECKPOINT"]).open("a") as f:
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
    woman_done = load_woman_checkpoint()
    todo = [p for p in professions if p not in done]

    # If pass 1 is fully cached we can already work out which woman-stories are
    # outstanding; otherwise we compute this after pass 1 completes below.
    woman_todo = None
    if not todo:
        male = male_professions(done, professions)
        woman_todo = [p for p in male if p not in woman_done]

    if not todo and not woman_todo:
        print("[gen] nothing to generate; all professions + woman-stories cached.", flush=True)
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

    # ---- Pass 1: single + story for every profession -------------------------
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

    # Now that pass 1 is complete, determine which male roles still need a
    # woman-story (deferred from above when pass 1 had outstanding work).
    if woman_todo is None:
        done = load_checkpoint()
        male = male_professions(done, professions)
        woman_todo = [p for p in male if p not in woman_done]

    # ---- Pass 2: woman-in-a-male-role counterfactual story -------------------
    if woman_todo:
        print(f"[gen] generating woman-stories for {len(woman_todo)} stereotypically-male roles ...", flush=True)
        n_wdone = len(woman_done)
        for chunk in chunked(woman_todo, CONFIG["CHUNK_SIZE"]):
            woman_prompts = [render(WOMAN_STORY_TMPL.format(role=p)) for p in chunk]
            woman_out = llm.generate(woman_prompts, story_sp)
            wrecords = []
            for p, wo in zip(chunk, woman_out):
                woman_story = wo.outputs[0].text.strip()
                if not woman_story:
                    raise RuntimeError(f"Empty woman-story for profession {p!r}")
                wrecords.append({"profession": p, "woman_story": woman_story})
            append_woman_checkpoint(wrecords)
            n_wdone += len(wrecords)
            print(f"[gen] {n_wdone} woman-stories done", flush=True)
            if _STOP:
                print("[gen] stopping early due to preemption; progress checkpointed.", flush=True)
                sys.exit(0)
    else:
        print("[gen] no outstanding woman-stories.", flush=True)


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

    # Report woman-story coverage for the verified-male roles.
    woman_done = load_woman_checkpoint()
    male_missing_woman = [p for p in male if p not in woman_done]

    report = {
        "n_professions": len(professions),
        "classified_male": len(male),
        "classified_female": len(female),
        "discarded": len(discarded),
        "discarded_professions": discarded,
        "woman_stories_present": len(male) - len(male_missing_woman),
        "male_missing_woman_story": male_missing_woman,
    }
    print(f"[verify] kept {len(kept)}/{len(professions)} "
          f"(male={len(male)}, female={len(female)}); discarded {len(discarded)}", flush=True)
    if male_missing_woman:
        print(f"[verify] WARNING: {len(male_missing_woman)} male roles still lack a woman-story "
              f"(re-run the generate stage), e.g. {male_missing_woman[:5]}", flush=True)
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
    # Steering files: keyed on the SAME (id -> male_role) mapping as the 8     #
    # paired files above. Each row is one length-matched pair's male role, at  #
    # the identical pair idx; male roles that were not paired do not appear.   #
    # Prompt held fixed (the stereotypically-male profession); only the        #
    # completion's gender changes. The long files contrast the original        #
    # male-character story against the woman-in-a-male-role story.             #
    # ----------------------------------------------------------------------- #
    # Only the train portion (idx < LIMIT) uses woman-stories, so only require
    # those to be present.
    steer_male_roles = [mr for i, (mr, _f, _L) in enumerate(pairs) if i < LIMIT]
    woman_done = load_woman_checkpoint()
    male_missing_woman = [p for p in steer_male_roles if p not in woman_done]
    if male_missing_woman:
        raise RuntimeError(
            f"BUILD: {len(male_missing_woman)} paired male roles missing woman-stories "
            f"(re-run the generate stage), e.g. {male_missing_woman[:5]}")

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

    for idx, (male_role, _female_role, _L) in enumerate(pairs):
        if idx >= LIMIT:
            continue  # remainder handled by the test files below
        original_male_response = results[male_role]["story"]          # male-character story
        new_male_response = woman_done[male_role]["woman_story"]      # woman-in-a-male-role story
        single_q = SINGLE_TMPL.format(role=male_role)
        story_q = STORY_TMPL.format(role=male_role)

        emit_steer("male-single-steering.jsonl", idx, single_q, "he")
        emit_steer("female-single-steering.jsonl", idx, single_q, "she")
        emit_steer("male-long-steering.jsonl", idx, story_q, original_male_response)
        emit_steer("female-long-steering.jsonl", idx, story_q, new_male_response)

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
    Path(CONFIG["WOMAN_CHECKPOINT"]).parent.mkdir(parents=True, exist_ok=True)
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
