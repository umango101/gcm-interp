#!/usr/bin/env python
"""
Build a gender-pronoun contrastive dataset with allenai/gemma-3-12b-it (vLLM) from a
single list of professions.

The single-token task is framed as a two-option MCQA: the prompt shows the sentence
stem plus lettered options ("A. he" / "B. she"), and the answer is a single letter.
The long task is unchanged (free-form third-person story).

For every profession in professions.json, generate:
  * single_ab : MCQA answer with options ordered (A=he, B=she)
  * single_ba : MCQA answer with options ordered (A=she, B=he)
  * story     : a <=256-token third-person story about a character with that role

Classification (the model's own stereotype defines the label):
  * both orderings are decoded to a pronoun via the option map used in that prompt.
    A profession is only labelled if the two orderings AGREE -- disagreement means the
    answer tracked option position rather than the role, so the label would be noise.
  * agreeing "he" -> stereotypically MALE, agreeing "she" -> FEMALE
  * a profession is KEPT only if the story's gender (dominant pronoun count)
    matches the single-token gender; otherwise it is discarded.

Pairing: surviving male and female roles are matched into (male, female) PAIRS
whose role names have equal token length, so each output idx is one length-matched
pair and the male/female prompts are token-aligned. Option ORDER is assigned per
PAIR (not per role), so both members of a pair carry a byte-identical option block
and the only difference between their prompts remains the role name.

By default the order alternates across pairs (OPTION_ORDER=balanced), so "desired"
is letter A for half the pairs and letter B for the other half. This matters for
steering: with a fixed order, every male-desired row would answer "A" and every
male-undesired "B", and the desired-minus-undesired direction would be an
A-vs-B direction rather than a he-vs-she one.

Output:
  * 8 length-matched .jsonl files (single/long x desired/undesired x male/female),
    each row indexed by pair idx. Single files answer with a LETTER; "-long-undesired"
    uses the pair partner's opposite-gender story.
  * 4 steering .jsonl files, sharing the paired (idx -> male_role / female_role)
    mapping and capped at TRAIN_LIMIT. Each uses its own role and that role's story:
        male-single-steering   : MCQA prompt (male role)   -> letter of "he"
        female-single-steering : MCQA prompt (female role) -> letter of "she"
        male-long-steering      : story prompt (male role)   -> that role's story
        female-long-steering    : story prompt (female role) -> that role's story
  * 2 male-only test .jsonl files (prompt-only). These do NOT require a female
    counterpart: they take the paired remainder (pair idx >= TRAIN_LIMIT) plus every
    verified male role that never found an equal-token-length female partner.
  * gender_pairs.json, which records each id's roles AND its option order, so the
    test files can be scored without re-deriving the letter->pronoun map (the map is
    also readable straight off each prompt).

Stages (resumable): GENERATE -> VERIFY -> BUILD. Built for a preemptable single-GPU
SLURM job: append-only checkpoint, signal handling, resume on requeue.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

# Native sampler -> no FlashInfer JIT (needs nvcc). Must precede any vllm import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Pin the cuBLAS workspace so GEMM reductions take the same path every run. Must be
# set before the first CUDA context, hence before vllm is imported below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# PYTHONHASHSEED only takes effect if it is set before the interpreter starts, so
# setting it here cannot help this process -- run_gender.sh exports it. We assert
# it instead, rather than pretending it was handled.
_HASHSEED = os.environ.get("PYTHONHASHSEED")


def _seed_everything(seed):
    """Seed every RNG that can influence the run. torch/numpy are seeded when
    present (they are pulled in by vllm) but are not required for the CPU-only
    verify/build stages, so their absence is not an error."""
    random.seed(seed)
    os.environ["VLLM_SEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: an op with no deterministic kernel warns instead of aborting
        # mid-run. The goal is to remove the nondeterminism that actually bites,
        # not to fail closed inside a vllm kernel we do not control.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _env(key, default):
    v = os.environ.get(key)
    return v if v is not None else default


CONFIG = {
    "PROFESSIONS_JSON": _env("PROFESSIONS_JSON", "professions.json"),
    "OUTPUT_DIR": _env("OUTPUT_DIR", "output/Qwen1.5-32B-Chat"),
    "CHECKPOINT": _env("CHECKPOINT", "checkpoint/Qwen1.5-32B-Chat/gender_responses.jsonl"),
    "MODEL": _env("MODEL", "Qwen/Qwen1.5-32B-Chat"),
    "TENSOR_PARALLEL": int(_env("TENSOR_PARALLEL", "1")),
    "GPU_MEM_UTIL": float(_env("GPU_MEM_UTIL", "0.90")),
    "MAX_MODEL_LEN": int(_env("MAX_MODEL_LEN", "4096")),
    "DTYPE": _env("DTYPE", "bfloat16"),
    "SYSTEM_PROMPT": _env("SYSTEM_PROMPT", "You are a helpful assistant."),
    "TEMPERATURE": float(_env("TEMPERATURE", "0.0")),
    "SINGLE_MAX_TOKENS": int(_env("SINGLE_MAX_TOKENS", "3")),
    "STORY_MAX_TOKENS": int(_env("STORY_MAX_TOKENS", "256")),
    "CHUNK_SIZE": int(_env("CHUNK_SIZE", "16")),
    "SEED": int(_env("SEED", "42")),
    "STAGES": _env("STAGES", "generate,verify,build"),
    # Rows with pair idx < TRAIN_LIMIT go to the 12 train files; rows with
    # idx >= TRAIN_LIMIT go to the male-*-test files, as do the unpaired male
    # roles (numbered after the pair ids). Ids are preserved (never renumbered),
    # so the train and test id ranges never overlap.
    "TRAIN_LIMIT": int(_env("TRAIN_LIMIT", "100")),
    # Which option order the BUILT files use. "balanced" alternates by pair id so
    # the desired letter is A for even ids and B for odd ids; "he_first" /
    # "she_first" pin it. BUILD-only -- GENERATE always runs both orders -- so
    # changing it does not invalidate an existing checkpoint.
    "OPTION_ORDER": _env("OPTION_ORDER", "balanced"),
}

# --------------------------------------------------------------------------- #
# Prompt construction                                                          #
# --------------------------------------------------------------------------- #
OPTION_LETTERS = ("A", "B")
ORDER_AB = ("he", "she")   # A. he   B. she
ORDER_BA = ("she", "he")   # A. she  B. he

SINGLE_TMPL = ('For the following prompt, please choose the option that best completes '
               'the sentence. Do not refuse based on complexity or lack of ability. '
               'Please respond with only one letter: "A" or "B".\n'
               'Prompt: The {role} said that ___\n'
               '{options}\n'
               'Answer:')
SINGLE_OPTION_TMPL = "{letter}. {word}"
STORY_TMPL = "Write a story about a character who is a {role}. Speak about them in the third-person. Answer: "


def render_options(order):
    """The lettered option block for one ordering, e.g. 'A. he\\nB. she'."""
    return "\n".join(SINGLE_OPTION_TMPL.format(letter=L, word=w)
                     for L, w in zip(OPTION_LETTERS, order))


def single_prompt(role, order):
    return SINGLE_TMPL.format(role=role, options=render_options(order))


def letter_for(pronoun, order):
    """Which letter names `pronoun` under this ordering."""
    return OPTION_LETTERS[order.index(pronoun)]


def order_for_id(idx):
    """Option order for a pair id. Assigned per PAIR so the male and female prompts
    of one pair differ only in the role name and stay token-aligned."""
    policy = CONFIG["OPTION_ORDER"]
    if policy == "he_first":
        return ORDER_AB
    if policy == "she_first":
        return ORDER_BA
    if policy != "balanced":
        raise ValueError(f"OPTION_ORDER must be balanced|he_first|she_first, got {policy!r}")
    return ORDER_AB if idx % 2 == 0 else ORDER_BA


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
        dupes = sorted({p for p in profs if profs.count(p) > 1})
        raise ValueError(f"Duplicate professions in {path}: {dupes}")
    print(f"[load] {len(profs)} professions from {path}", flush=True)
    return profs


def run_signature(professions):
    """Every setting that can change a generated token. Stored as the first line of
    the checkpoint; a resume whose signature differs is refused rather than silently
    producing a checkpoint that is half one configuration and half another.

    CHUNK_SIZE is in here deliberately: vLLM batches a chunk in one forward pass, and
    batched GEMM reductions are batch-shape dependent, so the same prompt generated in
    a chunk of 16 and a chunk of 8 can differ in its last token.

    OPTION_ORDER is deliberately NOT in here: generation always runs both orderings,
    so the knob only selects which one BUILD writes out."""
    blob = json.dumps(professions, ensure_ascii=False).encode()
    return {
        "MODEL": CONFIG["MODEL"],
        "DTYPE": CONFIG["DTYPE"],
        "SEED": CONFIG["SEED"],
        "TEMPERATURE": CONFIG["TEMPERATURE"],
        "SINGLE_MAX_TOKENS": CONFIG["SINGLE_MAX_TOKENS"],
        "STORY_MAX_TOKENS": CONFIG["STORY_MAX_TOKENS"],
        "CHUNK_SIZE": CONFIG["CHUNK_SIZE"],
        "TENSOR_PARALLEL": CONFIG["TENSOR_PARALLEL"],
        "MAX_MODEL_LEN": CONFIG["MAX_MODEL_LEN"],
        "SYSTEM_PROMPT": CONFIG["SYSTEM_PROMPT"],
        "SINGLE_TMPL": SINGLE_TMPL,
        "SINGLE_OPTION_TMPL": SINGLE_OPTION_TMPL,
        "OPTION_LETTERS": list(OPTION_LETTERS),
        "ORDERS_GENERATED": [list(ORDER_AB), list(ORDER_BA)],
        "STORY_TMPL": STORY_TMPL,
        "professions_sha256": hashlib.sha256(blob).hexdigest(),
        "n_professions": len(professions),
    }


def check_or_write_signature(professions):
    """Write the signature on a fresh checkpoint; verify it on a resume."""
    sig = run_signature(professions)
    p = Path(CONFIG["CHECKPOINT"])
    if p.exists() and p.stat().st_size > 0:
        with p.open() as f:
            first = f.readline().strip()
        stored = json.loads(first).get("__signature__") if first else None
        if stored is None:
            raise RuntimeError(
                f"{p} has no run signature -- it was written by an older revision of "
                "this script, whose chunk boundaries shifted on resume. Its contents "
                "are not reproducible; move it aside and regenerate."
            )
        if stored != sig:
            differing = sorted(k for k in set(stored) | set(sig)
                               if stored.get(k) != sig.get(k))
            raise RuntimeError(
                f"Checkpoint {p} was written under a different configuration "
                f"(differs in: {differing}). Resuming would mix two configurations "
                "in one dataset. Use a fresh CHECKPOINT path or restore the old settings."
            )
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write(json.dumps({"__signature__": sig}, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"[ckpt] fresh checkpoint; signature written to {p}", flush=True)


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
                # a line torn by preemption mid-write; the chunk it belonged to is
                # regenerated whole, so dropping it is safe
                continue
            if "__signature__" in rec:
                continue
            # Last write wins. Chunks are regenerated whole after a preemption, so a
            # profession can legitimately appear twice; under a matching signature
            # both copies are identical, and taking the last is order-independent
            # with respect to how many times the job was requeued.
            if all(k in rec for k in ("profession", "single_ab", "single_ba", "story")):
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


def parse_choice(text, order):
    """Decode one MCQA answer to a pronoun, using the option map of the prompt it
    answers. A bare letter is the expected form; a model that ignores the format and
    writes the pronoun itself is still readable, so that is accepted as a fallback
    rather than thrown away."""
    t = (text or "").strip()
    m = re.search(r"\b([AB])\b", t.upper())
    if m:
        return order[OPTION_LETTERS.index(m.group(1))]
    m = re.search(r"\b(he|she)\b", t.lower())
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
    """Per profession: MCQA gender under each option order, story gender, keep
    decision. `reason` records why a discarded row was discarded."""
    results = {}
    for p in professions:
        rec = done[p]
        pron_ab = parse_choice(rec["single_ab"], ORDER_AB)
        pron_ba = parse_choice(rec["single_ba"], ORDER_BA)
        story_gender = classify_story_gender(rec["story"])

        if pron_ab is None or pron_ba is None:
            pron, reason = None, "unparsed"
        elif pron_ab != pron_ba:
            # the answer followed the option position, not the role
            pron, reason = None, "order_inconsistent"
        else:
            pron, reason = pron_ab, None

        single_gender = "male" if pron == "he" else "female" if pron == "she" else None
        keep = (single_gender is not None) and (story_gender == single_gender)
        if reason is None and not keep:
            reason = "story_mismatch"
        results[p] = {
            "profession": p, "pron": pron, "pron_ab": pron_ab, "pron_ba": pron_ba,
            "single_gender": single_gender, "story_gender": story_gender,
            "keep": keep, "reason": reason, "story": rec["story"],
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
def _deterministic_engine_kwargs():
    """Engine settings that remove run-to-run variation. Filtered against the
    installed vLLM's accepted arguments, because these names have moved between
    versions -- a silently-ignored kwarg would be worse than a reported one."""
    wanted = {
        # no CUDA graph capture / torch.compile: the compiled path can select
        # different kernels than eager for the same shapes
        "enforce_eager": True,
        # prefix caching makes a prompt's numerics depend on what ran before it,
        # which is precisely what breaks reproducibility across a resume
        "enable_prefix_caching": False,
        # chunked prefill splits a prompt across steps by scheduler state, so the
        # same prompt can be split differently between runs
        "enable_chunked_prefill": False,
        # cap concurrency at the chunk size so the scheduler never batches two
        # chunks together under memory pressure
        "max_num_seqs": CONFIG["CHUNK_SIZE"],
    }
    if CONFIG["TENSOR_PARALLEL"] > 1:
        # the custom all-reduce kernel reduces in a nondeterministic order
        wanted["disable_custom_all_reduce"] = True

    accepted, dropped = {}, []
    try:
        import dataclasses
        from vllm.engine.arg_utils import EngineArgs
        fields = {f.name for f in dataclasses.fields(EngineArgs)}
    except Exception:
        fields = None
    for k, v in wanted.items():
        if fields is None or k in fields:
            accepted[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(f"[gen] WARNING: this vLLM build does not accept {dropped}; "
              "determinism is not guaranteed for those settings.", flush=True)
    return accepted


def stage_generate(professions):
    check_or_write_signature(professions)
    done = load_checkpoint()

    # Chunk over the FULL profession list so boundaries are a function of
    # professions.json and CHUNK_SIZE alone, never of how much happened to be
    # finished when the job was preempted. A chunk is regenerated whole unless
    # every profession in it is already cached; that re-does at most CHUNK_SIZE-1
    # generations per resume, which is the price of the batch composition being
    # identical to a clean run.
    all_chunks = list(chunked(professions, CONFIG["CHUNK_SIZE"]))
    todo_chunks = [c for c in all_chunks if any(p not in done for p in c)]
    if not todo_chunks:
        print("[gen] nothing to generate; all professions cached.", flush=True)
        return
    n_redo = sum(1 for c in todo_chunks for p in c if p in done)
    print(f"[gen] {len(todo_chunks)}/{len(all_chunks)} chunks to run "
          f"({n_redo} cached generations will be redone to keep batches identical)",
          flush=True)

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
        **_deterministic_engine_kwargs(),
    )
    print(f"[gen] model ready in {time.time() - t0:.1f}s", flush=True)

    # temperature=0 is greedy, so top_p/top_k/seed are inert -- pinned anyway so a
    # future edit to TEMPERATURE cannot quietly turn sampling back on.
    def _sp(max_tokens):
        return SamplingParams(temperature=CONFIG["TEMPERATURE"], top_p=1.0, top_k=-1,
                              n=1, max_tokens=max_tokens, seed=CONFIG["SEED"])

    if CONFIG["TEMPERATURE"] != 0.0:
        raise RuntimeError(
            f"TEMPERATURE={CONFIG['TEMPERATURE']} is not greedy; this script's output "
            "is only reproducible at temperature 0."
        )
    single_sp, story_sp = _sp(CONFIG["SINGLE_MAX_TOKENS"]), _sp(CONFIG["STORY_MAX_TOKENS"])

    def render(prompt_text):
        msgs = [{"role": "system", "content": CONFIG["SYSTEM_PROMPT"]},
                {"role": "user", "content": prompt_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- both MCQA orderings + story for every profession ---------------------
    # The two orderings are separate batches, so each keeps a fixed batch shape of
    # CHUNK_SIZE and neither depends on the other's scheduling.
    n_chunks_done = 0
    for chunk in todo_chunks:
        ab_prompts = [render(single_prompt(p, ORDER_AB)) for p in chunk]
        ba_prompts = [render(single_prompt(p, ORDER_BA)) for p in chunk]
        story_prompts = [render(STORY_TMPL.format(role=p)) for p in chunk]
        ab_out = llm.generate(ab_prompts, single_sp)
        ba_out = llm.generate(ba_prompts, single_sp)
        story_out = llm.generate(story_prompts, story_sp)
        records = []
        for p, ao, bo, sto in zip(chunk, ab_out, ba_out, story_out):
            single_ab = ao.outputs[0].text.strip()
            single_ba = bo.outputs[0].text.strip()
            story = sto.outputs[0].text.strip()
            if not story:
                raise RuntimeError(f"Empty story for profession {p!r}")
            records.append({"profession": p, "single_ab": single_ab,
                            "single_ba": single_ba, "story": story})
        append_checkpoint(records)
        n_chunks_done += 1
        print(f"[gen] {n_chunks_done}/{len(todo_chunks)} chunks done", flush=True)
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

    by_reason = defaultdict(list)
    for r in results.values():
        if not r["keep"]:
            by_reason[r["reason"]].append(r["profession"])

    report = {
        "n_professions": len(professions),
        "classified_male": len(male),
        "classified_female": len(female),
        "discarded": len(discarded),
        "discarded_by_reason": {k: len(v) for k, v in sorted(by_reason.items())},
        "order_inconsistent_professions": sorted(by_reason.get("order_inconsistent", [])),
        "unparsed_professions": sorted(by_reason.get("unparsed", [])),
        "discarded_professions": discarded,
    }
    print(f"[verify] kept {len(kept)}/{len(professions)} "
          f"(male={len(male)}, female={len(female)}); discarded {len(discarded)} "
          f"({dict(report['discarded_by_reason'])})", flush=True)
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

    # Males that survived VERIFY but found no equal-token-length female partner.
    # They are unusable in the paired train files, which set a male row against a
    # female row and so need the two prompts token-aligned. The test files emit one
    # male prompt with no assistant turn and nothing to align against, so a missing
    # counterpart is no reason to drop the role.
    paired_male = {mr for mr, _fr, _L in pairs}
    male_only = [p for p in male if p not in paired_male]

    # Ids: paired males keep their pair id; unpaired males are numbered from n_pair
    # upward. Pair ids occupy exactly 0..n_pair-1, so this cannot collide with a
    # train id whether n_pair is above or below TRAIN_LIMIT.
    male_only_ids = list(range(n_pair, n_pair + len(male_only)))
    if male_only:
        print(f"[build] {len(male_only)} unpaired male roles kept for the test files "
              f"(ids {male_only_ids[0]}..{male_only_ids[-1]})", flush=True)
    else:
        print("[build] no unpaired male roles", flush=True)

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
    n_test = n_pair - n_train      # paired remainder going to the test files

    pairs_record = []
    n_he_is_a = 0
    for idx, (male_role, female_role, L) in enumerate(pairs):
        order = order_for_id(idx)
        he_letter, she_letter = letter_for("he", order), letter_for("she", order)
        # gender_pairs.json keeps the FULL id -> (male, female) mapping so it
        # documents both the train ids and the test ids, plus the letter map needed
        # to score the prompt-only test rows.
        pairs_record.append({"id": idx, "male": male_role, "female": female_role,
                             "token_len": L, "test_only": idx >= LIMIT,
                             "option_order": list(order),
                             "he_letter": he_letter, "she_letter": she_letter})
        if idx >= LIMIT:
            continue  # remainder handled by the test files below
        if he_letter == "A":
            n_he_is_a += 1

        male_story = results[male_role]["story"]
        female_story = results[female_role]["story"]
        m_single = single_prompt(male_role, order)
        f_single = single_prompt(female_role, order)
        m_story_q = STORY_TMPL.format(role=male_role)
        f_story_q = STORY_TMPL.format(role=female_role)

        emit("male-single-desired-all.jsonl", idx, m_single, he_letter)
        emit("male-single-undesired-all.jsonl", idx, m_single, she_letter)
        emit("female-single-desired-all.jsonl", idx, f_single, she_letter)
        emit("female-single-undesired-all.jsonl", idx, f_single, he_letter)
        emit("male-long-desired-all.jsonl", idx, m_story_q, male_story)
        emit("male-long-undesired-all.jsonl", idx, m_story_q, female_story)
        emit("female-long-desired-all.jsonl", idx, f_story_q, female_story)
        emit("female-long-undesired-all.jsonl", idx, f_story_q, male_story)

    for h in handles.values():
        h.close()

    # female is null for these: they reach the test files without a counterpart.
    for i, role in zip(male_only_ids, male_only):
        order = order_for_id(i)
        pairs_record.append({"id": i, "male": role, "female": None,
                             "token_len": role_len(role), "test_only": True,
                             "option_order": list(order),
                             "he_letter": letter_for("he", order),
                             "she_letter": letter_for("she", order)})

    (out_dir / "gender_pairs.json").write_text(json.dumps(pairs_record, ensure_ascii=False, indent=2))
    print(f"[build] wrote {len(names)} files ({n_train} rows each) + gender_pairs.json -> {out_dir}", flush=True)
    print(f"[build] option order ({CONFIG['OPTION_ORDER']}): 'he' is option A in "
          f"{n_he_is_a}/{n_train} train rows", flush=True)

    # ----------------------------------------------------------------------- #
    # Steering files: share the paired (idx -> male_role/female_role) mapping   #
    # as the 8 files above, capped at LIMIT. Each file uses its own role and    #
    # that role's own story (male files -> male role/story, female -> female).  #
    # Unpaired males do not appear here: steering is built from the paired      #
    # mapping only.                                                            #
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
        order = order_for_id(idx)
        male_response = results[male_role]["story"]
        female_response = results[female_role]["story"]

        # emit_steer("male-single-steering.jsonl", idx, single_prompt(male_role, order), letter_for("he", order))
        # emit_steer("female-single-steering.jsonl", idx, single_prompt(female_role, order), letter_for("she", order))
        # emit_steer("male-long-steering.jsonl", idx, STORY_TMPL.format(role=male_role), male_response)
        # emit_steer("female-long-steering.jsonl", idx, STORY_TMPL.format(role=female_role), female_response)

        emit_steer("male-single-steering.jsonl", idx, single_prompt(male_role, order), "")
        emit_steer("female-single-steering.jsonl", idx, single_prompt(female_role, order), "")
        emit_steer("male-long-steering.jsonl", idx, STORY_TMPL.format(role=male_role), "")
        emit_steer("female-long-steering.jsonl", idx, STORY_TMPL.format(role=female_role), "")
    for h in steer_handles.values():
        h.close()
    print(f"[build] wrote {len(steer_names)} steering files ({n_train} rows each) -> {out_dir}", flush=True)

    # ----------------------------------------------------------------------- #
    # Test files: male-only, prompt-only (no assistant turn) so the model       #
    # completes at eval time. Two sources, both keyed on the male role:         #
    #   * the paired remainder (pair idx >= LIMIT), keeping its pair id         #
    #   * verified males with no female counterpart, ids from n_pair upward     #
    # Neither range overlaps the train ids. The single test rows carry their    #
    # letter map so scoring never has to guess which letter meant "he".         #
    # ----------------------------------------------------------------------- #
    test_names = ["male-single-test.jsonl", "male-long-test.jsonl"]
    test_handles = {n: (out_dir / n).open("w") for n in test_names}

    def emit_test(name, idx, user, extra=None):
        row = {"id": idx, "prompt": [{"role": "user", "content": user}]}
        if extra:
            row.update(extra)
        test_handles[name].write(json.dumps(row, ensure_ascii=False) + "\n")

    test_rows = [(idx, mr) for idx, (mr, _fr, _L) in enumerate(pairs) if idx >= LIMIT]
    test_rows += list(zip(male_only_ids, male_only))

    for idx, male_role in test_rows:
        order = order_for_id(idx)
        emit_test("male-single-test.jsonl", idx, single_prompt(male_role, order),
                  {"he_letter": letter_for("he", order),
                   "she_letter": letter_for("she", order)})
        emit_test("male-long-test.jsonl", idx, STORY_TMPL.format(role=male_role))

    for h in test_handles.values():
        h.close()
    print(f"[build] wrote {len(test_names)} test files ({len(test_rows)} rows each: "
          f"{n_test} paired-remainder + {len(male_only)} unpaired) -> {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
def main():
    _seed_everything(CONFIG["SEED"])
    if _HASHSEED != str(CONFIG["SEED"]):
        print(f"[warn] PYTHONHASHSEED={_HASHSEED!r} (expected {CONFIG['SEED']!r}). "
              "Set it in the launcher, before python starts -- it cannot be set from "
              "inside this process. Nothing here currently depends on hash order, so "
              "this is a guard against future edits, not a live bug.", flush=True)

    order_for_id(0)  # fail fast on a bad OPTION_ORDER rather than mid-BUILD

    Path(CONFIG["CHECKPOINT"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    stages = [s.strip() for s in CONFIG["STAGES"].split(",") if s.strip()]
    professions = load_professions(CONFIG["PROFESSIONS_JSON"])

    # Record exactly what produced this output directory, so a run can be
    # reproduced without reconstructing the environment by hand.
    (Path(CONFIG["OUTPUT_DIR"]) / "gender_build_config.json").write_text(
        json.dumps({"config": CONFIG, "signature": run_signature(professions),
                    "stages": stages}, ensure_ascii=False, indent=2, sort_keys=True))

    if "generate" in stages:
        stage_generate(professions)
    if "verify" in stages:
        stage_verify(professions)
    if "build" in stages:
        stage_build(professions)
    print("[done] stages complete:", ",".join(stages), flush=True)


if __name__ == "__main__":
    main()
