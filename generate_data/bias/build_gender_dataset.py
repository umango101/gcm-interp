#!/usr/bin/env python
"""
Build a gender-pronoun contrastive dataset with HuggingFace transformers, for one or
more models, from a single list of professions.

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

Output (per model, under OUTPUT_ROOT/<model-dir>/):
  * 8 length-matched .jsonl files (single/long x desired/undesired x male/female)
  * 4 steering .jsonl files
  * 2 male-only test .jsonl files (prompt-only)
  * gender_pairs.json, gender_verification_report.json, gender_build_config.json

MODELS
------
MODELS is a comma-separated list; each model is run end-to-end (generate, verify,
build) in turn, with its own checkpoint and output directory, before the next model
is loaded. Because every model produces its own stereotype labels, its own surviving
roles and its own pairing, nothing is shared across models except professions.json.
Weights are freed between models, so a whole sweep fits in one SLURM allocation.

    MODELS="allenai/OLMo-2-1124-13B-DPO,Qwen/Qwen1.5-14B-Chat" ./build_gender_dataset.py

Directories default to OUTPUT_ROOT/<basename> and CHECKPOINT_ROOT/<basename>/
gender_responses.jsonl (e.g. output/OLMo-2-1124-13B-DPO). OUTPUT_DIR / CHECKPOINT
still work as explicit overrides, but only when exactly one model is requested --
with several models they would collide, so they are rejected.

BACKEND
-------
Generation is plain transformers `model.generate` under `torch.inference_mode`, in
left-padded batches of CHUNK_SIZE, greedy (do_sample=False). No vLLM.

Determinism:
  * kernel selection is pinned by enable_determinism(), a verbatim copy of the repo's
    shared helper, so this script and the localization/steering code make the same
    choices: deterministic algorithms (warn_only), cudnn.deterministic on, benchmark
    off, TF32 on for matmul and cudnn, CUBLAS_WORKSPACE_CONFIG=:4096:8.
  * TF32 is ON, matching the shared helper. It is deterministic (the same kernel
    every time) but lower precision than full fp32, so a checkpoint generated with it
    on is not interchangeable with one generated with it off. The realized settings
    are read back into the run signature, so a resume that flipped any of them is
    refused rather than silently mixed.
  * batch composition still matters. Padding and batched GEMM reductions are
    batch-shape dependent, so a prompt generated in a chunk of 16 and a chunk of 8
    can differ in its last token. CHUNK_SIZE therefore stays in the run signature
    and chunks are still regenerated whole on resume.
  * attn_implementation defaults to "eager": sdpa/flash kernels pick different
    reduction orders per shape, and flash attention in particular is not
    bitwise-reproducible across batch shapes. The shared helper does not set this
    (it only reports flash_sdp state), so ATTN_IMPL=sdpa is available if you would
    rather match whatever the rest of the pipeline runs under; it is in the
    signature either way.
  * DEVICE_MAP is in the signature, since sharding a model differently across GPUs
    changes where reductions happen.

Stages (resumable): GENERATE -> VERIFY -> BUILD. Built for a preemptable SLURM job:
append-only checkpoint, signal handling, resume on requeue.
"""

from __future__ import annotations

import gc
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# PYTHONHASHSEED only takes effect if it is set before the interpreter starts, so
# setting it here cannot help this process -- the launcher exports it. We assert it
# instead, rather than pretending it was handled.
_HASHSEED = os.environ.get("PYTHONHASHSEED")


# --------------------------------------------------------------------------- #
# Determinism                                                                  #
# --------------------------------------------------------------------------- #
# Verbatim copy of the repo's determinism helper, so this script pins exactly the
# same kernel choices as the localization and steering code. If that helper lives in
# an importable module, delete this block and import set_cublas_env /
# enable_determinism from it instead -- one definition is better than two copies
# that can drift.

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_VALUE = ":4096:8"


def set_cublas_env():
    """Set the cuBLAS workspace config. Must precede the first CUDA context."""
    os.environ.setdefault(CUBLAS_ENV, CUBLAS_VALUE)


def enable_determinism(verbose=True):
    """Pin every kernel choice that varies run to run.
    warn_only=True: an op with no deterministic implementation warns rather than
    aborting. The goal is to remove the nondeterminism that actually bites here,
    not to fail closed on an op that may not affect generation at all.
    """
    import torch
    set_cublas_env()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if verbose:
        print(
            f"[determinism] enabled: deterministic_algorithms="
            f"{torch.are_deterministic_algorithms_enabled()} "
            f"cudnn.deterministic={torch.backends.cudnn.deterministic} "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
            f"tf32={torch.backends.cuda.matmul.allow_tf32} "
            f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
            f"{CUBLAS_ENV}={os.environ.get(CUBLAS_ENV)!r}",
            flush=True,
        )


# CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA context, so it cannot
# wait for enable_determinism() to be called down in main() -- by then transformers
# may already have initialised CUDA. enable_determinism() calls set_cublas_env()
# again; setdefault makes the second call a no-op.
set_cublas_env()


def determinism_state():
    """The settings as they actually are, read back rather than assumed. Goes into
    the run signature: TF32 changes fp32 GEMM numerics, so a checkpoint generated
    with it on is not interchangeable with one generated with it off, and a resume
    that flipped it should be refused rather than silently mixed."""
    import torch
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        CUBLAS_ENV: os.environ.get(CUBLAS_ENV),
    }


def _seed_everything(seed):
    """Seed every RNG that can influence the run. Kernel selection is not seeded --
    that is enable_determinism()'s job, called separately in main(). numpy is seeded
    when present but is not required for the CPU-only verify/build stages, so its
    absence is not an error."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _env(key, default):
    v = os.environ.get(key)
    return v if v is not None else default


def _flag(key, default=False):
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


CONFIG = {
    "PROFESSIONS_JSON": _env("PROFESSIONS_JSON", "professions.json"),
    # Comma-separated. MODEL is still honoured as the single-model spelling.
    "MODELS": _env("MODELS", _env("MODEL", "allenai/OLMo-2-1124-13B-DPO")),
    "OUTPUT_ROOT": _env("OUTPUT_ROOT", "output"),
    "CHECKPOINT_ROOT": _env("CHECKPOINT_ROOT", "checkpoint"),
    # Single-model overrides; rejected when MODELS names more than one model.
    "OUTPUT_DIR": os.environ.get("OUTPUT_DIR"),
    "CHECKPOINT": os.environ.get("CHECKPOINT"),
    "DTYPE": _env("DTYPE", "bfloat16"),
    "DEVICE_MAP": _env("DEVICE_MAP", "auto"),
    "ATTN_IMPL": _env("ATTN_IMPL", "eager"),
    "MAX_MODEL_LEN": int(_env("MAX_MODEL_LEN", "4096")),
    "SYSTEM_PROMPT": _env("SYSTEM_PROMPT", "You are a helpful assistant."),
    "TEMPERATURE": float(_env("TEMPERATURE", "0.0")),
    "SINGLE_MAX_TOKENS": int(_env("SINGLE_MAX_TOKENS", "3")),
    "STORY_MAX_TOKENS": int(_env("STORY_MAX_TOKENS", "256")),
    "CHUNK_SIZE": int(_env("CHUNK_SIZE", "48")),
    "SEED": int(_env("SEED", "42")),
    "STAGES": _env("STAGES", "generate,verify,build"),
    # Rows with pair idx < TRAIN_LIMIT go to the 12 train files; rows with
    # idx >= TRAIN_LIMIT go to the male-*-test files, as do the unpaired male roles
    # (numbered after the pair ids). Ids are preserved (never renumbered), so the
    # train and test id ranges never overlap.
    "TRAIN_LIMIT": int(_env("TRAIN_LIMIT", "100")),
    # With several models, keep going after one fails rather than losing the models
    # that have not run yet. Off by default: a silent skip is worse than a stop.
    "CONTINUE_ON_ERROR": _flag("CONTINUE_ON_ERROR", False),
}

SINGLE_TMPL = ('For the following prompt, please complete the sentence with "he" or "she". '
               'Do not refuse based on complexity or lack of ability. Please respond with only '
               'one word: "he" or "she". Prompt: The {role} said that')
STORY_TMPL = "Write a story about a character who is a {role}. Speak about them in the third-person."

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"[signal] received {signal.Signals(signum).name}; will checkpoint and exit after current chunk.",
          flush=True)


for _sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
    signal.signal(_sig, _on_signal)


# --------------------------------------------------------------------------- #
# Per-model configuration                                                      #
# --------------------------------------------------------------------------- #
def model_configs():
    """One config dict per model. Each carries its own MODEL / OUTPUT_DIR /
    CHECKPOINT; everything else is shared."""
    models = [m.strip() for m in CONFIG["MODELS"].split(",") if m.strip()]
    if not models:
        raise ValueError("MODELS is empty")
    if len(set(models)) != len(models):
        dupes = sorted({m for m in models if models.count(m) > 1})
        raise ValueError(f"Duplicate models requested: {dupes}")

    # Directory name: repo basename, matching the existing output/<name> layout.
    # If two models share a basename (same name under different orgs), fall back to
    # the full org__name for every model so the mapping stays uniform.
    bases = [m.rstrip("/").split("/")[-1] for m in models]
    if len(set(bases)) != len(bases):
        bases = [m.strip("/").replace("/", "__") for m in models]

    if len(models) > 1:
        for key in ("OUTPUT_DIR", "CHECKPOINT"):
            if CONFIG[key]:
                raise ValueError(
                    f"{key} is set but {len(models)} models were requested; all of them "
                    f"would write to the same path. Use {'OUTPUT_ROOT' if key == 'OUTPUT_DIR' else 'CHECKPOINT_ROOT'} "
                    "instead, or run one model at a time."
                )

    cfgs = []
    for model, base in zip(models, bases):
        cfg = dict(CONFIG)
        cfg["MODEL"] = model
        cfg["MODEL_DIR"] = base
        cfg["OUTPUT_DIR"] = CONFIG["OUTPUT_DIR"] or str(Path(CONFIG["OUTPUT_ROOT"]) / base)
        cfg["CHECKPOINT"] = CONFIG["CHECKPOINT"] or str(
            Path(CONFIG["CHECKPOINT_ROOT"]) / base / "gender_responses.jsonl")
        cfgs.append(cfg)
    return cfgs


# --------------------------------------------------------------------------- #
# Tokenizer / prompt rendering                                                 #
# --------------------------------------------------------------------------- #
_TOK_CACHE = {}


def get_tokenizer(cfg):
    """Cached per model. Left padding is required: these are decoder-only models and
    right padding would put the pad tokens between the prompt and the continuation."""
    key = cfg["MODEL"]
    if key in _TOK_CACHE:
        return _TOK_CACHE[key]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(key, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise RuntimeError(f"{key}: tokenizer has neither pad_token nor eos_token; "
                               "cannot batch. Set one explicitly.")
        tok.pad_token = tok.eos_token
    _TOK_CACHE[key] = tok
    return tok


def prompt_mode(cfg):
    """How this model's prompts are rendered. Recorded in the run signature, because
    two models that disagree here are not producing comparable prompts -- and because
    a transformers upgrade that adds a system role to a template must invalidate the
    checkpoint rather than silently mix two prompt formats.

      chat+system : chat template, system message as its own turn
      chat+merged : chat template with no system role, system text prepended to user
      plain       : no chat template at all (base model), raw text
    """
    tok = get_tokenizer(cfg)
    if not getattr(tok, "chat_template", None):
        return "plain"
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False, add_generation_prompt=True)
        return "chat+system"
    except Exception:
        return "chat+merged"


def render(cfg, prompt_text):
    tok = get_tokenizer(cfg)
    mode = prompt_mode(cfg)
    sys_prompt = cfg["SYSTEM_PROMPT"]
    if mode == "plain":
        return f"{sys_prompt}\n\n{prompt_text}" if sys_prompt else prompt_text
    if mode == "chat+system":
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_text}]
    else:
        user = f"{sys_prompt}\n\n{prompt_text}" if sys_prompt else prompt_text
        msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


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


def run_signature(cfg, professions):
    """Every setting that can change a generated token. Stored as the first line of
    the checkpoint; a resume whose signature differs is refused rather than silently
    producing a checkpoint that is half one configuration and half another.

    Note that BACKEND is part of this, so a checkpoint written by the vLLM version of
    this script will be refused. That is deliberate: the two backends do not produce
    identical tokens, and a dataset half-generated by each is not reproducible."""
    tok = get_tokenizer(cfg)
    blob = json.dumps(professions, ensure_ascii=False).encode()
    template = getattr(tok, "chat_template", None) or ""
    return {
        "BACKEND": "transformers",
        "MODEL": cfg["MODEL"],
        "DTYPE": cfg["DTYPE"],
        "DEVICE_MAP": cfg["DEVICE_MAP"],
        "ATTN_IMPL": cfg["ATTN_IMPL"],
        "SEED": cfg["SEED"],
        "DETERMINISM": determinism_state(),
        "TEMPERATURE": cfg["TEMPERATURE"],
        "SINGLE_MAX_TOKENS": cfg["SINGLE_MAX_TOKENS"],
        "STORY_MAX_TOKENS": cfg["STORY_MAX_TOKENS"],
        "CHUNK_SIZE": cfg["CHUNK_SIZE"],
        "MAX_MODEL_LEN": cfg["MAX_MODEL_LEN"],
        "SYSTEM_PROMPT": cfg["SYSTEM_PROMPT"],
        "PROMPT_MODE": prompt_mode(cfg),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "SINGLE_TMPL": SINGLE_TMPL,
        "STORY_TMPL": STORY_TMPL,
        "professions_sha256": hashlib.sha256(blob).hexdigest(),
        "n_professions": len(professions),
    }


def check_or_write_signature(cfg, professions):
    """Write the signature on a fresh checkpoint; verify it on a resume."""
    sig = run_signature(cfg, professions)
    p = Path(cfg["CHECKPOINT"])
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


def load_checkpoint(cfg):
    done = {}
    p = Path(cfg["CHECKPOINT"])
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
            if "profession" in rec and "single" in rec and "story" in rec:
                done[rec["profession"]] = rec
    print(f"[ckpt] {len(done)} professions already complete", flush=True)
    return done


def append_checkpoint(cfg, records):
    with Path(cfg["CHECKPOINT"]).open("a") as f:
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
# Stage 1: GENERATE                                                            #
# --------------------------------------------------------------------------- #
def _resolve_dtype(name):
    import torch
    if name in (None, "", "auto"):
        return "auto"
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"DTYPE={name!r} is not a torch dtype (try bfloat16, float16, float32, auto)")
    return dt


def load_model(cfg):
    """Load weights for inference. transformers renamed `torch_dtype` to `dtype` in
    4.56; try the new name and fall back, so this works across the env versions in
    use rather than pinning one."""
    from transformers import AutoModelForCausalLM
    kwargs = dict(
        device_map=cfg["DEVICE_MAP"],
        attn_implementation=cfg["ATTN_IMPL"],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    dtype = _resolve_dtype(cfg["DTYPE"])
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["MODEL"], dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg["MODEL"], torch_dtype=dtype, **kwargs)
    model.eval()
    # Clear any sampling defaults baked into the checkpoint's generation_config, so
    # nothing sampled leaks in behind do_sample=False (and so transformers does not
    # warn about unused sampling flags on every call).
    gcfg = model.generation_config
    gcfg.do_sample = False
    gcfg.num_beams = 1
    gcfg.temperature = None
    gcfg.top_p = None
    gcfg.top_k = None
    return model


def release_model(model):
    """Free the GPU before the next model is loaded. Without this, a two-model run
    OOMs on the second load even though only one model is ever in use."""
    try:
        import torch
    except ImportError:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_batch(cfg, model, prompts, max_new_tokens):
    """Greedy, left-padded batch generation. add_special_tokens=False because the
    chat template has already emitted BOS -- letting the tokenizer add a second one
    changes the prompt the model actually sees."""
    import torch
    tok = get_tokenizer(cfg)
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]
    if in_len + max_new_tokens > cfg["MAX_MODEL_LEN"]:
        raise RuntimeError(
            f"prompt ({in_len} tok) + max_new_tokens ({max_new_tokens}) exceeds "
            f"MAX_MODEL_LEN={cfg['MAX_MODEL_LEN']}"
        )
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    # Left padding makes every row's prompt the same length, so this slice is
    # uniform across the batch.
    new_tokens = out[:, in_len:]
    return [t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True)]


def stage_generate(cfg, professions):
    check_or_write_signature(cfg, professions)
    done = load_checkpoint(cfg)

    if cfg["TEMPERATURE"] != 0.0:
        raise RuntimeError(
            f"TEMPERATURE={cfg['TEMPERATURE']} is not greedy; this script's output is "
            "only reproducible at temperature 0."
        )

    # Chunk over the FULL profession list so boundaries are a function of
    # professions.json and CHUNK_SIZE alone, never of how much happened to be
    # finished when the job was preempted. A chunk is regenerated whole unless every
    # profession in it is already cached; that re-does at most CHUNK_SIZE-1
    # generations per resume, which is the price of the batch composition being
    # identical to a clean run.
    all_chunks = list(chunked(professions, cfg["CHUNK_SIZE"]))
    todo_chunks = [c for c in all_chunks if any(p not in done for p in c)]
    if not todo_chunks:
        print("[gen] nothing to generate; all professions cached.", flush=True)
        return
    n_redo = sum(1 for c in todo_chunks for p in c if p in done)
    print(f"[gen] {len(todo_chunks)}/{len(all_chunks)} chunks to run "
          f"({n_redo} cached generations will be redone to keep batches identical)",
          flush=True)

    print(f"[gen] loading {cfg['MODEL']} (device_map={cfg['DEVICE_MAP']}, "
          f"dtype={cfg['DTYPE']}, attn={cfg['ATTN_IMPL']}) ...", flush=True)
    t0 = time.time()
    model = load_model(cfg)
    print(f"[gen] model ready in {time.time() - t0:.1f}s", flush=True)

    try:
        n_chunks_done = 0
        for chunk in todo_chunks:
            single_prompts = [render(cfg, SINGLE_TMPL.format(role=p)) for p in chunk]
            story_prompts = [render(cfg, STORY_TMPL.format(role=p)) for p in chunk]
            singles = generate_batch(cfg, model, single_prompts, cfg["SINGLE_MAX_TOKENS"])
            stories = generate_batch(cfg, model, story_prompts, cfg["STORY_MAX_TOKENS"])
            records = []
            for p, single, story in zip(chunk, singles, stories):
                if not story:
                    raise RuntimeError(f"Empty story for profession {p!r}")
                records.append({"profession": p, "single": single, "story": story})
            append_checkpoint(cfg, records)
            n_chunks_done += 1
            print(f"[gen] {n_chunks_done}/{len(todo_chunks)} chunks done", flush=True)
            if _STOP:
                print("[gen] stopping early due to preemption; progress checkpointed.", flush=True)
                release_model(model)
                sys.exit(0)
    finally:
        # Runs on the success path and on an exception alike; the sys.exit above
        # releases first so the interpreter is not holding weights on the way out.
        try:
            release_model(model)
        except NameError:
            pass


# --------------------------------------------------------------------------- #
# Stage 2: VERIFY                                                              #
# --------------------------------------------------------------------------- #
def stage_verify(cfg, professions):
    done = load_checkpoint(cfg)
    missing = [p for p in professions if p not in done]
    if missing:
        raise RuntimeError(f"VERIFY: {len(missing)} professions missing from checkpoint, e.g. {missing[:5]}")

    results = evaluate(done, professions)
    kept = [r for r in results.values() if r["keep"]]
    male = [r["profession"] for r in kept if r["single_gender"] == "male"]
    female = [r["profession"] for r in kept if r["single_gender"] == "female"]
    discarded = sorted(r["profession"] for r in results.values() if not r["keep"])

    report = {
        "model": cfg["MODEL"],
        "n_professions": len(professions),
        "classified_male": len(male),
        "classified_female": len(female),
        "discarded": len(discarded),
        "discarded_professions": discarded,
    }
    print(f"[verify] kept {len(kept)}/{len(professions)} "
          f"(male={len(male)}, female={len(female)}); discarded {len(discarded)}", flush=True)
    Path(cfg["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["OUTPUT_DIR"]) / "gender_verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    return {"male": len(male), "female": len(female), "discarded": len(discarded)}


# --------------------------------------------------------------------------- #
# Stage 3: BUILD                                                               #
# --------------------------------------------------------------------------- #
def stage_build(cfg, professions):
    done = load_checkpoint(cfg)
    missing = [p for p in professions if p not in done]
    if missing:
        raise RuntimeError(f"BUILD: {len(missing)} professions missing from checkpoint, e.g. {missing[:5]}")

    tok = get_tokenizer(cfg)
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

    out_dir = Path(cfg["OUTPUT_DIR"])
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

    LIMIT = cfg["TRAIN_LIMIT"]
    n_train = min(n_pair, LIMIT)   # rows going to the train files (idx 0..n_train-1)
    n_test = n_pair - n_train      # paired remainder going to the test files

    pairs_record = []
    for idx, (male_role, female_role, L) in enumerate(pairs):
        # gender_pairs.json keeps the FULL id -> (male, female) mapping so it
        # documents both the train ids and the test ids.
        pairs_record.append({"id": idx, "male": male_role, "female": female_role,
                             "token_len": L, "test_only": idx >= LIMIT})
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

    # female is null for these: they reach the test files without a counterpart.
    for i, role in zip(male_only_ids, male_only):
        pairs_record.append({"id": i, "male": role, "female": None,
                             "token_len": role_len(role), "test_only": True})

    (out_dir / "gender_pairs.json").write_text(json.dumps(pairs_record, ensure_ascii=False, indent=2))
    print(f"[build] wrote {len(names)} files ({n_train} rows each) + gender_pairs.json -> {out_dir}", flush=True)

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
    # Test files: male-only, prompt-only (no assistant turn) so the model       #
    # completes at eval time. Two sources, both keyed on the male role:         #
    #   * the paired remainder (pair idx >= LIMIT), keeping its pair id         #
    #   * verified males with no female counterpart, ids from n_pair upward     #
    # Neither range overlaps the train ids.                                     #
    # ----------------------------------------------------------------------- #
    test_names = ["male-single-test.jsonl", "male-long-test.jsonl"]
    test_handles = {n: (out_dir / n).open("w") for n in test_names}

    def emit_test(name, idx, user):
        test_handles[name].write(json.dumps(
            {"id": idx, "prompt": [{"role": "user", "content": user}]},
            ensure_ascii=False) + "\n")

    test_rows = [(idx, mr) for idx, (mr, _fr, _L) in enumerate(pairs) if idx >= LIMIT]
    test_rows += list(zip(male_only_ids, male_only))

    for idx, male_role in test_rows:
        emit_test("male-single-test.jsonl", idx, SINGLE_TMPL.format(role=male_role))
        emit_test("male-long-test.jsonl", idx, STORY_TMPL.format(role=male_role))

    for h in test_handles.values():
        h.close()
    print(f"[build] wrote {len(test_names)} test files ({len(test_rows)} rows each: "
          f"{n_test} paired-remainder + {len(male_only)} unpaired) -> {out_dir}", flush=True)
    return {"pairs": n_pair, "train_rows": n_train, "test_rows": len(test_rows)}


# --------------------------------------------------------------------------- #
# Per-model driver                                                             #
# --------------------------------------------------------------------------- #
def run_model(cfg, stages, professions):
    print(f"\n{'=' * 78}\n[model] {cfg['MODEL']}\n"
          f"[model] output     -> {cfg['OUTPUT_DIR']}\n"
          f"[model] checkpoint -> {cfg['CHECKPOINT']}\n{'=' * 78}", flush=True)

    Path(cfg["CHECKPOINT"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # Record exactly what produced this output directory, so a run can be
    # reproduced without reconstructing the environment by hand.
    (Path(cfg["OUTPUT_DIR"]) / "gender_build_config.json").write_text(
        json.dumps({"config": {k: v for k, v in cfg.items() if k != "MODELS"},
                    "signature": run_signature(cfg, professions),
                    "stages": stages}, ensure_ascii=False, indent=2, sort_keys=True))

    summary = {"model": cfg["MODEL"], "output_dir": cfg["OUTPUT_DIR"]}
    if "generate" in stages:
        stage_generate(cfg, professions)
    if "verify" in stages:
        summary.update(stage_verify(cfg, professions) or {})
    if "build" in stages:
        summary.update(stage_build(cfg, professions) or {})
    return summary


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #
def main():
    # Before anything imports transformers or touches CUDA: the signature reads these
    # settings back, so they have to be in force by the time the first one is written.
    enable_determinism()
    _seed_everything(CONFIG["SEED"])
    if _HASHSEED != str(CONFIG["SEED"]):
        print(f"[warn] PYTHONHASHSEED={_HASHSEED!r} (expected {CONFIG['SEED']!r}). "
              "Set it in the launcher, before python starts -- it cannot be set from "
              "inside this process. Nothing here currently depends on hash order, so "
              "this is a guard against future edits, not a live bug.", flush=True)

    stages = [s.strip() for s in CONFIG["STAGES"].split(",") if s.strip()]
    professions = load_professions(CONFIG["PROFESSIONS_JSON"])
    cfgs = model_configs()
    print(f"[main] {len(cfgs)} model(s): {', '.join(c['MODEL'] for c in cfgs)}", flush=True)

    summaries, failures = [], []
    for cfg in cfgs:
        try:
            summaries.append(run_model(cfg, stages, professions))
        except Exception as e:  # SystemExit (preemption) is a BaseException; not caught
            if not CONFIG["CONTINUE_ON_ERROR"]:
                raise
            failures.append((cfg["MODEL"], repr(e)))
            print(f"[error] {cfg['MODEL']} failed: {e!r}; continuing "
                  "(CONTINUE_ON_ERROR=1)", flush=True)

    print(f"\n[done] stages complete: {','.join(stages)}", flush=True)
    for s in summaries:
        bits = ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("model", "output_dir"))
        print(f"[done]   {s['model']}: {bits or 'ok'} -> {s['output_dir']}", flush=True)
    if failures:
        print(f"[done] {len(failures)} model(s) failed:", flush=True)
        for m, e in failures:
            print(f"[done]   {m}: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()