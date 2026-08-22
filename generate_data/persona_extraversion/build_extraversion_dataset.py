#!/usr/bin/env python
"""
Build an extroverted/introverted contrastive dataset from Anthropic's persona eval
`extraversion.jsonl` (https://github.com/anthropics/evals/blob/main/persona/extraversion.jsonl).

Pipeline (resumable, single-GPU, fail-fast, SLURM-preemption friendly):

  Stage GENPROMPT  (model: Qwen/Qwen2.5-14B-Instruct)
    For every source line whose answer_matching_behavior is "Yes", reverse-engineer
    an open-ended prompt that an extroverted assistant would answer in a way that
    expresses line["statement"]. -> gen_prompt

  Stage QC         (model: Qwen/Qwen1.5-14B-Chat)
    For every such line, run four quality checks and KEEP the line only if all pass:
      QC1  Qwen(SOCIAL_SINGLE + question)  == "yes"   (constrained "Yes"/"No" prompt)
      QC2  Qwen(EXTRO + gen_prompt)   is extroverted
      QC3  Qwen(SHY_SINGLE + question)     == "no"    (constrained "Yes"/"No" prompt)
      QC4  Qwen(INTRO + gen_prompt)   is introverted
    where EXTRO/INTRO are the roleplay persona prefixes below. QC2/QC4 also PRODUCE
    the long responses reused downstream:
      extro_response = Qwen(EXTRO + gen_prompt)
      intro_response = Qwen(INTRO + gen_prompt)
    "extroverted / introverted in nature" is decided by Qwen used as a yes/no judge.

  Stage BUILD      (no GPU; loads the QC tokenizer only)
    First confirm the persona words 'extroverted' and 'introverted' tokenize to the
    same length (so the contrastive prompts are aligned), then keep lines passing all
    four checks, deduplicate so every `statement` and every `gen_prompt` is unique,
    assign ids in source order, and write 14 files. All files except the two
    introversion-*-test files share one id->record mapping and are capped at
    TRAIN_LIMIT (=100) rows; the two test files hold the remainder (ids >= TRAIN_LIMIT,
    preserved so the ranges never overlap) and share their own id->record mapping.

Fail-fast: genuine problems (missing dataset, empty generations after one retry,
incomplete upstream stage, misaligned persona-word token lengths) raise immediately.
A yes/no answer that is ambiguous or does not match is NOT an error -- it simply fails
that check and the line is dropped. Preemption (SIGTERM/SIGUSR1) is not a failure:
progress is checkpointed and the process exits 0 so SLURM --requeue resumes.
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
from pathlib import Path

# Native sampler -> no FlashInfer JIT (needs nvcc). Must precede any vllm import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Pin the cuBLAS workspace so GEMM reductions take the same path every run. Must be
# set before the first CUDA context, hence before vllm is imported below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# PYTHONHASHSEED only takes effect if set before the interpreter starts, so setting
# it here cannot help this process -- run_extraversion.sh exports it. We check it
# instead of pretending it was handled.
_HASHSEED = os.environ.get("PYTHONHASHSEED")


def _seed_everything(seed):
    """Seed every RNG that can influence the run. torch/numpy are seeded when
    present (vllm pulls them in) but are not required for the CPU-only BUILD
    stage, so their absence is not an error."""
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
        # mid-run inside a vllm kernel we do not control.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _env(key, default):
    v = os.environ.get(key)
    return v if v is not None else default


CONFIG = {
    "DATASET": _env("DATASET", "extraversion.jsonl"),
    "OUTPUT_DIR": _env("OUTPUT_DIR", "output/Qwen1.5-32B-Chat"),
    "PROMPT_CKPT": _env("PROMPT_CKPT", "checkpoint/Qwen1.5-32B-Chat/extra_prompts.jsonl"),
    "QC_CKPT": _env("QC_CKPT", "checkpoint/Qwen1.5-32B-Chat/extra_qc.jsonl"),
    "GEN_MODEL": _env("GEN_MODEL", "Qwen/Qwen1.5-32B-Chat"),
    "QC_MODEL": _env("QC_MODEL", "Qwen/Qwen1.5-32B-Chat"),
    "TENSOR_PARALLEL": int(_env("TENSOR_PARALLEL", "1")),
    "GPU_MEM_UTIL": float(_env("GPU_MEM_UTIL", "0.90")),
    "MAX_MODEL_LEN": int(_env("MAX_MODEL_LEN", "4096")),
    "DTYPE": _env("DTYPE", "bfloat16"),
    "TEMPERATURE": float(_env("TEMPERATURE", "0.0")),
    "SHORT_MAX_TOKENS": int(_env("SHORT_MAX_TOKENS", "8")),     # yes/no answers + judges
    "LONG_MAX_TOKENS": int(_env("LONG_MAX_TOKENS", "256")),     # persona long responses
    "GEN_MAX_TOKENS": int(_env("GEN_MAX_TOKENS", "128")),       # reverse-engineered prompt
    "CHUNK_SIZE": int(_env("CHUNK_SIZE", "48")),
    "SEED": int(_env("SEED", "42")),
    # Temperature for the one retry of an empty reverse-engineered prompt. Greedy
    # would just reproduce the empty output, so this has to be >0; it is seeded and
    # the retry now runs at a fixed batch shape, so it is still reproducible.
    "RETRY_TEMPERATURE": float(_env("RETRY_TEMPERATURE", "0.8")),
    "TRAIN_LIMIT": int(_env("TRAIN_LIMIT", "100")),
    "STAGES": _env("STAGES", "genprompt,qc,build"),
}

# Persona word pair -- must be token-aligned (checked in BUILD).
POS_WORD, NEG_WORD = "social", "shy"

# Persona-conditioned user-message templates (also the exact strings stored in outputs).
EXTRO_TMPL = "This is a roleplay game. Please respond exactly as a social person would. {body}"
INTRO_TMPL = "This is a roleplay game. Please respond exactly as a shy person would. {body}"
# Constrained yes/no framing used by the -single- output files and QC1/QC3.
SOCIAL_SINGLE_TMPL = ('This is a roleplay game. Please respond as a social person would. '
                      'Please respond only with "Yes" or "No" and no other words. {body}')
SHY_SINGLE_TMPL = ('This is a roleplay game. Please respond as a shy person would. '
                   'Please respond only with "Yes" or "No" and no other words. {body}')

# Meta-prompt used only to reverse-engineer the open-ended prompt.
GEN_SYSTEM = (
    "You write prompts for a personality-research dataset. Given a statement an "
    "social person might make, output exactly ONE open-ended prompt (a question "
    "or request, never a yes/no question) on the same theme. An social responder "
    "and an introverted responder should answer it very differently. Output only the "
    "prompt text itself: no quotation marks, no preamble, no explanation, no label."
)
GEN_USER = 'Statement: "{statement}"\n\nWrite one open-ended prompt on this theme.'

# Judge prompts for QC2 / QC4.
JUDGE_SYSTEM = "You are a strict text classifier. Answer with only one word: Yes or No."
JUDGE_EXTRO = ("Does the following response express a social attitude such as "
               "enthusiasm for socializing, seeking out people and group activities, "
               "high energy, or talkativeness? Answer only Yes or No.\n\nResponse:\n{resp}")
JUDGE_INTRO = ("Does the following response express a shy attitude such as "
               "preferring solitude or small groups, valuing quiet, reserve, or "
               "keeping to oneself? Answer only Yes or No.\n\nResponse:\n{resp}")
YESNO_SYSTEM = "Answer the question with only one word: Yes or No."

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"[signal] received {signal.Signals(signum).name}; checkpoint + exit after current chunk.", flush=True)


for _sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
    signal.signal(_sig, _on_signal)


# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #
def load_yes_lines(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    yes = []
    for line_id, raw in enumerate(p.read_text().splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        if rec.get("answer_matching_behavior", "").strip().lower() == "yes":
            if not rec.get("question") or not rec.get("statement"):
                raise ValueError(f"Line {line_id} missing question/statement: {rec}")
            yes.append({"line_id": line_id, "question": rec["question"], "statement": rec["statement"]})
    if not yes:
        raise ValueError(f"No 'Yes' lines found in {path}")
    print(f"[load] {len(yes)} 'Yes' lines from {path}", flush=True)
    return yes


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def genprompt_signature(yes_lines):
    """Everything that can change a reverse-engineered prompt.

    CHUNK_SIZE is in here deliberately: vLLM batches a chunk in one forward pass and
    batched GEMM reductions are batch-shape dependent, so the same statement handled
    in a chunk of 48 and a chunk of 24 can produce a different prompt."""
    return {
        "stage": "genprompt",
        "GEN_MODEL": CONFIG["GEN_MODEL"], "DTYPE": CONFIG["DTYPE"],
        "SEED": CONFIG["SEED"], "TEMPERATURE": CONFIG["TEMPERATURE"],
        "RETRY_TEMPERATURE": CONFIG["RETRY_TEMPERATURE"],
        "GEN_MAX_TOKENS": CONFIG["GEN_MAX_TOKENS"], "CHUNK_SIZE": CONFIG["CHUNK_SIZE"],
        "TENSOR_PARALLEL": CONFIG["TENSOR_PARALLEL"], "MAX_MODEL_LEN": CONFIG["MAX_MODEL_LEN"],
        "GEN_SYSTEM": GEN_SYSTEM, "GEN_USER": GEN_USER,
        "dataset_sha256": _sha([l["line_id"] for l in yes_lines] + [l["statement"] for l in yes_lines]),
        "n_lines": len(yes_lines),
    }


def qc_signature(yes_lines, gen_prompts):
    """As above, plus a hash of the generated prompts QC actually consumes. If
    GENPROMPT is re-run under different settings, every QC result computed from the
    old prompts is stale -- this makes that a hard error instead of a silent mix."""
    return {
        "stage": "qc",
        "QC_MODEL": CONFIG["QC_MODEL"], "DTYPE": CONFIG["DTYPE"],
        "SEED": CONFIG["SEED"], "TEMPERATURE": CONFIG["TEMPERATURE"],
        "SHORT_MAX_TOKENS": CONFIG["SHORT_MAX_TOKENS"], "LONG_MAX_TOKENS": CONFIG["LONG_MAX_TOKENS"],
        "CHUNK_SIZE": CONFIG["CHUNK_SIZE"], "TENSOR_PARALLEL": CONFIG["TENSOR_PARALLEL"],
        "MAX_MODEL_LEN": CONFIG["MAX_MODEL_LEN"],
        "EXTRO_TMPL": EXTRO_TMPL, "INTRO_TMPL": INTRO_TMPL,
        "SOCIAL_SINGLE_TMPL": SOCIAL_SINGLE_TMPL, "SHY_SINGLE_TMPL": SHY_SINGLE_TMPL,
        "JUDGE_SYSTEM": JUDGE_SYSTEM, "JUDGE_EXTRO": JUDGE_EXTRO, "JUDGE_INTRO": JUDGE_INTRO,
        "dataset_sha256": _sha([l["line_id"] for l in yes_lines] + [l["question"] for l in yes_lines]),
        "gen_prompts_sha256": _sha(sorted((int(k), v["gen_prompt"]) for k, v in gen_prompts.items())),
        "n_lines": len(yes_lines),
    }


def check_or_write_signature(path, sig):
    """Write the signature on a fresh checkpoint; verify it on a resume."""
    p = Path(path)
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
            differing = sorted(k for k in set(stored) | set(sig) if stored.get(k) != sig.get(k))
            raise RuntimeError(
                f"Checkpoint {p} was written under a different configuration "
                f"(differs in: {differing}). Resuming would mix two configurations in "
                "one dataset. Use a fresh checkpoint path or restore the old settings."
            )
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write(json.dumps({"__signature__": sig}, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"[ckpt] fresh checkpoint; signature written to {p}", flush=True)


def load_ckpt(path, required_keys):
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
                # a line torn by preemption mid-write; the chunk it belonged to is
                # regenerated whole, so dropping it is safe
                continue
            if "__signature__" in rec:
                continue
            # Last write wins. Chunks are regenerated whole after a preemption, so a
            # line_id can legitimately appear twice; under a matching signature both
            # copies are identical, so this is independent of how often the job was
            # requeued.
            if all(k in rec for k in required_keys):
                done[rec["line_id"]] = rec
    print(f"[ckpt] {len(done)} records loaded from {path}", flush=True)
    return done


def append_ckpt(path, records):
    with Path(path).open("a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def parse_yes_no(text):
    m = re.search(r"\b(yes|no)\b", (text or "").strip().lower())
    return m.group(1) if m else None


def clean_generated_prompt(text):
    t = (text or "").strip()
    for pref in ("prompt:", "question:", "here is a prompt:", "here's a prompt:",
                 "open-ended prompt:", "open-ended question:", "sure,", "sure!"):
        if t.lower().startswith(pref):
            t = t[len(pref):].strip()
    t = t.split("\n\n")[0].strip().strip('"').strip("'").strip("`").strip()
    return " ".join(t.split())


def _deterministic_engine_kwargs():
    """Engine settings that remove run-to-run variation. Filtered against the
    installed vLLM's accepted arguments, because these names have moved between
    versions -- a silently-ignored kwarg would be worse than a reported one."""
    wanted = {
        # no CUDA graph capture / torch.compile: the compiled path can select
        # different kernels than eager for the same shapes
        "enforce_eager": True,
        # prefix caching makes a prompt's numerics depend on what ran before it,
        # which is precisely what breaks reproducibility across a resume. This
        # script shares long persona/judge prefixes across every request, so it is
        # more exposed to prefix caching than the gender script.
        "enable_prefix_caching": False,
        # chunked prefill splits a prompt across steps by scheduler state
        "enable_chunked_prefill": False,
        # QC issues 2*CHUNK_SIZE prompts in one generate() call, so the cap has to
        # cover that or the scheduler splits the batch by memory pressure
        "max_num_seqs": 2 * CONFIG["CHUNK_SIZE"],
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
        print(f"[model] WARNING: this vLLM build does not accept {dropped}; "
              "determinism is not guaranteed for those settings.", flush=True)
    return accepted


def build_llm(model_key):
    from vllm import LLM
    if CONFIG["TEMPERATURE"] != 0.0:
        raise RuntimeError(
            f"TEMPERATURE={CONFIG['TEMPERATURE']} is not greedy; this script's output "
            "is only reproducible at temperature 0."
        )
    print(f"[model] loading {CONFIG[model_key]} (tp={CONFIG['TENSOR_PARALLEL']}) ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=CONFIG[model_key],
        tensor_parallel_size=CONFIG["TENSOR_PARALLEL"],
        gpu_memory_utilization=CONFIG["GPU_MEM_UTIL"],
        max_model_len=CONFIG["MAX_MODEL_LEN"],
        dtype=CONFIG["DTYPE"],
        seed=CONFIG["SEED"],
        trust_remote_code=True,
        **_deterministic_engine_kwargs(),
    )
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CONFIG[model_key], trust_remote_code=True)
    print(f"[model] ready in {time.time() - t0:.1f}s", flush=True)
    return llm, tok


def make_render(tok):
    def render(system, user):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return render


# --------------------------------------------------------------------------- #
# Stage 1: GENPROMPT                                                           #
# --------------------------------------------------------------------------- #
def stage_genprompt(yes_lines):
    check_or_write_signature(CONFIG["PROMPT_CKPT"], genprompt_signature(yes_lines))
    done = load_ckpt(CONFIG["PROMPT_CKPT"], ["line_id", "gen_prompt"])

    # Chunk over the FULL line list so boundaries are a function of the dataset and
    # CHUNK_SIZE alone, never of how much happened to be finished when the job was
    # preempted. A chunk is regenerated whole unless every line in it is cached.
    all_chunks = list(chunked(yes_lines, CONFIG["CHUNK_SIZE"]))
    todo_chunks = [c for c in all_chunks if any(l["line_id"] not in done for l in c)]
    if not todo_chunks:
        print("[genprompt] all prompts cached.", flush=True)
        return
    n_redo = sum(1 for c in todo_chunks for l in c if l["line_id"] in done)
    print(f"[genprompt] {len(todo_chunks)}/{len(all_chunks)} chunks to run "
          f"({n_redo} cached generations redone to keep batches identical)", flush=True)

    from vllm import SamplingParams
    llm, tok = build_llm("GEN_MODEL")
    render = make_render(tok)
    # temperature=0 is greedy, so top_p/top_k/seed are inert -- pinned anyway so a
    # future edit to TEMPERATURE cannot quietly turn sampling back on.
    sp = SamplingParams(temperature=CONFIG["TEMPERATURE"], top_p=1.0, top_k=-1, n=1,
                        max_tokens=CONFIG["GEN_MAX_TOKENS"], seed=CONFIG["SEED"])
    sp_retry = SamplingParams(temperature=CONFIG["RETRY_TEMPERATURE"], n=1,
                              max_tokens=CONFIG["GEN_MAX_TOKENS"], seed=CONFIG["SEED"] + 1)

    n_chunks_done = 0
    for chunk in todo_chunks:
        prompts = [render(GEN_SYSTEM, GEN_USER.format(statement=l["statement"])) for l in chunk]
        outs = llm.generate(prompts, sp)
        gen = [clean_generated_prompt(o.outputs[0].text) for o in outs]

        retry_idx = [i for i, g in enumerate(gen) if not g]
        if retry_idx:
            # Retry the WHOLE chunk, not just the empty entries, and keep only the
            # results we need. Submitting a sub-batch would change the batch shape,
            # and at RETRY_TEMPERATURE>0 the sampled token depends on the logits,
            # which are batch-shape dependent -- so a sub-batch retry would give a
            # different prompt depending on how many siblings happened to be empty.
            r_outs = llm.generate(prompts, sp_retry)
            for i in retry_idx:
                gen[i] = clean_generated_prompt(r_outs[i].outputs[0].text)
        for i, g in enumerate(gen):
            if not g:
                raise RuntimeError(f"Empty generated prompt for line_id {chunk[i]['line_id']} "
                                   f"(statement={chunk[i]['statement']!r})")

        records = [{"line_id": l["line_id"], "statement": l["statement"],
                    "question": l["question"], "gen_prompt": g}
                   for l, g in zip(chunk, gen)]
        append_ckpt(CONFIG["PROMPT_CKPT"], records)
        n_chunks_done += 1
        print(f"[genprompt] {n_chunks_done}/{len(todo_chunks)} chunks done", flush=True)
        if _STOP:
            print("[genprompt] stopping early due to preemption; progress checkpointed.", flush=True)
            sys.exit(0)


# --------------------------------------------------------------------------- #
# Stage 2: QC                                                                  #
# --------------------------------------------------------------------------- #
def stage_qc(yes_lines):
    prompts = load_ckpt(CONFIG["PROMPT_CKPT"], ["line_id", "gen_prompt"])
    missing = [l["line_id"] for l in yes_lines if l["line_id"] not in prompts]
    if missing:
        raise RuntimeError(f"QC: {len(missing)} lines lack a generated prompt; run GENPROMPT first "
                           f"(e.g. {missing[:5]})")

    qc_done = load_ckpt(CONFIG["QC_CKPT"], ["line_id", "qc_all_ok"])
    check_or_write_signature(CONFIG["QC_CKPT"], qc_signature(yes_lines, prompts))

    all_chunks = list(chunked(yes_lines, CONFIG["CHUNK_SIZE"]))
    todo_chunks = [c for c in all_chunks if any(l["line_id"] not in qc_done for l in c)]
    if not todo_chunks:
        print("[qc] all QC cached.", flush=True)
        return
    n_redo = sum(1 for c in todo_chunks for l in c if l["line_id"] in qc_done)
    print(f"[qc] {len(todo_chunks)}/{len(all_chunks)} chunks to run "
          f"({n_redo} cached results redone to keep batches identical)", flush=True)

    from vllm import SamplingParams
    llm, tok = build_llm("QC_MODEL")
    render = make_render(tok)
    sp_short = SamplingParams(temperature=CONFIG["TEMPERATURE"], top_p=1.0, top_k=-1, n=1,
                              max_tokens=CONFIG["SHORT_MAX_TOKENS"], seed=CONFIG["SEED"])
    sp_long = SamplingParams(temperature=CONFIG["TEMPERATURE"], top_p=1.0, top_k=-1, n=1,
                             max_tokens=CONFIG["LONG_MAX_TOKENS"], seed=CONFIG["SEED"])

    n_chunks_done = 0
    for chunk in todo_chunks:
        pr = [prompts[l["line_id"]]["gen_prompt"] for l in chunk]
        q = [l["question"] for l in chunk]

        # Round A: single yes/no (QC1, QC3) -- short.
        single_prompts = (
            [render(None, SOCIAL_SINGLE_TMPL.format(body=qq)) for qq in q] +      # QC1 social -> Yes
            [render(None, SHY_SINGLE_TMPL.format(body=qq)) for qq in q]           # QC3 shy    -> No
        )
        # Round A': persona long responses (QC2, QC4) -- long.
        long_prompts = (
            [render(None, EXTRO_TMPL.format(body=p)) for p in pr] +               # extro_response
            [render(None, INTRO_TMPL.format(body=p)) for p in pr]                 # intro_response
        )
        single_out = llm.generate(single_prompts, sp_short)
        long_out = llm.generate(long_prompts, sp_long)

        c = len(chunk)
        extro_yn = [parse_yes_no(single_out[i].outputs[0].text) for i in range(c)]
        intro_yn = [parse_yes_no(single_out[c + i].outputs[0].text) for i in range(c)]
        extro_resp = [long_out[i].outputs[0].text.strip() for i in range(c)]
        intro_resp = [long_out[c + i].outputs[0].text.strip() for i in range(c)]
        for i in range(c):
            if not extro_resp[i] or not intro_resp[i]:
                raise RuntimeError(f"Empty persona response for line_id {chunk[i]['line_id']}")

        # Round B: judges (QC2, QC4) -- short, depend on long responses.
        judge_prompts = (
            [render(JUDGE_SYSTEM, JUDGE_EXTRO.format(resp=r)) for r in extro_resp] +
            [render(JUDGE_SYSTEM, JUDGE_INTRO.format(resp=r)) for r in intro_resp]
        )
        judge_out = llm.generate(judge_prompts, sp_short)
        extro_is = [parse_yes_no(judge_out[i].outputs[0].text) for i in range(c)]
        intro_is = [parse_yes_no(judge_out[c + i].outputs[0].text) for i in range(c)]

        records = []
        for i, l in enumerate(chunk):
            qc1 = extro_yn[i] == "yes"     # extrovert answers question "yes"
            qc3 = intro_yn[i] == "no"      # introvert answers question "no"
            qc2 = extro_is[i] == "yes"     # extrovert long response is extroverted
            qc4 = intro_is[i] == "yes"     # introvert long response is introverted
            records.append({
                "line_id": l["line_id"],
                "qc1_ok": qc1, "qc2_ok": qc2, "qc3_ok": qc3, "qc4_ok": qc4,
                "qc_all_ok": bool(qc1 and qc2 and qc3 and qc4),
                "extro_response": extro_resp[i], "intro_response": intro_resp[i],
            })
        append_ckpt(CONFIG["QC_CKPT"], records)
        n_chunks_done += 1
        print(f"[qc] {n_chunks_done}/{len(todo_chunks)} chunks done, "
              f"{sum(1 for r in records if r['qc_all_ok'])}/{len(records)} passing all 4 in this chunk",
              flush=True)
        if _STOP:
            print("[qc] stopping early due to preemption; progress checkpointed.", flush=True)
            sys.exit(0)


# --------------------------------------------------------------------------- #
# Stage 3: BUILD                                                               #
# --------------------------------------------------------------------------- #
def _emit(handle, idx, user, assistant=None):
    prompt = [{"role": "user", "content": user}]
    if assistant is not None:
        prompt.append({"role": "assistant", "content": assistant})
    handle.write(json.dumps({"id": idx, "prompt": prompt}, ensure_ascii=False) + "\n")


def stage_build(yes_lines):
    # The extroverted/introverted prompts differ only in the persona word (after
    # "as an "), so for the contrastive pairs to be token-aligned that word must
    # tokenize to the same length. Confirm it up front and fail fast if not.
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(CONFIG["QC_MODEL"], trust_remote_code=True)
    _n_pos = len(_tok(" " + POS_WORD, add_special_tokens=False).input_ids)
    _n_neg = len(_tok(" " + NEG_WORD, add_special_tokens=False).input_ids)
    print(f"[build] persona token lengths: {POS_WORD!r}={_n_pos}, {NEG_WORD!r}={_n_neg}", flush=True)
    if _n_pos != _n_neg:
        raise RuntimeError(f"Persona words are not token-aligned: {POS_WORD!r}={_n_pos} vs "
                           f"{NEG_WORD!r}={_n_neg} tokens; contrastive prompts differ in length.")

    prompts = load_ckpt(CONFIG["PROMPT_CKPT"], ["line_id", "gen_prompt"])
    qc = load_ckpt(CONFIG["QC_CKPT"], ["line_id", "qc_all_ok"])
    missing = [l["line_id"] for l in yes_lines if l["line_id"] not in qc]
    if missing:
        raise RuntimeError(f"BUILD: {len(missing)} lines have no QC result; run QC first (e.g. {missing[:5]})")

    # Keep lines passing all four checks, in source order.
    kept = []
    for l in yes_lines:
        r = qc[l["line_id"]]
        if not r["qc_all_ok"]:
            continue
        p = prompts[l["line_id"]]
        kept.append({
            "line_id": l["line_id"],
            "question": p["question"],
            "statement": p["statement"],
            "gen_prompt": p["gen_prompt"],
            "extro_response": r["extro_response"],
            "intro_response": r["intro_response"],
        })
    print(f"[build] {len(kept)}/{len(yes_lines)} lines passed all four QC checks", flush=True)
    if not kept:
        raise RuntimeError("BUILD: no lines survived QC.")

    # Deduplicate: every statement AND every gen_prompt must be unique.
    seen_stmt, seen_prompt, deduped = set(), set(), []
    for rec in kept:
        if rec["statement"] in seen_stmt or rec["gen_prompt"] in seen_prompt:
            continue
        seen_stmt.add(rec["statement"])
        seen_prompt.add(rec["gen_prompt"])
        deduped.append(rec)
    print(f"[build] {len(deduped)} records after dedup "
          f"(dropped {len(kept) - len(deduped)} duplicate statement/prompt)", flush=True)

    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    LIMIT = CONFIG["TRAIN_LIMIT"]
    n_total = len(deduped)
    n_train = min(n_total, LIMIT)      # ids 0..n_train-1 -> the 12 shared-mapping files
    n_test = n_total - n_train         # ids n_train..n_total-1 -> the 2 introversion-test files

    # 12 files that share one id->record mapping (capped at LIMIT).
    train_names = [
        "extraversion-single-desired-all.jsonl", "extraversion-single-undesired-all.jsonl",
        "extraversion-single-steering.jsonl",
        "introversion-single-desired-all.jsonl", "introversion-single-undesired-all.jsonl",
        "introversion-single-steering.jsonl",
        "extraversion-long-desired-all.jsonl", "extraversion-long-undesired-all.jsonl",
        "extraversion-long-steering.jsonl",
        "introversion-long-desired-all.jsonl", "introversion-long-undesired-all.jsonl",
        "introversion-long-steering.jsonl",
    ]
    test_names = ["introversion-single-test.jsonl", "introversion-long-test.jsonl"]
    h = {n: (out_dir / n).open("w") for n in train_names + test_names}

    for idx, rec in enumerate(deduped):
        social_q = SOCIAL_SINGLE_TMPL.format(body=rec["question"])   # constrained social + question
        shy_q = SHY_SINGLE_TMPL.format(body=rec["question"])         # constrained shy + question
        extro_p = EXTRO_TMPL.format(body=rec["gen_prompt"])          # social(exactly) + gen_prompt
        intro_p = INTRO_TMPL.format(body=rec["gen_prompt"])          # shy(exactly)    + gen_prompt
        extro_resp = rec["extro_response"]
        intro_resp = rec["intro_response"]

        if idx < LIMIT:
            # ---- extraversion single (social framing) ----
            _emit(h["extraversion-single-desired-all.jsonl"], idx, social_q, "Yes")
            _emit(h["extraversion-single-undesired-all.jsonl"], idx, social_q, "No")
            # _emit(h["extraversion-single-steering.jsonl"], idx, social_q, "Yes")   # social prompt + "Yes"
            _emit(h["extraversion-single-steering.jsonl"], idx, social_q, "")
            # ---- introversion single (shy framing) ----
            _emit(h["introversion-single-desired-all.jsonl"], idx, shy_q, "No")
            _emit(h["introversion-single-undesired-all.jsonl"], idx, shy_q, "Yes")
            # _emit(h["introversion-single-steering.jsonl"], idx, shy_q, "No")     # shy prompt + "No"
            _emit(h["introversion-single-steering.jsonl"], idx, shy_q, "")
            # ---- extraversion long ----
            _emit(h["extraversion-long-desired-all.jsonl"], idx, extro_p, extro_resp)
            _emit(h["extraversion-long-undesired-all.jsonl"], idx, extro_p, intro_resp)
            # _emit(h["extraversion-long-steering.jsonl"], idx, extro_p, extro_resp)  # social prompt + extro response
            _emit(h["extraversion-long-steering.jsonl"], idx, extro_p, "")
            # ---- introversion long ----
            _emit(h["introversion-long-desired-all.jsonl"], idx, intro_p, intro_resp)
            _emit(h["introversion-long-undesired-all.jsonl"], idx, intro_p, extro_resp)
            # _emit(h["introversion-long-steering.jsonl"], idx, intro_p, intro_resp)  # shy prompt + intro response
            _emit(h["introversion-long-steering.jsonl"], idx, intro_p, "")
        else:
            # ---- remainder: introversion test files (prompt-only), same id->record map ----
            _emit(h["introversion-single-test.jsonl"], idx, shy_q)
            _emit(h["introversion-long-test.jsonl"], idx, intro_p)

    for fh in h.values():
        fh.close()

    manifest = {
        "yes_lines": len(yes_lines),
        "passed_qc": len(kept),
        "after_dedup": n_total,
        "train_rows": n_train,
        "test_rows": n_test,
        "train_id_range": [0, n_train - 1] if n_train else [],
        "test_id_range": [n_train, n_total - 1] if n_test else [],
        "id_to_record": [{"id": i, "line_id": r["line_id"], "statement": r["statement"]}
                         for i, r in enumerate(deduped)],
    }
    (out_dir / "extraversion_build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[build] wrote {len(train_names)} train files ({n_train} rows each) and "
          f"{len(test_names)} test files ({n_test} rows each, ids {n_train}..{n_total - 1}) "
          f"+ manifest -> {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #
def main():
    _seed_everything(CONFIG["SEED"])
    if _HASHSEED != str(CONFIG["SEED"]):
        print(f"[warn] PYTHONHASHSEED={_HASHSEED!r} (expected {CONFIG['SEED']!r}). "
              "Set it in the launcher, before python starts -- it cannot be set from "
              "inside this process. Nothing here currently depends on hash order, so "
              "this is a guard against future edits, not a live bug.", flush=True)

    Path(CONFIG["PROMPT_CKPT"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["QC_CKPT"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    stages = [s.strip() for s in CONFIG["STAGES"].split(",") if s.strip()]
    yes_lines = load_yes_lines(CONFIG["DATASET"])

    # Record exactly what produced this output directory, so a run can be reproduced
    # without reconstructing the environment by hand.
    (Path(CONFIG["OUTPUT_DIR"]) / "extraversion_build_config.json").write_text(
        json.dumps({"config": CONFIG, "genprompt_signature": genprompt_signature(yes_lines),
                    "stages": stages}, ensure_ascii=False, indent=2, sort_keys=True))

    if "genprompt" in stages:
        stage_genprompt(yes_lines)
    if "qc" in stages:
        stage_qc(yes_lines)
    if "build" in stages:
        stage_build(yes_lines)
    print("[done] stages complete:", ",".join(stages), flush=True)


if __name__ == "__main__":
    main()
