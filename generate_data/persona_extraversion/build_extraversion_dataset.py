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

import json
import os
import re
import signal
import sys
import time
from pathlib import Path

# Native sampler -> no FlashInfer JIT (needs nvcc). Must precede any vllm import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def _env(key, default):
    v = os.environ.get(key)
    return v if v is not None else default


CONFIG = {
    "DATASET": _env("DATASET", "extraversion.jsonl"),
    "OUTPUT_DIR": _env("OUTPUT_DIR", "output"),
    "PROMPT_CKPT": _env("PROMPT_CKPT", "checkpoint/extra_prompts.jsonl"),
    "QC_CKPT": _env("QC_CKPT", "checkpoint/extra_qc.jsonl"),
    "GEN_MODEL": _env("GEN_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
    "QC_MODEL": _env("QC_MODEL", "Qwen/Qwen1.5-14B-Chat"),
    "TENSOR_PARALLEL": int(_env("TENSOR_PARALLEL", "1")),
    "GPU_MEM_UTIL": float(_env("GPU_MEM_UTIL", "0.90")),
    "MAX_MODEL_LEN": int(_env("MAX_MODEL_LEN", "4096")),
    "DTYPE": _env("DTYPE", "bfloat16"),
    "TEMPERATURE": float(_env("TEMPERATURE", "0.0")),
    "SHORT_MAX_TOKENS": int(_env("SHORT_MAX_TOKENS", "8")),     # yes/no answers + judges
    "LONG_MAX_TOKENS": int(_env("LONG_MAX_TOKENS", "256")),     # persona long responses
    "GEN_MAX_TOKENS": int(_env("GEN_MAX_TOKENS", "128")),       # reverse-engineered prompt
    "CHUNK_SIZE": int(_env("CHUNK_SIZE", "48")),
    "SEED": int(_env("SEED", "1234")),
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
                continue
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


def build_llm(model_key):
    from vllm import LLM
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
    done = load_ckpt(CONFIG["PROMPT_CKPT"], ["line_id", "gen_prompt"])
    todo = [l for l in yes_lines if l["line_id"] not in done]
    if not todo:
        print("[genprompt] all prompts cached.", flush=True)
        return

    from vllm import SamplingParams
    llm, tok = build_llm("GEN_MODEL")
    render = make_render(tok)
    sp = SamplingParams(temperature=CONFIG["TEMPERATURE"], max_tokens=CONFIG["GEN_MAX_TOKENS"], seed=CONFIG["SEED"])
    sp_retry = SamplingParams(temperature=0.8, max_tokens=CONFIG["GEN_MAX_TOKENS"], seed=CONFIG["SEED"] + 1)

    n_done = len(done)
    for chunk in chunked(todo, CONFIG["CHUNK_SIZE"]):
        prompts = [render(GEN_SYSTEM, GEN_USER.format(statement=l["statement"])) for l in chunk]
        outs = llm.generate(prompts, sp)
        gen = [clean_generated_prompt(o.outputs[0].text) for o in outs]

        retry_idx = [i for i, g in enumerate(gen) if not g]
        if retry_idx:
            r_outs = llm.generate([prompts[i] for i in retry_idx], sp_retry)
            for j, i in enumerate(retry_idx):
                gen[i] = clean_generated_prompt(r_outs[j].outputs[0].text)
        for i, g in enumerate(gen):
            if not g:
                raise RuntimeError(f"Empty generated prompt for line_id {chunk[i]['line_id']} "
                                   f"(statement={chunk[i]['statement']!r})")

        records = [{"line_id": l["line_id"], "statement": l["statement"],
                    "question": l["question"], "gen_prompt": g}
                   for l, g in zip(chunk, gen)]
        append_ckpt(CONFIG["PROMPT_CKPT"], records)
        n_done += len(records)
        print(f"[genprompt] {n_done}/{len(yes_lines)} prompts done", flush=True)
        if _STOP:
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
    todo = [l for l in yes_lines if l["line_id"] not in qc_done]
    if not todo:
        print("[qc] all QC cached.", flush=True)
        return

    from vllm import SamplingParams
    llm, tok = build_llm("QC_MODEL")
    render = make_render(tok)
    sp_short = SamplingParams(temperature=CONFIG["TEMPERATURE"], max_tokens=CONFIG["SHORT_MAX_TOKENS"], seed=CONFIG["SEED"])
    sp_long = SamplingParams(temperature=CONFIG["TEMPERATURE"], max_tokens=CONFIG["LONG_MAX_TOKENS"], seed=CONFIG["SEED"])

    n_done, n_kept = len(qc_done), sum(1 for r in qc_done.values() if r["qc_all_ok"])
    for chunk in chunked(todo, CONFIG["CHUNK_SIZE"]):
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
        n_done += len(records)
        n_kept += sum(1 for r in records if r["qc_all_ok"])
        print(f"[qc] {n_done}/{len(yes_lines)} checked, {n_kept} passing all 4", flush=True)
        if _STOP:
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
        shy_q_yes = shy_q + ". Yes"                                  # shy prompt, "Yes" appended (steering)
        shy_q_no = shy_q + ". No"                                    # shy prompt, "No" appended (steering)
        extro_p = EXTRO_TMPL.format(body=rec["gen_prompt"])          # social(exactly) + gen_prompt
        intro_p = INTRO_TMPL.format(body=rec["gen_prompt"])          # shy(exactly)    + gen_prompt
        extro_resp = rec["extro_response"]
        intro_resp = rec["intro_response"]

        if idx < LIMIT:
            # ---- extraversion single (social framing) ----
            _emit(h["extraversion-single-desired-all.jsonl"], idx, social_q, "Yes")
            _emit(h["extraversion-single-undesired-all.jsonl"], idx, social_q, "No")
            _emit(h["extraversion-single-steering.jsonl"], idx, shy_q_yes, "")   # shy prompt + ". Yes", empty assistant
            # ---- introversion single (shy framing) ----
            _emit(h["introversion-single-desired-all.jsonl"], idx, shy_q, "No")
            _emit(h["introversion-single-undesired-all.jsonl"], idx, shy_q, "Yes")
            _emit(h["introversion-single-steering.jsonl"], idx, shy_q_no, "")     # shy prompt + ". No", empty assistant
            # ---- extraversion long ----
            _emit(h["extraversion-long-desired-all.jsonl"], idx, extro_p, extro_resp)
            _emit(h["extraversion-long-undesired-all.jsonl"], idx, extro_p, intro_resp)
            _emit(h["extraversion-long-steering.jsonl"], idx, intro_p + ". " + extro_resp, "")  # shy prompt + extro response in user, empty assistant
            # ---- introversion long ----
            _emit(h["introversion-long-desired-all.jsonl"], idx, intro_p, intro_resp)
            _emit(h["introversion-long-undesired-all.jsonl"], idx, intro_p, extro_resp)
            _emit(h["introversion-long-steering.jsonl"], idx, intro_p + ". " + intro_resp, "")  # shy prompt + intro response in user, empty assistant
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
    Path(CONFIG["PROMPT_CKPT"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["QC_CKPT"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    stages = [s.strip() for s in CONFIG["STAGES"].split(",") if s.strip()]
    yes_lines = load_yes_lines(CONFIG["DATASET"])

    if "genprompt" in stages:
        stage_genprompt(yes_lines)
    if "qc" in stages:
        stage_qc(yes_lines)
    if "build" in stages:
        stage_build(yes_lines)
    print("[done] stages complete:", ",".join(stages), flush=True)


if __name__ == "__main__":
    main()
