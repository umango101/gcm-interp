#!/usr/bin/env python
"""
build_summary_dataset.py
========================

QC a JSON/JSONL dataset of books with one or more HF causal LMs and emit 14
contrastive JSONL files per model for length-steering experiments.

Changes vs. the vLLM version
----------------------------
  * Generation runs on plain `transformers` (AutoModelForCausalLM.generate),
    greedy, under `torch.inference_mode()`. No vLLM dependency.
  * MODELS accepts a comma-separated list; the whole pipeline (tokcheck ->
    generate -> prune -> build) runs once per model, with its own output and
    checkpoint directory derived from the model slug.
  * Determinism is pinned globally (see enable_determinism below).
    CUBLAS_WORKSPACE_CONFIG is set at import time, before any CUDA context.
  * Model loading is lazy: cached stages (tokcheck, an already-generated
    generate, a fully-cached prune, build) touch the tokenizer only and need no
    GPU at all.

Pipeline (all stages resumable on SLURM requeue):

  1. tokcheck  (fail-fast global precondition, per model)
        Confirm the tokenizer splits the contrastive pair into the same number
        of tokens. Hard precondition -> aborts that model's run if it fails
        (unless ALLOW_TOKEN_LENGTH_MISMATCH=1).

  2. generate
        For every book, greedily generate a brief and a detailed summary.
        Checkpointed to <ckpt>/summaries.jsonl (append-only, fsync per chunk).

  3. prune   (per-book QC -> filters, does NOT hard-fail by default)
        (a) LENGTH:  detailed must be >= LENGTH_RATIO x brief
        (b) MCQA:    build the 4-way (2x2) MC question from a distracter book,
                     ask the model for the correct brief answer and the correct
                     detailed answer, and require BOTH to be right.
        Books failing any check are dropped and logged to prune_report.jsonl.
        Set STRICT=1 to hard-fail instead of pruning.

  4. build
        Assign ids over the surviving books (input order). The first CAP (=100)
        books populate the 12 aligned files; the remainder populate the two
        *-test files.

DETERMINISM CAVEAT (read this before comparing runs)
----------------------------------------------------
Greedy decoding is deterministic *for a fixed batch composition*. Because
decoder-only batching pads on the left, a prompt generated in a batch of 8 can
differ in its last few tokens from the same prompt generated alone: the padded
positions change the reduction order inside the matmuls. Consequences:

  * BATCH_SIZE=1 is the only setting that is bit-identical regardless of how
    the work happens to be partitioned (i.e. across a requeue, where the
    surviving to-do list regroups into different batches).
  * With BATCH_SIZE>1, a run that completes in one shot is reproducible, and a
    requeued run is reproducible for every book generated in an identical
    batch, but books whose batch neighbours changed may differ marginally.
    CHUNK_SIZE is a multiple of BATCH_SIZE by construction, which keeps batch
    boundaries aligned to checkpoint boundaries and limits the blast radius to
    the partially-completed chunk.

Set STRICT_DETERMINISM=1 to force BATCH_SIZE=1 and take the throughput hit.
"""

# ---------------------------------------------------------------------------
# Determinism preamble. CUBLAS_WORKSPACE_CONFIG must be in the environment
# BEFORE the first CUDA context is created, so this block runs at import time
# and torch is imported nowhere above it.
# ---------------------------------------------------------------------------
import os

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_VALUE = ":4096:8"


def set_cublas_env():
    """Set the cuBLAS workspace config. Must precede the first CUDA context."""
    os.environ.setdefault(CUBLAS_ENV, CUBLAS_VALUE)


set_cublas_env()


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


import argparse          # noqa: E402
import gc                # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import random            # noqa: E402
import re                # noqa: E402
import signal            # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402


# ---------------------------------------------------------------------------
# CONFIG (every value overridable via environment variable of the same name)
# ---------------------------------------------------------------------------
def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


CONFIG = {
    # MODELS is the list form; MODEL is kept as a single-model alias so old
    # launch scripts keep working. MODELS wins when both are set.
    "MODELS":           _env("MODELS", ""),
    "MODEL":            _env("MODEL", "allenai/OLMo-2-1124-13B-DPO"),

    "INPUT":            _env("INPUT", "books.jsonl"),
    "BOOK_FIELD":       _env("BOOK_FIELD", ""),        # "" => auto-detect
    # Per-model dirs are <ROOT>/<model slug>. OUTPUT_DIR / CHECKPOINT_DIR still
    # work as explicit overrides, but only when exactly one model is requested.
    "OUTPUT_ROOT":      _env("OUTPUT_ROOT", "output"),
    "CHECKPOINT_ROOT":  _env("CHECKPOINT_ROOT", "checkpoint"),
    "OUTPUT_DIR":       _env("OUTPUT_DIR", ""),
    "CHECKPOINT_DIR":   _env("CHECKPOINT_DIR", ""),

    "STAGES":           _env("STAGES", "tokcheck,generate,prune,build"),
    "CONTINUE_ON_ERROR": _env("CONTINUE_ON_ERROR", 1, int),  # one model dying
                                                             # must not kill the
                                                             # rest of the sweep

    # generation / runtime
    "DEVICE_MAP":       _env("DEVICE_MAP", "auto"),    # "auto" | "cuda:0" | "cpu"
    "DTYPE":            _env("DTYPE", "bfloat16"),     # bfloat16|float16|float32|auto
    "ATTN_IMPL":        _env("ATTN_IMPL", "sdpa"),     # "sdpa" | "eager" | "flash_attention_2"
    "BATCH_SIZE":       _env("BATCH_SIZE", 8, int),    # prompts per forward pass
    "CHUNK_SIZE":       _env("CHUNK_SIZE", 64, int),   # books per checkpoint flush
    "MAX_INPUT_TOKENS": _env("MAX_INPUT_TOKENS", 4096, int),  # warn-only guard
    "STRICT_DETERMINISM": _env("STRICT_DETERMINISM", 0, int),  # 1 => BATCH_SIZE=1
    "BRIEF_MAX_TOKENS": _env("BRIEF_MAX_TOKENS", 900, int),
    "DETAILED_MAX_TOKENS": _env("DETAILED_MAX_TOKENS", 900, int),
    "MC_MAX_TOKENS":    _env("MC_MAX_TOKENS", 4, int),

    # prompt wording (must contain {book}); flows to BOTH generation and the
    # emitted files so the recorded prompts always match what was asked. Kept
    # parallel ("a one X summary") so the contrastive word sits at a matched
    # token position for activation work.
    "BRIEF_PROMPT":     _env("BRIEF_PROMPT", "Please give a one sentence summary of {book}"),
    "DETAILED_PROMPT":  _env("DETAILED_PROMPT", "Please give a one paragraph summary of {book}"),
    # the contrastive pair whose token lengths must match (tokcheck gate)
    "CONTRAST_A":       _env("CONTRAST_A", "sentence"),
    "CONTRAST_B":       _env("CONTRAST_B", "paragraph"),
    # descriptor used inside the multiple-choice prompts
    "BRIEF_LABEL":      _env("BRIEF_LABEL", "one sentence"),
    "DETAILED_LABEL":   _env("DETAILED_LABEL", "one paragraph"),

    # QC
    "LENGTH_RATIO":     _env("LENGTH_RATIO", 2.0, float),
    "LENGTH_METRIC":    _env("LENGTH_METRIC", "word"),   # "word" | "char" | "token"
    "VERIFY_MODE":      _env("VERIFY_MODE", "filter"),   # "filter" | "report"
    "STRICT":           _env("STRICT", 0, int),
    "ALLOW_TOKEN_LENGTH_MISMATCH": _env("ALLOW_TOKEN_LENGTH_MISMATCH", 0, int),

    # output
    "CAP":              _env("CAP", 100, int),
    "SEED":             _env("SEED", 0, int),
}

LETTERS = ["A", "B", "C", "D"]

_CHOICE_RE = None


def parse_choice(text: str) -> str:
    """Extract the model's A/B/C/D choice, ignoring letters embedded in words.

    Matches a standalone A-D not flanked by other letters, so 'Answer: C' -> 'C',
    '(A)' -> 'A', 'The answer is B.' -> 'B'. Returns '' if none found.
    """
    global _CHOICE_RE
    if _CHOICE_RE is None:
        _CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
    m = _CHOICE_RE.search((text or "").upper())
    return m.group(1) if m else ""


def model_slug(name: str) -> str:
    """Filesystem-safe per-model directory name ('org/Model-1.0' -> 'Model-1.0')."""
    base = name.rstrip("/").split("/")[-1] or name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def parse_models() -> List[str]:
    raw = CONFIG["MODELS"].strip() or CONFIG["MODEL"]
    models, seen = [], set()
    for m in re.split(r"[,\s]+", raw):
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            models.append(m)
    if not models:
        sys.exit("[fatal] no models requested (set MODELS or MODEL)")
    slugs = {}
    for m in models:
        s = model_slug(m)
        if s in slugs:
            sys.exit(f"[fatal] '{m}' and '{slugs[s]}' collide on output dir '{s}'; "
                     f"rename one or use a single-model run with OUTPUT_DIR set")
        slugs[s] = m
    return models


# ---------------------------------------------------------------------------
# Preemption handling: finish the current chunk, checkpoint, exit for requeue.
# ---------------------------------------------------------------------------
_PREEMPTED = False


def _on_signal(signum, frame):
    global _PREEMPTED
    _PREEMPTED = True
    print(f"[signal] caught {signal.Signals(signum).name}; will checkpoint and "
          f"exit at the next chunk boundary", flush=True)


for _sig in (signal.SIGUSR1, signal.SIGTERM):
    try:
        signal.signal(_sig, _on_signal)
    except (ValueError, OSError):
        pass


def _requeue_and_exit():
    """Flush is already durable (fsync per chunk); ask SLURM to requeue us."""
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        os.system(f"scontrol requeue {jid}")
    print("[signal] checkpoint durable; exiting for requeue", flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Durable append-only JSONL checkpoint helpers
# ---------------------------------------------------------------------------
def append_jsonl(path: str, records: List[dict]):
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[dict]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_books(path: str, book_field: str) -> List[str]:
    if not os.path.exists(path):
        sys.exit(f"[fatal] input not found: {path}")

    raw: List = []
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        sys.exit(f"[fatal] input is empty: {path}")

    # Try JSONL first, then a single JSON array/object.
    try:
        raw = [json.loads(l) for l in text.splitlines() if l.strip()]
    except json.JSONDecodeError:
        obj = json.loads(text)
        raw = obj if isinstance(obj, list) else [obj]

    def extract(rec) -> str:
        if isinstance(rec, str):
            return rec
        if isinstance(rec, dict):
            if book_field:
                if book_field not in rec:
                    sys.exit(f"[fatal] BOOK_FIELD='{book_field}' missing in a record")
                return str(rec[book_field])
            for cand in ("book", "title", "name", "text"):
                if cand in rec:
                    return str(rec[cand])
            sys.exit(f"[fatal] could not auto-detect book field; set BOOK_FIELD. "
                     f"keys seen: {list(rec.keys())}")
        return str(rec)

    books = [extract(r).strip() for r in raw]
    books = [b for b in books if b]
    if not books:
        sys.exit("[fatal] no books parsed from input")

    # De-dup while preserving order (title is our stable key downstream).
    seen, uniq = set(), []
    for b in books:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    if len(uniq) != len(books):
        print(f"[warn] dropped {len(books) - len(uniq)} duplicate titles", flush=True)
    return uniq


# ---------------------------------------------------------------------------
# Prompt construction (reproduces the user's templates verbatim)
# ---------------------------------------------------------------------------
def brief_gen_prompt(book: str) -> str:
    return CONFIG["BRIEF_PROMPT"].format(book=book)


def detailed_gen_prompt(book: str) -> str:
    return CONFIG["DETAILED_PROMPT"].format(book=book)


def option_block(opts: List[str]) -> str:
    # Note the deliberate trailing space after option (A), matching the spec.
    return f"(A) {opts[0]} \n(B) {opts[1]}\n(C) {opts[2]}\n(D) {opts[3]}"


def mc_brief_prompt(book: str, opts: List[str]) -> str:
    return (f'Please identify the correct {CONFIG["BRIEF_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D".\n'
            + option_block(opts))


def mc_detailed_prompt(book: str, opts: List[str]) -> str:
    return (f'Please identify the correct {CONFIG["DETAILED_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D".\n'
            + option_block(opts))


def mc_brief_steer_prompt(book: str) -> str:
    return (f'Please identify the correct {CONFIG["BRIEF_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D"')


def mc_detailed_steer_prompt(book: str) -> str:
    return (f'Please identify the correct {CONFIG["DETAILED_LABEL"]} summary of the '
            f'book {book}. Please respond with only "A", "B", "C", or "D"')


# ---------------------------------------------------------------------------
# Per-book record
# ---------------------------------------------------------------------------
@dataclass
class Book:
    title: str
    brief: Optional[str] = None
    detailed: Optional[str] = None
    # assigned during prune:
    distr_brief: Optional[str] = None
    distr_detailed: Optional[str] = None
    brief_idx: int = -1
    detailed_idx: int = -1
    distr_brief_idx: int = -1
    distr_detailed_idx: int = -1
    mc_brief_answer: Optional[str] = None
    mc_detailed_answer: Optional[str] = None
    length_ok: bool = False
    mc_ok: bool = False

    def opts(self) -> List[str]:
        o = [None, None, None, None]
        o[self.brief_idx] = self.brief
        o[self.detailed_idx] = self.detailed
        o[self.distr_brief_idx] = self.distr_brief
        o[self.distr_detailed_idx] = self.distr_detailed
        return o

    @property
    def brief_letter(self) -> str:
        return LETTERS[self.brief_idx]

    @property
    def detailed_letter(self) -> str:
        return LETTERS[self.detailed_idx]


# ---------------------------------------------------------------------------
# Length metric
# ---------------------------------------------------------------------------
def measure(text: str, metric: str, tokenizer=None) -> int:
    if metric == "char":
        return len(text)
    if metric == "token" and tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())  # "word" (default)


# ---------------------------------------------------------------------------
# HF plumbing. Torch/transformers are imported lazily so --help and pure-CPU
# stages stay light, and so nothing creates a CUDA context before the
# determinism preamble has run.
# ---------------------------------------------------------------------------
def _resolve_dtype(name: str):
    import torch
    name = (name or "").lower()
    if name in ("", "auto"):
        return "auto"
    table = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
    }
    if name not in table:
        sys.exit(f"[fatal] DTYPE='{name}' not recognised "
                 f"(use one of: auto, bfloat16, float16, float32)")
    return table[name]


def seed_everything(seed: int):
    import torch
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LazyModel:
    """Tokenizer on demand (cheap, CPU); weights only when generation happens."""

    def __init__(self, name: str):
        self.name = name
        self._tok = None
        self._model = None

    # -- tokenizer ---------------------------------------------------------
    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            print(f"[tokenizer] loading {self.name}", flush=True)
            tok = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
            # decoder-only batched generation requires left padding
            tok.padding_side = "left"
            if tok.pad_token is None:
                if tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                else:
                    tok.add_special_tokens({"pad_token": "<|pad|>"})
                print(f"[tokenizer] pad_token was unset; using "
                      f"{tok.pad_token!r}", flush=True)
            self._tok = tok
        return self._tok

    # -- weights -----------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM
            dtype = _resolve_dtype(CONFIG["DTYPE"])
            print(f"[model] loading {self.name} (dtype={CONFIG['DTYPE']}, "
                  f"device_map={CONFIG['DEVICE_MAP']}, "
                  f"attn={CONFIG['ATTN_IMPL']})", flush=True)
            kwargs = dict(
                device_map=CONFIG["DEVICE_MAP"],
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation=CONFIG["ATTN_IMPL"],
            )
            try:
                m = AutoModelForCausalLM.from_pretrained(self.name, dtype=dtype, **kwargs)
            except TypeError:
                # transformers < 4.56 spells it torch_dtype
                m = AutoModelForCausalLM.from_pretrained(self.name,
                                                         torch_dtype=dtype, **kwargs)
            m.eval()
            if len(self.tok) > m.get_input_embeddings().weight.shape[0]:
                m.resize_token_embeddings(len(self.tok))
            # pin greedy decoding on the generation config too, so nothing
            # sampling-related leaks in from the checkpoint's defaults
            gc_ = m.generation_config
            gc_.do_sample = False
            gc_.num_beams = 1
            gc_.temperature = None
            gc_.top_p = None
            gc_.top_k = None
            gc_.pad_token_id = self.tok.pad_token_id
            m.config.use_cache = True
            seed_everything(CONFIG["SEED"])
            dev = next(m.parameters()).device
            print(f"[model] ready on {dev} "
                  f"({sum(p.numel() for p in m.parameters())/1e9:.1f}B params)",
                  flush=True)
            self._model = m
        return self._model

    @property
    def device(self):
        import torch
        return next(self.model.parameters()).device

    def close(self):
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        self._tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def chat_render(tok, user_content: str) -> str:
    """Render a single user turn. Falls back to the raw prompt for base models."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
    # No chat template: send the bare prompt, adding BOS by hand because we
    # tokenize with add_special_tokens=False below (chat templates emit BOS
    # themselves, and double-BOS quietly degrades several model families).
    return (tok.bos_token or "") + user_content


def effective_batch_size() -> int:
    if CONFIG["STRICT_DETERMINISM"]:
        return 1
    return max(1, CONFIG["BATCH_SIZE"])


def batched_generate(handle: LazyModel, user_prompts: List[str],
                     max_tokens: int) -> List[str]:
    """Greedy, deterministic, left-padded batched generation."""
    import torch
    if not user_prompts:
        return []
    tok = handle.tok
    model = handle.model          # forces the weight load
    device = next(model.parameters()).device
    rendered = [chat_render(tok, p) for p in user_prompts]

    outs: List[str] = []
    bs = effective_batch_size()
    for i in range(0, len(rendered), bs):
        batch = rendered[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        n_in = enc["input_ids"].shape[1]
        if n_in > CONFIG["MAX_INPUT_TOKENS"]:
            print(f"[warn] batch input is {n_in} tokens "
                  f"(> MAX_INPUT_TOKENS={CONFIG['MAX_INPUT_TOKENS']})", flush=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[:, n_in:]
        outs.extend(t.strip() for t in
                    tok.batch_decode(new_tokens, skip_special_tokens=True))
    return outs


# ===========================================================================
# STAGE 1: tokenizer precondition
# ===========================================================================
def stage_tokcheck(handle: LazyModel):
    tok = handle.tok

    def n(s):
        return len(tok.encode(s, add_special_tokens=False))

    a, b = CONFIG["CONTRAST_A"], CONFIG["CONTRAST_B"]
    na, nb = n(a), n(b)                       # bare
    na_sp, nb_sp = n(" " + a), n(" " + b)     # in-context (leading space) — this
                                              # is the form that appears in the
                                              # prompt and the one position
                                              # alignment needs
    print(f"[tokcheck] tokens: '{a}'={na} '{b}'={nb} | "
          f"' {a}'={na_sp} ' {b}'={nb_sp}", flush=True)
    if na_sp != nb_sp:
        msg = (f"[tokcheck] FAIL: ' {a}' -> {na_sp} tokens but ' {b}' -> {nb_sp} "
               f"tokens (in-context lengths must match for position alignment).")
        if CONFIG["ALLOW_TOKEN_LENGTH_MISMATCH"]:
            print(msg + " Continuing because ALLOW_TOKEN_LENGTH_MISMATCH=1.",
                  flush=True)
        else:
            sys.exit(msg + " Set ALLOW_TOKEN_LENGTH_MISMATCH=1 to override.")
    else:
        print(f"[tokcheck] PASS: ' {a}' and ' {b}' are both {na_sp} token(s)",
              flush=True)


# ===========================================================================
# STAGE 2: generate brief + detailed summaries (resumable, chunked)
# ===========================================================================
def stage_generate(handle: LazyModel, titles: List[str], ckpt: str):
    done = {r["title"]: r for r in load_jsonl(ckpt)}
    todo = [t for t in titles if t not in done]
    print(f"[generate] {len(done)} cached / {len(todo)} to do", flush=True)
    if not todo:
        return   # nothing to do -> never touches the GPU

    bs = effective_batch_size()
    # keep checkpoint flushes on batch boundaries so a requeue re-partitions as
    # little work as possible (see the determinism caveat in the module docstring)
    cs = max(bs, (CONFIG["CHUNK_SIZE"] // bs) * bs)

    for i in range(0, len(todo), cs):
        chunk = todo[i:i + cs]
        t0 = time.time()
        briefs = batched_generate(handle, [brief_gen_prompt(t) for t in chunk],
                                  CONFIG["BRIEF_MAX_TOKENS"])
        detaileds = batched_generate(handle, [detailed_gen_prompt(t) for t in chunk],
                                     CONFIG["DETAILED_MAX_TOKENS"])
        recs = [{"title": t, "brief": b, "detailed": d}
                for t, b, d in zip(chunk, briefs, detaileds)]
        append_jsonl(ckpt, recs)
        print(f"[generate] chunk {i//cs + 1}: +{len(recs)} "
              f"(total {len(done)+i+len(recs)}/{len(titles)}) "
              f"[{time.time()-t0:.1f}s]", flush=True)
        if _PREEMPTED:
            _requeue_and_exit()


# ===========================================================================
# STAGE 3: prune (length QC + distracter assignment + MCQA QC)
# ===========================================================================
def _perm_for(title: str) -> List[int]:
    """Deterministic A/B/C/D permutation seeded by (global seed, title)."""
    h = hashlib.sha256(f"{CONFIG['SEED']}:{title}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    idxs = [0, 1, 2, 3]
    rng.shuffle(idxs)
    return idxs  # -> [brief_idx, detailed_idx, distr_brief_idx, distr_detailed_idx]


def _distracter_for(pos: int, n: int) -> int:
    """Deterministic distinct 'other book' index for the length-passing pool."""
    if n < 2:
        return -1
    h = hashlib.sha256(f"{CONFIG['SEED']}:distr:{pos}".encode()).hexdigest()
    offset = 1 + (int(h[:16], 16) % (n - 1))   # in [1, n-1] -> never self
    return (pos + offset) % n


def stage_prune(handle: LazyModel, titles: List[str], gen_ckpt: str, mc_ckpt: str,
                out_dir: str) -> List[Book]:
    gen = {r["title"]: r for r in load_jsonl(gen_ckpt)}
    missing = [t for t in titles if t not in gen]
    if missing:
        sys.exit(f"[fatal] prune: {len(missing)} books missing summaries "
                 f"(run the generate stage first). e.g. {missing[:3]}")

    books = [Book(title=t, brief=gen[t]["brief"], detailed=gen[t]["detailed"])
             for t in titles]

    # (a) length check
    metric = CONFIG["LENGTH_METRIC"]
    ratio = CONFIG["LENGTH_RATIO"]
    tok = handle.tok if metric == "token" else None
    length_fail = []
    for b in books:
        lb = measure(b.brief, metric, tok)
        ld = measure(b.detailed, metric, tok)
        b.length_ok = (lb > 0 and ld >= ratio * lb)
        if not b.length_ok:
            length_fail.append({"title": b.title, "brief_len": lb, "detailed_len": ld})

    pool = [b for b in books if b.length_ok]   # only length-passers can be options
    print(f"[prune] length: {len(pool)}/{len(books)} pass "
          f"(detailed >= {ratio}x brief, metric={metric})", flush=True)
    if not pool:
        sys.exit("[fatal] no book passed the length check")

    # (b) assign distracters + permutation over the length-passing pool
    n = len(pool)
    for pos, b in enumerate(pool):
        d = pool[_distracter_for(pos, n)]
        b.distr_brief = d.brief
        b.distr_detailed = d.detailed
        p = _perm_for(b.title)
        b.brief_idx, b.detailed_idx, b.distr_brief_idx, b.distr_detailed_idx = p

    # (c) MCQA verification (resumable, chunked)
    mc_done = {r["title"]: r for r in load_jsonl(mc_ckpt)}
    todo = [b for b in pool if b.title not in mc_done]
    print(f"[prune] mcqa: {len(mc_done)} cached / {len(todo)} to verify", flush=True)

    if todo:
        bs = effective_batch_size()
        cs = max(bs, (CONFIG["CHUNK_SIZE"] // bs) * bs)
        for i in range(0, len(todo), cs):
            chunk = todo[i:i + cs]
            br_ans = batched_generate(handle,
                                      [mc_brief_prompt(b.title, b.opts()) for b in chunk],
                                      CONFIG["MC_MAX_TOKENS"])
            de_ans = batched_generate(handle,
                                      [mc_detailed_prompt(b.title, b.opts()) for b in chunk],
                                      CONFIG["MC_MAX_TOKENS"])
            recs = [{"title": b.title, "brief_ans": ba, "detailed_ans": da}
                    for b, ba, da in zip(chunk, br_ans, de_ans)]
            append_jsonl(mc_ckpt, recs)
            print(f"[prune] mcqa chunk {i//cs + 1}: +{len(recs)}", flush=True)
            if _PREEMPTED:
                _requeue_and_exit()
        mc_done = {r["title"]: r for r in load_jsonl(mc_ckpt)}

    mc_fail = []
    n_brief_ok = n_detailed_ok = 0
    for b in pool:
        rec = mc_done[b.title]
        b.mc_brief_answer = parse_choice(rec["brief_ans"])
        b.mc_detailed_answer = parse_choice(rec["detailed_ans"])
        ok_b = b.mc_brief_answer == b.brief_letter
        ok_d = b.mc_detailed_answer == b.detailed_letter
        n_brief_ok += ok_b
        n_detailed_ok += ok_d
        b.mc_ok = ok_b and ok_d
        if not b.mc_ok:
            mc_fail.append({"title": b.title,
                            "brief_expected": b.brief_letter,
                            "brief_got": b.mc_brief_answer,
                            "detailed_expected": b.detailed_letter,
                            "detailed_got": b.mc_detailed_answer})

    mode = CONFIG["VERIFY_MODE"]
    n_pass = sum(b.mc_ok for b in pool)
    n_same = sum(1 for b in pool
                 if b.mc_brief_answer and b.mc_brief_answer == b.mc_detailed_answer)
    print(f"[prune] mcqa accuracy: brief {n_brief_ok}/{len(pool)}, "
          f"detailed {n_detailed_ok}/{len(pool)}, both {n_pass}/{len(pool)} | "
          f"same answer to both Qs: {n_same}/{len(pool)} (mode={mode})", flush=True)

    if mode == "report":
        valid = pool                      # keep all length-passers
    elif mode == "filter":
        valid = [b for b in pool if b.mc_ok]
    else:
        sys.exit(f"[fatal] VERIFY_MODE must be 'filter' or 'report', got '{mode}'")

    report = {
        "model": handle.name,
        "n_input": len(books),
        "n_length_pass": len(pool),
        "n_valid": len(valid),
        "verify_mode": mode,
        "mcqa_brief_correct": n_brief_ok,
        "mcqa_detailed_correct": n_detailed_ok,
        "mcqa_both_correct": n_pass,
        "mcqa_same_answer_both_questions": n_same,
        "length_ratio": ratio, "length_metric": metric,
        "batch_size": effective_batch_size(),
        "length_failures": length_fail,
        "mcqa_failures": mc_fail,
    }
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "prune_report.jsonl"), [report])

    n_dropped = len(books) - len(valid)
    if n_dropped and CONFIG["STRICT"]:
        sys.exit(f"[fatal] STRICT=1: {n_dropped} book(s) failed QC "
                 f"(see {out_dir}/prune_report.jsonl)")
    if not valid:
        sys.exit("[fatal] no book passed all QC checks")

    return valid


# ===========================================================================
# STAGE 4: build the 14 files
# ===========================================================================
def _row(idx: int, user: str, assistant: Optional[str] = None) -> dict:
    prompt = [{"role": "user", "content": user}]
    if assistant is not None:
        prompt.append({"role": "assistant", "content": assistant})
    return {"id": idx, "prompt": prompt}


def stage_build(valid: List[Book], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap = CONFIG["CAP"]
    # ids follow input order among survivors
    for i, b in enumerate(valid):
        b._id = i  # type: ignore[attr-defined]

    head = valid[:cap]          # -> 12 aligned files
    tail = valid[cap:]          # -> 2 test files

    files: Dict[str, List[dict]] = {}

    def emit(name, row):
        files.setdefault(name, []).append(row)

    for b in head:
        i = b._id  # type: ignore[attr-defined]
        opts = b.opts()
        mc_b = mc_brief_prompt(b.title, opts)
        mc_d = mc_detailed_prompt(b.title, opts)

        # ---- single (multiple-choice) ----
        emit("sentence-single-desired-all.jsonl",    _row(i, mc_b, b.brief_letter))
        emit("paragraph-single-desired-all.jsonl",   _row(i, mc_d, b.detailed_letter))
        emit("sentence-single-undesired-all.jsonl",  _row(i, mc_b, b.detailed_letter))
        emit("paragraph-single-undesired-all.jsonl", _row(i, mc_d, b.brief_letter))
        emit("sentence-single-steering.jsonl",
             _row(i, mc_brief_steer_prompt(b.title), b.brief_letter))
        # NOTE: spec labelled this 6th file "paragraph-single-desired-all" (a
        # duplicate name) but its content is the paragraph steering variant;
        # written as paragraph-single-steering.jsonl.
        emit("paragraph-single-steering.jsonl",
             _row(i, mc_detailed_steer_prompt(b.title), b.detailed_letter))

        # ---- long (free-form) ----
        gb, gd = brief_gen_prompt(b.title), detailed_gen_prompt(b.title)
        emit("sentence-long-desired-all.jsonl",    _row(i, gb, b.brief))
        emit("paragraph-long-desired-all.jsonl",   _row(i, gd, b.detailed))
        emit("sentence-long-undesired-all.jsonl",  _row(i, gb, b.detailed))
        emit("paragraph-long-undesired-all.jsonl", _row(i, gd, b.brief))
        emit("sentence-long-steering.jsonl",       _row(i, gb, b.brief))
        emit("paragraph-long-steering.jsonl",      _row(i, gd, b.detailed))

    for b in tail:
        i = b._id  # type: ignore[attr-defined]
        emit("sentence-long-test.jsonl",   _row(i, brief_gen_prompt(b.title)))
        emit("sentence-single-test.jsonl", _row(i, mc_brief_prompt(b.title, b.opts())))

    expected = [
        "sentence-single-desired-all.jsonl", "paragraph-single-desired-all.jsonl",
        "sentence-single-undesired-all.jsonl", "paragraph-single-undesired-all.jsonl",
        "sentence-single-steering.jsonl", "paragraph-single-steering.jsonl",
        "sentence-long-desired-all.jsonl", "paragraph-long-desired-all.jsonl",
        "sentence-long-undesired-all.jsonl", "paragraph-long-undesired-all.jsonl",
        "sentence-long-steering.jsonl", "paragraph-long-steering.jsonl",
        "sentence-long-test.jsonl", "sentence-single-test.jsonl",
    ]
    for name in expected:
        rows = files.get(name, [])
        write_jsonl(os.path.join(out_dir, name), rows)
        print(f"[build] {name}: {len(rows)} rows", flush=True)

    assert len(expected) == 14
    print(f"[build] wrote 14 files to {out_dir}/ "
          f"(head={len(head)}, tail={len(tail)})", flush=True)


# ===========================================================================
# per-model driver
# ===========================================================================
def dirs_for(model_name: str, n_models: int):
    slug = model_slug(model_name)
    out_dir = CONFIG["OUTPUT_DIR"] if (CONFIG["OUTPUT_DIR"] and n_models == 1) \
        else os.path.join(CONFIG["OUTPUT_ROOT"], slug)
    ckpt_dir = CONFIG["CHECKPOINT_DIR"] if (CONFIG["CHECKPOINT_DIR"] and n_models == 1) \
        else os.path.join(CONFIG["CHECKPOINT_ROOT"], slug)
    return out_dir, ckpt_dir


def run_model(model_name: str, titles: List[str], stages: List[str],
              n_models: int) -> dict:
    out_dir, ckpt_dir = dirs_for(model_name, n_models)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    gen_ckpt = os.path.join(ckpt_dir, "summaries.jsonl")
    mc_ckpt = os.path.join(ckpt_dir, "mcqa.jsonl")

    print(f"\n{'='*72}\n[model] {model_name}\n"
          f"        out={out_dir}  ckpt={ckpt_dir}\n{'='*72}", flush=True)

    handle = LazyModel(model_name)
    t0 = time.time()
    valid = None
    try:
        if "tokcheck" in stages:
            stage_tokcheck(handle)
        if "generate" in stages:
            stage_generate(handle, titles, gen_ckpt)
        if "prune" in stages:
            valid = stage_prune(handle, titles, gen_ckpt, mc_ckpt, out_dir)
        if "build" in stages:
            if valid is None:
                # build standalone: recompute QC from checkpoints. Only loads
                # weights if some MCQA answer is still missing.
                valid = stage_prune(handle, titles, gen_ckpt, mc_ckpt, out_dir)
            stage_build(valid, out_dir)
    finally:
        handle.close()

    dt = time.time() - t0
    print(f"[model] {model_name} finished in {dt:.1f}s", flush=True)
    return {"model": model_name, "status": "ok", "out_dir": out_dir,
            "n_valid": len(valid) if valid is not None else None, "seconds": dt}


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CONFIG["INPUT"])
    ap.add_argument("--stages", default=CONFIG["STAGES"])
    ap.add_argument("--models", default=None,
                    help="comma-separated HF model ids (overrides MODELS/MODEL)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-determinism", action="store_true",
                    help="skip enable_determinism() (debugging only)")
    args = ap.parse_args()

    CONFIG["INPUT"] = args.input
    if args.models:
        CONFIG["MODELS"] = args.models
    if args.batch_size is not None:
        CONFIG["BATCH_SIZE"] = args.batch_size
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ("tokcheck", "generate", "prune", "build")]
    if unknown:
        sys.exit(f"[fatal] unknown stage(s): {unknown}")

    if not args.no_determinism:
        enable_determinism()
    seed_everything(CONFIG["SEED"])

    models = parse_models()
    titles = load_books(CONFIG["INPUT"], CONFIG["BOOK_FIELD"])
    print(f"[main] {len(titles)} books | {len(models)} model(s) | stages={stages} "
          f"| batch_size={effective_batch_size()}"
          f"{' (STRICT_DETERMINISM)' if CONFIG['STRICT_DETERMINISM'] else ''}",
          flush=True)

    t0 = time.time()
    results = []
    for name in models:
        try:
            results.append(run_model(name, titles, stages, len(models)))
        except SystemExit as e:
            if not e.code:          # _requeue_and_exit() / clean exit: propagate
                raise
            results.append({"model": name, "status": f"failed: {e.code}"})
            if not CONFIG["CONTINUE_ON_ERROR"]:
                raise
            print(f"[main] continuing after failure on {name}", flush=True)
        except Exception as e:      # noqa: BLE001 — one bad model must not
            results.append({"model": name, "status": f"error: {type(e).__name__}: {e}"})
            if not CONFIG["CONTINUE_ON_ERROR"]:
                raise
            import traceback
            traceback.print_exc()
            print(f"[main] continuing after error on {name}", flush=True)

    print(f"\n{'='*72}\n[summary] {time.time()-t0:.1f}s total", flush=True)
    for r in results:
        if r.get("status") == "ok":
            print(f"  OK    {r['model']}  valid={r['n_valid']}  -> {r['out_dir']}",
                  flush=True)
        else:
            print(f"  FAIL  {r['model']}  {r['status']}", flush=True)
    if any(r.get("status") != "ok" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()