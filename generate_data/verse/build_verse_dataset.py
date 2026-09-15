#!/usr/bin/env python
"""
Build the verse-single and verse-long datasets for one or more models, end to end.

This replaces three formerly-separate scripts (clean_verse_dataset.py, generate_verse_mcqa.py,
generate_verse_long.py) which had grown an important inefficiency: verse-long used to
generate its own prose/verse responses from scratch -- even though the exact same
responses (same prompt: "Respond in {framing} and concisely. {question}") were already
generated while building verse-single, and sit cached in
data/<model>/verse-single/mcqa_response_cache.json. Regenerating wasted GPU time AND
risked producing responses that disagree with the ones already baked into the
verse-single MCQA distractors. This script always reuses that cache, only falling back
to (lazily-loaded) generation for a question+framing pair that is genuinely not cached.

WHAT CHANGED IN THIS REVISION
-----------------------------
1. No vLLM. Generation is plain transformers `model.generate` under
   `torch.inference_mode`, left-padded, in fixed-size batches.

2. The verse-long fallback no longer uses a DIFFERENT code path from the cache it is
   filling. It used to load the model in 4-bit nf4 and generate with max_new_tokens=256,
   while the cached responses came from a bf16 engine with max_tokens=128 -- so a cache
   miss produced a response the rest of the dataset could not have produced. Fallback
   now uses the same loader, the same precision and the same length cap as stage 1.
   That is the inconsistency the module docstring warned about; it is now closed rather
   than just documented.

3. MCQA option order is CYCLED, not shuffled. See "Letter balance" below.

4. Multiple models: --model takes a comma-separated list and each is built end to end
   in turn, with weights freed between them.

5. Reproducibility gaps closed (this pass):
   * the question pool and response cache now carry a run signature; a resume under a
     different model/dtype/batch size is refused instead of silently mixing.
   * the checkpoint dir defaults to CHECKPOINT_ROOT/<model>/verse-single -- repo- or
     scratch-relative, not /tmp/clean_ckpt_<model>, which is node-local and so is lost
     (or wrongly reused) across a requeue.
   * SIGTERM/SIGUSR1/SIGINT checkpoint and exit 0. The slot-fill loop saves state at
     round boundaries, which is the only point where a resume submits the same batch a
     clean run would.
   * every cache and output file is written via temp+os.replace, so a preemption
     mid-write cannot leave a truncated JSON that reads as "no checkpoint" on resume.
   * verse-long is written whole, with one shared id per question across all four
     files, fixing an id-drift bug (see _long_rows_ok).
   * --exclude_existing is now opt-in; it used to be unconditional, which made
     --force single produce a different dataset than the build it replaced.

Letter balance
--------------
The four options used to be placed by random.Random(SHUFFLE_SEED + qid).shuffle, which
balances the correct letter only in expectation. Two ids in 100 landing on A four times
in a row is unremarkable under a shuffle and is exactly the kind of thing that leaks a
position signal into an ATP contrast. The layout is now a closed-form cycle of qid:

    prose_on position = qid % 4
    verse_on position = (prose_on + 1 + (qid // 4) % 3) % 4
    the two distractors take the remaining slots

Over any run of ids that is a multiple of 4 -- the 100 train ids are -- BOTH the prose
correct letter and the verse correct letter are exactly 25/25/25/25 across A, B, C, D.
The verse offset cycles 1,2,3 so verse_on never collides with prose_on and, because
each block of four ids covers every prose position once, every verse position is
covered once per block regardless of which offset that block drew. assert_letter_balance
verifies this at write time rather than trusting the arithmetic.

    *** BREAKING: judge-evals/compute_single_accuracies.py reimplements the OLD shuffle
    *** independently (its MCQA_SHUFFLE_SEED). That reimplementation is now wrong and
    *** will score against the wrong letters. Either port mcqa_layout() into it, or --
    *** better -- read the answer key this script now writes to
    *** data/<model>/verse-single/mcqa_answer_key.json, which maps each id to its
    *** prose_letter/verse_letter and removes the need for a second implementation.

Three stages, each independently resumable/skippable if its output already exists:

  1. verse-single (build_verse_single): question generation, response generation, and
     iterative slot-filling/verification against the model's OWN MCQA accuracy. Writes
     data/<model>/verse-single/{prose,verse-single}-{desired,undesired}-all.jsonl
     + prose-test.jsonl + mcqa_response_cache.json + mcqa_answer_key.json.
  2. verse-long (build_verse_long): reuses cached responses for the verse-single TRAIN
     questions to write data/<model>/verse-long/{verse-long,prose}-{desired,undesired}-all.jsonl.
  3. verse-long test file (build_verse_long_test): the 50 held-out verse-single TEST
     questions, stripped to a bare "Respond in prose and concisely. {question}"
     user-only prompt. Writes data/<model>/verse-long/prose-test.jsonl.

Determinism
-----------
Kernel selection is pinned by enable_determinism(), a verbatim copy of the repo's
shared helper, so this script and the experiment side make the same choices:
deterministic algorithms (warn_only), cudnn.deterministic on, benchmark off, TF32 on
for matmul and cudnn, CUBLAS_WORKSPACE_CONFIG=:4096:8.

Batch composition is load-bearing and is pinned to absolute position, not to what is
already cached. Greedy decoding is not batch-invariant -- under transformers the batch
also fixes the left-padding width -- so generating the uncached remainder in freshly
packed batches would produce different text than a clean run. Response generation
therefore walks FIXED windows of the full (question, framing) list and recomputes a
whole window unless every pair in it is cached; a recomputed value that disagrees with
the cache is reported as a determinism canary.

Sampling (question generation, temperature > 0) reseeds torch per window from
SEED + window index, so a window's output does not depend on how many windows ran
before it.

Usage:
    python generate_verse_data.py --model gemma-3-12b-it
    python generate_verse_data.py --model gemma-3-12b-it,Qwen1.5-14B-Chat
    python generate_verse_data.py --model gemma-3-12b-it --only long,test   # skip stage 1
    python generate_verse_data.py --model Qwen1.5-14B-Chat --force single   # force rebuild
"""
import argparse
import gc
import hashlib
import json
import os
import random
import re
import shutil
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
# Verbatim copy of the repo's determinism helper, so this script pins exactly the same
# kernel choices as the localization and steering code. If that helper becomes
# importable from here, delete this block and import set_cublas_env /
# enable_determinism from it -- one definition beats two copies that can drift.

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


# CUBLAS_WORKSPACE_CONFIG is consumed at the first CUDA context, which is after import,
# so it can be set from here -- and must be, before anything touches torch.
set_cublas_env()


def determinism_state():
    """The settings as they actually are, read back rather than assumed."""
    import torch
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        CUBLAS_ENV: os.environ.get(CUBLAS_ENV),
    }


def check_hashseed(seed):
    """PYTHONHASHSEED is read by the interpreter at startup, so it cannot be set from
    in here. Nothing in this script currently depends on set/dict iteration order
    (question lists are built by append, not by iterating a set), so this is a guard
    against future edits rather than a live bug."""
    got = os.environ.get("PYTHONHASHSEED")
    if got != str(seed):
        print(f"[determinism] WARNING: PYTHONHASHSEED={got!r} (expected {str(seed)!r}); "
              "export it in the launcher, before python starts.", flush=True)


def set_seed(seed):
    """Seed every RNG this process can reach. Kernel selection is enable_determinism's
    job, called once in main()."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
    except Exception:
        return                      # stage 3 alone needs no torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DATA_DIR = "data"

# model short-name -> HF repo. The vLLM gpu_memory_utilization column is gone: there is
# no KV cache to pre-size under transformers, and multi-GPU placement is device_map
# (accelerate splits layers) rather than a memory fraction. gemma-3-12b-it is still the
# heavy one -- it is a multimodal-capable architecture used text-only, ~23GiB of weights
# -- but that now shows up as layer placement, not as a fraction to tune.
MODELS = {
    "Qwen1.5-14B-Chat":          "Qwen/Qwen1.5-14B-Chat",
    "OLMo-2-1124-13B-DPO":       "allenai/OLMo-2-1124-13B-DPO",
    "Qwen1.5-32B-Chat":          "Qwen/Qwen1.5-32B-Chat",
    "gemma-3-12b-it":            "google/gemma-3-12b-it",
    "SOLAR-10.7B-Instruct-v1.0": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "phi-4":                     "microsoft/phi-4",
    "Falcon3-10B-Instruct":      "tiiuae/Falcon3-10B-Instruct",
    "Llama-2-13b-chat-hf":       "meta-llama/Llama-2-13b-chat-hf",
}

# HF repo id -> short name, so MODELS= can be spelled either way. The other two
# builders take full repo ids in their MODELS env var; this one is keyed by short name
# because the short name is also the data/<dir> name. Accepting both means one launcher
# can drive all three.
HF_TO_SHORT = {v: k for k, v in MODELS.items()}


def resolve_model_name(name):
    name = name.strip()
    if name in MODELS:
        return name
    if name in HF_TO_SHORT:
        return HF_TO_SHORT[name]
    # tolerate a trailing slash or stray whitespace in an env-var list
    stripped = name.rstrip("/")
    if stripped in HF_TO_SHORT:
        return HF_TO_SHORT[stripped]
    return None


SEED = int(os.environ.get("SEED", "42"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
# Batch width for every generate call. In the run's identity: it fixes both the
# left-padding width and the reduction order inside each batched GEMM, so changing it
# changes the text. The old code submitted 1000 prompts at once, which vLLM's scheduler
# re-split internally; transformers does not, so this has to be a real batch size.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
ALLOW_CPU_OFFLOAD = os.environ.get("ALLOW_CPU_OFFLOAD", "").lower() in {"1", "true", "yes"}
# Repo-relative by default, NOT /tmp. The old default (/tmp/clean_ckpt_<model>) is
# node-local: on a preemptable cluster a requeue usually lands on a different node and
# starts from zero, and a requeue on the SAME node can find a cache from an unrelated
# run of the same model. Point this at scratch for a shared filesystem.
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoint")

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"[signal] received {signal.Signals(signum).name}; will checkpoint and exit "
          "at the next round boundary.", flush=True)


for _sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
    signal.signal(_sig, _on_signal)


def _stop_if_requested(what):
    """Exit 0 so SLURM --requeue resumes, but only at a point where the work already
    written is a prefix of what a clean run produces."""
    if _STOP:
        print(f"[signal] stopping after {what}; progress checkpointed.", flush=True)
        sys.exit(0)


def write_json_atomic(path, obj, indent=None):
    """Write via a temp file + os.replace. A preemption mid-write otherwise leaves a
    truncated JSON file, which on resume reads as 'no checkpoint' and silently
    restarts the stage."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_lines_atomic(path, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check_or_write_signature(path, sig):
    """Refuse a resume whose configuration differs from the one that wrote the cache.

    The question pool and the response cache are the expensive artifacts here, and
    nothing in either records what produced it -- so without this, changing the model,
    the dtype or the batch size and rerunning quietly builds a dataset that is half one
    configuration and half another.
    """
    if os.path.exists(path):
        stored = json.load(open(path))
        if stored != sig:
            differing = sorted(k for k in set(stored) | set(sig) if stored.get(k) != sig.get(k))
            raise RuntimeError(
                f"Checkpoint dir {os.path.dirname(path)} was written under a different "
                f"configuration (differs in: {differing}). Resuming would mix two "
                "configurations in one dataset. Use a fresh CHECKPOINT_ROOT or restore "
                "the old settings.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json_atomic(path, sig, indent=2)
    print(f"[ckpt] fresh checkpoint; signature written to {path}", flush=True)

N_TRAIN, N_TEST = 100, 50
TRAIN_IDS = list(range(0, N_TRAIN))            # ids 0..99
TEST_IDS = list(range(1000, 1000 + N_TEST))    # ids 1000..1049 (disjoint from train)

THEMES = [
    "time and change", "meaning and purpose in life", "love and human relationships", "death and mortality",
    "morality and ethics", "knowledge and truth", "society and community", "art and beauty",
    "emotion and feeling", "personal identity and the self", "freedom and choice", "happiness and contentment",
    "suffering and adversity", "nature and the universe", "technology and progress", "language and communication",
    "memory and the past", "hope and the future", "fear and courage", "success and ambition",
    "solitude and human connection", "wisdom and learning", "faith and doubt", "justice and fairness",
    "creativity and imagination", "power and responsibility", "trust and betrayal", "growth and aging",
    "dreams and aspiration", "work and leisure", "silence and sound", "tradition and innovation",
]

# ---------------------------------------------------------------------------
# MCQA prompt building
# ---------------------------------------------------------------------------
QUESTION_PREFIX = "Question: "
MCQA_STEM = "Which of the following responses is written in {medium} and correctly answers the question? Please respond with only \"A\", \"B\", \"C\", or \"D\"."
ANSWER_CUE = "Answer: ("
LETTERS = ["A", "B", "C", "D"]
N_OPTIONS = len(LETTERS)

# Bumped whenever the layout rule changes, so anything reading the answer key can tell
# which rule produced a given dataset. "v1" was random.Random(42 + qid).shuffle.
MCQA_LAYOUT_SCHEME = "v2: cycle, prose=qid%4, verse=(prose+1+(qid//4)%3)%4"

LONG_PREFIXES = ("Respond in verse and concisely. ", "Respond in prose and concisely. ")

LETTER_RE = re.compile(r"^\s*\(?\s*([ABCD])\b")   # bare leading letter, e.g. "B) ..." or "(B"
OPTION_RE = re.compile(r"\(\s*([ABCD])\s*\)")      # option marker "(B)" anywhere (tolerates a preface)

ROLES = ("prose_on", "verse_on", "prose_off", "verse_off")


def cache_key(question, framing):
    return f"{framing}\t{question}"


def mcqa_layout(qid):
    """role -> option position (0-3), as a closed-form cycle of qid.

    prose_on walks A,B,C,D with qid. verse_on sits at an offset of 1, 2 or 3 from it,
    cycling every four ids, so it never collides with prose_on and every block of four
    ids covers all four verse positions. The two distractors take the remaining slots,
    swapped every other block so neither 'the leftover pair is always prose-then-verse'
    nor its opposite is a standing cue.
    """
    p = qid % N_OPTIONS
    v = (p + 1 + (qid // N_OPTIONS) % 3) % N_OPTIONS
    rest = [s for s in range(N_OPTIONS) if s not in (p, v)]
    if (qid // N_OPTIONS) % 2:
        rest = rest[::-1]
    return {"prose_on": p, "verse_on": v, "prose_off": rest[0], "verse_off": rest[1]}


def build_mcqa(qid, question, responses, medium):
    """
    Build the MCQA user prompt and return (prompt_str, prose_letter, verse_letter).

    `responses` maps each of the four roles to its text:
        'prose_on', 'verse_on', 'prose_off', 'verse_off'
    Option placement is deterministic in qid and independent of `medium`, so for a
    given id every file shares the same A/B/C/D layout and the prompts differ only in
    the `medium` word ('prose'/'verse') in the stem.
    """
    layout = mcqa_layout(qid)
    slots = [None] * N_OPTIONS
    for role, pos in layout.items():
        slots[pos] = (role, responses[role])

    lines = [QUESTION_PREFIX + question, MCQA_STEM.format(medium=medium)]
    prose_letter = verse_letter = None
    for letter, (tag, text) in zip(LETTERS, slots):
        lines.append(f"({letter}) {text}")
        if tag == "prose_on":
            prose_letter = letter
        elif tag == "verse_on":
            verse_letter = letter
    prompt = "\n".join(lines) + "\n" + ANSWER_CUE
    return prompt, prose_letter, verse_letter


def letters_for(qid):
    """(prose_letter, verse_letter) without building the prompt."""
    layout = mcqa_layout(qid)
    return LETTERS[layout["prose_on"]], LETTERS[layout["verse_on"]]


def assert_letter_balance(ids, label, strict=True):
    """Confirm the cycle actually balanced, rather than trusting the arithmetic.

    Both correct letters have to be balanced independently: the prose files are scored
    on prose_letter and the verse files on verse_letter, so a layout that balanced only
    one of them would still hand a position cue to half the dataset.
    """
    from collections import Counter
    prose = Counter(); verse = Counter()
    for qid in ids:
        pL, vL = letters_for(qid)
        prose[pL] += 1
        verse[vL] += 1
    counts = {"prose": dict(sorted(prose.items())), "verse": dict(sorted(verse.items()))}
    balanced = len(set(prose.values())) == 1 and len(set(verse.values())) == 1 \
        and len(prose) == N_OPTIONS and len(verse) == N_OPTIONS
    print(f"[mcqa] {label} letter balance: prose={counts['prose']} verse={counts['verse']}"
          f" -> {'balanced' if balanced else 'NOT balanced'}", flush=True)
    if not balanced and strict:
        raise RuntimeError(
            f"{label}: correct letters are not evenly spread ({counts}). The cycle is "
            f"exact only when len(ids) is a multiple of {N_OPTIONS} and the ids are "
            "contiguous; check TRAIN_IDS.")
    return counts


def parse_letter(r):
    """Extract the chosen option letter (prefer a leading letter, fall back to '(X)')."""
    r = (r or "").upper()
    m = LETTER_RE.match(r)
    if m:
        return m.group(1)
    m = OPTION_RE.search(r)
    return m.group(1) if m else None


def assistant(role, medium, pL, vL):
    correct = pL if medium == "prose" else vL
    other = vL if medium == "prose" else pL
    return other if role == "undesired" else correct


# ---------------------------------------------------------------------------
# Generation backend (transformers)
# ---------------------------------------------------------------------------
_TOK_CACHE = {}


def get_tokenizer(model_id, hf_token=""):
    """Cached per model. Left padding is required: these are decoder-only models and
    right padding would put the pad tokens between the prompt and the continuation."""
    if model_id in _TOK_CACHE:
        return _TOK_CACHE[model_id]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token or None, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise RuntimeError(f"{model_id}: tokenizer has neither pad_token nor eos_token.")
        tok.pad_token = tok.eos_token
    _TOK_CACHE[model_id] = tok
    return tok


_MODE_CACHE = {}


def prompt_mode(model_id):
    """chat / plain. Only 'user' turns are used here, so the system-role question does
    not arise -- but a base model with no template still needs the plain path."""
    if model_id in _MODE_CACHE:
        return _MODE_CACHE[model_id]
    tok = get_tokenizer(model_id)
    _MODE_CACHE[model_id] = "chat" if getattr(tok, "chat_template", None) else "plain"
    return _MODE_CACHE[model_id]


def render(model_id, user):
    if prompt_mode(model_id) == "plain":
        return user
    tok = get_tokenizer(model_id)
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def _resolve_dtype(name):
    import torch
    if name in (None, "", "auto"):
        return "auto"
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"DTYPE={name!r} is not a torch dtype")
    return dt


def load_model(model_id, hf_token=""):
    """One loader for both stages. The verse-long fallback used to load in 4-bit nf4
    while the cache it was topping up came from a bf16 engine -- quantisation changes
    greedy output, so a cache miss produced a response the rest of the dataset could
    not have produced. Same path, same precision, for both now."""
    import torch
    from transformers import AutoModelForCausalLM
    print(f"[model] loading {model_id} (device_map={DEVICE_MAP}, dtype={DTYPE}, "
          f"attn={ATTN_IMPL}) ...", flush=True)
    kwargs = dict(device_map=DEVICE_MAP, attn_implementation=ATTN_IMPL,
                  trust_remote_code=True, low_cpu_mem_usage=True,
                  token=hf_token or None)
    dtype = _resolve_dtype(DTYPE)
    # transformers renamed torch_dtype -> dtype in 4.56; try the new name, fall back.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    model.eval()
    dmap = getattr(model, "hf_device_map", None) or {}
    off = sorted({str(v) for v in dmap.values() if str(v) in ("cpu", "disk")})
    if off:
        msg = (f"{model_id}: accelerate placed some layers on {off} -- the model does not "
               "fit on the visible GPU(s). Generation will be slow and partly off-GPU.")
        if not ALLOW_CPU_OFFLOAD:
            raise RuntimeError(msg + "\nUse more GPUs or set ALLOW_CPU_OFFLOAD=1.")
        print(f"[model] WARNING: {msg} (ALLOW_CPU_OFFLOAD=1)", flush=True)
    gcfg = model.generation_config
    gcfg.do_sample = False
    gcfg.num_beams = 1
    gcfg.temperature = None
    gcfg.top_p = None
    gcfg.top_k = None
    print("[model] ready", flush=True)
    return model


def free_model(model):
    """Release before the next model loads."""
    try:
        del model
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def windows(seq, n):
    """Fixed windows over `seq`, yielded as (window_index, window)."""
    for i in range(0, len(seq), n):
        yield i // n, seq[i:i + n]


def chat(model_id, model, prompts, temperature, max_new_tokens, seed=0, batch_size=None):
    """Batched generation. Returns one string per prompt, in submission order.

    Batches are fixed windows of `prompts`, so the caller controls batch composition by
    controlling the prompt list -- which is why the response stage iterates over ALL
    question/framing pairs rather than only the uncached ones.

    For temperature > 0 the torch RNG is reseeded per window from seed + window index,
    so a window's sample does not depend on how many windows preceded it.
    """
    import torch
    tok = get_tokenizer(model_id)
    bs = batch_size or BATCH_SIZE
    out_all = []
    for wi, window in windows(prompts, bs):
        rendered = [render(model_id, p) for p in window]
        enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        dev = getattr(model, "device", None) or next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        if in_len + max_new_tokens > MAX_MODEL_LEN:
            raise RuntimeError(f"prompt ({in_len}) + max_new_tokens ({max_new_tokens}) "
                               f"exceeds MAX_MODEL_LEN={MAX_MODEL_LEN}")
        kwargs = dict(max_new_tokens=max_new_tokens, num_beams=1,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if temperature and temperature > 0:
            torch.manual_seed(seed + wi)
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = model.generate(**enc, **kwargs)
        # Left padding makes every row's prompt the same length, so this slice is
        # uniform across the batch.
        out_all += [t.strip() for t in tok.batch_decode(out[:, in_len:], skip_special_tokens=True)]
    return out_all


# ---------------------------------------------------------------------------
# Stage 1: verse-single
# ---------------------------------------------------------------------------
VERSE_SINGLE_FILES = [
    "prose-desired-all.jsonl", "prose-undesired-all.jsonl",
    "verse-single-desired-all.jsonl", "verse-single-undesired-all.jsonl",
    "prose-test.jsonl",
]


def verse_single_dir(model_dir):
    return os.path.join(DATA_DIR, model_dir, "verse-single")


def verse_single_done(model_dir):
    d = verse_single_dir(model_dir)
    return all(os.path.exists(os.path.join(d, f)) for f in VERSE_SINGLE_FILES)


def load_existing_verse_single_questions(model_dir):
    qs = set()
    d = verse_single_dir(model_dir)
    if not os.path.isdir(d):
        return qs
    for f in sorted(os.listdir(d)):          # sorted: os.listdir order is filesystem-dependent
        if f.endswith(".jsonl"):
            for line in open(os.path.join(d, f)):
                c = json.loads(line)["prompt"][0]["content"]
                m = re.search(r"Question:\s*(.*?)\n", c)
                if m:
                    qs.add(m.group(1).strip().lower())
    return qs


def stage1_signature(model_dir, need, max_q_rounds, exclude_existing):
    """Everything that can change a generated token in stage 1."""
    themes = hashlib.sha256("\n".join(THEMES).encode()).hexdigest()
    return {
        "BACKEND": "transformers",
        "MODEL": MODELS[model_dir],
        "DTYPE": DTYPE,
        "DEVICE_MAP": DEVICE_MAP,
        "ATTN_IMPL": ATTN_IMPL,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_MODEL_LEN": MAX_MODEL_LEN,
        "SEED": SEED,
        "NEED": need,
        "MAX_Q_ROUNDS": max_q_rounds,
        "EXCLUDE_EXISTING": bool(exclude_existing),
        "MCQA_LAYOUT_SCHEME": MCQA_LAYOUT_SCHEME,
        "MCQA_STEM": MCQA_STEM,
        "PROMPT_MODE": prompt_mode(MODELS[model_dir]),
        "DETERMINISM": determinism_state(),
        "themes_sha256": themes,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
    }


def build_verse_single(model_dir, need=650, max_q_rounds=8, fill_rounds=60, force=False,
                       hf_token="", exclude_existing=False):
    if verse_single_done(model_dir) and not force:
        print(f"[verse-single] already built for {model_dir} (all 5 files present) -- skipping. "
              f"Pass --force single to rebuild.")
        return
    if model_dir not in MODELS:
        sys.exit(f"[verse-single] unknown model {model_dir!r}, add it to MODELS")
    hf_id = MODELS[model_dir]

    D = verse_single_dir(model_dir) + "/"
    os.makedirs(D, exist_ok=True)
    CKPT = os.path.join(CHECKPOINT_ROOT, model_dir, "verse-single")
    os.makedirs(CKPT, exist_ok=True)
    check_or_write_signature(os.path.join(CKPT, "signature.json"),
                             stage1_signature(model_dir, need, max_q_rounds, exclude_existing))
    print(f"[verse-single] checkpoint dir: {CKPT}", flush=True)

    model = load_model(hf_id, hf_token)

    try:
        def generate_questions(need, exclude):
            qpath = f"{CKPT}/questions.json"
            have = json.load(open(qpath)) if os.path.exists(qpath) else []
            seen = set(exclude) | {q.lower() for q in have}
            rnd = 0
            while len(have) < need:
                prompts = [
                    (f"Generate 25 short, open-ended, reflective questions about {t}. "
                     "Each must be a single sentence ending in '?', the kind a person might "
                     "write a thoughtful essay about. Style examples:\n"
                     "- What is the nature of time?\n- What is the value of silence?\n"
                     "- What role does fate play in our lives?\n"
                     "Output ONLY the questions, one per line, no numbering or extra text.")
                    for t in THEMES
                ]
                # seed = SEED + round: each round samples differently, and rerunning a
                # round from scratch reproduces it.
                resp = chat(hf_id, model, prompts, temperature=0.95, max_new_tokens=900,
                            seed=SEED + rnd)
                for r in resp:
                    for line in r.split("\n"):
                        line = line.strip().lstrip("-•*0123456789. ").strip()
                        if line.endswith("?") and 12 <= len(line) <= 160 and line.lower() not in seen:
                            seen.add(line.lower())
                            have.append(line)
                rnd += 1
                write_json_atomic(qpath, have, indent=1)
                print(f"  questions: {len(have)} (round {rnd})", flush=True)
                _stop_if_requested(f"question round {rnd}")
                if rnd >= max_q_rounds:
                    break
            return have

        def generate_responses(questions):
            """Fill the prose/verse response cache over FIXED windows of all pairs.

            The old version built `todo` from the uncached pairs and batched THAT, so a
            resume packed different batches than a clean run and could produce different
            greedy text for the same prompt. Here the windows are positions in the full
            pair list; a window is skipped only when every pair in it is cached, and
            otherwise regenerated whole.
            """
            cpath = f"{CKPT}/cache.json"
            cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
            pairs = [(q, fr) for q in questions for fr in ("prose", "verse")]
            todo_windows = [(wi, w) for wi, w in windows(pairs, BATCH_SIZE)
                            if any(cache_key(q, fr) not in cache for q, fr in w)]
            n_redo = sum(1 for _, w in todo_windows for q, fr in w if cache_key(q, fr) in cache)
            print(f"  responses: {len(pairs)} pairs, {len(todo_windows)} windows to run "
                  f"({n_redo} cached values recomputed to keep batches identical)", flush=True)
            for n, (wi, window) in enumerate(todo_windows):
                prompts = [f"Respond in {fr} and concisely. {q}" for q, fr in window]
                resp = chat(hf_id, model, prompts, temperature=0.0, max_new_tokens=128)
                for (q, fr), r in zip(window, resp):
                    k = cache_key(q, fr)
                    if k in cache and cache[k] != r:
                        # Recomputed only to hold the window together: same prompt, same
                        # batch, so a difference is nondeterminism that survived the pins.
                        print(f"  [determinism] response changed on recompute for {k!r}\n"
                              f"    cached:     {cache[k][:80]!r}\n"
                              f"    recomputed: {r[:80]!r}\n"
                              f"    (keeping the cached value)", flush=True)
                        continue
                    cache[k] = r
                write_json_atomic(cpath, cache)
                print(f"  responses: window {n + 1}/{len(todo_windows)} done", flush=True)
                _stop_if_requested(f"response window {n + 1}/{len(todo_windows)}")
            return cache

        def responses_for(sid, q, cache, heldout):
            off = random.Random(98765 + sid).choice(heldout)
            return {"prose_on": cache[cache_key(q, "prose")], "verse_on": cache[cache_key(q, "verse")],
                    "prose_off": cache[cache_key(off, "prose")], "verse_off": cache[cache_key(off, "verse")]}

        def slot_items(ids, id2q, cache, heldout):
            items = []
            for sid in ids:
                r = responses_for(sid, id2q[sid], cache, heldout)
                pp, pL, vL = build_mcqa(sid, id2q[sid], r, "prose")
                vp, _, _ = build_mcqa(sid, id2q[sid], r, "verse")
                items.append((sid, "prose", pp, pL))
                items.append((sid, "verse", vp, vL))
            return items

        def fill(ids, cand, ptr, cache, heldout, label):
            """Iteratively swap in fresh questions until the model answers both MCQA
            framings correctly for every id.

            Checkpointed at ROUND boundaries, which is the only place a resume is
            exactly equivalent to a clean run: a round's batch is the set of ids still
            dirty, so restoring (id2q, clean, ptr, round) reproduces the same batch the
            clean run would have submitted. Mid-round there is no such equivalence, so
            a preemption there costs that round and no more.

            This is still the least positionally-pinned part of the pipeline: round n's
            composition depends on round n-1's verdicts, so it cannot be expressed as
            fixed windows the way the response cache can.
            """
            fpath = f"{CKPT}/fill_{label}.json"
            start_round = 0
            if os.path.exists(fpath):
                st = json.load(open(fpath))
                id2q = {int(k): v for k, v in st["id2q"].items()}
                clean = set(st["clean"])
                ptr = st["ptr"]
                start_round = st["round"]
                print(f"  [{label}] resuming at round {start_round}: "
                      f"{len(clean)}/{len(ids)} already clean, ptr={ptr}", flush=True)
            else:
                id2q = {sid: cand[ptr + k] for k, sid in enumerate(ids)}
                ptr += len(ids)
                clean = set()

            def save(rnd):
                write_json_atomic(fpath, {"id2q": {str(k): v for k, v in id2q.items()},
                                          "clean": sorted(clean), "ptr": ptr,
                                          "round": rnd}, indent=1)

            for it in range(start_round, fill_rounds):
                todo = [sid for sid in ids if sid not in clean]
                if not todo:
                    save(it)
                    return id2q, ptr
                items = slot_items(todo, id2q, cache, heldout)
                preds = chat(hf_id, model, [p for *_, p, _ in items], temperature=0.0,
                             max_new_tokens=24)
                ok = {sid: {} for sid in todo}
                for (sid, kind, _, tgt), pr in zip(items, preds):
                    ok[sid][kind] = (parse_letter(pr) == tgt)
                clean.update(sid for sid in todo if ok[sid]["prose"] and ok[sid]["verse"])
                fails = [sid for sid in todo if sid not in clean]
                print(f"  [{label}] round {it}: clean {len(clean)}/{len(ids)}, retry {len(fails)}", flush=True)
                for sid in fails:
                    if ptr >= len(cand):
                        raise RuntimeError(f"ran out of candidates filling {label} ({ptr} used)")
                    id2q[sid] = cand[ptr]
                    ptr += 1
                save(it + 1)
                _stop_if_requested(f"{label} fill round {it}")
            raise RuntimeError(f"{label} did not converge")

        def write_files(train_id2q, test_id2q, cache, heldout):
            # The cycle is exact only over a contiguous run whose length is a multiple
            # of 4; check before writing rather than discovering it in a results plot.
            train_counts = assert_letter_balance(TRAIN_IDS, "train", strict=True)
            test_counts = assert_letter_balance(TEST_IDS, "test", strict=False)

            backup = D + "_pre_clean_backup"
            os.makedirs(backup, exist_ok=True)
            out = {f: [] for f in VERSE_SINGLE_FILES}
            key = {}

            def emit(sid, q):
                r = responses_for(sid, q, cache, heldout)
                pp, pL, vL = build_mcqa(sid, q, r, "prose")
                vp, _, _ = build_mcqa(sid, q, r, "verse")
                return pp, vp, pL, vL

            for sid in TRAIN_IDS:
                pp, vp, pL, vL = emit(sid, train_id2q[sid])
                key[sid] = {"question": train_id2q[sid], "split": "train",
                            "prose_letter": pL, "verse_letter": vL,
                            "layout": {r: LETTERS[p] for r, p in mcqa_layout(sid).items()}}
                out["prose-desired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": pp},
                    {"role": "assistant", "content": assistant("desired", "prose", pL, vL)}]})
                out["prose-undesired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": pp},
                    {"role": "assistant", "content": assistant("undesired", "prose", pL, vL)}]})
                out["verse-single-desired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": vp},
                    {"role": "assistant", "content": assistant("desired", "verse", pL, vL)}]})
                out["verse-single-undesired-all.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": vp},
                    {"role": "assistant", "content": assistant("undesired", "verse", pL, vL)}]})
            for sid in TEST_IDS:
                pp, vp, pL, vL = emit(sid, test_id2q[sid])
                key[sid] = {"question": test_id2q[sid], "split": "test",
                            "prose_letter": pL, "verse_letter": vL,
                            "layout": {r: LETTERS[p] for r, p in mcqa_layout(sid).items()}}
                out["prose-test.jsonl"].append({"id": sid, "prompt": [{"role": "user", "content": pp},
                    {"role": "assistant", "content": pL}]})  # forced correct prose answer
            for f, rows in out.items():
                if os.path.exists(D + f) and not os.path.exists(f"{backup}/{f}"):
                    shutil.copy2(D + f, f"{backup}/{f}")
                write_lines_atomic(D + f, rows)
                print(f"  wrote {f}: {len(rows)} rows", flush=True)

            kept = set(train_id2q.values()) | set(test_id2q.values()) | set(heldout)
            newcache = {k: v for k, v in cache.items() if k.split("\t", 1)[1] in kept}
            write_json_atomic(D + "mcqa_response_cache.json", newcache, indent=2)

            # Written so nothing downstream has to reimplement the layout. Anything
            # scoring these files should read this instead of recomputing letters.
            write_json_atomic(D + "mcqa_answer_key.json", {
                "scheme": MCQA_LAYOUT_SCHEME,
                "letters": LETTERS,
                "model": model_dir,
                "balance": {"train": train_counts, "test": test_counts},
                "determinism": determinism_state(),
                "by_id": {str(k): v for k, v in sorted(key.items())},
            }, indent=2)
            print(f"  wrote mcqa_answer_key.json: {len(key)} ids", flush=True)

        # The old script ALWAYS excluded questions already present in the output dir,
        # which meant --force single deliberately produced a DIFFERENT dataset -- an
        # original build could only be reproduced from an empty directory. Exclusion is
        # now opt-in (--exclude_existing) and recorded in the signature, so a forced
        # rebuild against the same checkpoint reproduces what it rebuilt.
        existing = load_existing_verse_single_questions(model_dir) if exclude_existing else set()
        print(f"[verse-single] existing questions to exclude: {len(existing)}"
              f"{'' if exclude_existing else ' (--exclude_existing off: rebuild is reproducible)'}",
              flush=True)
        questions = generate_questions(need=need, exclude=existing)
        cache = generate_responses(questions)
        cand = list(questions)
        random.Random(7).shuffle(cand)
        heldout, slot_cand = cand[:60], cand[60:]
        print(f"[verse-single] slot-filling train ({N_TRAIN})...", flush=True)
        train_id2q, ptr = fill(TRAIN_IDS, slot_cand, 0, cache, heldout, "train")
        print(f"[verse-single] slot-filling test ({N_TEST})...", flush=True)
        test_id2q, ptr = fill(TEST_IDS, slot_cand, ptr, cache, heldout, "test")
        write_json_atomic(f"{CKPT}/final.json",
                          {"train": {str(k): v for k, v in train_id2q.items()},
                           "test": {str(k): v for k, v in test_id2q.items()},
                           "heldout": heldout}, indent=1)
        print("[verse-single] writing data files...", flush=True)
        write_files(train_id2q, test_id2q, cache, heldout)
        print("[verse-single] DONE.", flush=True)
    finally:
        free_model(model)


# ---------------------------------------------------------------------------
# Stage 2: verse-long
# ---------------------------------------------------------------------------
def _long_rows_ok(files, n_expected):
    """All four files present, equal length, and agreeing question-by-question on id.

    This is the invariant the old append-per-question loop could not hold. It appended
    to four files inside the loop, took each id from that file's OWN line count, and
    decided what was already done by reading only verse-long-desired-all. An
    interruption between the first and fourth append left the other three a row short,
    and on resume that question counted as done -- so every later id in the short files
    referred to a different question than the same id in verse-long-desired-all,
    silently breaking the pairing the contrastive setup depends on. --force long hid it
    by deleting all four.
    """
    if not all(os.path.exists(f) for f in files.values()):
        return False
    rows = {}
    for k, f in files.items():
        with open(f) as fh:
            rows[k] = [json.loads(line) for line in fh]
        if len(rows[k]) != n_expected:
            return False
    ref = rows["verse_desired"]
    for k, rs in rows.items():
        for a, b in zip(ref, rs):
            if a["id"] != b["id"] or _question_of(a) != _question_of(b):
                return False
    return True


def _question_of(obj):
    content = obj["prompt"][0]["content"]
    for prefix in LONG_PREFIXES:
        if content.startswith(prefix):
            return content[len(prefix):]
    return None


def long_entry(question, framing, response, entry_id):
    return {
        "id": entry_id,
        "prompt": [
            {"role": "user", "content": f"Respond in {framing} and concisely. {question}"},
            {"role": "assistant", "content": response},
        ],
    }


def get_line_count(fpath):
    if not os.path.exists(fpath):
        return 0
    with open(fpath) as f:
        return sum(1 for _ in f)


def get_verse_single_train_questions(model_dir):
    """Extract the 100 train questions from verse-single-desired-all.jsonl's MCQA prompts
    ("Question: {q}\nWhich of the following ...")."""
    fpath = os.path.join(verse_single_dir(model_dir), "verse-single-desired-all.jsonl")
    questions = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            content = obj["prompt"][0]["content"]
            m = re.search(r"^Question:\s*(.*?)\s*\n", content)
            if m:
                questions.append(m.group(1).strip())
    return questions


def load_verse_single_response_cache(model_dir):
    fpath = os.path.join(verse_single_dir(model_dir), "mcqa_response_cache.json")
    if not os.path.exists(fpath):
        return {}
    with open(fpath) as f:
        return json.load(f)


def build_verse_long(model_dir, hf_token="", max_new_tokens=128, force=False):
    long_dir = os.path.join(DATA_DIR, model_dir, "verse-long")
    os.makedirs(long_dir, exist_ok=True)
    files = {
        "verse_desired": os.path.join(long_dir, "verse-long-desired-all.jsonl"),
        "verse_undesired": os.path.join(long_dir, "verse-long-undesired-all.jsonl"),
        "prose_desired": os.path.join(long_dir, "prose-desired-all.jsonl"),
        "prose_undesired": os.path.join(long_dir, "prose-undesired-all.jsonl"),
    }

    all_questions = get_verse_single_train_questions(model_dir)
    if not all_questions:
        sys.exit(f"[verse-long] no train questions found for {model_dir}; run verse-single first.")

    if not force and _long_rows_ok(files, len(all_questions)):
        print(f"[verse-long] all four files present, {len(all_questions)} rows each, ids "
              "aligned -- nothing to do.")
        return

    response_cache = load_verse_single_response_cache(model_dir)
    misses = [(q, fr) for q in all_questions for fr in ("verse", "prose")
              if cache_key(q, fr) not in response_cache]
    n_pairs = 2 * len(all_questions)

    print(f"[verse-long] model:                       {model_dir}")
    print(f"[verse-long] questions in verse-single train: {len(all_questions)}")
    print(f"[verse-long] reusable from response cache:  {n_pairs - len(misses)}/{n_pairs} "
          f"(verse-single's mcqa_response_cache.json)")
    print(f"[verse-long] needing fresh generation:     {len(misses)}/{n_pairs}")

    # All four files are rebuilt from the cache every time rather than appended to.
    # Regenerating is free when the cache covers everything, and it is the only way the
    # four files are guaranteed to agree: they are written from one in-memory list.
    if misses:
        if model_dir not in MODELS:
            sys.exit(f"[verse-long] unknown model {model_dir!r}, add it to MODELS")
        print(f"[verse-long] loading the model to fill {len(misses)} missing pairs. Same "
              "loader, precision and length cap as stage 1 -- a fallback response is "
              "meant to be indistinguishable from a cached one.", flush=True)
        model = load_model(MODELS[model_dir], hf_token)
        try:
            prompts = [f"Respond in {fr} and concisely. {q}" for q, fr in misses]
            outs = chat(MODELS[model_dir], model, prompts, temperature=0.0,
                        max_new_tokens=max_new_tokens)
            for (q, fr), r in zip(misses, outs):
                response_cache[cache_key(q, fr)] = r
            print(f"[verse-long] generated {len(misses)} missing responses", flush=True)
        finally:
            free_model(model)

    rows = {k: [] for k in files}
    for idx, question in enumerate(all_questions):
        verse_resp = response_cache[cache_key(question, "verse")]
        prose_resp = response_cache[cache_key(question, "prose")]
        # One id per question, shared by all four files, so id N is the same question
        # everywhere -- which is what the desired/undesired pairing assumes.
        rows["verse_desired"].append(long_entry(question, "verse", verse_resp, idx))
        rows["prose_undesired"].append(long_entry(question, "prose", verse_resp, idx))
        rows["prose_desired"].append(long_entry(question, "prose", prose_resp, idx))
        rows["verse_undesired"].append(long_entry(question, "verse", prose_resp, idx))

    for k, f in files.items():
        write_lines_atomic(f, rows[k])

    if not _long_rows_ok(files, len(all_questions)):
        raise RuntimeError("[verse-long] post-write check failed: the four files do not "
                           "agree on ids/questions.")
    print("[verse-long] final counts:")
    for k, v in files.items():
        print(f"  {k}: {get_line_count(v)} entries")
    print("[verse-long] wrote all four files whole; ids aligned across files.")
    print("[verse-long] DONE.")


# ---------------------------------------------------------------------------
# Stage 3: verse-long/prose-test.jsonl -- bare held-out eval prompts
# ---------------------------------------------------------------------------
def build_verse_long_test(model_dir, force=False):
    """The 50 verse-single TEST questions (ids 1000-1049), stripped to a bare
    "Respond in prose and concisely. {question}" user-only prompt: no MCQA wrapping,
    no assistant response. Same underlying 50 questions already generated for
    verse-single's prose-test.jsonl -- just a different (simpler) prompt format, not a
    new generation."""
    src = os.path.join(verse_single_dir(model_dir), "prose-test.jsonl")
    dst_dir = os.path.join(DATA_DIR, model_dir, "verse-long")
    dst = os.path.join(dst_dir, "prose-test.jsonl")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src):
        print(f"[verse-long-test] {src} not found -- run verse-single first. Skipping.")
        return
    if os.path.exists(dst) and not force:
        print(f"[verse-long-test] {dst} already exists -- skipping. Pass --force test to rebuild.")
        return

    rows = []
    with open(src) as f:
        for line in f:
            obj = json.loads(line)
            content = obj["prompt"][0]["content"]
            m = re.search(r"^Question:\s*(.*?)\s*\n", content)
            if not m:
                raise ValueError(f"Could not extract question from: {content!r}")
            question = m.group(1).strip()
            rows.append({
                "id": obj["id"],
                "prompt": [{"role": "user", "content": f"Respond in prose and concisely. {question}"}],
            })
    with open(dst, "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[verse-long-test] wrote {dst}: {len(rows)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build verse-single + verse-long datasets")
    # Defaults come from the environment so the same SLURM launcher can drive this
    # script and the other two builders, which are env-driven throughout. An explicit
    # flag always wins over the env.
    ap.add_argument("--model", default=os.environ.get("MODELS", os.environ.get("MODEL", "")),
                    help="Model short name OR full HF repo id, or a comma-separated list. "
                         f"Defaults to $MODELS / $MODEL. Known: {', '.join(MODELS)}")
    ap.add_argument("--only", default=os.environ.get("ONLY", os.environ.get("STAGES", "single,long,test")),
                    help="Comma-separated subset of stages to run: single,long,test "
                         "(default: all, or $ONLY / $STAGES)")
    ap.add_argument("--force", default=os.environ.get("FORCE", ""),
                    help="Comma-separated stages to force-rebuild even if already done")
    ap.add_argument("--need", type=int, default=650, help="[single] candidate-question pool size")
    ap.add_argument("--max_q_rounds", type=int, default=8, help="[single] max question-generation rounds")
    ap.add_argument("--fill_rounds", type=int, default=60, help="[single] max slot-fill retry rounds")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), help="HF token")
    ap.add_argument("--exclude_existing", action="store_true",
                    help="[single] exclude questions already present in the output dir from "
                         "the candidate pool. The old script did this unconditionally, which "
                         "made --force single produce a DIFFERENT dataset; off by default so "
                         "a rebuild is reproducible.")
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="[long] cap for fallback generation. Defaults to 128 to match the "
                         "cap stage 1 used for the cached responses; raising it means a "
                         "fallback response can be longer than any cached one.")
    ap.add_argument("--continue_on_error", action="store_true",
                    default=os.environ.get("CONTINUE_ON_ERROR", "").lower() in {"1", "true", "yes", "on"},
                    help="With several models, keep going after one fails ($CONTINUE_ON_ERROR)")
    args = ap.parse_args()

    enable_determinism()
    check_hashseed(SEED)
    set_seed(SEED)

    if not args.model.strip():
        sys.exit("no model given: pass --model, or export MODELS/MODEL. Known short "
                 f"names: {', '.join(MODELS)}")
    raw = [m.strip() for m in args.model.split(",") if m.strip()]
    resolved = [(r, resolve_model_name(r)) for r in raw]
    unknown = [r for r, got in resolved if got is None]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}.\nKnown short names: {', '.join(MODELS)}\n"
                 f"Known repo ids:   {', '.join(MODELS.values())}\n"
                 "Add an entry to MODELS if this is a new model.")
    models = [got for _r, got in resolved]
    if len(set(models)) != len(models):
        sys.exit(f"duplicate models after resolving names: {models} (from {args.model!r}) "
                 "-- a short name and its repo id both name the same model.")

    stages = {s.strip() for s in args.only.split(",") if s.strip()}
    forced = {s.strip() for s in args.force.split(",") if s.strip()}
    print(f"[main] {len(models)} model(s): {', '.join(models)} | stages: {sorted(stages)}",
          flush=True)

    failures = []
    for model_dir in models:
        print(f"\n{'=' * 78}\n[main] {model_dir}\n{'=' * 78}", flush=True)
        try:
            # Per model, so a model sees the same RNG state whether or not another ran
            # before it. random.Random(7)/Random(98765+sid) are already independent of
            # the global stream; this covers torch and anything added later.
            set_seed(SEED)
            if "single" in stages:
                build_verse_single(model_dir, need=args.need, max_q_rounds=args.max_q_rounds,
                                   fill_rounds=args.fill_rounds, force="single" in forced,
                                   hf_token=args.hf_token,
                                   exclude_existing=args.exclude_existing)
            if "long" in stages:
                build_verse_long(model_dir, hf_token=args.hf_token,
                                 max_new_tokens=args.max_new_tokens, force="long" in forced)
            if "test" in stages:
                build_verse_long_test(model_dir, force="test" in forced)
        except Exception as e:
            if not args.continue_on_error:
                raise
            failures.append((model_dir, repr(e)))
            print(f"[error] {model_dir} failed: {e!r}; continuing (--continue_on_error)",
                  flush=True)

    if failures:
        print(f"\n[done] {len(failures)} model(s) failed:", flush=True)
        for m, e in failures:
            print(f"  {m}: {e}", flush=True)
        sys.exit(1)
    print("\n[done] all models complete", flush=True)


if __name__ == "__main__":
    main()