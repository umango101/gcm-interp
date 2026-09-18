#!/usr/bin/env python
"""
build_gender_dataset.py
=======================

Generate male/female story pairs for every role with one or more HF causal LMs,
QC them, and emit 14 contrastive JSONL files per model for gender-steering
experiments.

Pipeline (all stages resumable on SLURM requeue):

  1. tokcheck  (fail-fast global precondition)
        Confirm the tokenizer splits " female" and " male" into the same number
        of tokens. This is what makes the two prompts token-aligned, so the
        contrastive word sits at a matched position for activation work. Hard
        precondition -> aborts unless ALLOW_TOKEN_LENGTH_MISMATCH=1.
        NOTE: most BPE vocabularies fail this pair. See the note at the bottom
        of this docstring.

  2. generate
        For every role, greedily generate a female story and a male story.
        Checkpointed to <ckpt>/stories.jsonl (append-only, fsync per chunk).

  3. qc   (per-role quality check -> filters, does NOT hard-fail by default)
        (a) LEXICAL: the female story must actually be about a female character
            and the male story about a male one, judged by gendered pronoun and
            noun counts (word-boundary matched). Both directions must pass.
        (b) MCQA:    show the model the two stories as options A/B under the
            female prompt and under the male prompt, and require it to pick the
            matching one both times. This validates the emitted item itself,
            not a paraphrase of it.
        Roles failing any enabled check are dropped and logged to qc_report.jsonl.
        QC_MODE selects which checks run. STRICT=1 hard-fails instead.

  4. build
        Assign ids over the surviving roles (input order). The first CAP (=100)
        roles populate the 12 aligned files; the next TEST_CAP (=50) populate
        the test files. All aligned files share the same id->role mapping.

Files written (14 by default):

  female-single-desired-all.jsonl     MC prompt, answer = female option
  female-single-undesired-all.jsonl   MC prompt, answer = male option
  female-single-steering.jsonl        MC prompt, empty assistant turn
  male-single-desired-all.jsonl       MC prompt, answer = male option
  male-single-undesired-all.jsonl     MC prompt, answer = female option
  male-single-steering.jsonl          MC prompt, empty assistant turn
  male-single-test.jsonl              MC prompt, no assistant turn
  female-long-desired-all.jsonl       story prompt, answer = female story
  female-long-undesired-all.jsonl     story prompt, answer = male story
  female-long-steering.jsonl          story prompt, empty assistant turn
  male-long-desired-all.jsonl         story prompt, answer = male story
  male-long-undesired-all.jsonl       story prompt, answer = female story
  male-long-steering.jsonl            story prompt, empty assistant turn
  male-long-test.jsonl                story prompt, no assistant turn

  (+ female-single-test.jsonl / female-long-test.jsonl when EMIT_FEMALE_TEST=1)

SPEC DEVIATIONS — each is one env var away from the literal spec
----------------------------------------------------------------
  1. The spec's "long" block reuses the "-single-" filenames for what is
     plainly the long-format content (the same duplicate-name slip as the
     summary pipeline's 6th file). Written here as -long-.
  2. The spec's generation prompt omits ". Speak about them in the
     third-person." while the emitted prompt includes it. Recording a prompt
     that is not the prompt that produced the response poisons the long-desired
     targets, and the third-person instruction is also what makes the response
     carry the pronouns QC relies on. Default: generate with the full emitted
     prompt. Set GEN_PROMPT_BARE=1 for the literal spec behaviour.
  3. Three of the four steering examples carry an empty assistant turn; the
     female-long one carries no assistant turn at all. Default: empty turn
     everywhere. Set STEERING_ASSISTANT=none to drop the turn everywhere.

DETERMINISM CAVEAT
------------------
Greedy decoding is deterministic *for a fixed batch composition*. Left padding
means a prompt generated in a batch of 8 can differ in its last tokens from the
same prompt generated alone, and a requeue re-partitions the surviving to-do
list into different batches. CHUNK_SIZE is rounded to a multiple of BATCH_SIZE
so flushes land on batch boundaries; STRICT_DETERMINISM=1 forces BATCH_SIZE=1
for bit-exactness across requeues, at a throughput cost.

TOKCHECK CAVEAT
---------------
" female" and " male" are usually different token counts (" female" often
splits, " male" usually does not). If the gate fails for your tokenizers you
have three options: set ALLOW_TOKEN_LENGTH_MISMATCH=1 and index activations
from the end of the contrastive span rather than a fixed offset; pick a pair
that does tokenize equally (CONTRAST_A / CONTRAST_B are configurable, e.g.
"woman"/"man" via GENDER_WORD_F / GENDER_WORD_M); or accept the misalignment
and handle it downstream. The gate exists so you find out now, not after the
steering vectors look strange.
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
from dataclasses import dataclass, field  # noqa: E402
from typing import Dict, List, Optional   # noqa: E402


# ---------------------------------------------------------------------------
# CONFIG (every value overridable via environment variable of the same name)
# ---------------------------------------------------------------------------
def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


CONFIG = {
    "MODELS":           _env("MODELS", "allenai/OLMo-2-1124-13B-DPO,google/gemma-3-12b-it,tiiuae/Falcon3-10B-Instruct,Qwen/Qwen1.5-14B-Chat,Qwen/Qwen1.5-32B-Chat"),
    "MODEL":            _env("MODEL", "allenai/OLMo-2-1124-13B-DPO"),

    "INPUT":            _env("INPUT", "roles.json"),
    "ROLE_FIELD":       _env("ROLE_FIELD", ""),        # "" => auto-detect
    "OUTPUT_ROOT":      _env("OUTPUT_ROOT", "output"),
    "CHECKPOINT_ROOT":  _env("CHECKPOINT_ROOT", "checkpoint"),
    "OUTPUT_DIR":       _env("OUTPUT_DIR", ""),        # single-model override
    "CHECKPOINT_DIR":   _env("CHECKPOINT_DIR", ""),

    "STAGES":           _env("STAGES", "tokcheck,generate,qc,build"),
    "CONTINUE_ON_ERROR": _env("CONTINUE_ON_ERROR", 1, int),

    # runtime
    "DEVICE_MAP":       _env("DEVICE_MAP", "auto"),
    "DTYPE":            _env("DTYPE", "bfloat16"),
    "ATTN_IMPL":        _env("ATTN_IMPL", "sdpa"),
    "BATCH_SIZE":       _env("BATCH_SIZE", 8, int),
    "CHUNK_SIZE":       _env("CHUNK_SIZE", 16, int),
    "MAX_INPUT_TOKENS": _env("MAX_INPUT_TOKENS", 4096, int),
    "STRICT_DETERMINISM": _env("STRICT_DETERMINISM", 0, int),
    "STORY_MAX_TOKENS": _env("STORY_MAX_TOKENS", 600, int),
    "MC_MAX_TOKENS":    _env("MC_MAX_TOKENS", 4, int),

    # ---- prompt wording -------------------------------------------------
    # The emitted long prompt. Must contain {gender} and {role}.
    "LONG_PROMPT":      _env("LONG_PROMPT",
                             "Write a story about a character who is a {gender} {role}. "
                             "Speak about them in the third-person."),
    # Spec deviation 2: by default we generate with the prompt we emit.
    # GEN_PROMPT_BARE=1 generates with the style sentence stripped instead.
    "GEN_PROMPT_BARE":  _env("GEN_PROMPT_BARE", 0, int),
    "GEN_PROMPT":       _env("GEN_PROMPT",
                             "Write a story about a character who is a {gender} {role}"),
    "MC_HEADER":        _env("MC_HEADER",
                             'For the following question, choose the option that best '
                             'answers the prompt. Please respond with only one letter: '
                             '"A" or "B".'),
    "MC_TAIL":          _env("MC_TAIL", "Answer:"),
    "GENDER_WORD_F":    _env("GENDER_WORD_F", "female"),
    "GENDER_WORD_M":    _env("GENDER_WORD_M", "male"),
    # tokcheck pair; defaults follow the gender words actually used in prompts
    "CONTRAST_A":       _env("CONTRAST_A", ""),        # "" => GENDER_WORD_F
    "CONTRAST_B":       _env("CONTRAST_B", ""),        # "" => GENDER_WORD_M
    "ALLOW_TOKEN_LENGTH_MISMATCH": _env("ALLOW_TOKEN_LENGTH_MISMATCH", 0, int),

    # ---- QC --------------------------------------------------------------
    "QC_MODE":          _env("QC_MODE", "both"),       # lexical | mcqa | both | none
    "MIN_GENDER_SIGNAL": _env("MIN_GENDER_SIGNAL", 3, int),   # min gendered hits
    "GENDER_RATIO":     _env("GENDER_RATIO", 2.0, float),     # target vs other
    "MIN_STORY_WORDS":  _env("MIN_STORY_WORDS", 30, int),
    "DROP_TRUNCATED":   _env("DROP_TRUNCATED", 0, int),  # drop stories with no
                                                         # terminal punctuation
    "VERIFY_MODE":      _env("VERIFY_MODE", "filter"),   # filter | report
    "STRICT":           _env("STRICT", 0, int),

    # ---- output ----------------------------------------------------------
    "CAP":              _env("CAP", 100, int),          # aligned files
    "TEST_CAP":         _env("TEST_CAP", 50, int),      # test files
    "TEST_DISJOINT":    _env("TEST_DISJOINT", 1, int),  # 0 => test reuses the head
    "EMIT_FEMALE_TEST": _env("EMIT_FEMALE_TEST", 0, int),
    # Spec deviation 3: "empty" => assistant turn with "", "none" => no turn.
    "STEERING_ASSISTANT": _env("STEERING_ASSISTANT", "empty"),
    # A=female / B=male is fixed in the spec, which confounds the answer token
    # with gender. PERMUTE_OPTIONS=1 draws a per-role A/B order (seeded by role,
    # shared across every file so the variants stay aligned) instead.
    "PERMUTE_OPTIONS":  _env("PERMUTE_OPTIONS", 0, int),
    "SEED":             _env("SEED", 0, int),
}

if not CONFIG["CONTRAST_A"]:
    CONFIG["CONTRAST_A"] = CONFIG["GENDER_WORD_F"]
if not CONFIG["CONTRAST_B"]:
    CONFIG["CONTRAST_B"] = CONFIG["GENDER_WORD_M"]

FEMALE, MALE = "female", "male"     # internal keys, not prompt text


# ---------------------------------------------------------------------------
# Gendered lexicon for the lexical QC
# ---------------------------------------------------------------------------
FEM_PRONOUNS = ["she", "her", "hers", "herself"]
MASC_PRONOUNS = ["he", "him", "his", "himself"]
FEM_NOUNS = [
    "woman", "women", "girl", "girls", "female", "females", "lady", "ladies",
    "mother", "mom", "mum", "sister", "daughter", "wife", "aunt", "niece",
    "grandmother", "granddaughter", "queen", "heroine", "mrs", "ms", "miss",
    "madam", "madame",
]
MASC_NOUNS = [
    "man", "men", "boy", "boys", "male", "males", "gentleman", "gentlemen",
    "father", "dad", "brother", "son", "husband", "uncle", "nephew",
    "grandfather", "grandson", "king", "hero", "mr", "sir",
]


def _word_re(words):
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b",
                      re.IGNORECASE)


_RE_FEM_PRON = _word_re(FEM_PRONOUNS)
_RE_MASC_PRON = _word_re(MASC_PRONOUNS)
_RE_FEM_NOUN = _word_re(FEM_NOUNS)
_RE_MASC_NOUN = _word_re(MASC_NOUNS)


def gender_counts(text: str) -> dict:
    t = text or ""
    fp = len(_RE_FEM_PRON.findall(t))
    mp = len(_RE_MASC_PRON.findall(t))
    fn = len(_RE_FEM_NOUN.findall(t))
    mn = len(_RE_MASC_NOUN.findall(t))
    return {"fem_pron": fp, "masc_pron": mp, "fem_noun": fn, "masc_noun": mn,
            "fem_score": fp + fn, "masc_score": mp + mn}


def lexical_gender_ok(text: str, want: str) -> (bool, dict):
    """True iff `text` reads as being about a character of gender `want`."""
    c = gender_counts(text)
    if want == FEMALE:
        target, other = c["fem_score"], c["masc_score"]
    else:
        target, other = c["masc_score"], c["fem_score"]
    ok = (target >= CONFIG["MIN_GENDER_SIGNAL"] and
          target >= CONFIG["GENDER_RATIO"] * other)
    c["want"] = want
    c["ok"] = ok
    return ok, c


_CHOICE_RE = None


def parse_choice(text: str) -> str:
    """Extract a standalone A/B, ignoring letters embedded in words."""
    global _CHOICE_RE
    if _CHOICE_RE is None:
        _CHOICE_RE = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")
    m = _CHOICE_RE.search((text or "").upper())
    return m.group(1) if m else ""


def model_slug(name: str) -> str:
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
            sys.exit(f"[fatal] '{m}' and '{slugs[s]}' collide on output dir '{s}'")
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
def load_roles(path: str, role_field: str) -> List[str]:
    if not os.path.exists(path):
        sys.exit(f"[fatal] input not found: {path}")
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        sys.exit(f"[fatal] input is empty: {path}")

    try:
        raw = [json.loads(l) for l in text.splitlines() if l.strip()]
    except json.JSONDecodeError:
        obj = json.loads(text)
        raw = obj if isinstance(obj, list) else [obj]

    def extract(rec) -> str:
        if isinstance(rec, str):
            return rec
        if isinstance(rec, dict):
            if role_field:
                if role_field not in rec:
                    sys.exit(f"[fatal] ROLE_FIELD='{role_field}' missing in a record")
                return str(rec[role_field])
            for cand in ("role", "occupation", "job", "profession", "name", "text"):
                if cand in rec:
                    return str(rec[cand])
            sys.exit(f"[fatal] could not auto-detect role field; set ROLE_FIELD. "
                     f"keys seen: {list(rec.keys())}")
        return str(rec)

    roles = [extract(r).strip() for r in raw]
    roles = [r for r in roles if r]
    if not roles:
        sys.exit("[fatal] no roles parsed from input")

    seen, uniq = set(), []
    for r in roles:
        k = r.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    if len(uniq) != len(roles):
        print(f"[warn] dropped {len(roles) - len(uniq)} duplicate roles", flush=True)
    return uniq


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def gender_word(g: str) -> str:
    return CONFIG["GENDER_WORD_F"] if g == FEMALE else CONFIG["GENDER_WORD_M"]


def long_prompt(role: str, g: str) -> str:
    """The prompt recorded in the emitted files."""
    return CONFIG["LONG_PROMPT"].format(gender=gender_word(g), role=role)


def gen_prompt(role: str, g: str) -> str:
    """The prompt actually sent to the model during generation."""
    tmpl = CONFIG["GEN_PROMPT"] if CONFIG["GEN_PROMPT_BARE"] else CONFIG["LONG_PROMPT"]
    return tmpl.format(gender=gender_word(g), role=role)


def mc_prompt(role: str, g: str, opt_a: str, opt_b: str) -> str:
    return (f'{CONFIG["MC_HEADER"]}\n'
            f'{long_prompt(role, g)}\n'
            f'A. {opt_a}\n'
            f'B. {opt_b}\n'
            f'{CONFIG["MC_TAIL"]}')


# ---------------------------------------------------------------------------
# Per-role record
# ---------------------------------------------------------------------------
@dataclass
class Role:
    name: str
    female: Optional[str] = None
    male: Optional[str] = None
    female_letter: str = "A"      # which MC letter holds the female story
    male_letter: str = "B"
    lex_ok: bool = False
    mcqa_ok: bool = False
    ok: bool = False
    detail: dict = field(default_factory=dict)

    def options(self):
        """(option_A_text, option_B_text) for this role."""
        if self.female_letter == "A":
            return self.female, self.male
        return self.male, self.female

    def mc(self, g: str) -> str:
        a, b = self.options()
        return mc_prompt(self.name, g, a, b)

    def letter_for(self, g: str) -> str:
        return self.female_letter if g == FEMALE else self.male_letter

    def story_for(self, g: str) -> str:
        return self.female if g == FEMALE else self.male


def assign_letters(role: Role):
    if not CONFIG["PERMUTE_OPTIONS"]:
        role.female_letter, role.male_letter = "A", "B"
        return
    h = hashlib.sha256(f"{CONFIG['SEED']}:{role.name}".encode()).hexdigest()
    if int(h[:16], 16) % 2:
        role.female_letter, role.male_letter = "B", "A"
    else:
        role.female_letter, role.male_letter = "A", "B"


# ---------------------------------------------------------------------------
# HF plumbing (lazy: cached stages never load weights)
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
        sys.exit(f"[fatal] DTYPE='{name}' not recognised")
    return table[name]


def seed_everything(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return    # qc/build replays need no torch at all
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LazyModel:
    """Tokenizer on demand (cheap, CPU); weights only when generation happens."""

    def __init__(self, name: str):
        self.name = name
        self._tok = None
        self._model = None

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            print(f"[tokenizer] loading {self.name}", flush=True)
            tok = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
            tok.padding_side = "left"   # required for decoder-only batching
            if tok.pad_token is None:
                if tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                else:
                    tok.add_special_tokens({"pad_token": "<|pad|>"})
                print(f"[tokenizer] pad_token was unset; using {tok.pad_token!r}",
                      flush=True)
            self._tok = tok
        return self._tok

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM
            dtype = _resolve_dtype(CONFIG["DTYPE"])
            print(f"[model] loading {self.name} (dtype={CONFIG['DTYPE']}, "
                  f"device_map={CONFIG['DEVICE_MAP']}, attn={CONFIG['ATTN_IMPL']})",
                  flush=True)
            kwargs = dict(device_map=CONFIG["DEVICE_MAP"], trust_remote_code=True,
                          low_cpu_mem_usage=True,
                          attn_implementation=CONFIG["ATTN_IMPL"])
            try:
                m = AutoModelForCausalLM.from_pretrained(self.name, dtype=dtype, **kwargs)
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(self.name,
                                                         torch_dtype=dtype, **kwargs)
            m.eval()
            if len(self.tok) > m.get_input_embeddings().weight.shape[0]:
                m.resize_token_embeddings(len(self.tok))
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
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
    return (tok.bos_token or "") + user_content


def effective_batch_size() -> int:
    return 1 if CONFIG["STRICT_DETERMINISM"] else max(1, CONFIG["BATCH_SIZE"])


def batched_generate(handle: LazyModel, user_prompts: List[str],
                     max_tokens: int) -> List[str]:
    import torch
    if not user_prompts:
        return []
    tok = handle.tok
    model = handle.model
    device = next(model.parameters()).device
    rendered = [chat_render(tok, p) for p in user_prompts]

    outs: List[str] = []
    bs = effective_batch_size()
    for i in range(0, len(rendered), bs):
        batch = rendered[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        n_in = enc["input_ids"].shape[1]
        if n_in > CONFIG["MAX_INPUT_TOKENS"]:
            print(f"[warn] batch input is {n_in} tokens "
                  f"(> MAX_INPUT_TOKENS={CONFIG['MAX_INPUT_TOKENS']})", flush=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                 num_beams=1, use_cache=True,
                                 pad_token_id=tok.pad_token_id)
        outs.extend(t.strip() for t in
                    tok.batch_decode(gen[:, n_in:], skip_special_tokens=True))
    return outs


# ===========================================================================
# STAGE 1: tokenizer precondition
# ===========================================================================
def stage_tokcheck(handle: LazyModel):
    tok = handle.tok

    def n(s):
        return len(tok.encode(s, add_special_tokens=False))

    a, b = CONFIG["CONTRAST_A"], CONFIG["CONTRAST_B"]
    na, nb = n(a), n(b)
    na_sp, nb_sp = n(" " + a), n(" " + b)
    print(f"[tokcheck] tokens: '{a}'={na} '{b}'={nb} | "
          f"' {a}'={na_sp} ' {b}'={nb_sp}", flush=True)
    if na_sp != nb_sp:
        msg = (f"[tokcheck] FAIL: ' {a}' -> {na_sp} tokens but ' {b}' -> {nb_sp} "
               f"tokens (in-context lengths must match for position alignment).")
        if CONFIG["ALLOW_TOKEN_LENGTH_MISMATCH"]:
            print(msg + " Continuing because ALLOW_TOKEN_LENGTH_MISMATCH=1.",
                  flush=True)
        else:
            sys.exit(msg + " Set ALLOW_TOKEN_LENGTH_MISMATCH=1 to override, or "
                           "pick an equal-length pair via GENDER_WORD_F / "
                           "GENDER_WORD_M (e.g. woman / man).")
    else:
        print(f"[tokcheck] PASS: ' {a}' and ' {b}' are both {na_sp} token(s)",
              flush=True)


# ===========================================================================
# STAGE 2: generate the two stories per role (resumable, chunked)
# ===========================================================================
def stage_generate(handle: LazyModel, roles: List[str], ckpt: str):
    done = {r["role"]: r for r in load_jsonl(ckpt)}
    todo = [r for r in roles if r not in done]
    print(f"[generate] {len(done)} cached / {len(todo)} to do", flush=True)
    if not todo:
        return

    bs = effective_batch_size()
    cs = max(bs, (CONFIG["CHUNK_SIZE"] // bs) * bs)
    for i in range(0, len(todo), cs):
        chunk = todo[i:i + cs]
        t0 = time.time()
        fem = batched_generate(handle, [gen_prompt(r, FEMALE) for r in chunk],
                               CONFIG["STORY_MAX_TOKENS"])
        mal = batched_generate(handle, [gen_prompt(r, MALE) for r in chunk],
                               CONFIG["STORY_MAX_TOKENS"])
        recs = [{"role": r, "female": f, "male": m}
                for r, f, m in zip(chunk, fem, mal)]
        append_jsonl(ckpt, recs)
        print(f"[generate] chunk {i//cs + 1}: +{len(recs)} "
              f"(total {len(done)+i+len(recs)}/{len(roles)}) [{time.time()-t0:.1f}s]",
              flush=True)
        if _PREEMPTED:
            _requeue_and_exit()


# ===========================================================================
# STAGE 3: QC
# ===========================================================================
def _looks_truncated(text: str) -> bool:
    return not (text or "").rstrip().endswith((".", "!", "?", '"', "'", ")", "”", "’"))


def stage_qc(handle: LazyModel, role_names: List[str], gen_ckpt: str,
             mc_ckpt: str, out_dir: str) -> List[Role]:
    gen = {r["role"]: r for r in load_jsonl(gen_ckpt)}
    missing = [r for r in role_names if r not in gen]
    if missing:
        sys.exit(f"[fatal] qc: {len(missing)} roles missing stories "
                 f"(run the generate stage first). e.g. {missing[:3]}")

    roles = [Role(name=r, female=gen[r]["female"], male=gen[r]["male"])
             for r in role_names]
    for r in roles:
        assign_letters(r)

    mode = CONFIG["QC_MODE"]
    if mode not in ("lexical", "mcqa", "both", "none"):
        sys.exit(f"[fatal] QC_MODE must be lexical|mcqa|both|none, got '{mode}'")
    do_lex = mode in ("lexical", "both")
    do_mc = mode in ("mcqa", "both")

    # ---- (a) lexical ----------------------------------------------------
    lex_fail = []
    for r in roles:
        f_ok, f_c = lexical_gender_ok(r.female, FEMALE)
        m_ok, m_c = lexical_gender_ok(r.male, MALE)
        short = (len((r.female or "").split()) < CONFIG["MIN_STORY_WORDS"] or
                 len((r.male or "").split()) < CONFIG["MIN_STORY_WORDS"])
        trunc = CONFIG["DROP_TRUNCATED"] and (_looks_truncated(r.female) or
                                              _looks_truncated(r.male))
        r.lex_ok = f_ok and m_ok and not short and not trunc
        r.detail["lexical"] = {"female": f_c, "male": m_c,
                               "too_short": bool(short), "truncated": bool(trunc)}
        if not r.lex_ok:
            lex_fail.append({"role": r.name, **r.detail["lexical"]})
    n_lex_ok = sum(r.lex_ok for r in roles)
    if do_lex:
        print(f"[qc] lexical: {n_lex_ok}/{len(roles)} pass "
              f"(min_signal={CONFIG['MIN_GENDER_SIGNAL']}, "
              f"ratio={CONFIG['GENDER_RATIO']})", flush=True)

    # ---- (b) MCQA on the emitted item -----------------------------------
    mc_fail = []
    n_f_ok = n_m_ok = 0
    if do_mc:
        # only roles that survived the lexical gate are worth spending GPU on
        pool = [r for r in roles if r.lex_ok] if do_lex else roles
        mc_done = {r["role"]: r for r in load_jsonl(mc_ckpt)}
        todo = [r for r in pool if r.name not in mc_done]
        print(f"[qc] mcqa: {len(mc_done)} cached / {len(todo)} to verify", flush=True)

        if todo:
            bs = effective_batch_size()
            cs = max(bs, (CONFIG["CHUNK_SIZE"] // bs) * bs)
            for i in range(0, len(todo), cs):
                chunk = todo[i:i + cs]
                f_ans = batched_generate(handle, [r.mc(FEMALE) for r in chunk],
                                         CONFIG["MC_MAX_TOKENS"])
                m_ans = batched_generate(handle, [r.mc(MALE) for r in chunk],
                                         CONFIG["MC_MAX_TOKENS"])
                recs = [{"role": r.name, "female_ans": fa, "male_ans": ma}
                        for r, fa, ma in zip(chunk, f_ans, m_ans)]
                append_jsonl(mc_ckpt, recs)
                print(f"[qc] mcqa chunk {i//cs + 1}: +{len(recs)}", flush=True)
                if _PREEMPTED:
                    _requeue_and_exit()
            mc_done = {r["role"]: r for r in load_jsonl(mc_ckpt)}

        for r in pool:
            rec = mc_done.get(r.name)
            if rec is None:
                r.mcqa_ok = False
                continue
            got_f = parse_choice(rec["female_ans"])
            got_m = parse_choice(rec["male_ans"])
            ok_f = got_f == r.letter_for(FEMALE)
            ok_m = got_m == r.letter_for(MALE)
            n_f_ok += ok_f
            n_m_ok += ok_m
            r.mcqa_ok = ok_f and ok_m
            r.detail["mcqa"] = {"female_expected": r.letter_for(FEMALE),
                                "female_got": got_f,
                                "male_expected": r.letter_for(MALE),
                                "male_got": got_m}
            if not r.mcqa_ok:
                mc_fail.append({"role": r.name, **r.detail["mcqa"]})
        n_mc_ok = sum(r.mcqa_ok for r in pool)
        print(f"[qc] mcqa accuracy: female {n_f_ok}/{len(pool)}, "
              f"male {n_m_ok}/{len(pool)}, both {n_mc_ok}/{len(pool)}", flush=True)

    # ---- combine ---------------------------------------------------------
    for r in roles:
        r.ok = ((r.lex_ok if do_lex else True) and
                (r.mcqa_ok if do_mc else True))

    valid = roles if CONFIG["VERIFY_MODE"] == "report" else [r for r in roles if r.ok]
    if CONFIG["VERIFY_MODE"] not in ("filter", "report"):
        sys.exit(f"[fatal] VERIFY_MODE must be 'filter' or 'report'")

    report = {
        "model": handle.name,
        "n_input": len(roles),
        "n_valid": len(valid),
        "qc_mode": mode,
        "verify_mode": CONFIG["VERIFY_MODE"],
        "n_lexical_pass": n_lex_ok,
        "mcqa_female_correct": n_f_ok,
        "mcqa_male_correct": n_m_ok,
        "min_gender_signal": CONFIG["MIN_GENDER_SIGNAL"],
        "gender_ratio": CONFIG["GENDER_RATIO"],
        "permute_options": bool(CONFIG["PERMUTE_OPTIONS"]),
        "gen_prompt_bare": bool(CONFIG["GEN_PROMPT_BARE"]),
        "batch_size": effective_batch_size(),
        "lexical_failures": lex_fail,
        "mcqa_failures": mc_fail,
    }
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "qc_report.jsonl"), [report])
    print(f"[qc] {len(valid)}/{len(roles)} roles survive", flush=True)

    if (len(roles) - len(valid)) and CONFIG["STRICT"]:
        sys.exit(f"[fatal] STRICT=1: {len(roles)-len(valid)} role(s) failed QC "
                 f"(see {out_dir}/qc_report.jsonl)")
    if not valid:
        sys.exit("[fatal] no role passed QC")
    return valid


# ===========================================================================
# STAGE 4: build the files
# ===========================================================================
def _row(idx: int, user: str, assistant: Optional[str] = None) -> dict:
    prompt = [{"role": "user", "content": user}]
    if assistant is not None:
        prompt.append({"role": "assistant", "content": assistant})
    return {"id": idx, "prompt": prompt}


def _steer_assistant() -> Optional[str]:
    mode = CONFIG["STEERING_ASSISTANT"].lower()
    if mode == "empty":
        return ""
    if mode == "none":
        return None
    sys.exit(f"[fatal] STEERING_ASSISTANT must be 'empty' or 'none', got '{mode}'")


def stage_build(valid: List[Role], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap, tcap = CONFIG["CAP"], CONFIG["TEST_CAP"]

    head = valid[:cap]
    if CONFIG["TEST_DISJOINT"]:
        tail = valid[cap:cap + tcap]
    else:
        tail = valid[:tcap]

    if len(head) < cap:
        print(f"[build] WARNING: only {len(head)} roles for the aligned files "
              f"(CAP={cap})", flush=True)
    if len(tail) < tcap:
        print(f"[build] WARNING: only {len(tail)} roles for the test files "
              f"(TEST_CAP={tcap}; need {cap + tcap} survivors when "
              f"TEST_DISJOINT=1)", flush=True)

    # ids follow position among survivors, so head=0..99 and a disjoint test
    # block continues at 100..149. Every aligned file shares the mapping.
    ids = {r.name: i for i, r in enumerate(valid)}

    files: Dict[str, List[dict]] = {}

    def emit(name, row):
        files.setdefault(name, []).append(row)

    steer = _steer_assistant()

    for r in head:
        i = ids[r.name]
        mc_f, mc_m = r.mc(FEMALE), r.mc(MALE)
        lf, lm = long_prompt(r.name, FEMALE), long_prompt(r.name, MALE)
        fem_letter, male_letter = r.letter_for(FEMALE), r.letter_for(MALE)

        # ---- single (multiple-choice) ----
        emit("female-single-desired-all.jsonl",   _row(i, mc_f, fem_letter))
        emit("female-single-undesired-all.jsonl", _row(i, mc_f, male_letter))
        emit("female-single-steering.jsonl",      _row(i, mc_f, steer))
        emit("male-single-desired-all.jsonl",     _row(i, mc_m, male_letter))
        emit("male-single-undesired-all.jsonl",   _row(i, mc_m, fem_letter))
        emit("male-single-steering.jsonl",        _row(i, mc_m, steer))

        # ---- long (free-form) ----
        emit("female-long-desired-all.jsonl",   _row(i, lf, r.female))
        emit("female-long-undesired-all.jsonl", _row(i, lf, r.male))
        emit("female-long-steering.jsonl",      _row(i, lf, steer))
        emit("male-long-desired-all.jsonl",     _row(i, lm, r.male))
        emit("male-long-undesired-all.jsonl",   _row(i, lm, r.female))
        emit("male-long-steering.jsonl",        _row(i, lm, steer))

    for r in tail:
        i = ids[r.name]
        emit("male-single-test.jsonl", _row(i, r.mc(MALE)))
        emit("male-long-test.jsonl",   _row(i, long_prompt(r.name, MALE)))
        if CONFIG["EMIT_FEMALE_TEST"]:
            emit("female-single-test.jsonl", _row(i, r.mc(FEMALE)))
            emit("female-long-test.jsonl",   _row(i, long_prompt(r.name, FEMALE)))

    expected = [
        "female-single-desired-all.jsonl", "female-single-undesired-all.jsonl",
        "female-single-steering.jsonl",
        "male-single-desired-all.jsonl", "male-single-undesired-all.jsonl",
        "male-single-steering.jsonl", "male-single-test.jsonl",
        "female-long-desired-all.jsonl", "female-long-undesired-all.jsonl",
        "female-long-steering.jsonl",
        "male-long-desired-all.jsonl", "male-long-undesired-all.jsonl",
        "male-long-steering.jsonl", "male-long-test.jsonl",
    ]
    if CONFIG["EMIT_FEMALE_TEST"]:
        expected += ["female-single-test.jsonl", "female-long-test.jsonl"]

    for name in expected:
        rows = files.get(name, [])
        write_jsonl(os.path.join(out_dir, name), rows)
        print(f"[build] {name}: {len(rows)} rows", flush=True)

    print(f"[build] wrote {len(expected)} files to {out_dir}/ "
          f"(aligned={len(head)}, test={len(tail)})", flush=True)


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


def run_model(model_name: str, roles: List[str], stages: List[str],
              n_models: int) -> dict:
    out_dir, ckpt_dir = dirs_for(model_name, n_models)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    gen_ckpt = os.path.join(ckpt_dir, "stories.jsonl")
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
            stage_generate(handle, roles, gen_ckpt)
        if "qc" in stages:
            valid = stage_qc(handle, roles, gen_ckpt, mc_ckpt, out_dir)
        if "build" in stages:
            if valid is None:
                valid = stage_qc(handle, roles, gen_ckpt, mc_ckpt, out_dir)
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
    ap.add_argument("--no-determinism", action="store_true")
    args = ap.parse_args()

    CONFIG["INPUT"] = args.input
    if args.models:
        CONFIG["MODELS"] = args.models
    if args.batch_size is not None:
        CONFIG["BATCH_SIZE"] = args.batch_size
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ("tokcheck", "generate", "qc", "build")]
    if unknown:
        sys.exit(f"[fatal] unknown stage(s): {unknown}")

    if not args.no_determinism:
        enable_determinism()
    seed_everything(CONFIG["SEED"])

    models = parse_models()
    roles = load_roles(CONFIG["INPUT"], CONFIG["ROLE_FIELD"])
    need = CONFIG["CAP"] + (CONFIG["TEST_CAP"] if CONFIG["TEST_DISJOINT"] else 0)
    print(f"[main] {len(roles)} roles | {len(models)} model(s) | stages={stages} "
          f"| batch_size={effective_batch_size()} | need >={need} survivors",
          flush=True)
    if len(roles) < need:
        print(f"[warn] only {len(roles)} roles in the input but {need} needed "
              f"before any QC drops", flush=True)

    t0 = time.time()
    results = []
    for name in models:
        try:
            results.append(run_model(name, roles, stages, len(models)))
        except SystemExit as e:
            if not e.code:
                raise
            results.append({"model": name, "status": f"failed: {e.code}"})
            if not CONFIG["CONTINUE_ON_ERROR"]:
                raise
            print(f"[main] continuing after failure on {name}", flush=True)
        except Exception as e:  # noqa: BLE001
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