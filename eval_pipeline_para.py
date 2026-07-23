"""
Consolidated paragraph->sentence evaluation pipeline (multi-cell).

Replaces the five-file chain:
    one-giant-eval-file.py
    gen-judge-qs-dataframe.py
    gen-relevance-fluency-dataframe.py
    umang_accuracies.py
    umang_plots.py

EVERYTHING is driven by the single CONFIG block below. The experiment family
(paragraph->sentence), the model, the method, and the grid are hardcoded in ONE
place, so you cannot cross experiment families by accident.

Within that family the pipeline sweeps every CELL, where a cell is one
combination of:
    LOCALIZATION (from_{SOURCE}_to_{BASE})  x  EVAL_SUB_DIR  x  STEER_SUB_DIR

For paragraph->sentence that is the full 2 x 2 x 2 = 8 cells:
    {paragraph-long, paragraph-single} localization
    x {paragraph-long_eval, paragraph-single_eval}
    x {paragraph-long_steer, paragraph-single_steer}

Per-cell subtleties (verified against the data and handled below):
  * Gen-file KEYS  (old_/edit_X) track the LOCALIZATION base  -> drives BASE.
  * Gen FILENAME   suffix tracks the EVAL dir's source         -> drives regex.
  * Test QUERIES   come from the EVAL dir's source test jsonl.

TWO EVALUATION METHODS -- the single/long split
-----------------------------------------------
'single' and 'long' name the RESPONSE FORMAT, not sentence-vs-paragraph:

  * SINGLE-eval: the test prompt is an MCQA item and the model emits ONE token,
    a letter. Scored by the answer key below (did the letter move?). The 70B
    length / relevance / fluency judge is meaningless on a bare "B" and is NOT
    run for these cells by default.

  * LONG-eval: the test prompt is a bare generation prompt ("Please give a one
    sentence summary of The Baron in the Trees") and the model emits free text.
    There is no MCQA option list and therefore NO ANSWER KEY. Scored purely by
    the 70B judge (length / relevance / fluency).

The discriminator is `cell.is_single_eval`, derived from `cell.eval_source`
(the EVAL subdir) -- NOT from `cell.base`. Those two disagree in 4 of the 8
cells because the grid is a full cross product, so gating on the localization
base would run the answer key on long-eval cells. That was the old bug:
`extract_title` was reached with a bare generation prompt and raised.

Single-eval scoring (MCQA answer key)
-------------------------------------
The MCQA items shuffle their four options per book, so there is no fixed
expected letter and no substring rule can work. Two of the four options
summarize the target book -- one DETAILED (long) and one BRIEF (short) -- and
the other two summarize a different book. The key is therefore derived from the
test prompts themselves: among the two options that name the book, the longer
is the detailed summary and the shorter is the brief one.

A generation counts as a success when the model's answer MOVED from the
brief letter (unsteered) to the detailed letter (post-intervention). See
MCQA_FROM / MCQA_TO / MCQA_SCORE_MODE to flip or relax that.

Stages
------
1. merge          : glob gen.json -> one merged_eval_outputs.csv PER CELL
2. validate_key   : sanity-check the derived answer key (SINGLE-eval cells only)
3. build_prompts  : judge / relevance / fluency prompt columns (judge cells only)
4. judge          : run the vLLM 70B judge over those columns, needs GPU
5. accuracies     : sweep (N x topk) -> per-cell accuracy json
6. plots          : per-cell json -> seaborn heatmaps

Run all stages, all cells:   python eval_pipeline_para.py
Run a subset of stages:      python eval_pipeline_para.py --stages merge accuracies
Check the answer key only:   python eval_pipeline_para.py --stages validate_key
Run only the GPU step:       python eval_pipeline_para.py --stages judge

With no --stages, the stage list is derived from the SELECTED cells: single-eval
cells run CPU-only (merge / validate_key / accuracies / plots), long-eval cells
add build_prompts + judge because the judge is their only metric.

Any missing input file or empty filter is a HARD ERROR (fail fast), by design.
"""

import os
import re
import glob
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
#                                  CONFIG
#        The ONLY place experiment identity is defined. Edit here only.
# =============================================================================

# --- experiment family (paragraph -> sentence) -------------------------------
MODEL_ID = "Qwen1.5-14B-Chat"
METHOD   = "atp"

# Localizations to sweep. Each is (SOURCE, BASE); BASE drives the gen keys.
LOCALIZATIONS = [
    ("paragraphMCQA-long",   "sentenceMCQA-long"),
    ("paragraphMCQA-single", "sentenceMCQA-single"),
]
# Eval / steer subdirs to sweep (full cross product with the localizations).
EVAL_SUB_DIRS  = ["paragraphMCQA-long_eval",  "paragraphMCQA-single_eval"]
STEER_SUB_DIRS = ["paragraphMCQA-long_steer", "paragraphMCQA-single_steer"]

# Suffix on the EVAL source that marks a single-token (MCQA letter) eval.
SINGLE_EVAL_SUFFIX = "-single"

# --- sweep grid (must match what was actually generated) ---------------------
NS           = [1, 2, 4, 5, 6, 8, 10]
TOP_KS       = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
STEER_METHOD = "steer"          # 'steer' or 'mean'; part of the gen filename

# --- single-eval scoring (MCQA answer key) -----------------------------------
# Success = the answer moved FROM one summary letter TO the other. Default is
# brief -> detailed; swap the two to score the opposite direction.
MCQA_FROM = "brief"              # letter the UNSTEERED model is expected to give
MCQA_TO   = "detailed"           # letter the STEERED model should give

# 'changed'  : denominator = rows whose unsteered answer == MCQA_FROM letter.
#              numerator   = those whose steered answer == MCQA_TO letter.
#              This is the literal "the answer changed from X to Y".
# 'to_only'  : denominator = every scorable row, regardless of the baseline.
#              Constant denominator across the grid; easier to compare cells.
MCQA_SCORE_MODE = "changed"

# Rows the baseline answer cannot be parsed from are dropped under 'changed'
# and counted as incorrect under 'to_only'. Both are reported per grid point in
# mcqa_diagnostics.json so a small denominator can never hide.

# --- LLM judge ---------------------------------------------------------------
# LONG-eval cells ALWAYS use the judge: it is their only metric, so this flag
# cannot switch it off. The flag controls only whether the judge additionally
# runs on SINGLE-eval cells, where it grades a one-letter response and is
# almost never what you want.
JUDGE_METRICS_ON_SINGLE_EVAL = False

# Length judge ("response (1) is longer") is on a 1-5 scale; report at several thresholds.
# Relevance & fluency are 0-2; count rating == 2.
JUDGE_THRESHOLDS = [3, 4, 5]
RELEVANCE_TARGET = 2
FLUENCY_TARGET   = 2
COMBINED_JUDGE_TARGET = 5        # combined: judge>=5 AND fluency==2 AND relevance==2

# --- single source of truth for every directory ------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")          # *_gen.json live here (input)
DATA_DIR    = os.path.join(BASE_DIR, "data")             # *-test.jsonl live here (input)
EVAL_ROOT   = os.path.join(BASE_DIR, "eval_pipeline")    # intermediate CSV/JSON (output)
OUT_ROOT    = os.path.join(BASE_DIR, "results_pipeline") # accuracies + plots (output)

JUDGE_MODEL_NAME = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
PROMPT_TOKENIZER = "meta-llama/Llama-3.1-70B-Instruct"


# =============================================================================
#                                   CELL
# =============================================================================

def eval_source_of(eval_sub_dir):
    """'paragraph-single_eval' -> 'paragraph-single' (drives filename + test file)."""
    if not eval_sub_dir.endswith("_eval"):
        raise ValueError(f"EVAL_SUB_DIR must end with '_eval': {eval_sub_dir}")
    return eval_sub_dir[: -len("_eval")]


def base_for_eval_source(eval_source):
    """'paragraph-single' -> 'sentence-single' (the eval dataset's base/test name)."""
    return eval_source.replace("paragraphMCQA", "sentenceMCQA")


class Cell:
    """One (localization x eval x steer) analysis cell. All paths derive from it."""

    def __init__(self, source, base, eval_sub_dir, steer_sub_dir):
        self.source = source                # localization source (gen keys' partner)
        self.base = base                    # localization base   (gen keys: old_/edit_{base})
        self.eval_sub_dir = eval_sub_dir
        self.steer_sub_dir = steer_sub_dir
        self.eval_source = eval_source_of(eval_sub_dir)        # gen filename suffix + test file
        self.eval_base = base_for_eval_source(self.eval_source)

        # THE discriminator for which evaluation method applies. Derived from the
        # EVAL source, never from the localization base -- see module docstring.
        self.is_single_eval = self.eval_source.endswith(SINGLE_EVAL_SUFFIX)

        self.localization = f"from_{source}_to_{base}"
        rel = os.path.join(MODEL_ID, self.localization, METHOD, eval_sub_dir, steer_sub_dir)

        # input: the one eval folder of gen files for this cell
        self.gen_dir = os.path.join(
            RESULTS_DIR, MODEL_ID, self.localization, METHOD,
            eval_sub_dir, steer_sub_dir, "eval",
        )
        # gen filename suffix tracks the EVAL source, not the localization
        self.gen_re = re.compile(
            r"^(?P<N>\d+)_targeted_(?P<STEERING_METHOD>steer|mean)_"
            r"(?P<topk>\d+(?:\.\d+)?)_" + re.escape(self.eval_source) + r"_gen\.json$"
        )
        # test queries come from the EVAL source's dataset
        self.test_jsonl = os.path.join(
            DATA_DIR, MODEL_ID, self.eval_source, f"{self.eval_base}-test.jsonl"
        )

        # outputs (one folder per cell -> never cross cells)
        self.eval_dir = os.path.join(EVAL_ROOT, rel)
        self.out_dir = os.path.join(OUT_ROOT, rel)
        self.merged_csv = os.path.join(self.eval_dir, "merged_eval_outputs.csv")
        self.prompts_csv = os.path.join(self.eval_dir, "eval_prompts.csv")
        self.judge_out = os.path.join(self.eval_dir, "judge.judge_outputs.json")
        self.relevance_out = os.path.join(self.eval_dir, "relevance.judge_outputs.json")
        self.fluency_out = os.path.join(self.eval_dir, "fluency.judge_outputs.json")
        self.accuracy_dir = os.path.join(self.out_dir, "accuracy")
        self.plots_dir = os.path.join(self.out_dir, "plots")

    @property
    def eval_kind(self):
        return "single" if self.is_single_eval else "long"

    def __str__(self):
        return (f"{self.localization} | {self.eval_sub_dir} | {self.steer_sub_dir} "
                f"[{self.eval_kind}-eval]")


def all_cells():
    cells = []
    for source, base in LOCALIZATIONS:
        for eval_sub_dir in EVAL_SUB_DIRS:
            for steer_sub_dir in STEER_SUB_DIRS:
                cells.append(Cell(source, base, eval_sub_dir, steer_sub_dir))
    return cells


def cell_uses_answer_key(cell):
    """Only single-token MCQA evals have an option list to key against."""
    return cell.is_single_eval


def cell_uses_judge(cell):
    """Long-eval cells always; single-eval cells only if explicitly enabled."""
    return (not cell.is_single_eval) or JUDGE_METRICS_ON_SINGLE_EVAL


# =============================================================================
#                                  HELPERS
# =============================================================================

def _require(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{what}] required path does not exist: {path}")
    return path


def _validate_record_values(record):
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            raise ValueError(f"Invalid value for '{key}': NaN")
        if not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Invalid type for '{key}': {type(value).__name__} ({value}) "
                "- must be str, int, or float"
            )


# =============================================================================
#                          STAGE 1 - MERGE GEN FILES
# =============================================================================

def load_test_queries(cell):
    _require(cell.test_jsonl, "test-queries")
    queries = []
    with open(cell.test_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line)["prompt"][-1]["content"])
    return queries


def stage_merge(cell):
    _require(cell.gen_dir, "results-eval-folder")
    gen_files = sorted(glob.glob(os.path.join(cell.gen_dir, "*_gen.json")))
    gen_files = [f for f in gen_files if cell.gen_re.match(Path(f).name)]
    if not gen_files:
        raise FileNotFoundError(
            f"No gen files matching {cell.gen_re.pattern!r} under {cell.gen_dir}"
        )

    queries = load_test_queries(cell)
    old_key, edit_key = f"old_{cell.base}", f"edit_{cell.base}"

    output = []
    for gpath in gen_files:
        md = cell.gen_re.match(Path(gpath).name).groupdict()
        with open(gpath) as f:
            items = json.load(f)
        if old_key not in items[0] or edit_key not in items[0]:
            raise KeyError(
                f"Expected keys '{old_key}'/'{edit_key}' in {gpath}; "
                f"found {list(items[0].keys())}"
            )
        if len(items) != len(queries):
            raise ValueError(
                f"Row count mismatch in {gpath}: {len(items)} gen items vs "
                f"{len(queries)} test queries ({cell.test_jsonl}). Refusing to misalign."
            )
        for i, item in enumerate(items):
            record = {
                "query": item["query"].strip().replace("\r", "\n"),
                "post-intervention-response": item[edit_key].strip().replace("\r", "\n"),
                "original-response": item[old_key].strip().replace("\r", "\n"),
                "filename": Path(gpath).name,
                "data_path_query": queries[i].strip().replace("\r", "\n"),
                "MODEL_ID": MODEL_ID,
                "METHOD": METHOD,
                "LOCALIZATION": cell.localization,
                "EVAL_SUB_DIR": cell.eval_sub_dir,
                "STEER_SUB_DIR": cell.steer_sub_dir,
                "EVAL_KIND": cell.eval_kind,
                "N": int(md["N"]),
                "REPS": "targeted",
                "STEERING_METHOD": md["STEERING_METHOD"],
                "topk": float(md["topk"]),
                "SOURCE": cell.source,
                "BASE": cell.base,
            }
            _validate_record_values(record)
            output.append(record)

    df = pd.DataFrame(output)
    if df.isna().any().any():
        bad = df[df.isna().any(axis=1)]
        raise ValueError("NaNs detected in merged dataframe!\n" + bad.to_string(index=False))

    os.makedirs(cell.eval_dir, exist_ok=True)
    df.to_csv(cell.merged_csv, index=False)
    print(f"    merge: {len(gen_files)} files -> {df.shape} rows  ({cell.merged_csv})")
    return df


# =============================================================================
#                       STAGE 2 - BUILD JUDGE PROMPTS
# =============================================================================

LENGTH_JUDGE = (
    "Read the following conversation between a user and an AI assistant, and "
    "answer the following question.\n\n[BEGIN CONVERSATION]\n{conversation}\n"
    "[END CONVERSATION]\n\nQ. Response (1) is longer as compared to Response (2).\n"
    " (1) Strongly disagree (2) Disagree (3) Neutral (4) Agree (5) Strongly agree\n"
    "Respond with a number from 1 to 5 that best matches your response. Do not "
    "include extra characters, such as parentheses. Respond only with a number from 1 to 5."
)

FLUENCY_SYSTEM = (
    "Please act as an impartial judge and evaluate the fluency of the sentence "
    "fragment provided below. Focus solely on fluency, disregarding its "
    "completeness, relevance, coherence with any broader context, or "
    "informativeness. Begin your evaluation by briefly describing the fluency "
    "of the sentence, noting any unnatural phrasing, awkward transitions, "
    "grammatical errors, or repetitive structures that may hinder readability. "
    "After providing your explanation, rate the sentence's fluency on a scale "
    "from 0 to 2, where 0 indicates the sentence is not fluent and highly "
    "unnatural (e.g., incomprehensible or repetitive), 1 indicates it is "
    "somewhat fluent but contains noticeable errors or awkward phrasing, and 2 "
    "indicates the sentence is fluent and almost perfect. Provide your rating "
    "using this exact format: \u201cRating: [[score]]\u201d."
)

RELEVANCE_SYSTEM = (
    "Please act as an impartial judge and evaluate whether the sentence "
    "fragment provided below is related to the instruction. Focus solely on the "
    "degree of relatedness in terms of topic, regardless of grammar, coherence, "
    "or informativeness. Begin your evaluation by providing a brief explanation "
    "of whether the sentence is related to the instruction, and point out "
    "references related to the instruction. After providing your explanation, "
    "rate the instruction relevance on a scale from 0 to 2, where 0 indicates "
    "the sentence is unrelated to the instruction, 1 indicates it is somewhat "
    "related but only minimally or indirectly relevant in terms of topic, and 2 "
    "indicates it is more clearly and directly related to the instruction. "
    "Provide your rating using this exact format: \u201cRating: [[score]]\u201d."
)

# Module-level tokenizer cache so we load it once across all cells.
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(PROMPT_TOKENIZER)
    return _TOKENIZER


def stage_build_prompts(cell):
    if not cell_uses_judge(cell):
        print(f"    build_prompts: SKIP (single-eval; the length/relevance/fluency "
              f"judge grades a one-letter response). Set "
              f"JUDGE_METRICS_ON_SINGLE_EVAL=True to force it.")
        return None

    tok = _get_tokenizer()
    _require(cell.merged_csv, "merged-csv")
    df = pd.read_csv(cell.merged_csv, keep_default_na=False)

    def length_prompt(new_r, old_r, query):
        for name, v in (("new", new_r), ("old", old_r), ("query", query)):
            if not isinstance(v, str):
                raise TypeError(f"length judge: {name} is not a string: {v!r}")
        conv = f"{query}\nResponse (1): {new_r}\nResponse (2): {old_r}"
        chat = [{"role": "user", "content": LENGTH_JUDGE.format(conversation=conv)}]
        return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def fluency_prompt(sentence):
        user = f"[Sentence Fragment Start]\n{sentence}\n[Sentence Fragment End]"
        chat = [
            {"role": "system", "content": FLUENCY_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "Rating: [["},
        ]
        p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return p[: -len("<|eot_id|>")]

    def relevance_prompt(instruction, sentence):
        user = (
            f"[Instruction Start]\n{instruction}\n[Instruction End]\n"
            f"[Sentence Fragment Start]\n{sentence}\n[Sentence Fragment End]"
        )
        chat = [
            {"role": "system", "content": RELEVANCE_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "Rating: [["},
        ]
        p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return p[: -len("<|eot_id|>")]

    df["judge_prompt"] = df.apply(
        lambda r: length_prompt(
            r["post-intervention-response"], r["original-response"], r["query"]
        ),
        axis=1,
    )
    df["fluency_prompt"] = df["post-intervention-response"].apply(fluency_prompt)
    df["relevance_prompt"] = df.apply(
        lambda r: relevance_prompt(r["data_path_query"], r["post-intervention-response"]),
        axis=1,
    )

    if df.isna().any().any():
        bad = df[df.isna().any(axis=1)]
        raise ValueError("NaNs after building prompts!\n" + bad.to_string(index=False))

    df.to_csv(cell.prompts_csv, index=False)
    print(f"    build_prompts: {df.shape} -> {cell.prompts_csv}")
    return df


# =============================================================================
#                       STAGE 3 - RUN THE vLLM JUDGE
# =============================================================================

PASSTHROUGH_COLS = [
    "query", "post-intervention-response", "original-response", "filename",
    "data_path_query", "MODEL_ID", "SOURCE", "BASE", "METHOD", "LOCALIZATION",
    "EVAL_SUB_DIR", "STEER_SUB_DIR", "EVAL_KIND", "N", "REPS", "STEERING_METHOD", "topk",
]

_RATING_RE = re.compile(r"\d+")


def extract_rating(text):
    if not isinstance(text, str):
        return -1
    m = _RATING_RE.search(text.replace("(", "").replace(")", ""))
    return int(m.group(0)) if m else -1


def _judge_pass_done(out_path, expected_rows):
    """A pass is complete iff its output exists and has the expected row count."""
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path) as f:
            return len(json.load(f)) == expected_rows
    except (json.JSONDecodeError, ValueError):
        return False  # partial/corrupt -> redo


def _run_one_judge_pass(llm, sampling_params, df, prompt_col, out_path, batch_size,
                        resume=True):
    if prompt_col not in df.columns:
        raise KeyError(f"prompt column '{prompt_col}' missing from prompts csv")
    if resume and _judge_pass_done(out_path, len(df)):
        print(f"      judge pass '{prompt_col}': already complete ({len(df)} rows), skipping")
        return
    prompts = df[prompt_col].tolist()

    enriched = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        results = llm.generate(batch, sampling_params)
        rows = df.iloc[i: i + len(batch)]
        for (_, row), res in zip(rows.iterrows(), results):
            out_text = res.outputs[0].text
            enriched.append({
                **{c: row[c] for c in PASSTHROUGH_COLS if c in row},
                prompt_col: row[prompt_col],
                "judge_output": out_text,
                "judge_rating": extract_rating(out_text),
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:        # single valid JSON array
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"      judge pass '{prompt_col}': {len(enriched)} rows -> {out_path}")


def stage_judge_all(cells, batch_size=16, resume=True):
    """Load the judge ONCE, then run all three passes for every cell that needs it."""
    judge_cells = [c for c in cells if cell_uses_judge(c)]
    skipped = [c for c in cells if not cell_uses_judge(c)]
    for c in skipped:
        print(f"  skipping judge for single-eval cell: {c}")
    if not judge_cells:
        print("  no cells need the judge -> not loading the 70B model (no GPU required).")
        return

    import torch
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() == 0:
        raise RuntimeError("No GPUs detected - the judge stage needs a GPU.")
    print(f"  loading 4-bit judge on {torch.cuda.device_count()} GPU(s)...")
    llm = LLM(
        model=JUDGE_MODEL_NAME, quantization="bitsandbytes",
        tensor_parallel_size=1, pipeline_parallel_size=1, dtype="auto",
        max_num_seqs=64, max_model_len=4096,
    )
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=5)

    for cell in judge_cells:
        print(f"  judging cell: {cell}")
        _require(cell.prompts_csv, "prompts-csv")
        df = pd.read_csv(cell.prompts_csv)
        _run_one_judge_pass(llm, sampling_params, df, "judge_prompt",     cell.judge_out,     batch_size, resume)
        _run_one_judge_pass(llm, sampling_params, df, "relevance_prompt", cell.relevance_out, batch_size, resume)
        _run_one_judge_pass(llm, sampling_params, df, "fluency_prompt",   cell.fluency_out,   batch_size, resume)


# =============================================================================
#                      MCQA ANSWER KEY (derived from the prompts)
#            SINGLE-EVAL ONLY -- long-eval prompts have no option list.
# =============================================================================

LETTERS = ("A", "B", "C", "D")

# "...summary of the book The Baron in the Trees. Please respond with only..."
_TITLE_RE = re.compile(r"summary of the book\s+(.+?)\s*\.\s*Please respond", re.S)
# older prompt shape: "...summary of the book {title}\n(A) ..."
_TITLE_RE_FALLBACK = re.compile(r"summary of the book\s+(.+?)\s*\n", re.S)
_OPTION_RE = re.compile(r"\(([ABCD])\)\s*(.*?)(?=\n\([ABCD]\)|\Z)", re.S)


def _norm(text):
    """Casefold to alphanumerics so titles match inside option prose."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def extract_title(query):
    """Book title out of an MCQA test prompt. Raises on an unrecognized shape.

    Only ever called for SINGLE-eval cells. If this raises on something like
    'Please give a one sentence summary of X', the caller is a long-eval cell
    and the answer-key path should not have been entered at all -- check the
    `cell.is_single_eval` gate, not this regex.
    """
    for rx in (_TITLE_RE, _TITLE_RE_FALLBACK):
        m = rx.search(query)
        if m:
            return m.group(1).strip().strip('"')
    raise ValueError(f"Could not parse a book title from query: {query[:160]!r}")


def extract_letter(text):
    """Parse an answer letter, never matching a letter embedded in a word.

    Mirrors verify_mcqa_old._letter_from_text so verification and evaluation
    agree. Returns None when no letter can be identified.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    m = re.fullmatch(r"\(?\s*([ABCD])\s*[\).:]?\s*", s)              # "A" / "(A)" / "A."
    if m:
        return m.group(1)
    m = re.search(r"answer\b[^A-Da-d]{0,12}([ABCD])\b", s, re.IGNORECASE)  # "answer is (B)"
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([ABCD])\)", s)                                 # "(C)" anywhere
    if m:
        return m.group(1)
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", s)            # standalone letter
    return m.group(1) if m else None


class AnswerKey:
    """title -> {'brief': 'D', 'detailed': 'A'}, derived from the test prompts."""

    def __init__(self, by_title, source):
        self.by_title = by_title
        self.source = source
        self._lookup = {_norm(t): t for t in by_title}
        if len(self._lookup) != len(by_title):
            raise ValueError(f"{source}: titles collide after normalization.")

    def __len__(self):
        return len(self.by_title)

    def for_query(self, query):
        """Gold pair for a test query. Raises on a miss - never guess silently."""
        title = extract_title(query)
        hit = self._lookup.get(_norm(title))
        if hit is None:
            raise KeyError(
                f"Title {title!r} (from the test prompt) is not in the answer key "
                f"({self.source}, {len(self)} entries)."
            )
        return self.by_title[hit]


def build_answer_key(test_jsonl):
    """Derive the key from a *-test.jsonl.

    Two of the four options name the target book; of those the SHORTER is the
    brief summary and the LONGER is the detailed one. This works for held-out
    books, which is why the key is not read from a separate answers file.
    """
    _require(test_jsonl, "test-queries")
    by_title = {}
    with open(test_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            query = json.loads(line)["prompt"][-1]["content"]
            title = extract_title(query)
            options = [(m.group(1), m.group(2).strip()) for m in _OPTION_RE.finditer(query)]
            if len(options) != len(LETTERS):
                raise ValueError(
                    f"{test_jsonl}: expected {len(LETTERS)} options for {title!r}, "
                    f"parsed {len(options)}"
                )
            norm_title = _norm(title)
            hits = [(letter, body) for letter, body in options if norm_title in _norm(body)]
            if len(hits) != 2:
                raise ValueError(
                    f"{test_jsonl}: {title!r} matched {len(hits)} options by title, "
                    "expected exactly 2 (one brief, one detailed). The title is not "
                    "verbatim in both of its summaries - this book needs a manual key."
                )
            hits.sort(key=lambda lb: len(lb[1]))
            if title in by_title:
                raise ValueError(f"{test_jsonl}: duplicate title {title!r}")
            by_title[title] = {"brief": hits[0][0], "detailed": hits[1][0]}
    if not by_title:
        raise ValueError(f"{test_jsonl}: no rows -> empty answer key")
    return AnswerKey(by_title, source=os.path.basename(test_jsonl))


_KEY_CACHE = {}


def answer_key_for(cell):
    """One key per test file, cached across the cells that share it.

    Hard-refuses long-eval cells: their prompts are bare generation prompts with
    no options, so any 'key' built from them would be meaningless rather than
    merely unparseable.
    """
    if not cell_uses_answer_key(cell):
        raise AssertionError(
            f"answer_key_for() called on a long-eval cell ({cell}). Long-eval "
            "prompts have no MCQA options and no answer key; they are scored by "
            "the 70B judge. This is a gating bug in the caller."
        )
    if cell.test_jsonl not in _KEY_CACHE:
        _KEY_CACHE[cell.test_jsonl] = build_answer_key(cell.test_jsonl)
    return _KEY_CACHE[cell.test_jsonl]


def score_rows(rows, key):
    """Score merged-CSV rows: did the answer move MCQA_FROM -> MCQA_TO?

    Each row needs 'data_path_query' (or 'query'), 'original-response' and
    'post-intervention-response'. Returns a stats dict whose 'accuracy' is None
    when nothing survived the filters.
    """
    if MCQA_SCORE_MODE not in ("changed", "to_only"):
        raise ValueError(f"MCQA_SCORE_MODE must be 'changed' or 'to_only', got {MCQA_SCORE_MODE!r}")
    for label in (MCQA_FROM, MCQA_TO):
        if label not in ("brief", "detailed"):
            raise ValueError(f"MCQA_FROM/MCQA_TO must be 'brief' or 'detailed', got {label!r}")

    stats = {
        "n_rows": 0,
        "correct": 0,
        "total": 0,
        "dropped_unparsed_baseline": 0,   # no letter in the unsteered response
        "dropped_baseline_mismatch": 0,   # unsteered answer != MCQA_FROM letter
        "unparsed_steered": 0,            # scored incorrect, but worth watching
    }

    for row in rows:
        stats["n_rows"] += 1
        query = row.get("data_path_query") or row["query"]
        gold = key.for_query(query)

        old_letter = extract_letter(row["original-response"])
        new_letter = extract_letter(row["post-intervention-response"])
        if new_letter is None:
            stats["unparsed_steered"] += 1

        if MCQA_SCORE_MODE == "changed":
            if old_letter is None:
                stats["dropped_unparsed_baseline"] += 1
                continue
            if old_letter != gold[MCQA_FROM]:
                stats["dropped_baseline_mismatch"] += 1
                continue

        stats["total"] += 1
        if new_letter == gold[MCQA_TO]:
            stats["correct"] += 1

    stats["accuracy"] = (stats["correct"] / stats["total"]) if stats["total"] else None
    return stats


def stage_validate_key(cell):
    """Cheap guard: does the key agree with what the UNSTEERED model answers?

    Every derived key is only as good as the option-parsing heuristic, so before
    trusting a heatmap, check that the baseline lands on the expected letter far
    more often than the 25% you would get by chance.

    SINGLE-eval only; long-eval cells have no key to validate.
    """
    if not cell_uses_answer_key(cell):
        print("    validate_key: SKIP (long-eval; no MCQA options, no answer key).")
        return

    key = answer_key_for(cell)
    _require(cell.merged_csv, "merged-csv")
    df = pd.read_csv(cell.merged_csv, keep_default_na=False)

    counts = {"brief": 0, "detailed": 0, "other": 0, "unparsed": 0}
    for row in df.to_dict("records"):
        gold = key.for_query(row.get("data_path_query") or row["query"])
        letter = extract_letter(row["original-response"])
        if letter is None:
            counts["unparsed"] += 1
        elif letter == gold["brief"]:
            counts["brief"] += 1
        elif letter == gold["detailed"]:
            counts["detailed"] += 1
        else:
            counts["other"] += 1

    total = max(len(df), 1)
    print(f"    key: {key.source} ({len(key)} books)")
    print(f"    unsteered answers over {total} rows: "
          f"brief={counts['brief']} ({counts['brief']/total:.1%})  "
          f"detailed={counts['detailed']} ({counts['detailed']/total:.1%})  "
          f"other={counts['other']}  unparsed={counts['unparsed']}")
    expected = counts[MCQA_FROM] / total
    if expected < 0.4:
        print(f"    WARNING: only {expected:.1%} of baselines give the MCQA_FROM "
              f"('{MCQA_FROM}') letter. Under MCQA_SCORE_MODE='changed' that is the "
              "whole denominator. Check the key, or flip MCQA_FROM / MCQA_TO.")


# =============================================================================
#                       STAGE 4 - COMPUTE ACCURACIES
# =============================================================================

def _load_judge_json(path, what):
    _require(path, what)
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def _filter(df, cell, n, top_k):
    mask = (
        (df["EVAL_SUB_DIR"] == cell.eval_sub_dir)
        & (df["STEER_SUB_DIR"] == cell.steer_sub_dir)
        & (df["SOURCE"] == cell.source)
        & (df["BASE"] == cell.base)
        & (df["N"] == n)
        & (np.isclose(df["topk"].astype(float), float(top_k)))
    )
    return df[mask]


def _accuracy(df, cell, n, top_k, col, target):
    sub = _filter(df, cell, n, top_k)
    if sub.empty:
        raise ValueError(f"No rows for N={n}, topk={top_k} in '{col}' ({cell}).")
    return float((sub[col] >= target).sum() / len(sub))


def _mcqa_stats(mdf, cell, n, top_k, key):
    sub = _filter(mdf, cell, n, top_k)
    if sub.empty:
        raise ValueError(f"No merged rows for N={n}, topk={top_k} ({cell}).")
    stats = score_rows(sub.to_dict("records"), key)
    if stats["accuracy"] is None:
        raise ValueError(
            f"Empty denominator for N={n}, topk={top_k} ({cell}): "
            f"rows={stats['n_rows']} "
            f"unparsed_baseline={stats['dropped_unparsed_baseline']} "
            f"baseline_mismatch={stats['dropped_baseline_mismatch']}. "
            "Run --stages validate_key."
        )
    return stats


def _combined_accuracy_judge(jdf, fdf, rdf, cell, n, top_k):
    """LONG-eval: length judge AND fluency AND relevance, joined per query."""
    fj, ff, fr = (_filter(jdf, cell, n, top_k),
                  _filter(fdf, cell, n, top_k),
                  _filter(rdf, cell, n, top_k))
    if fj.empty or ff.empty or fr.empty:
        raise ValueError(f"Empty filter in combined accuracy for N={n}, topk={top_k} ({cell}).")

    merged = (
        fj[["query", "judge_rating"]].rename(columns={"judge_rating": "jud"})
        .merge(ff[["query", "judge_rating"]].rename(columns={"judge_rating": "flu"}), on="query")
        .merge(fr[["query", "judge_rating"]].rename(columns={"judge_rating": "rel"}), on="query")
    )
    if merged.empty:
        raise ValueError(f"Combined merge empty for N={n}, topk={top_k} ({cell}).")
    ok = (
        (merged["jud"] >= COMBINED_JUDGE_TARGET)
        & (merged["flu"] >= FLUENCY_TARGET)
        & (merged["rel"] >= RELEVANCE_TARGET)
    )
    return float(ok.sum() / len(merged))


def _combined_accuracy_mcqa(mdf, fdf, rdf, cell, n, top_k, key):
    """SINGLE-eval with the judge forced on: mcqa success AND fluency AND relevance."""
    fm, ff, fr = (_filter(mdf, cell, n, top_k),
                  _filter(fdf, cell, n, top_k),
                  _filter(rdf, cell, n, top_k))
    if fm.empty or ff.empty or fr.empty:
        raise ValueError(f"Empty filter in combined accuracy for N={n}, topk={top_k} ({cell}).")

    rows = []
    for row in fm.to_dict("records"):
        gold = key.for_query(row.get("data_path_query") or row["query"])
        old_letter = extract_letter(row["original-response"])
        if MCQA_SCORE_MODE == "changed" and old_letter != gold[MCQA_FROM]:
            continue
        rows.append({
            "query": row["query"],
            "mcqa_ok": extract_letter(row["post-intervention-response"]) == gold[MCQA_TO],
        })
    if not rows:
        raise ValueError(f"No scorable rows in combined for N={n}, topk={top_k} ({cell}).")

    merged = (
        pd.DataFrame(rows)
        .merge(ff[["query", "judge_rating"]].rename(columns={"judge_rating": "flu"}), on="query")
        .merge(fr[["query", "judge_rating"]].rename(columns={"judge_rating": "rel"}), on="query")
    )
    if merged.empty:
        raise ValueError(f"Combined merge empty for N={n}, topk={top_k} ({cell}).")
    ok = (
        merged["mcqa_ok"]
        & (merged["flu"] >= FLUENCY_TARGET)
        & (merged["rel"] >= RELEVANCE_TARGET)
    )
    return float(ok.sum() / len(merged))


def _cell_filename(name, n, top_k):
    return f"{n}_targeted_{STEER_METHOD}_topk_{top_k}_gen_accuracy_{name}.json.accuracy.json"


def metric_names(cell):
    """Metrics written by stage_accuracies and plotted by stage_plots, per cell.

    Single-eval and long-eval produce DIFFERENT metric sets, so this is a
    function of the cell -- plotting a 'mcqa' heatmap for a long-eval cell would
    just _require() a file that was never written.
    """
    names = []
    if cell_uses_answer_key(cell):
        names.append("mcqa")
    if cell_uses_judge(cell):
        names += [f"judge_{t}" for t in JUDGE_THRESHOLDS] + ["rel", "flu", "comb"]
    if not names:
        raise ValueError(f"No metrics enabled for {cell} - nothing to compute.")
    return names


def stage_accuracies(cell):
    _require(cell.merged_csv, "merged-csv")
    mdf = pd.read_csv(cell.merged_csv, keep_default_na=False)
    mdf["N"] = mdf["N"].astype(int)

    use_key = cell_uses_answer_key(cell)
    use_judge = cell_uses_judge(cell)

    key = None
    if use_key:
        key = answer_key_for(cell)
        print(f"    key={key.source} ({len(key)} books)  "
              f"{MCQA_FROM} -> {MCQA_TO}  mode={MCQA_SCORE_MODE}")
    else:
        print("    long-eval: no answer key; scoring by the 70B judge only.")

    jdf = fdf = rdf = None
    if use_judge:
        jdf = _load_judge_json(cell.judge_out, "judge-outputs")
        fdf = _load_judge_json(cell.fluency_out, "fluency-outputs")
        rdf = _load_judge_json(cell.relevance_out, "relevance-outputs")

    os.makedirs(cell.accuracy_dir, exist_ok=True)
    diagnostics = {}

    for n in NS:
        for top_k in TOP_KS:
            to_write = {}

            if use_key:
                stats = _mcqa_stats(mdf, cell, n, top_k, key)
                diagnostics[f"N={n},topk={top_k}"] = stats
                to_write["mcqa"] = stats["accuracy"]

            if use_judge:
                for thr in JUDGE_THRESHOLDS:
                    to_write[f"judge_{thr}"] = _accuracy(jdf, cell, n, top_k, "judge_rating", thr)
                to_write["rel"] = _accuracy(rdf, cell, n, top_k, "judge_rating", RELEVANCE_TARGET)
                to_write["flu"] = _accuracy(fdf, cell, n, top_k, "judge_rating", FLUENCY_TARGET)
                if use_key:
                    to_write["comb"] = _combined_accuracy_mcqa(mdf, fdf, rdf, cell, n, top_k, key)
                else:
                    to_write["comb"] = _combined_accuracy_judge(jdf, fdf, rdf, cell, n, top_k)

            for name, value in to_write.items():
                with open(os.path.join(cell.accuracy_dir, _cell_filename(name, n, top_k)), "w") as f:
                    json.dump({"q1": value}, f, indent=2)

    # Under mode='changed' the denominator varies across the grid, so keep the
    # per-cell counts next to the accuracies instead of throwing them away.
    if diagnostics:
        with open(os.path.join(cell.accuracy_dir, "mcqa_diagnostics.json"), "w") as f:
            json.dump(diagnostics, f, indent=2)
    print(f"    accuracies ({', '.join(metric_names(cell))}) -> {cell.accuracy_dir}")


# =============================================================================
#                          STAGE 5 - PLOT HEATMAPS
# =============================================================================

def _load_acc_cell(cell, name, n, top_k):
    path = os.path.join(cell.accuracy_dir, _cell_filename(name, n, top_k))
    _require(path, f"accuracy-cell:{name}")
    with open(path) as f:
        return json.load(f)["q1"]


def _heatmap(cell, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = {top_k: {n: _load_acc_cell(cell, metric, n, top_k) for n in NS} for top_k in TOP_KS}
    df = pd.DataFrame(data)  # rows = N, cols = topk

    os.makedirs(cell.plots_dir, exist_ok=True)
    df.to_csv(os.path.join(cell.plots_dir, f"{metric}_dataset.csv"))

    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(df, annot=True, vmin=0, vmax=1, cmap="Reds", fmt=".1f")
    ax.set_title(f"{MODEL_ID} - {metric} [{cell.eval_kind}-eval]\n{cell.localization}\n"
                 f"eval={cell.eval_sub_dir}  steer={cell.steer_sub_dir}")
    ax.set_ylabel("Steering Factor (N)")
    ax.set_xlabel("top_k")
    plt.tight_layout()
    plt.savefig(os.path.join(cell.plots_dir, f"{metric}_heatmap.png"))
    plt.close()


def stage_plots(cell):
    for metric in metric_names(cell):
        _heatmap(cell, metric)
    print(f"    plots -> {cell.plots_dir}")


# =============================================================================
#                                   MAIN
# =============================================================================

# Per-cell stages (run once per cell). 'judge' is handled separately so the
# 70B model is loaded only once across all cells.
PER_CELL_STAGES = {
    "merge":         stage_merge,
    "validate_key":  stage_validate_key,
    "build_prompts": stage_build_prompts,
    "accuracies":    stage_accuracies,
    "plots":         stage_plots,
}
ALL_STAGES = ["merge", "validate_key", "build_prompts", "judge", "accuracies", "plots"]


def default_stages(cells):
    """Stage list implied by the SELECTED cells.

    Single-eval-only runs stay CPU-only; any long-eval cell pulls in the judge,
    because for those cells the judge is the only source of a metric.
    """
    stages = ["merge"]
    if any(cell_uses_answer_key(c) for c in cells):
        stages.append("validate_key")
    if any(cell_uses_judge(c) for c in cells):
        stages += ["build_prompts", "judge"]
    stages += ["accuracies", "plots"]
    return stages


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None,
                    help="Which stages to run, in order. Default: derived from the "
                         "selected cells (see default_stages).")
    ap.add_argument("--batch_size", type=int, default=16, help="Judge batch size.")
    ap.add_argument("--cells", nargs="+", type=int, default=None,
                    help="0-based indices of cells to run (for SLURM array sharding). "
                         "Default: all cells.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-run judge passes even if a complete output already exists.")
    args = ap.parse_args()

    cells = all_cells()
    if args.cells is not None:
        bad = [i for i in args.cells if i < 0 or i >= len(cells)]
        if bad:
            raise IndexError(f"--cells {bad} out of range (have {len(cells)} cells, 0..{len(cells)-1})")
        cells = [cells[i] for i in args.cells]

    stages = args.stages if args.stages is not None else default_stages(cells)

    n_single = sum(1 for c in cells if c.is_single_eval)
    print(f"Family: paragraph->sentence  model={MODEL_ID}  method={METHOD}")
    print(f"Running {len(cells)} of {len(all_cells())} cells "
          f"({len(LOCALIZATIONS)} localizations x {len(EVAL_SUB_DIRS)} eval "
          f"x {len(STEER_SUB_DIRS)} steer): "
          f"{n_single} single-eval, {len(cells) - n_single} long-eval")
    for i, c in enumerate(cells):
        print(f"  - {c}  metrics={','.join(metric_names(c))}")
    print(f"Stages: {' -> '.join(stages)}")

    for stage in stages:
        print("=" * 70)
        print(f"STAGE: {stage}")
        if stage == "judge":
            stage_judge_all(cells, batch_size=args.batch_size, resume=not args.no_resume)
        else:
            fn = PER_CELL_STAGES[stage]
            for cell in cells:
                print(f"  cell: {cell}")
                fn(cell)

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
