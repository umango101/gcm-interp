"""
Consolidated extraversion->introversion evaluation pipeline (multi-cell).
 
Replaces the five-file chain:
    one-giant-eval-file.py
    gen-judge-qs-dataframe.py
    gen-relevance-fluency-dataframe.py
    umang_accuracies.py
    umang_plots.py
 
EVERYTHING is driven by the single CONFIG block below. The experiment family
(extraversion->introversion), the model, the method, and the grid are hardcoded in ONE
place, so you cannot cross experiment families by accident.
 
Within that family the pipeline sweeps every CELL, where a cell is one
combination of:
    LOCALIZATION (from_{SOURCE}_to_{BASE})  x  EVAL_SUB_DIR  x  STEER_SUB_DIR
 
For extraversion->introversion that is the full 2 x 2 x 2 = 8 cells:
    {extraversion-long, extraversion-single} localization
    x {extraversion-long_eval, extraversion-single_eval}
    x {extraversion-long_steer, extraversion-single_steer}
 
Per-cell subtleties (verified against the data and handled below):
  * Gen-file KEYS  (old_/edit_X) track the LOCALIZATION base  -> drives BASE.
  * Gen FILENAME   suffix tracks the EVAL dir's source         -> drives regex.
  * Test QUERIES   come from the EVAL dir's source test jsonl.
 
Stages
------
1. merge          : glob gen.json -> one merged_eval_outputs.csv PER CELL
2. build_prompts  : add judge / relevance / fluency prompt columns PER CELL
3. judge          : run the vLLM 70B judge over the prompt columns (needs GPU)
4. accuracies     : sweep (N x topk) -> per-cell accuracy json
5. plots          : per-cell json -> seaborn heatmaps
 
Run all stages, all cells:   python pipeline.py
Run a subset of stages:      python pipeline.py --stages merge build_prompts
Run only the GPU step:       python pipeline.py --stages judge
 
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
 
# --- experiment family (extraversion -> introversion) -------------------------------
MODEL_ID = "Qwen1.5-14B-Chat"
METHOD   = "atp"
 
# Localizations to sweep. Each is (SOURCE, BASE); BASE drives the gen keys.
LOCALIZATIONS = [
    ("extraversion-long",   "introversion-long"),
    ("extraversion-single", "introversion-single"),
]
# Eval / steer subdirs to sweep (full cross product with the localizations).
EVAL_SUB_DIRS  = ["extraversion-long_eval",  "extraversion-single_eval"]
STEER_SUB_DIRS = ["extraversion-long_steer", "extraversion-single_steer"]
 
# --- sweep grid (must match what was actually generated) ---------------------
NS           = [1, 2, 4, 5, 6, 8, 10]
TOP_KS       = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
STEER_METHOD = "steer"          # 'steer' or 'mean'; part of the gen filename
 
# --- judge thresholds --------------------------------------------------------
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
    """'extraversion-single_eval' -> 'extraversion-single' (drives filename + test file)."""
    if not eval_sub_dir.endswith("_eval"):
        raise ValueError(f"EVAL_SUB_DIR must end with '_eval': {eval_sub_dir}")
    return eval_sub_dir[: -len("_eval")]
 
 
def base_for_eval_source(eval_source):
    """'extraversion-single' -> 'introversion-single' (the eval dataset's base/test name)."""
    return eval_source.replace("extraversion", "introversion")
 
 
class Cell:
    """One (localization x eval x steer) analysis cell. All paths derive from it."""
 
    def __init__(self, source, base, eval_sub_dir, steer_sub_dir):
        self.source = source                # localization source (gen keys' partner)
        self.base = base                    # localization base   (gen keys: old_/edit_{base})
        self.eval_sub_dir = eval_sub_dir
        self.steer_sub_dir = steer_sub_dir
        self.eval_source = eval_source_of(eval_sub_dir)        # gen filename suffix + test file
        self.eval_base = base_for_eval_source(self.eval_source)
 
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
 
    def __str__(self):
        return f"{self.localization} | {self.eval_sub_dir} | {self.steer_sub_dir}"
 
 
def all_cells():
    cells = []
    for source, base in LOCALIZATIONS:
        for eval_sub_dir in EVAL_SUB_DIRS:
            for steer_sub_dir in STEER_SUB_DIRS:
                cells.append(Cell(source, base, eval_sub_dir, steer_sub_dir))
    return cells
 
 
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
 
JUDGE_PROMPT = (
    "Read the following conversation between a user and an AI assistant, and "
    "answer the following question.\n\n[BEGIN CONVERSATION]\n{conversation}\n"
    "[END CONVERSATION]\n\nQ. Response (1) displays more extroverted traits "
    "as compared to Response (2).\n (1) Strongly disagree "
    "(2) Disagree (3) Neutral (4) Agree (5) Strongly agree\nRespond with a "
    "number from 1 to 5 that best matches your response. Do not include extra "
    "characters, such as parentheses. Respond only with a number from 1 to 5."
)
 
FLUENCY_SYSTEM = (
    "Please act as an impartial judge and evaluate the fluency of the sentence "
    "fragment provided below. Focus solely on fluency, disregarding its "
    "completeness, relevance, coherence with any broader context, or "
    "informativeness. Begin your evaluation by briefly describing the fluency "
    "of the sentence, noting any unnatural phrasing, awkward transitions, "
    "grammatical errors, or repetitive structures that may hinder readability. "
    "After providing your explanation, rate the sentence's fluency on a scale "
    "from 0 to 2, where 0 indicates the introversion is not fluent and highly "
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
    tok = _get_tokenizer()
    _require(cell.merged_csv, "merged-csv")
    df = pd.read_csv(cell.merged_csv, keep_default_na=False)
 
    def judge_prompt(new_r, old_r, query):
        for name, v in (("new", new_r), ("old", old_r), ("query", query)):
            if not isinstance(v, str):
                raise TypeError(f"length judge: {name} is not a string: {v!r}")
        conv = f"{query}\nResponse (1): {new_r}\nResponse (2): {old_r}"
        chat = [{"role": "user", "content": JUDGE_PROMPT.format(conversation=conv)}]
        return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
 
    def fluency_prompt(introversion):
        user = f"[Sentence Fragment Start]\n{introversion}\n[Sentence Fragment End]"
        chat = [
            {"role": "system", "content": FLUENCY_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "Rating: [["},
        ]
        p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return p[: -len("<|eot_id|>")]
 
    def relevance_prompt(instruction, introversion):
        user = (
            f"[Instruction Start]\n{instruction}\n[Instruction End]\n"
            f"[Sentence Fragment Start]\n{introversion}\n[Sentence Fragment End]"
        )
        chat = [
            {"role": "system", "content": RELEVANCE_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "Rating: [["},
        ]
        p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return p[: -len("<|eot_id|>")]
 
    df["judge_prompt"] = df.apply(
        lambda r: judge_prompt(
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
    "EVAL_SUB_DIR", "STEER_SUB_DIR", "N", "REPS", "STEERING_METHOD", "topk",
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
    """Load the judge ONCE, then run all three passes for every cell."""
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
 
    for cell in cells:
        print(f"  judging cell: {cell}")
        _require(cell.prompts_csv, "prompts-csv")
        df = pd.read_csv(cell.prompts_csv)
        _run_one_judge_pass(llm, sampling_params, df, "judge_prompt",     cell.judge_out,     batch_size, resume)
        _run_one_judge_pass(llm, sampling_params, df, "relevance_prompt", cell.relevance_out, batch_size, resume)
        _run_one_judge_pass(llm, sampling_params, df, "fluency_prompt",   cell.fluency_out,   batch_size, resume)
 
 
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
 
 
def _combined_accuracy(jdf, fdf, rdf, cell, n, top_k):
    fj, ff, fr = (_filter(jdf, cell, n, top_k),
                  _filter(fdf, cell, n, top_k),
                  _filter(rdf, cell, n, top_k))
    if fj.empty or ff.empty or fr.empty:
        raise ValueError(f"Empty filter in combined accuracy for N={n}, topk={top_k} ({cell}).")
    merged = (
        fj[["query", "judge_rating"]]
        .merge(ff[["query", "judge_rating"]], on="query", suffixes=("_judge", "_flu"))
        .merge(fr[["query", "judge_rating"]].rename(columns={"judge_rating": "judge_rating_rel"}),
               on="query")
    )
    if merged.empty:
        raise ValueError(f"Combined merge empty for N={n}, topk={top_k} ({cell}).")
    ok = (
        (merged["judge_rating_judge"] >= COMBINED_JUDGE_TARGET)
        & (merged["judge_rating_flu"] >= FLUENCY_TARGET)
        & (merged["judge_rating_rel"] >= RELEVANCE_TARGET)
    )
    return float(ok.sum() / len(merged))
 
 
def _cell_filename(name, n, top_k):
    return f"{n}_targeted_{STEER_METHOD}_topk_{top_k}_gen_accuracy_{name}.json.accuracy.json"
 
 
def stage_accuracies(cell):
    jdf = _load_judge_json(cell.judge_out, "judge-outputs")
    fdf = _load_judge_json(cell.fluency_out, "fluency-outputs")
    rdf = _load_judge_json(cell.relevance_out, "relevance-outputs")
 
    os.makedirs(cell.accuracy_dir, exist_ok=True)
    for n in NS:
        for top_k in TOP_KS:
            to_write = {}
            for thr in JUDGE_THRESHOLDS:
                to_write[f"judge_{thr}"] = _accuracy(jdf, cell, n, top_k, "judge_rating", thr)
            to_write["rel"] = _accuracy(rdf, cell, n, top_k, "judge_rating", RELEVANCE_TARGET)
            to_write["flu"] = _accuracy(fdf, cell, n, top_k, "judge_rating", FLUENCY_TARGET)
            to_write["comb"] = _combined_accuracy(jdf, fdf, rdf, cell, n, top_k)
            for name, value in to_write.items():
                with open(os.path.join(cell.accuracy_dir, _cell_filename(name, n, top_k)), "w") as f:
                    json.dump({"q1": value}, f, indent=2)
    print(f"    accuracies -> {cell.accuracy_dir}")
 
 
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
    ax.set_title(f"{MODEL_ID} - {metric}\n{cell.localization}\n"
                 f"eval={cell.eval_sub_dir}  steer={cell.steer_sub_dir}")
    ax.set_ylabel("Steering Factor (N)")
    ax.set_xlabel("top_k")
    plt.tight_layout()
    plt.savefig(os.path.join(cell.plots_dir, f"{metric}_heatmap.png"))
    plt.close()
 
 
def stage_plots(cell):
    metrics = [f"judge_{t}" for t in JUDGE_THRESHOLDS] + ["rel", "flu", "comb"]
    for metric in metrics:
        _heatmap(cell, metric)
    print(f"    plots -> {cell.plots_dir}")
 
 
# =============================================================================
#                                   MAIN
# =============================================================================
 
# Per-cell stages (run once per cell). 'judge' is handled separately so the
# 70B model is loaded only once across all cells.
PER_CELL_STAGES = {
    "merge":         stage_merge,
    "build_prompts": stage_build_prompts,
    "accuracies":    stage_accuracies,
    "plots":         stage_plots,
}
DEFAULT_ORDER = ["merge", "build_prompts", "judge", "accuracies", "plots"]
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+",
                    choices=["merge", "build_prompts", "judge", "accuracies", "plots"],
                    default=DEFAULT_ORDER, help="Which stages to run, in order. Default: all.")
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
 
    print(f"Family: extraversion->introversion  model={MODEL_ID}  method={METHOD}")
    print(f"Running {len(cells)} of {len(all_cells())} cells "
          f"({len(LOCALIZATIONS)} localizations x {len(EVAL_SUB_DIRS)} eval "
          f"x {len(STEER_SUB_DIRS)} steer):")
    for c in cells:
        print(f"  - {c}")
 
    for stage in args.stages:
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
