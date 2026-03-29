import os
import json
import glob
import pandas as pd
from pathlib import Path
import re
import numpy as np
import math
BASE_DIR = '/mnt/align4_drive/arunas/multi-token/gcm-interp'

RUNS_DIR = f"{BASE_DIR}/results"
DATA_DIR = f"{BASE_DIR}/data"

GEN_RE = re.compile(
    r"""
    (?P<N>\d+)_
    (?P<REPS>random)_
    (?P<STEERING_METHOD>steer|mean)_
    (?P<topk>\d\.\d+)_
    (?P<TEST_FILE>.+?-long)
    _gen\.json$
    """,
    re.VERBOSE
)

def extract_path_metadata(path):
    parts = Path(path).parts
    print('Path parts:', parts)
    runs_idx = parts.index("results")
    MODEL_ID = parts[runs_idx + 1]
    from_to = parts[runs_idx + 2]
    _, source, _, base = from_to.split("_")
    METHOD = parts[runs_idx + 3]
    if METHOD not in ['acp', 'atp', 'atp-zero', 'probes', 'random']:
        raise ValueError(f"Unexpected METHOD: {METHOD} in path: {path}")
    EVAL_SUB_DIR = parts[runs_idx + 4]
    STEER_SUB_DIR = parts[runs_idx + 5]
    SUB_DIR = parts[runs_idx + 6]
    if SUB_DIR not in ['eval']:
        raise ValueError(f"Unexpected SUB_DIR: {SUB_DIR} in path: {path}")

    filename = parts[runs_idx + 7]
    m = GEN_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    md = m.groupdict()

    return {
        "MODEL_ID": MODEL_ID,
        "SOURCE": source,
        "BASE": base,
        "METHOD": METHOD,
        "EVAL_SUB_DIR": EVAL_SUB_DIR,
        "STEER_SUB_DIR": STEER_SUB_DIR,
        **md,
        "filename": filename,
    }


def load_logits_queries(MODEL_ID, SOURCE, BASE):
    logits_path = f"{DATA_DIR}/{MODEL_ID}/{SOURCE}/{BASE}-test.jsonl"
    queries = []
    with open(logits_path) as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj["prompt"][-1]['content'])
    return queries


def load_rating_file(path):
    """
    Load one rating file.
    File format: a JSON list of strings (your example).
    """
    with open(path) as f:
        return json.load(f)


def main():

    output = []

    # ---------------------------
    # Glob gen.json files
    # ---------------------------
    print("Globbing generation files from:", RUNS_DIR, " pattern: ", f"{RUNS_DIR}/**/*long_gen.json")
    gen_files = sorted(glob.glob(f"{RUNS_DIR}/**/*long_gen.json", recursive=True))
    print(f"Found {len(gen_files)} gen files")
    gen_files = [f for f in gen_files if GEN_RE.search(Path(f).name)]
    print(f"Found {len(gen_files)} gen files after filtering with regex")
    # ---------------------------
    # Glob matching rating files
    # (Adjust pattern if your filenames differ!)
    # ---------------------------
    # rating_files = [g.replace('_gen.json', '_gen_accuracy_responses.json') for g in gen_files]
    # print(f"Found {len(rating_files)} rating files")

    # if len(gen_files) != len(rating_files):
    #     print("WARNING: gen files and rating files counts do NOT match!")
    logits_cache = {}

    for gpath in gen_files:
        try:
            meta = extract_path_metadata(gpath)
        except ValueError as e:
            print(f"Skipping {gpath}: {e}")
            continue
        MODEL_ID = meta["MODEL_ID"]
        SOURCE = meta["SOURCE"]
        BASE = meta["BASE"]

        # Load gen.json items
        with open(gpath) as f:
            items = json.load(f)

        # Load logits
        cache_key = (MODEL_ID, SOURCE, BASE)
        if cache_key not in logits_cache:
            logits_cache[cache_key] = load_logits_queries(MODEL_ID, SOURCE, BASE)
        logits_queries = logits_cache[cache_key]

        old_key = f"old_{BASE}"
        edit_key = f"edit_{BASE}"

        # ---------------------------
        # Merge + filter by rating
        # ---------------------------
        for i, item in enumerate(items):
            record = {
                "query": item["query"].strip().replace("\r", "\n"),
                "post-intervention-response": item[edit_key].strip().replace("\r", "\n"),
                "original-response": item[old_key].strip().replace("\r", "\n"),
                "filename": meta["filename"].strip().replace("\r", "\n"),
                "data_path_query": logits_queries[i].strip().replace("\r", "\n"),
                **{k: v for k, v in meta.items() if k not in ["BASE", "SOURCE"]},  # optional
                "BASE": BASE,
                "SOURCE": SOURCE
            }
            for key, value in record.items():

                # Reject NaNs
                if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
                    raise ValueError(f"Invalid value for '{key}': NaN")

                # Keep ints, floats, strings
                if isinstance(value, (str, int, float)):
                    continue

                # Anything else is invalid
                raise TypeError(
                    f"Invalid type for '{key}': {type(value).__name__} ({value}) — "
                    "must be str, int, or float"
                )
            output.append(record)

    df = pd.DataFrame(output)
    if df.isna().any().any():
        # Identify WHERE the NaNs are
        nan_locations = df[df.isna().any(axis=1)]
        raise ValueError(
            "NaNs detected in dataframe!\n" +
            nan_locations.to_string(index=False)
        )

    df.to_csv("merged_eval_outputs.csv", index=False)

    print("Saved merged_eval_outputs.csv")
    print("Final shape:", df.shape)
    return df


if __name__ == "__main__":
    df = main()
