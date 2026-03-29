import json
import os
import pandas as pd
import re
import glob
from tqdm import tqdm

# ------------ Helper Functions ------------

def extract_jp_rating(judge_output):
    """Extract first integer inside judge_output."""
    if not isinstance(judge_output, str):
        return None
    match = re.search(r"\b(\d+)\b", judge_output)
    return int(match.group(1)) if match else None


def load_json_as_df(path):
    with open(path, "r") as f:
        return json.load(f)


# ------------ Find All Triplets Automatically ------------

jp_files = sorted(glob.glob("jp_*.judge_prompt.judge_outputs.json"))
# ------------ Load or Build DataFrames ------------
if not (
    os.path.exists('jp_ratings.csv') and 
    os.path.exists('rf_fluency_ratings.csv') and 
    os.path.exists('rf_relevance_ratings.csv')
):
    jp_df = []
    rf_flu_df = []
    rf_rel_df = []

    # for tid in triplet_ids:
    #     print(f"\nProcessing triplet {tid}...")

    jp_path     = f"judge_prompts.judge_prompt.judge_outputs.json"
    rf_flu_path = f"relevance_fluency_prompts.fluency_prompt.judge_outputs.json"
    rf_rel_path = f"relevance_fluency_prompts.relevance_prompt.judge_outputs.json"

    judge  = load_json_as_df(jp_path)
    rf_flu = load_json_as_df(rf_flu_path)
    rf_rel = load_json_as_df(rf_rel_path)

    jp_df.append(judge)
    rf_flu_df.append(rf_flu)
    rf_rel_df.append(rf_rel)

    # Flatten nested triplet structure
    jp_df     = [jp_df[i][j]     for i in range(len(jp_df))     for j in range(len(jp_df[i]))     ]
    rf_flu_df = [rf_flu_df[i][j] for i in range(len(rf_flu_df)) for j in range(len(rf_flu_df[i])) ]
    rf_rel_df = [rf_rel_df[i][j] for i in range(len(rf_rel_df)) for j in range(len(rf_rel_df[i])) ]

    jp_df     = pd.DataFrame(jp_df)
    rf_flu_df = pd.DataFrame(rf_flu_df)
    rf_rel_df = pd.DataFrame(rf_rel_df)

    jp_df.to_csv('jp_ratings.csv', index=False)
    rf_flu_df.to_csv('rf_fluency_ratings.csv', index=False)
    rf_rel_df.to_csv('rf_relevance_ratings.csv', index=False)

else:
    jp_df     = pd.read_csv('jp_ratings.csv')
    rf_flu_df = pd.read_csv('rf_fluency_ratings.csv')
    rf_rel_df = pd.read_csv('rf_relevance_ratings.csv')


# ------------ Extract and Rename Ratings ------------

jp_df["jp_rating"] = jp_df["judge_output"].apply(extract_jp_rating)
rf_flu_df = rf_flu_df.rename(columns={"judge_rating": "fluency_rating"})
rf_rel_df = rf_rel_df.rename(columns={"judge_rating": "relevance_rating"})


# ------------ Define Keys ------------

# Unique row identifier (row-level merge)
row_key_cols = [
    "MODEL_ID","SOURCE","BASE","METHOD","EVAL_SUB_DIR","STEER_SUB_DIR",
    "N","REPS","STEERING_METHOD","topk","data_path_query"
]

# Condition-level grouping (accuracy level)
group_cols = [
    "MODEL_ID","SOURCE","BASE","METHOD","EVAL_SUB_DIR","STEER_SUB_DIR",
    "N","REPS","STEERING_METHOD","topk"
]


# ------------ Merge on row-level metadata ------------

print("Merging DFs on metadata columns...")

merged = (
    jp_df[row_key_cols + ["jp_rating"]]
    .merge(
        rf_flu_df[row_key_cols + ["fluency_rating"]],
        on=row_key_cols,
        how="left"
    )
    .merge(
        rf_rel_df[row_key_cols + ["relevance_rating"]],
        on=row_key_cols,
        how="left"
    )
)

merged.to_csv("merged_ratings.csv", index=False)


# ------------ Grouping & Accuracy Computation ------------

print('Grouping and computing accuracies...')

grouped = merged.groupby(group_cols)

for key_values, group in tqdm(grouped):

    # === COUNT-based accuracy ===
    accuracy_without_rf = float((group["jp_rating"] == 5).mean())
    accuracy_with_rf = float(
        (
            (group["jp_rating"] == 5) &
            (group["fluency_rating"] == 2) &
            (group["relevance_rating"] == 2)
        ).mean()
    )

    # === If you prefer fraction accuracy instead, uncomment ===
    # total = len(group)
    # accuracy_without_rf /= total
    # accuracy_with_rf /= total

    # Row metadata (all identical for this group)
    row = group.iloc[0]

    base_dir = (
        f"accuracy/"
        f"{row['MODEL_ID']}/"
        f"from_{row['SOURCE']}_to_{row['BASE']}/"
        f"{row['METHOD']}/"
        f"{row['EVAL_SUB_DIR']}/"
        f"{row['STEER_SUB_DIR']}/"
    )
    os.makedirs(base_dir, exist_ok=True)

    # if row['topk'] == 1.0:
    #     print(row['topk'])
    #     fn_base = f"{row['N']}_{row['REPS']}_{row['STEERING_METHOD']}_topk_1"
    # else:
    fn_base = f"{row['N']}_{row['REPS']}_{row['STEERING_METHOD']}_topk_{row['topk']}"

    path_wo = os.path.join(base_dir, fn_base + "_gen_accuracy_wo_rf.json.accuracy.json")
    with open(path_wo, "w") as f:
        json.dump({"q1": accuracy_without_rf}, f, indent=2)

    # Save WITH relevance/fluency
    path_w = os.path.join(base_dir, fn_base + "_gen_accuracy_w_rf.json.accuracy.json")
    with open(path_w, "w") as f:
        json.dump({"q1": accuracy_with_rf}, f, indent=2)
