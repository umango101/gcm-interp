import json
import os
import pandas as pd
import re
import glob
from tqdm import tqdm

# ------------ Helper Functions ------------

def extract_jp_rating(judge_output):
    """Extract first integer inside judge_output."""
    if isinstance(judge_output, str):
        match = re.search(r"(\d+)", judge_output)
        return int(match.group(0)) if match else None
    elif isinstance(judge_output, int):
        return judge_output
    print("judge output not a string or int")
    return None
    

def load_json_as_df(path):
    with open(path, 'r') as f:
        content = f.read()
    # Merge concatenated JSON arrays into one
    content = content.strip().replace('][', ',')
    return json.loads(content)

# ------------ Find All Triplets Automatically ------------

jp_files = sorted(glob.glob("jp_*.judge_prompt.judge_outputs.json"))
# ------------ Load or Build DataFrames ------------
if not (
    os.path.exists('eval_para_long/Qwen1.5-14B-Chat/jp_paragraph_ratings.csv') and
    os.path.exists('eval_para_long/Qwen1.5-14B-Chat/rf_fluency_ratings.csv') and 
    os.path.exists('eval_para_long/Qwen1.5-14B-Chat/rf_relevance_ratings.csv')
):
    jp_p_df = []
    rf_flu_df = []
    rf_rel_df = []

    # for tid in triplet_ids:
    #     print(f"\nProcessing triplet {tid}...")

    jp_para_path     = f"eval_para_long/Qwen1.5-14B-Chat/judge_paragraph_prompts.judge_paragraph_prompt.judge_outputs.json"
    rf_flu_path         = f"eval_para_long/Qwen1.5-14B-Chat/relevance_fluency_prompts.fluency_prompt.judge_outputs.json"
    rf_rel_path         = f"eval_para_long/Qwen1.5-14B-Chat/relevance_fluency_prompts.relevance_prompt.judge_outputs.json"

    judge_para   = load_json_as_df(jp_para_path)
    rf_flu          = load_json_as_df(rf_flu_path)
    rf_rel          = load_json_as_df(rf_rel_path)

    jp_p_df.append(judge_para)
    rf_flu_df.append(rf_flu)
    rf_rel_df.append(rf_rel)

    jp_p_df = jp_p_df[0]
    rf_rel_df = rf_rel_df[0]
    rf_flu_df = rf_flu_df[0]

    print(len(jp_p_df))
    print(len(rf_flu_df))
    print(len(rf_rel_df))

    jp_p_df     = pd.DataFrame(jp_p_df)
    rf_flu_df   = pd.DataFrame(rf_flu_df)
    rf_rel_df   = pd.DataFrame(rf_rel_df)

    jp_p_df.to_csv('eval_para_long/Qwen1.5-14B-Chat/jp_paragraph_ratings.csv', index=False)
    rf_flu_df.to_csv('eval_para_long/Qwen1.5-14B-Chat/rf_fluency_ratings.csv', index=False)
    rf_rel_df.to_csv('eval_para_long/Qwen1.5-14B-Chat/rf_relevance_ratings.csv', index=False)

else:
    jp_p_df     = pd.read_csv('eval_para_long/Qwen1.5-14B-Chat/jp_paragraph_ratings.csv')
    rf_flu_df   = pd.read_csv('eval_para_long/Qwen1.5-14B-Chat/rf_fluency_ratings.csv')
    rf_rel_df   = pd.read_csv('eval_para_long/Qwen1.5-14B-Chat/rf_relevance_ratings.csv')


# ------------ Extract and Rename Ratings ------------

jp_p_df["jp_rating"] = jp_p_df["judge_output"].apply(extract_jp_rating)
rf_flu_df = rf_flu_df.rename(columns={"judge_rating": "fluency_rating"})
rf_rel_df = rf_rel_df.rename(columns={"judge_rating": "relevance_rating"})
# # print(jp_df.head())
# # print(jp_df.columns)

print(jp_p_df.columns.tolist())
print(rf_flu_df.columns.tolist())
print(rf_rel_df.columns.tolist())
print(jp_p_df.head(2))

# # ------------ Define Keys ------------

eval_dirs = ["paragraph-long_eval"]
# steer_dirs = ["paragraph-long_steer", "paragraph-single_steer"]
steer_dirs = ["paragraph-long_steer", "paragraph-single_steer"]
source_base_combos = [["paragraph-long", "sentence-long"], ["paragraph-single", "sentence-single"]]
method = "atp"
Ns = [1, 2, 4, 5, 6, 8, 10]
top_ks = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
# localization_dirs = ["from_paragraph-long_to_sentence-long", "from_paragraph-single_to_sentence-single"]

def calculate_accuracy(dataset, eval_dir, steer_dir, source, base, n, top_k, col_name, target_rating):
    filtered = dataset[
        (dataset["EVAL_SUB_DIR"] == eval_dir) &
        (dataset["STEER_SUB_DIR"] == steer_dir) &
        (dataset["SOURCE"] == source) &
        (dataset["BASE"] == base) &
        (dataset["N"] == n) &
        (dataset["topk"] == top_k)
    ]
    # print(dataset["judge_rating"].dtype)
    if filtered.empty:
        print("No data found for the given filters.")
        return -1
    accuracy = (filtered[col_name] >= target_rating).sum() / len(filtered)
    print(accuracy)
    return accuracy

# def calculate_combined_accuracy(dataset_jp, dataset_flu, dataset_rel, eval_dir, steer_dir, source, base, n, top_k, col_name_jp, col_name_flu, col_name_rel, target_rating_jp, target_rating_flu, target_rating_rel):
#     filtered_jp = dataset_jp[
#         (dataset_jp["EVAL_SUB_DIR"] == eval_dir) &
#         (dataset_jp["STEER_SUB_DIR"] == steer_dir) &
#         (dataset_jp["SOURCE"] == source) &
#         (dataset_jp["BASE"] == base) &
#         (dataset_jp["N"] == n) &
#         (dataset_jp["topk"] == top_k)
#     ]
#     filtered_flu = dataset_flu[
#         (dataset_flu["EVAL_SUB_DIR"] == eval_dir) &
#         (dataset_flu["STEER_SUB_DIR"] == steer_dir) &
#         (dataset_flu["SOURCE"] == source) &
#         (dataset_flu["BASE"] == base) &
#         (dataset_flu["N"] == n) &
#         (dataset_flu["topk"] == top_k)
#     ]
#     filtered_rel = dataset_rel[
#         (dataset_rel["EVAL_SUB_DIR"] == eval_dir) &
#         (dataset_rel["STEER_SUB_DIR"] == steer_dir) &
#         (dataset_rel["SOURCE"] == source) &
#         (dataset_rel["BASE"] == base) &
#         (dataset_rel["N"] == n) &
#         (dataset_rel["topk"] == top_k)
#     ]
#     condition1 = (filtered_jp[col_name_jp] >= target_rating_jp)
#     condition2 = (filtered_flu[col_name_flu] >= target_rating_flu)
#     condition3 = (filtered_rel[col_name_rel] >= target_rating_rel)
#     if filtered_jp.empty or filtered_flu.empty or filtered_rel.empty:
#         print("No data found for the given filters.")
#         return -1
#     combined_condition = condition1 & condition2 & condition3
#     count_of_matching_rows = combined_condition.sum()
#     accuracy = count_of_matching_rows / len(filtered_jp)
#     return accuracy

def calculate_combined_accuracy(dataset_jp, dataset_flu, dataset_rel,
                                eval_dir, steer_dir, source, base, n, top_k,
                                col_name_jp, col_name_flu, col_name_rel,
                                target_rating_jp, target_rating_flu, target_rating_rel):
    
    filter_cols = ["EVAL_SUB_DIR", "STEER_SUB_DIR", "SOURCE", "BASE", "N", "topk"]
    filter_vals = [eval_dir, steer_dir, source, base, n, top_k]
    
    def apply_filter(df):
        mask = pd.Series(True, index=df.index)
        for c, v in zip(filter_cols, filter_vals):
            mask &= (df[c] == v)
        return df[mask]
    
    f_jp  = apply_filter(dataset_jp)
    f_flu = apply_filter(dataset_flu)
    f_rel = apply_filter(dataset_rel)
    
    if f_jp.empty or f_flu.empty or f_rel.empty:
        return -1
    
    # You need a join key — whatever uniquely identifies a prompt/example.
    # Common candidates: "prompt", "prompt_id", "example_id", "behavior", etc.
    # Replace JOIN_KEYS with whatever actually identifies a row across the three files.
    JOIN_KEYS = ["query"]   # <-- adjust to match your data
    
    merged = (
        f_jp[JOIN_KEYS + [col_name_jp]]
        .merge(f_flu[JOIN_KEYS + [col_name_flu]], on=JOIN_KEYS, how="inner")
        .merge(f_rel[JOIN_KEYS + [col_name_rel]], on=JOIN_KEYS, how="inner")
    )
    
    if merged.empty:
        return -1
    
    combined = (
        (merged[col_name_jp]  >= target_rating_jp) &
        (merged[col_name_flu] >= target_rating_flu) &
        (merged[col_name_rel] >= target_rating_rel)
    )
    return combined.sum() / len(merged)
    

for source, base in source_base_combos:
    for steer_dir in steer_dirs:
        for eval_dir in eval_dirs:
            for n in Ns:
                for top_k in top_ks:
                    jp_p_accuracy_5 = calculate_accuracy(
                        jp_p_df, 
                        eval_dir,
                        steer_dir, 
                        source, 
                        base,
                        n,
                        top_k, 
                        "jp_rating",
                        5
                    )
                    jp_p_accuracy_4 = calculate_accuracy(
                        jp_p_df, 
                        eval_dir,
                        steer_dir, 
                        source, 
                        base,
                        n,
                        top_k, 
                        "jp_rating",
                        4
                    )
                    jp_p_accuracy_3 = calculate_accuracy(
                        jp_p_df, 
                        eval_dir,
                        steer_dir, 
                        source, 
                        base,
                        n,
                        top_k, 
                        "jp_rating",
                        3
                    )
                    rel_accuracy = calculate_accuracy(
                        rf_rel_df, 
                        eval_dir,
                        steer_dir, 
                        source, 
                        base,
                        n,
                        top_k, 
                        "relevance_rating",
                        2
                    )
                    flu_accuracy = calculate_accuracy(
                        rf_flu_df, 
                        eval_dir,
                        steer_dir, 
                        source, 
                        base,
                        n,
                        top_k, 
                        "fluency_rating",
                        2
                    )
                    comb_p_accuracy = calculate_combined_accuracy(
                        jp_p_df, 
                        rf_flu_df, 
                        rf_rel_df,
                        eval_dir,
                        steer_dir, 
                        source, 
                        base,
                        n,
                        top_k, 
                        "jp_rating",
                        "fluency_rating",
                        "relevance_rating",
                        5,
                        2,
                        2
                    )

                    output_file_p_jp_5 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_paragraph_5.json.accuracy.json"
                    output_file_p_jp_4 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_paragraph_4.json.accuracy.json"
                    output_file_p_jp_3 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_paragraph_3.json.accuracy.json"
                    output_file_rel = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_rel.json.accuracy.json"
                    output_file_flu = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_flu.json.accuracy.json"
                    output_file_p_comb = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_comb_paragraph.json.accuracy.json"
                    loc_dir = "from_" + source + "_to_" + base

                    output_dir = "results/accuracy/Qwen1.5-14B-Chat/" + loc_dir + "/" + method + "/" + eval_dir + "/" + steer_dir + "/"
                    os.makedirs(output_dir, exist_ok=True)
                    path_jp = os.path.join(output_dir, output_file_p_jp_5)
                    with open(path_jp, "w") as f:
                        json.dump({"q1": jp_p_accuracy_5}, f, indent=2)
                    path_jp = os.path.join(output_dir, output_file_p_jp_4)
                    with open(path_jp, "w") as f:
                        json.dump({"q1": jp_p_accuracy_4}, f, indent=2)
                    path_jp = os.path.join(output_dir, output_file_p_jp_3)
                    with open(path_jp, "w") as f:
                        json.dump({"q1": jp_p_accuracy_3}, f, indent=2)
                    path_rel = os.path.join(output_dir, output_file_rel)
                    with open(path_rel, "w") as f:
                        json.dump({"q1": rel_accuracy}, f, indent=2)
                    path_flu = os.path.join(output_dir, output_file_flu)
                    with open(path_flu, "w") as f:
                        json.dump({"q1": flu_accuracy}, f, indent=2)
                    path_comb = os.path.join(output_dir, output_file_p_comb)
                    with open(path_comb, "w") as f:
                        json.dump({"q1": comb_p_accuracy}, f, indent=2)