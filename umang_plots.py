import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
import matplotlib.patches as patches
import seaborn as sns
RM_INTERP_REPO = os.path.dirname(os.path.abspath(__file__))

# ===== Constants =====
steering_factors = [1, 2, 4, 5, 6, 8, 10]
topk_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
tasks = ["from_anti-long_to_pro-long", "from_anti-single_to_pro-single"]
task_dict = {
    "from_harmful-long_to_harmless": "Refusal\n(Long)",
    "from_harmful-single_to_harmless": "Refusal\n(Single)",
}
method = "atp"
model = "Qwen1.5-14B-Chat"
# models = ["SOLAR-10.7B-Instruct-v1.0-steering-last"]
steers = ["anti-long_steer", "anti-single_steer"]
root_dir = f"{RM_INTERP_REPO}/results/accuracy/{model}"

def make_plot(dataset, model, task, eval, steer, metric, save_dir):
    dataset.to_csv(f"{save_dir}{metric}_dataset.csv")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(dataset, annot=True, vmin=0, vmax=1, cmap='viridis', fmt=".1f")

    ax.set_title(f"{model} - {metric} Outputs\nLocalization: {task}\nEvaluation: {eval}\nSteering: {steer}")
    ax.set_ylabel("Steering Factor")
    ax.set_xlabel("top_k")

    plt.savefig(f"{save_dir}{metric}_heatmap.png")
    plt.close()

# ============================================================
#                        MAIN HEATMAP PIPELINE
# ============================================================

for task in tasks:
    for steer in steers:
        # do long first, then single
        jp_data_5 = {}
        jp_data_4 = {}
        jp_data_3 = {}
        fluency_data = {}
        relevance_data = {}
        comb_data = {}
        for top_k in topk_values:
            jp_row_5 = {}
            jp_row_4 = {}
            jp_row_3 = {}
            fluency_row = {}
            relevance_row = {}
            comb_row = {}
            for n in steering_factors:
                jp_filename_5 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_5.json.accuracy.json"
                jp_filename_4 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_4.json.accuracy.json"
                jp_filename_3 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_3.json.accuracy.json"
                flu_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_flu.json.accuracy.json"
                rel_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_rel.json.accuracy.json"
                comb_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_comb.json.accuracy.json"

                jp_path_5 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_filename_5
                jp_path_4 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_filename_4
                jp_path_3 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_filename_3
                flu_path = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + flu_filename
                rel_path = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + rel_filename
                comb_path = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + comb_filename

                with open(jp_path_5, "r") as f:
                    jp_accuracy_5 = json.load(f)["q1"]
                with open(jp_path_4, "r") as f:
                    jp_accuracy_4 = json.load(f)["q1"]
                with open(jp_path_3, "r") as f:
                    jp_accuracy_3 = json.load(f)["q1"]
                with open(flu_path, "r") as f:
                    flu_accuracy = json.load(f)["q1"]
                with open(rel_path, "r") as f:
                    rel_accuracy = json.load(f)["q1"]
                with open(comb_path, "r") as f:
                    comb_accuracy = json.load(f)["q1"]

                jp_row_5[n] = jp_accuracy_5
                jp_row_4[n] = jp_accuracy_4
                jp_row_3[n] = jp_accuracy_3
                fluency_row[n] = flu_accuracy
                relevance_row[n] = rel_accuracy
                comb_row[n] = comb_accuracy
            jp_data_5[top_k] = jp_row_5
            jp_data_4[top_k] = jp_row_4
            jp_data_3[top_k] = jp_row_3
            fluency_data[top_k] = fluency_row
            relevance_data[top_k] = relevance_row
            comb_data[top_k] = comb_row

        jp_df_5 = pd.DataFrame(jp_data_5)
        jp_df_4 = pd.DataFrame(jp_data_4)
        jp_df_3 = pd.DataFrame(jp_data_3)
        fluency_df = pd.DataFrame(fluency_data)
        relevance_df = pd.DataFrame(relevance_data)
        comb_df = pd.DataFrame(comb_data)
        save_dir = f"{RM_INTERP_REPO}/results/plots/{task}/anti-long_eval/{steer}/"
        os.makedirs(save_dir, exist_ok=True)
        make_plot(jp_df_5, model, task, "anti-long_eval", steer, "jp_5", save_dir)
        make_plot(jp_df_4, model, task, "anti-long_eval", steer, "jp_4", save_dir)
        make_plot(jp_df_3, model, task, "anti-long_eval", steer, "jp_3", save_dir)
        make_plot(fluency_df, model, task, "anti-long_eval", steer, "fluency", save_dir)
        make_plot(relevance_df, model, task, "anti-long_eval", steer, "relevance", save_dir)
        make_plot(comb_df, model, task, "anti-long_eval", steer, "comb", save_dir)
        
        # now do single
        single_data = {}
        for top_k in topk_values:
            row = {}
            for n in steering_factors:
                filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_w_rf.json.accuracy.json"

                path = root_dir + "/" + task + "/" + method + "/anti-single_eval/" + steer + "/" + filename

                with open(path, "r") as f:
                    accuracy = json.load(f)["q1"]

                row[n] = accuracy
            single_data[top_k] = row

        df = pd.DataFrame(single_data)
        save_dir = f"{RM_INTERP_REPO}/results/plots/{task}/anti-single_eval/{steer}/"
        os.makedirs(save_dir, exist_ok=True)
        make_plot(df, model, task, "anti-single_eval", steer, "single", save_dir)



        
