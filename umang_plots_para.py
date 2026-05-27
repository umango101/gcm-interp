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
tasks = ["from_paragraph-long_to_sentence-long", "from_paragraph-single_to_sentence-single"]
method = "atp"
model = "Qwen1.5-14B-Chat"
# models = ["Qwen1.5-14B-Chat-steering-last"]
steers = ["paragraph-long_steer", "paragraph-single_steer"]
root_dir = f"{RM_INTERP_REPO}/results/accuracy/{model}"

def make_plot(dataset, model, task, eval, steer, metric, save_dir):
    dataset.to_csv(f"{save_dir}{metric}_dataset.csv")
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(dataset, annot=True, vmin=0, vmax=1, cmap='Reds', fmt=".1f")

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
        steer_dir = steer
        # if steer == "paragraph-long_steer":
        #     steer_dir = "paragraph-long_steer"
        # do long first, then single
        jp_p_data_5 = {}
        jp_p_data_4 = {}
        jp_p_data_3 = {}
        fluency_data = {}
        relevance_data = {}
        comb_p_data = {}
        for top_k in topk_values:
            jp_p_row_5 = {}
            jp_p_row_4 = {}
            jp_p_row_3 = {}
            fluency_row = {}
            relevance_row = {}
            comb_p_row = {}
            for n in steering_factors:
                jp_p_filename_5 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_paragraph_5.json.accuracy.json"
                jp_p_filename_4 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_paragraph_4.json.accuracy.json"
                jp_p_filename_3 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_paragraph_3.json.accuracy.json"
                flu_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_flu.json.accuracy.json"
                rel_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_rel.json.accuracy.json"
                comb_p_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_comb_paragraph.json.accuracy.json"
                
                jp_p_path_5 = root_dir + "/" + task + "/" + method + "/paragraph-long_eval/" + steer_dir + "/" + jp_p_filename_5
                jp_p_path_4 = root_dir + "/" + task + "/" + method + "/paragraph-long_eval/" + steer_dir + "/" + jp_p_filename_4
                jp_p_path_3 = root_dir + "/" + task + "/" + method + "/paragraph-long_eval/" + steer_dir + "/" + jp_p_filename_3
                flu_path = root_dir + "/" + task + "/" + method + "/paragraph-long_eval/" + steer_dir + "/" + flu_filename
                rel_path = root_dir + "/" + task + "/" + method + "/paragraph-long_eval/" + steer_dir + "/" + rel_filename
                comb_p_path = root_dir + "/" + task + "/" + method + "/paragraph-long_eval/" + steer_dir + "/" + comb_p_filename
                
                with open(jp_p_path_5, "r") as f:
                    jp_p_accuracy_5 = json.load(f)["q1"]
                with open(jp_p_path_4, "r") as f:
                    jp_p_accuracy_4 = json.load(f)["q1"]
                with open(jp_p_path_3, "r") as f:
                    jp_p_accuracy_3 = json.load(f)["q1"]
                with open(flu_path, "r") as f:
                    flu_accuracy = json.load(f)["q1"]
                with open(rel_path, "r") as f:
                    rel_accuracy = json.load(f)["q1"]
                with open(comb_p_path, "r") as f:
                    comb_p_accuracy = json.load(f)["q1"]

                jp_p_row_5[n] = jp_p_accuracy_5
                jp_p_row_4[n] = jp_p_accuracy_4
                jp_p_row_3[n] = jp_p_accuracy_3
                fluency_row[n] = flu_accuracy
                relevance_row[n] = rel_accuracy
                comb_p_row[n] = comb_p_accuracy
            jp_p_data_5[top_k] = jp_p_row_5
            jp_p_data_4[top_k] = jp_p_row_4
            jp_p_data_3[top_k] = jp_p_row_3
            fluency_data[top_k] = fluency_row
            relevance_data[top_k] = relevance_row
            comb_p_data[top_k] = comb_p_row

        jp_p_df_5 = pd.DataFrame(jp_p_data_5)
        jp_p_df_4 = pd.DataFrame(jp_p_data_4)
        jp_p_df_3 = pd.DataFrame(jp_p_data_3)
        fluency_df = pd.DataFrame(fluency_data)
        relevance_df = pd.DataFrame(relevance_data)
        comb_p_df = pd.DataFrame(comb_p_data)
        save_dir = f"{RM_INTERP_REPO}/results/plots/{model}/{task}/paragraph-long_eval/{steer}/"
        os.makedirs(save_dir, exist_ok=True)
        make_plot(jp_p_df_5, model, task, "paragraph-long_eval", steer, "jp_paragraph_5", save_dir)
        make_plot(jp_p_df_4, model, task, "paragraph-long_eval", steer, "jp_paragraph_4", save_dir)
        make_plot(jp_p_df_3, model, task, "paragraph-long_eval", steer, "jp_paragraph_3", save_dir)
        make_plot(fluency_df, model, task, "paragraph-long_eval", steer, "fluency", save_dir)
        make_plot(relevance_df, model, task, "paragraph-long_eval", steer, "relevance", save_dir)
        make_plot(comb_p_df, model, task, "paragraph-long_eval", steer, "comb_paragraph", save_dir)
        
        # now do single
        single_data = {}
        print("finished long")
        for top_k in topk_values:
            print(top_k)
            row = {}
            for n in steering_factors:
                filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_w_rf.json.accuracy.json"
                
                path = root_dir + "/" + task + "/" + method + "/paragraph-single_eval/" + steer + "/" + filename

                with open(path, "r") as f:
                    accuracy = json.load(f)["q1"]


                row[n] = accuracy
            single_data[top_k] = row

        df = pd.DataFrame(single_data)
        print(df)
        save_dir = f"{RM_INTERP_REPO}/results/plots/{model}/{task}/paragraph-single_eval/{steer}/"
        os.makedirs(save_dir, exist_ok=True)
        make_plot(df, model, task, "paragraph-single_eval", steer, "single", save_dir)



        
