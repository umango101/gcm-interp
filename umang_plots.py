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
# models = ["Qwen1.5-14B-Chat-steering-last"]
steers = ["anti-long_steer", "anti-single_steer"]
root_dir = f"{RM_INTERP_REPO}/results_jailbreak/accuracy/{model}"

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
        # do long first, then single
        jp_r_data_5 = {}
        jp_r_data_4 = {}
        jp_r_data_3 = {}
        jp_s_data_5 = {}
        jp_s_data_4 = {}
        jp_s_data_3 = {}
        fluency_data = {}
        relevance_data = {}
        comb_r_data = {}
        comb_s_data = {}
        for top_k in topk_values:
            jp_r_row_5 = {}
            jp_r_row_4 = {}
            jp_r_row_3 = {}
            jp_s_row_5 = {}
            jp_s_row_4 = {}
            jp_s_row_3 = {}
            fluency_row = {}
            relevance_row = {}
            comb_r_row = {}
            comb_s_row = {}
            for n in steering_factors:
                jp_r_filename_5 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_refusal_5.json.accuracy.json"
                jp_r_filename_4 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_refusal_4.json.accuracy.json"
                jp_r_filename_3 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_refusal_3.json.accuracy.json"
                jp_s_filename_5 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_sentiment_5.json.accuracy.json"
                jp_s_filename_4 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_sentiment_4.json.accuracy.json"
                jp_s_filename_3 = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_judge_sentiment_3.json.accuracy.json"
                flu_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_flu.json.accuracy.json"
                rel_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_rel.json.accuracy.json"
                comb_r_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_comb_refusal.json.accuracy.json"
                comb_s_filename = f"{n}_targeted_steer_topk_{top_k}_gen_accuracy_comb_sentiment.json.accuracy.json"

                jp_r_path_5 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_r_filename_5
                jp_r_path_4 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_r_filename_4
                jp_r_path_3 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_r_filename_3
                jp_s_path_5 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_s_filename_5
                jp_s_path_4 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_s_filename_4
                jp_s_path_3 = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + jp_s_filename_3
                flu_path = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + flu_filename
                rel_path = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + rel_filename
                comb_r_path = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + comb_r_filename
                comb_s_path = root_dir + "/" + task + "/" + method + "/anti-long_eval/" + steer + "/" + comb_s_filename

                with open(jp_r_path_5, "r") as f:
                    jp_r_accuracy_5 = json.load(f)["q1"]
                with open(jp_r_path_4, "r") as f:
                    jp_r_accuracy_4 = json.load(f)["q1"]
                with open(jp_r_path_3, "r") as f:
                    jp_r_accuracy_3 = json.load(f)["q1"]
                with open(jp_s_path_5, "r") as f:
                    jp_s_accuracy_5 = json.load(f)["q1"]
                with open(jp_s_path_4, "r") as f:
                    jp_s_accuracy_4 = json.load(f)["q1"]
                with open(jp_s_path_3, "r") as f:
                    jp_s_accuracy_3 = json.load(f)["q1"]
                with open(flu_path, "r") as f:
                    flu_accuracy = json.load(f)["q1"]
                with open(rel_path, "r") as f:
                    rel_accuracy = json.load(f)["q1"]
                with open(comb_r_path, "r") as f:
                    comb_r_accuracy = json.load(f)["q1"]
                with open(comb_s_path, "r") as f:
                    comb_s_accuracy = json.load(f)["q1"]

                jp_r_row_5[n] = jp_r_accuracy_5
                jp_r_row_4[n] = jp_r_accuracy_4
                jp_r_row_3[n] = jp_r_accuracy_3
                jp_s_row_5[n] = jp_s_accuracy_5
                jp_s_row_4[n] = jp_s_accuracy_4
                jp_s_row_3[n] = jp_s_accuracy_3
                fluency_row[n] = flu_accuracy
                relevance_row[n] = rel_accuracy
                comb_r_row[n] = comb_r_accuracy
                comb_s_row[n] = comb_s_accuracy
            jp_r_data_5[top_k] = jp_r_row_5
            jp_r_data_4[top_k] = jp_r_row_4
            jp_r_data_3[top_k] = jp_r_row_3
            jp_s_data_5[top_k] = jp_s_row_5
            jp_s_data_4[top_k] = jp_s_row_4
            jp_s_data_3[top_k] = jp_s_row_3
            fluency_data[top_k] = fluency_row
            relevance_data[top_k] = relevance_row
            comb_r_data[top_k] = comb_r_row
            comb_s_data[top_k] = comb_s_row

        jp_r_df_5 = pd.DataFrame(jp_r_data_5)
        jp_r_df_4 = pd.DataFrame(jp_r_data_4)
        jp_r_df_3 = pd.DataFrame(jp_r_data_3)
        jp_s_df_5 = pd.DataFrame(jp_s_data_5)
        jp_s_df_4 = pd.DataFrame(jp_s_data_4)
        jp_s_df_3 = pd.DataFrame(jp_s_data_3)
        fluency_df = pd.DataFrame(fluency_data)
        relevance_df = pd.DataFrame(relevance_data)
        comb_r_df = pd.DataFrame(comb_r_data)
        comb_s_df = pd.DataFrame(comb_s_data)
        save_dir = f"{RM_INTERP_REPO}/results_jailbreak/plots/{model}/{task}/anti-long_eval/{steer}/"
        os.makedirs(save_dir, exist_ok=True)
        make_plot(jp_r_df_5, model, task, "anti-long_eval", steer, "jp_refusal_5", save_dir)
        make_plot(jp_r_df_4, model, task, "anti-long_eval", steer, "jp_refusal_4", save_dir)
        make_plot(jp_r_df_3, model, task, "anti-long_eval", steer, "jp_refusal_3", save_dir)
        make_plot(jp_s_df_5, model, task, "anti-long_eval", steer, "jp_sentiment_5", save_dir)
        make_plot(jp_s_df_4, model, task, "anti-long_eval", steer, "jp_sentiment_4", save_dir)
        make_plot(jp_s_df_3, model, task, "anti-long_eval", steer, "jp_sentiment_3", save_dir)
        make_plot(fluency_df, model, task, "anti-long_eval", steer, "fluency", save_dir)
        make_plot(relevance_df, model, task, "anti-long_eval", steer, "relevance", save_dir)
        make_plot(comb_r_df, model, task, "anti-long_eval", steer, "comb_refusal", save_dir)
        make_plot(comb_s_df, model, task, "anti-long_eval", steer, "comb_sentiment", save_dir)
        
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
        print(df)
        save_dir = f"{RM_INTERP_REPO}/results_jailbreak/plots/{model}/{task}/anti-single_eval/{steer}/"
        os.makedirs(save_dir, exist_ok=True)
        make_plot(df, model, task, "anti-single_eval", steer, "single", save_dir)



        
