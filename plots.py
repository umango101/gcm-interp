import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
import matplotlib.patches as patches
RM_INTERP_REPO = os.path.dirname(os.path.abspath(__file__))

# ===== Constants =====
steering_factors = [10, 8, 6, 5, 4, 2, 1]
topk_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
tasks = ["from_harmful-long_to_harmless", "from_harmful-single_to_harmless", "from_verse-long_to_prose", "from_verse-single_to_prose"]
task_dict = {
    "from_harmful-long_to_harmless": "Refusal\n(Long)",
    "from_harmful-single_to_harmless": "Refusal\n(Single)",
    "from_verse-long_to_prose": "Verse\n(Long)",
    "from_verse-single_to_prose": "Verse\n(Single)"
}
methods = ["atp"]
method_dict = {
    'acp': 'Full Vector\nPatching [[GCM]]',
    'atp': 'Attribution\nPatching [[GCM]]',
    'atp-zero': 'Attention Head\nKnockouts [[GCM]]',
    'probes': 'Inference-Time\nInterventions (ITI)',
    'random': 'Randomly Selected\nHeads'
}
# models = ["SOLAR-10.7B-Instruct-v1.0"]
models = ["Qwen1.5-14B-Chat"]
ablation_dict = {
    'steer': 'Difference in Means Steering',
    'pyreft': 'Representation Fine-Tuning based steering',
    'mean': 'Means Steering'
}
save_dir = f"{RM_INTERP_REPO}/results-newer/accuracy/plots/"
os.makedirs(save_dir, exist_ok=True)

# CSV aggregation buffer
csv_rows = []

# ============================================================
#                        MAIN HEATMAP PIPELINE
# ============================================================

for model_id in models:
    for ablation in ["steer"]:
        if ablation in ["steer", "mean"]:
            root_dir = f"{RM_INTERP_REPO}/results-newer/accuracy/{model_id}"
        else:
            root_dir = f"{RM_INTERP_REPO}/results-newer/accuracy/{model_id}"

    for steer in ['long', 'single']:
        for eval in ['long', 'single']:
            # Setup heatmap figure (3 tasks × 5 methods)
            fig, axes = plt.subplots(
                nrows=len(task_dict.keys()), ncols=len(models), figsize=(35, 20), constrained_layout=True
            )

            print('LEN AXES ', len(axes))

            # --- NEW: add one big yellow banner behind GCM (cols 0–2) ---
            # fig.canvas.draw()   # ensures axis positions are finalized
            # gcm_left  = axes[0,0].get_position().x0 + 0.015
            # gcm_right = axes[0,len(models)-1].get_position().x1
            # gcm_top = axes[0,0].get_position().y1 + 0.015
            # gcm_bottom    = gcm_top - 0.05

            # banner = patches.Rectangle(
            #     (gcm_left, gcm_bottom),
            #     gcm_right - gcm_left,
            #     gcm_top - gcm_bottom,
            #     transform=fig.transFigure,
            #     facecolor=(1.0, 1.0, 0.6),   # soft yellow
            #     zorder=-100,                 # behind everything
            # )

            # fig.patches.append(banner)

            # gcm_left  = axes[0,len(models)-1].get_position().x1 + 0.001
            # gcm_right = axes[0,len(models)-1].get_position().x1 - 0.02
            # gcm_top = axes[0,0].get_position().y1 + 0.015
            # gcm_bottom    = gcm_top - 0.05

            # banner = patches.Rectangle(
            #     (gcm_left, gcm_bottom),
            #     gcm_right - gcm_left,
            #     gcm_top - gcm_bottom,
            #     transform=fig.transFigure,
            #     facecolor=(0.80, 0.87, 1.0),  # soft yellow
            #     zorder=-100,                 # behind everything
            # )
            # fig.patches.append(banner)

            # -------------------------------------------------------------

            row_images = []
            summary_results = {task: {} for task in tasks}

            # One colormap per row
            colormaps = [cm.get_cmap('Reds')] * 4

            # Loop over tasks + methods
            for row_idx, task in enumerate(tasks):
                source = task.split("_to_")[0].split("from_")[1]
                breakup_source = source.split("-")[0]
                base = task.split("_")[-1]
                print(f"Processing {model_id} - {task}...")
                cmap = colormaps[row_idx]
                for col_idx, method in enumerate(methods):
                    heatmap_data = np.zeros((len(steering_factors), len(topk_values)))
                    # Load heatmap JSONs
                    for i, sf in enumerate(steering_factors):
                        for j, topk in enumerate(topk_values):
                            # Correct folder structure by method type
                            # determine which method directory to read from
                            if topk == 1:
                                load_method = methods[0]     # ALWAYS read from acp dir when topk==1
                            else:
                                load_method = method    # use actual method otherwise
                            method_dir = os.path.join(root_dir, task, load_method, f"{breakup_source}-{eval}_eval/", f"{breakup_source}-{steer}_steer/")
                            # print('METHOD DIR ', method_dir)
                            filename = (
                                f"{sf}_targeted_{ablation}_topk_{topk}_gen_accuracy_wo_rf.json.accuracy.json"
                                if load_method != "random"
                                else f"{sf}_random_{ablation}_topk_{topk}_gen_accuracy_wo_rf.json.accuracy.json"
                            )
                            filepath = os.path.join(method_dir, filename)

                            try:
                                with open(filepath, "r") as f:
                                    data = json.load(f)
                                    accuracy = data.get("gen", {}).get("q1", np.nan)
                                    if accuracy is np.nan:
                                        accuracy = data.get("q1", np.nan)
                                    heatmap_data[i, j] = accuracy

                                    # ---- NEW: add CSV row ----
                                    csv_rows.append({
                                        "model_id": model_id,
                                        "method": method,
                                        "ablation": ablation,
                                        "task": task,
                                        "steering_factor": sf,
                                        "topk": topk,
                                        "accuracy": accuracy
                                    })

                            except FileNotFoundError:
                                print(f"File not found: {filepath}")
                                heatmap_data[i, j] = np.nan

                    # Draw heatmap
                    ax = axes[row_idx]
                    norm = plt.Normalize(vmin=0, vmax=1)
                    im = ax.imshow(
                        heatmap_data, aspect="auto", origin="lower",
                        cmap=cmap, norm=norm
                    )

                    if col_idx == len(methods) - 1:
                        row_images.append((im, norm, cmap))

                    # Add text annotations
                    for i in range(len(steering_factors)):
                        for j in range(len(topk_values)):
                            val = heatmap_data[i, j]
                            if not np.isnan(val):
                                rgba = cmap(norm(val))
                                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                                color = "black" if brightness > 0.5 else "white"
                                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                        fontsize=15, color=color)

                    # Axes/ticks formatting
                    if row_idx == len(tasks) - 1:
                        ax.set_xticks(range(len(topk_values)))
                        ax.set_xticklabels(topk_values, rotation=90, fontsize=24)
                    else:
                        ax.set_xticks([])

                    if col_idx == len(models) - 1:
                        ax.set_yticks(range(len(steering_factors)))
                        ax.set_yticklabels(steering_factors, fontsize=24)
                        ax.set_ylabel("Steering Factor", fontsize=24)
                    else:
                        ax.set_yticks([])

                    if row_idx == 0:
                        ax.set_title(method_dict[method], fontsize=24)

                # Add row label
                first_ax = axes[row_idx]
                pos = first_ax.get_position()
                if row_idx == 0:
                    fig.text(
                        pos.x0 - 0.15,
                        ((pos.y0 + pos.y1) / 2) + 0.04,
                        task_dict[task],
                        fontsize=28, weight="bold",
                        va="center", ha="center", rotation=90
                    )
                elif row_idx == 1:
                    fig.text(
                        pos.x0 - 0.15,
                        ((pos.y0 + pos.y1) / 2) + 0.02,
                        task_dict[task],
                        fontsize=28, weight="bold",
                        va="center", ha="center", rotation=90
                    )
                else:
                    fig.text(
                        pos.x0 - 0.15,
                        (pos.y0 + pos.y1) / 2,
                        task_dict[task],
                        fontsize=28, weight="bold",
                        va="center", ha="center", rotation=90
                    )

            # Colorbars
            for i, (im, norm, cmap) in enumerate(row_images):
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            # Save heatmap grid
            plt.suptitle(f'{model_id}\nLocalization: (Specified on Y-axis), Evaluation: {"Single-Token" if eval == "single" else "Long-Form"} Responses. Steering: {"No Prologue/Long-form response queries" if steer == "long" else "Single-Token Response Queries"}', fontsize=28)
            plt.figtext(0.5, -0.02, "Top-K % of concept-sensitive attention heads",
                        ha="center", fontsize=24)
            plt.figtext(1.01, 0.5, "Rate of successful steering",
                        ha="center", va="center", rotation=90, fontsize=24)

            plt.savefig(
                f"{save_dir}/{model_id}_{ablation}_{eval}-eval_{steer}-steer_task-wise_heatmaps_wo_rf.png",
                dpi=300, bbox_inches="tight"
            )
            plt.savefig(
                f"{save_dir}/{model_id}_{ablation}_{eval}-eval_{steer}-steer_task-wise_heatmaps_wo_rf.pdf",
                dpi=300, bbox_inches="tight"
            )
            plt.close()

# ============================================================
#                        SAVE CSV OUTPUT
# ============================================================

df = pd.DataFrame(csv_rows)
csv_path = os.path.join(save_dir, "steering_results_wo_rf.csv")
df.to_csv(csv_path, index=False)

print(f"\nCSV saved to: {csv_path}")
print(f"Total rows: {len(df)}")
