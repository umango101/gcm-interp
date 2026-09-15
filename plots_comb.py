import argparse
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RM_INTERP_REPO = os.path.dirname(os.path.abspath(__file__))

# ===== Defaults =====
DEFAULT_STEERING_FACTORS = [10, 8, 6, 5, 4, 2, 1]
DEFAULT_TOPK_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

ALL_TASKS = [
    # "from_extraversion-long_to_introversion-long",
    # "from_extraversion-single_to_introversion-single",
    "from_female-long_to_male-long",
    "from_female-single_to_male-single",
    # "from_lying-long_to_truthful-long",
    # "from_lying-single_to_truthful-single",
    "from_femaleDescribe-long_to_maleDescribe-long",
    "from_femaleDescribe-single_to_maleDescribe-single",
    "from_femaleGeneration-long_to_maleGeneration-long",
    "from_femaleGeneration-single_to_maleGeneration-single",
    "from_femaleMCQA-long_to_maleMCQA-long",
    "from_femaleMCQA-single_to_maleMCQA-single",
]
TASK_DICT = {
    "from_extraversion-long_to_introversion-long":          "Persona\n(Long)",
    "from_extraversion-single_to_introversion-single":      "Persona\n(Single)",
    "from_female-long_to_male-long":                        "Bias\n(Long)",
    "from_female-single_to_male-single":                    "Bias\n(Single)",
    "from_lying-long_to_truthful-long":                     "Factual Recall (Long)",
    "from_lying-single_to_truthful-single":                 "Factual Recall (Single)",
    "from_femaleDescribe-long_to_maleDescribe-long":        "Bias - Long, Description",
    "from_femaleDescribe-single_to_maleDescribe-single":    "Bias - Single, Description",
    "from_femaleGeneration-long_to_maleGeneration-long":    "Bias - Long, Generation",
    "from_femaleGeneration-single_to_maleGeneration-single":"Bias - Single, Generation",
    "from_femaleMCQA-long_to_maleMCQA-long":                "Bias - Long, MCQA",
    "from_femaleMCQA-single_to_maleMCQA-single":            "Bias - Single, MCQA",
}


# ALL_MODELS = ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "Qwen1.5-32B-Chat", "gemma-3-12b-it", "phi-4"]

ALL_MODELS = ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "gemma-3-12b-it", "Qwen1.5-32B-Chat", "Falcon3-10B-Instruct"]

MODEL_DISPLAY_NAMES = {
    "Qwen1.5-14B-Chat":            "Qwen 1.5-14B",
    "Qwen1.5-32B-Chat":            "Qwen 1.5-32B",
    "Llama-2-13b-chat-hf":         "Llama 2-13B",
    "OLMo-2-1124-13B-DPO":         "OLMo 2-13B",
    "Falcon3-10B-Instruct":        "Falcon 10B",
    "SOLAR-10.7B-Instruct-v1.0":   "SOLAR-10.7B",
    "gemma-3-12b-it":              "Gemma 3-12B",
    "phi-4":                       "Phi-4",
}

MODEL_COLORMAPS = {
    "Qwen1.5-14B-Chat":            "Blues",
    "Qwen1.5-32B-Chat":            "Purples",
    "Llama-2-13b-chat-hf":         "Oranges",
    "OLMo-2-1124-13B-DPO":         "Greens",
    "Falcon3-10B-Instruct":             "Greys",
    "SOLAR-10.7B-Instruct-v1.0":   "Reds",
    "gemma-3-12b-it":              "YlOrBr",
    "phi-4":                       "PuRd",
}

METHOD_DICT = {
    "acp": "Full Vector\nPatching [[GCM]]",
    "atp": "Attribution\nPatching [[GCM]]",
    "atp-zero": "Attention Head\nKnockouts [[GCM]]",
    "probes": "Inference-Time\nInterventions (ITI)",
    "random": "Randomly Selected\nHeads",
}

ABLATION_DICT = {
    "steer": "Difference in Means Steering",
    "pyreft": "Representation Fine-Tuning based steering",
    "mean": "Means Steering",
}


# ---------------------------------------------------------------------------
# Core heatmap builder
# ---------------------------------------------------------------------------

def load_accuracy(filepath: str) -> float:
    """Load accuracy value from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    acc = data.get("gen", {}).get("q1", np.nan)
    if acc is np.nan or acc != acc:
        acc = data.get("q1", np.nan)
    return acc


def build_heatmap(
    root_dir: str,
    model_id: str,
    task: str,
    method: str,
    ablation: str,
    eval_variant: str,
    steer_variant: str,
    rf_suffix: str,  # "w_rf" or "wo_rf"
    steering_factors: list,
    topk_values: list,
) -> tuple[np.ndarray, list[dict]]:
    """Load accuracy values and return (heatmap_data, csv_rows)."""
    source = task.split("_to_")[0].split("from_")[1]
    breakup_source = source.split("-")[0]

    heatmap_data = np.zeros((len(steering_factors), len(topk_values)))
    csv_rows = []

    for i, sf in enumerate(steering_factors):
        for j, topk in enumerate(topk_values):
            acp_dir = os.path.join(root_dir, task, "acp")
            load_method = "acp" if (topk == 1 and os.path.isdir(acp_dir)) else method
            method_dir = os.path.join(
                root_dir, task, load_method,
                f"female-{eval_variant}_eval/",
                f"female-{steer_variant}_steer/",
                "accuracy/"
            )
            if load_method != "random":
                filename = f"{sf}_targeted_{ablation}_topk_{topk}_gen_accuracy_{rf_suffix}.json.accuracy.json"
            else:
                filename = f"{sf}_random_{ablation}_topk_{topk}_gen_accuracy_{rf_suffix}.json.accuracy.json"

            filepath = os.path.join(method_dir, filename)
            try:
                accuracy = load_accuracy(filepath)
                heatmap_data[i, j] = accuracy
                csv_rows.append({
                    "model_id": model_id,
                    "method": method,
                    "ablation": ablation,
                    "task": task,
                    "eval_variant": eval_variant,
                    "steer_variant": steer_variant,
                    "rf_mode": rf_suffix,
                    "steering_factor": sf,
                    "topk": topk,
                    "accuracy": accuracy,
                })
            except FileNotFoundError:
                print(f"  Missing: {filepath}")
                heatmap_data[i, j] = np.nan

    return heatmap_data, csv_rows


def draw_heatmap(ax, heatmap_data, topk_values, steering_factors, cmap, norm,
                 row_idx, col_idx, n_rows, n_cols, task_label, model_id, fig):
    """Draw one heatmap cell with annotations and axis labels."""
    im = ax.imshow(heatmap_data, aspect="auto", origin="lower", cmap=cmap, norm=norm)

    for i in range(len(steering_factors)):
        for j in range(len(topk_values)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                rgba = cmap(norm(val))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if brightness > 0.5 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.tick_params(axis="both", which="both", length=0)

    if row_idx == n_rows - 1:
        ax.set_xticks(range(len(topk_values)))
        ax.set_xticklabels(topk_values, rotation=45, ha="right")
    else:
        ax.set_xticks([])

    if col_idx == 0:
        ax.set_yticks(range(len(steering_factors)))
        ax.set_yticklabels(steering_factors)
        ax.set_ylabel("Steering Factor")
    else:
        ax.set_yticks([])

    if row_idx == 0:
        display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        ax.set_title(display_name, fontweight="bold", pad=6)

    return im


def make_grid_plot(
    models: list[str],
    task: str,
    method: str,
    ablation: str,
    eval_variant: str,
    steer_variant: str,
    rf_suffix: str,
    steering_factors: list,
    topk_values: list,
    accuracy_dir: str,
    save_dir: str,
) -> list[dict]:
    """Build and save one heatmap grid for a single task (one file per task)."""
    n_rows = 1
    n_cols = len(models)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(4.5 * n_cols + 1, 4.5), constrained_layout=True, squeeze=False,
    )

    all_csv_rows = []
    norm = plt.Normalize(vmin=0, vmax=1)

    for col_idx, model_id in enumerate(models):
        root_dir = os.path.join("/home/ubansal/orcd/scratch/gcm-interp/results_pipeline/", model_id)
        print(f"  {model_id} | {task} | eval={eval_variant} steer={steer_variant} rf={rf_suffix}")

        cmap = cm.get_cmap(MODEL_COLORMAPS.get(model_id, "Reds"))
        heatmap_data, csv_rows = build_heatmap(
            root_dir, model_id, task, method, ablation,
            eval_variant, steer_variant, rf_suffix,
            steering_factors, topk_values,
        )
        all_csv_rows.extend(csv_rows)

        ax = axes[0, col_idx]
        task_label = TASK_DICT.get(task, task)
        im = draw_heatmap(ax, heatmap_data, topk_values, steering_factors,
                          cmap, norm, 0, col_idx, n_rows, n_cols,
                          task_label, model_id, fig)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=18)
        cbar.set_label("Steering success rate", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    is_single = eval_variant == "single"
    eval_str  = "Single-Token" if eval_variant == "single" else "Long-Form"
    steer_str = "Single-Token" if steer_variant == "single" else "Long-Form"
    rf_label  = "Token Matching" if is_single else (
        "w/ R+F Filter" if rf_suffix == "w_rf" else "combined"
    )

    task_label_title = TASK_DICT.get(task, task).replace("\n", " ")
    plt.suptitle(
        f"{task_label_title}  ·  {eval_str} Eval, {steer_str} Steer  ·  {rf_label}",
        fontsize=15, y=1.05, fontweight="bold",
    )
    plt.figtext(0.5, -0.04, "Top-K fraction of concept-sensitive attention heads",
                ha="center", fontsize=10)

    # Derive a short task slug for the filename (e.g. "sycophancy-long", "verse-single")
    task_slug = task.split("from_")[1].split("_to_")[0]
    rf_file_suffix = "token_matching" if is_single else rf_suffix
    fname = f"{ablation}_{task_slug}_{eval_variant}-eval_{steer_variant}-steer_heatmap_{rf_file_suffix}"
    for ext in ("png", "pdf"):
        out_path = os.path.join(save_dir, f"{fname}.{ext}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved {out_path}")
    plt.close()

    return all_csv_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate steering evaluation heatmap plots")
    p.add_argument("--models", nargs="*", default=None,
                   help="Models to include (default: all)")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Task names as short keys (e.g. sycophancy-long sycophancy-single) "
                        "or full names (from_sycophancy-long_to_non-sycophantic). Default: all")
    p.add_argument("--eval_variants", nargs="*", default=["single", "long"],
                   help="Eval variants: long single (default: both)")
    p.add_argument("--steer_variants", nargs="*", default=["single", "long"],
                   help="Steer variants: long single (default: both)")
    p.add_argument("--rf_mode", choices=["with", "without", "both"], default="both",
                   help="Which accuracy files to plot: with/without relevance+fluency filter, or both (default)")
    p.add_argument("--ablations", nargs="*", default=["steer"],
                   help="Ablation types (default: steer)")
    p.add_argument("--methods", nargs="*", default=["atp"],
                   help="Methods to plot (default: atp)")
    p.add_argument("--accuracy_dir", default=None,
                   help="Path to accuracy directory (default: judge-evals/accuracy/)")
    p.add_argument("--save_dir", default=None,
                   help="Where to save plots (default: judge-evals/accuracy/plots/)")
    return p.parse_args()


def resolve_tasks(task_args: list[str] | None) -> list[str]:
    """Resolve short task names to full task keys."""
    if task_args is None:
        return ALL_TASKS
    resolved = []
    for t in task_args:
        if t in TASK_DICT:
            resolved.append(t)
        else:
            # Try matching by short name (e.g. "sycophancy-long" → full key)
            matches = [k for k in TASK_DICT if f"from_{t}_to_" in k or t in k]
            if matches:
                resolved.extend(matches)
            else:
                print(f"Warning: unknown task '{t}', skipping")
    return resolved


def main():
    args = parse_args()

    accuracy_dir = args.accuracy_dir or os.path.join(RM_INTERP_REPO, "judge-evals", "accuracy")
    save_dir = args.save_dir or os.path.join(accuracy_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    models = args.models or ALL_MODELS
    tasks = resolve_tasks(args.tasks)
    methods = args.methods
    evals = args.eval_variants
    steering = args.steer_variants
    steering_factors = DEFAULT_STEERING_FACTORS
    topk_values = DEFAULT_TOPK_VALUES

    print(f"Models:        {models}")
    print(f"Tasks:         {tasks}")
    print(f"Ablations:     {args.ablations}")
    print(f"Methods:       {methods}")
    print(f"Accuracy dir:  {accuracy_dir}")
    print(f"Save dir:      {save_dir}")

    all_csv_rows = []

    for ablation in args.ablations:
        for method in methods:
            for task in tasks:
                for evalu in evals:
                    for steer in steering:
                        is_single = evalu == "single"
                        variant = "single" if is_single else "long"
                        rf_suffixes = ["w_rf"] if is_single else ["comb"]
        
                        for rf_suffix in rf_suffixes:
                            print(f"\n--- ablation={ablation} method={method} "
                                  f"task={task} variant={variant} rf={rf_suffix} ---")
                            csv_rows = make_grid_plot(
                                models=models,
                                task=task,
                                method=method,
                                ablation=ablation,
                                eval_variant=variant,
                                steer_variant=steer,
                                rf_suffix=rf_suffix,
                                steering_factors=steering_factors,
                                topk_values=topk_values,
                                accuracy_dir=accuracy_dir,
                                save_dir=save_dir,
                            )
                            all_csv_rows.extend(csv_rows)

    # Save aggregated CSV
    df = pd.DataFrame(all_csv_rows)
    csv_path = os.path.join(save_dir, "steering_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()