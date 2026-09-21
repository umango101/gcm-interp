"""
Steering evaluation plots -- MAX ENVELOPE line version.

Same data loading and directory layout as the heatmap script; only the drawing
changes. Each model gets one panel holding a SINGLE line: the best accuracy
achieved at each x position, maximised over the other sweep variable.

    --x_axis topk              (default) x = top-k fraction; each point is the
                               max over all steering factors at that top-k.
    --x_axis steering_factor   x = steering factor; each point is the max over
                               all top-k values at that steering factor.

    --overlay                  draw all models on ONE axes instead of a panel
                               each, which is usually what you want once every
                               panel holds a single line.
    --annotate_best            label each point with the steering factor (or
                               top-k) that won it.

CAVEAT, since the plot no longer shows it: a max over N noisy cells is an
upward-biased estimate of best achievable accuracy, and the bias grows with N.
That matters here because N is not constant -- a top-k where only 2 of 7
steering factors ran gets a systematically lower max than one where all 7 did,
purely from missing files. The script prints a warning when coverage is uneven
and writes the per-point contributing count to the envelope CSV. Check it
before reading a dip as a real effect.

NOTE on the x axis: the default TOPK_VALUES are not evenly spaced -- six values
crowded into 0.01-0.1 and then 0.5 and 1.0. On a linear axis the interesting
low-k region collapses into the left margin. So the default x scale is
'category' (evenly spaced ticks, exactly like the heatmap columns).
Use --x_scale log for a true numeric axis, or --x_scale linear if you
specifically want the real spacing.
"""

import argparse
import os
import json
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

# Absolute. A relative root silently resolves against the CWD and every single
# file reads as "Missing", which looks like an incomplete sweep rather than a
# bad path.
DEFAULT_RESULTS_ROOT = "/home/ubansal/orcd/scratch/gcm-interp/results_pipeline"

# ===== Defaults =====
DEFAULT_STEERING_FACTORS = [10, 8, 6, 5, 4, 2, 1]
DEFAULT_TOPK_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

ALL_TASKS = [
    # "from_extraversion-long_to_introversion-long",
    # "from_extraversion-single_to_introversion-single",
    # "from_female-long_to_male-long",
    # "from_female-single_to_male-single",
    # "from_lying-long_to_truthful-long",
    # "from_lying-single_to_truthful-single",
    # "from_femaleDescribe-long_to_maleDescribe-long",
    # "from_femaleDescribe-single_to_maleDescribe-single",
    # "from_femaleGeneration-long_to_maleGeneration-long",
    # "from_femaleGeneration-single_to_maleGeneration-single",
    # "from_femaleMCQA-long_to_maleMCQA-long",
    # "from_femaleMCQA-single_to_maleMCQA-single",
    "from_paragraph-long_to_sentence-long",
    "from_paragraph-single_to_sentence-single"
]
TASK_DICT = {
    # "from_extraversion-long_to_introversion-long":          "Persona\n(Long)",
    # "from_extraversion-single_to_introversion-single":      "Persona\n(Single)",
    # "from_female-long_to_male-long":                        "Bias\n(Long)",
    # "from_female-single_to_male-single":                    "Bias\n(Single)",
    # "from_lying-long_to_truthful-long":                     "Factual Recall (Long)",
    # "from_lying-single_to_truthful-single":                 "Factual Recall (Single)",
    # "from_femaleDescribe-long_to_maleDescribe-long":        "Bias - Long, Description",
    # "from_femaleDescribe-single_to_maleDescribe-single":    "Bias - Single, Description",
    # "from_femaleGeneration-long_to_maleGeneration-long":    "Bias - Long, Generation",
    # "from_femaleGeneration-single_to_maleGeneration-single":"Bias - Single, Generation",
    # "from_femaleMCQA-long_to_maleMCQA-long":                "Bias - Long, MCQA",
    # "from_femaleMCQA-single_to_maleMCQA-single":            "Bias - Single, MCQA",
    "from_paragraph-long_to_sentence-long":                 "Summarization (Long)",
    "from_paragraph-single_to_sentence-single":             "Summarization (Single)"
}

ALL_MODELS = ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "gemma-3-12b-it",
              "Qwen1.5-32B-Chat", "Falcon3-10B-Instruct"]

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
    "Falcon3-10B-Instruct":        "Greys",
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
# Data loading (unchanged from the heatmap version except for the eval/steer
# directory prefix, which now follows the task instead of assuming "female-")
# ---------------------------------------------------------------------------

def load_accuracy(filepath: str) -> float:
    """Load accuracy value from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    acc = data.get("gen", {}).get("q1", np.nan)
    if acc is np.nan or acc != acc:
        acc = data.get("q1", np.nan)
    return acc


def build_matrix(
    root_dir: str,
    model_id: str,
    task: str,
    method: str,
    ablation: str,
    eval_variant: str,
    steer_variant: str,
    rf_suffix: str,          # "w_rf" or "wo_rf"
    steering_factors: list,
    topk_values: list,
) -> tuple[np.ndarray, list[dict]]:
    """Load accuracies into a (steering_factor x topk) matrix + tidy csv rows."""
    source = task.split("_to_")[0].split("from_")[1]   # e.g. "femaleMCQA-single"
    breakup_source = source.split("-")[0]              # e.g. "femaleMCQA"

    data = np.full((len(steering_factors), len(topk_values)), np.nan)
    csv_rows = []

    for i, sf in enumerate(steering_factors):
        for j, topk in enumerate(topk_values):
            acp_dir = os.path.join(root_dir, task, "acp")
            load_method = "acp" if (topk == 1 and os.path.isdir(acp_dir)) else method
            method_dir = os.path.join(
                root_dir, task, load_method,
                f"paragraph-{eval_variant}_eval",
                f"paragraph-{steer_variant}_steer",
                "accuracy",
            )
            stem = "targeted" if load_method != "random" else "random"
            filename = (f"{sf}_{stem}_{ablation}_topk_{topk}"
                        f"_gen_accuracy_{rf_suffix}.json.accuracy.json")

            filepath = os.path.join(method_dir, filename)
            try:
                accuracy = load_accuracy(filepath)
            except FileNotFoundError:
                print(f"  Missing: {filepath}")
                continue
            data[i, j] = accuracy
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

    return data, csv_rows


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Max envelope + drawing
# ---------------------------------------------------------------------------

def max_envelope(panel, line_values):
    """Collapse a (line x x) matrix to the best value at each x position.

    Returns (best, winners, counts):
      best    max over the line axis, NaN where nothing loaded
      winners the line value that achieved it (None where best is NaN)
      counts  how many cells actually contributed -- the max of 2 cells is not
              comparable to the max of 7, so this is kept, not thrown away."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column
        best = np.nanmax(panel, axis=0)
    counts = np.isfinite(panel).sum(axis=0)
    winners = []
    for j in range(panel.shape[1]):
        col = panel[:, j]
        winners.append(line_values[int(np.nanargmax(col))]
                       if np.isfinite(col).any() else None)
    return best, winners, counts


def warn_uneven_coverage(counts, x_values, label, model_id):
    """A dip in a max envelope can be a real effect or just fewer runs at that
    x position. The two are indistinguishable in the plot, so say it out loud."""
    present = counts[counts > 0]
    if present.size and present.min() != present.max():
        detail = ", ".join(f"{v:g}:{c}" for v, c in zip(x_values, counts) if c)
        print(f"    WARNING: {model_id} has uneven coverage across {label} "
              f"({detail}) -- max values are not directly comparable.")


def draw_max_line(ax, best, x_values, x_scale, color, label=None,
                  winners=None, annotate=False, winner_fmt="{:g}"):
    """Draw the single max-envelope line on one axes."""
    positions = (np.arange(len(x_values)) if x_scale == "category"
                 else np.asarray(x_values, dtype=float))
    finite = np.isfinite(best)
    if finite.any():
        # Markers matter: an envelope with gaps (or one surviving point) draws
        # nothing at all with lines only.
        ax.plot(positions[finite], best[finite], marker="o", markersize=5,
                linewidth=2.0, color=color, label=label)
        if annotate and winners is not None:
            for pos, val, win in zip(positions, best, winners):
                if np.isfinite(val) and win is not None:
                    ax.annotate(winner_fmt.format(win), (pos, val),
                                textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=7, color=color)

    if x_scale == "category":
        labels = [f"{v:g}" for v in x_values]
        rotate = max(len(s) for s in labels) > 3
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45 if rotate else 0,
                           ha="right" if rotate else "center")
    elif x_scale == "log":
        ax.set_xscale("log")

    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_grid_plot(
    models: list,
    task: str,
    method: str,
    ablation: str,
    eval_variant: str,
    steer_variant: str,
    rf_suffix: str,
    steering_factors: list,
    topk_values: list,
    results_root: str,
    save_dir: str,
    x_axis: str,
    x_scale: str,
    share_y: bool,
    overlay: bool,
    annotate_best: bool,
) -> tuple:
    """Save one figure for a single task. Returns (raw_rows, envelope_rows)."""
    if x_axis == "topk":
        x_values, x_label = topk_values, "Top-K fraction of heads"
        winner_label, winner_fmt = "steering factor", "{:g}"
    else:
        x_values, x_label = steering_factors, "Steering factor"
        winner_label, winner_fmt = "top-k", "{:g}"

    n_cols = 1 if overlay else len(models)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_cols,
        figsize=((6.4, 4.4) if overlay else (3.3 * n_cols + 1.0, 3.9)),
        constrained_layout=True, squeeze=False, sharey=share_y,
    )

    raw_rows, envelope_rows = [], []
    drew_anything = False

    for idx, model_id in enumerate(models):
        root_dir = os.path.join(results_root, model_id)
        print(f"  {model_id} | {task} | eval={eval_variant} steer={steer_variant} rf={rf_suffix}")

        cmap = plt.get_cmap(MODEL_COLORMAPS.get(model_id, "Reds"))
        data, csv_rows = build_matrix(
            root_dir, model_id, task, method, ablation,
            eval_variant, steer_variant, rf_suffix,
            steering_factors, topk_values,
        )
        raw_rows.extend(csv_rows)
        drew_anything |= bool(csv_rows)

        # data is (steering_factor x topk). The line axis is whatever is NOT
        # on x, so the envelope is always "max over the other variable".
        panel = data if x_axis == "topk" else data.T
        line_values = steering_factors if x_axis == "topk" else topk_values
        best, winners, counts = max_envelope(panel, line_values)
        warn_uneven_coverage(counts, x_values, x_label.lower(), model_id)

        for xv, bv, wv, cv in zip(x_values, best, winners, counts):
            envelope_rows.append({
                "model_id": model_id, "method": method, "ablation": ablation,
                "task": task, "eval_variant": eval_variant,
                "steer_variant": steer_variant, "rf_mode": rf_suffix,
                "x_axis": x_axis, "x_value": xv,
                "max_accuracy": bv if np.isfinite(bv) else np.nan,
                # Fixed column names regardless of orientation, so the CSV has a
                # stable schema for downstream analysis.
                "maximised_over": winner_label,
                "argmax_value": wv,
                "n_contributing": int(cv),
            })

        ax = axes[0, 0] if overlay else axes[0, idx]
        draw_max_line(
            ax, best, x_values, x_scale,
            color=cmap(0.75),
            label=MODEL_DISPLAY_NAMES.get(model_id, model_id) if overlay else None,
            winners=winners, annotate=annotate_best, winner_fmt=winner_fmt,
        )
        ax.set_xlabel(x_label)
        if overlay:
            ax.set_ylabel("Best steering success rate")
        else:
            ax.set_title(MODEL_DISPLAY_NAMES.get(model_id, model_id),
                         fontweight="bold", pad=6)
            if idx == 0 or not share_y:
                ax.set_ylabel("Best steering success rate")

    if not drew_anything:
        print(f"  WARNING: no accuracy files found for task '{task}' under "
              f"{results_root} -- the figure will be blank. Check --results_root.")

    if overlay:
        axes[0, 0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                          frameon=False)

    is_single = eval_variant == "single"
    eval_str = "Single-Token" if is_single else "Long-Form"
    steer_str = "Single-Token" if steer_variant == "single" else "Long-Form"
    rf_label = "Token Matching" if is_single else (
        "w/ R+F Filter" if rf_suffix == "mcqa" else "combined"
    )

    task_label_title = TASK_DICT.get(task, task).replace("\n", " ")
    fig.suptitle(
        f"{task_label_title}  \u00b7  {eval_str} Eval, {steer_str} Steer  \u00b7  {rf_label}\n"
        f"best over all {winner_label}s",
        fontsize=13, fontweight="bold",
    )

    task_slug = task.split("from_")[1].split("_to_")[0]
    rf_file_suffix = "token_matching" if is_single else rf_suffix
    fname = (f"maxline_{ablation}_{task_slug}_{eval_variant}-eval_{steer_variant}-steer"
             f"_{x_axis}{'_overlay' if overlay else ''}_{rf_file_suffix}")
    for ext in ("png", "pdf"):
        out_path = os.path.join(save_dir, f"{fname}.{ext}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved {out_path}")
    plt.close(fig)

    return raw_rows, envelope_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate steering evaluation line plots")
    p.add_argument("--models", nargs="*", default=None,
                   help="Models to include (default: all)")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Task names as short keys (e.g. femaleMCQA-single) or full "
                        "names (from_femaleMCQA-single_to_maleMCQA-single). Default: all")
    p.add_argument("--eval_variants", nargs="*", default=["single", "long"],
                   help="Eval variants: long single (default: both)")
    p.add_argument("--steer_variants", nargs="*", default=["single", "long"],
                   help="Steer variants: long single (default: both)")
    p.add_argument("--ablations", nargs="*", default=["steer"],
                   help="Ablation types (default: steer)")
    p.add_argument("--methods", nargs="*", default=["atp"],
                   help="Methods to plot (default: atp)")
    p.add_argument("--x_axis", choices=["topk", "steering_factor"], default="topk",
                   help="What goes on the x axis; the other variable becomes the "
                        "lines (default: topk)")
    p.add_argument("--x_scale", choices=["category", "log", "linear"], default="category",
                   help="'category' (default) spaces x ticks evenly like the heatmap "
                        "columns; 'log'/'linear' use real numeric spacing")
    p.add_argument("--overlay", action="store_true",
                   help="Draw all models on one axes instead of a panel each. Usually "
                        "what you want now that each panel holds a single line.")
    p.add_argument("--annotate_best", action="store_true",
                   help="Label each point with the steering factor (or top-k) that "
                        "achieved it -- the max hides which one won.")
    p.add_argument("--no_share_y", action="store_true",
                   help="Give each model panel its own y range (default: shared 0-1, "
                        "so panels are comparable at a glance)")
    p.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT,
                   help=f"Root holding per-model result trees (default: {DEFAULT_RESULTS_ROOT})")
    p.add_argument("--save_dir", default=None,
                   help="Where to save plots (default: judge-evals/accuracy/plots/)")
    return p.parse_args()


def resolve_tasks(task_args):
    """Resolve short task names to full task keys."""
    if task_args is None:
        return ALL_TASKS
    resolved = []
    for t in task_args:
        if t in TASK_DICT:
            resolved.append(t)
        else:
            matches = [k for k in TASK_DICT if f"from_{t}_to_" in k or t in k]
            if matches:
                resolved.extend(matches)
            else:
                print(f"Warning: unknown task '{t}', skipping")
    return resolved


def main():
    args = parse_args()

    save_dir = args.save_dir or os.path.join(
        RM_INTERP_REPO, "judge-evals", "accuracy", "plots")
    os.makedirs(save_dir, exist_ok=True)

    models = args.models or ALL_MODELS
    tasks = resolve_tasks(args.tasks)
    steering_factors = DEFAULT_STEERING_FACTORS
    topk_values = DEFAULT_TOPK_VALUES

    if not os.path.isdir(args.results_root):
        raise NotADirectoryError(
            f"--results_root does not exist: {args.results_root}. Every file would "
            "read as 'Missing' and the plots would come out blank."
        )

    print(f"Models:        {models}")
    print(f"Tasks:         {tasks}")
    print(f"Ablations:     {args.ablations}")
    print(f"Methods:       {args.methods}")
    print(f"X axis:        {args.x_axis} ({args.x_scale} scale)")
    print(f"Results root:  {args.results_root}")
    print(f"Save dir:      {save_dir}")

    all_raw_rows, all_env_rows = [], []

    for ablation in args.ablations:
        for method in args.methods:
            for task in tasks:
                for evalu in args.eval_variants:
                    for steer in args.steer_variants:
                        is_single = evalu == "single"
                        rf_suffixes = ["mcqa"] if is_single else ["comb"]

                        for rf_suffix in rf_suffixes:
                            print(f"\n--- ablation={ablation} method={method} "
                                  f"task={task} eval={evalu} steer={steer} rf={rf_suffix} ---")
                            raw_rows, env_rows = make_grid_plot(
                                models=models,
                                task=task,
                                method=method,
                                ablation=ablation,
                                eval_variant=evalu,
                                steer_variant=steer,
                                rf_suffix=rf_suffix,
                                steering_factors=steering_factors,
                                topk_values=topk_values,
                                results_root=args.results_root,
                                save_dir=save_dir,
                                x_axis=args.x_axis,
                                x_scale=args.x_scale,
                                share_y=not args.no_share_y,
                                overlay=args.overlay,
                                annotate_best=args.annotate_best,
                            )
                            all_raw_rows.extend(raw_rows)
                            all_env_rows.extend(env_rows)

    # Two CSVs: the full sweep stays the data of record, since the envelope
    # throws away every cell that did not win.
    raw_path = os.path.join(save_dir, "steering_results.csv")
    pd.DataFrame(all_raw_rows).to_csv(raw_path, index=False)
    print(f"\nCSV saved to: {raw_path}  ({len(all_raw_rows)} rows, full sweep)")

    # Orientation in the filename: the two --x_axis runs produce different
    # envelopes and must not clobber each other.
    env_path = os.path.join(save_dir, f"steering_results_max_{args.x_axis}.csv")
    pd.DataFrame(all_env_rows).to_csv(env_path, index=False)
    print(f"CSV saved to: {env_path}  ({len(all_env_rows)} rows, plotted envelope)")


if __name__ == "__main__":
    main()