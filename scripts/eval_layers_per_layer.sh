#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 48:00:00
#SBATCH -J eval_per_layer
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

# Per-layer sweep: steer each layer INDIVIDUALLY and measure the effect, rather
# than selecting the top k layers and steering them together.
#
# This changes what the experiment tests. Top-k asks "are ATP's chosen layers
# better than random?" -- seven coarse points, confounding the ranking's quality
# with the fact that steering more layers does more. Per-layer measures all forty
# layers directly, so ATP's score becomes a PREDICTION to correlate against the
# measurement. That correlation is the localization claim, stated directly.
#
# Because every layer is measured, the random baseline is redundant here: there
# is no selection left to randomize.
#
# COST. Per cell: n_layers x n_vals generation runs. At 40 layers this is 40/7 =
# 5.7x the top-k sweep per N value, so N_VALS is trimmed below. Set it back to the
# full grid only if you need the N axis at layer resolution; with 4 values the
# 8-cell matrix is 40 x 4 x 8 = 1280 runs.
#
#   sbatch --export=ALL,LOC_PAIR=female-long_male-long     scripts/eval_layers_per_layer.sh
#   sbatch --export=ALL,LOC_PAIR=female-single_male-single scripts/eval_layers_per_layer.sh
#
# Results land under a '-per-layer' method directory, so they never collide with
# the top-k sweep and both can be scored from the same tree.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"

mkdir -p logs
source scripts/_preflight.sh
preflight_gpu

LOC_PAIR="${LOC_PAIR:-female-long_male-long}"
IFS='_' read -r source base <<< "$LOC_PAIR"

algo="atp"
model_id="Qwen/Qwen1.5-14B-Chat"
model_name="Qwen1.5-14B-Chat"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"

# Trimmed from the top-k grid to offset sweeping 40 layers instead of 7 counts.
N_VALS="${N_VALS:-2,5,8,10}"
N_SCALE=0.1

echo "=== per-layer sweep, localization from_${source}_to_${base} ==="

run_cell () {
  local eval_dir=$1 eval_base=$2 steer_dir=$3 steer_add=$4 steer_sub=$5
  echo "--- eval=${eval_dir} steer=${steer_dir} ---"
  python -m layers.run_layers \
    --model_id "$model_id" \
    --batch_size 8 \
    --patch_algo "$algo" \
    --source "$source" \
    --base "$base" \
    --device "$device" \
    --eval_model \
    --steering \
    --ablation steer \
    --eval_test "${data}/${eval_dir}/${eval_base}-test.jsonl" \
    --steering_add_path "${data}/${steer_dir}/${steer_add}-steering.jsonl" \
    --steering_sub_path "${data}/${steer_dir}/${steer_sub}-steering.jsonl" \
    --results_root ./results_layers \
    --sweep_mode per_layer \
    --n_vals "$N_VALS" \
    --n_scale "$N_SCALE" \
    --steering_scale relative \
    --sdp_backend default \
    --strict_determinism \
    --gen_mode prefill \
    --gen_batch_size 25
}

run_cell female-single male-single    female-single     female-single  male-single
run_cell female-single male-single    female-long       female-long    male-long
run_cell female-long   male-long      female-single     female-single  male-single
run_cell female-long   male-long      female-long       female-long    male-long
