#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 24:00:00
#SBATCH -J eval_layers_rnd
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

# Random-k-layers baseline. This is the arm the whole experiment turns on: if
# ATP is layer-informative, targeted layers should beat random layers at matched
# k; if it is layer-agnostic too, the curves coincide and the head-level result
# generalizes rather than being specific to head granularity.
#
# --patch_algo random needs no localization pass, but --source/--base still set
# the results directory so the random arm sits beside the targeted one it is
# read against. Layers are drawn seeded on (seed, k), so the k=3 draw is not a
# prefix of the k=5 draw -- otherwise the random arm's own k-curve is
# autocorrelated and cannot be compared against the targeted k-curve.
#
# Run several seeds to get a baseline band rather than a single line:
#   for s in 42 43 44 45 46; do
#     sbatch --export=ALL,SEED=$s,LOC_PAIR=female-long_male-long scripts/eval_layers_random.sh
#   done

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
SEED="${SEED:-42}"
IFS='_' read -r source base <<< "$LOC_PAIR"

model_id="Qwen/Qwen1.5-14B-Chat"
model_name="Qwen1.5-14B-Chat"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"

TOPK_LAYERS="1,2,3,5,7,9,10"
N_VALS="1,2,4,5,6,8,10"
N_SCALE=0.1

# Seeded draws must not overwrite each other, so each seed gets its own root.
RESULTS_ROOT="./results_layers_random_seed${SEED}"

run_cell () {
  local eval_dir=$1 eval_base=$2 steer_dir=$3 steer_add=$4 steer_sub=$5
  echo "--- RANDOM seed=${SEED} eval=${eval_dir} steer=${steer_dir} ---"
  python -m layers.run_layers \
    --model_id "$model_id" \
    --batch_size 8 \
    --patch_algo random \
    --source "$source" \
    --base "$base" \
    --device "$device" \
    --seed "$SEED" \
    --eval_model \
    --steering \
    --ablation steer \
    --eval_test "${data}/${eval_dir}/${eval_base}-test.jsonl" \
    --steering_add_path "${data}/${steer_dir}/${steer_add}-steering.jsonl" \
    --steering_sub_path "${data}/${steer_dir}/${steer_sub}-steering.jsonl" \
    --results_root "$RESULTS_ROOT" \
    --topk_layers "$TOPK_LAYERS" \
    --n_vals "$N_VALS" \
    --n_scale "$N_SCALE" \
    --steering_scale relative \
    --sdp_backend math \
    --strict_determinism
}

run_cell female-single male-single    female-single     female-single  male-single
run_cell female-single male-single    female-long       female-long    male-long
run_cell female-long   male-long      female-single     female-single  male-single
run_cell female-long   male-long      female-long       female-long    male-long
