#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -J eval_layers
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

# One localization x {2 eval sets} x {2 steering vectors} = 4 cells, each swept
# over layer counts and steering coefficients.
#
# The localization is chosen by LOC_PAIR, so the cross-localization arm is the
# same script with the other value -- there is no second copy of this file that
# can drift from this one:
#
#   sbatch --export=ALL,LOC_PAIR=female-long_male-long     scripts/eval_layers.sh
#   sbatch --export=ALL,LOC_PAIR=female-single_male-single scripts/eval_layers.sh
#
# Cells produced (eval set x steering vector):
#   single eval / single steer   <- matched, single-token
#   single eval / long   steer   <- cross-format steering vector
#   long   eval / single steer   <- cross-format steering vector
#   long   eval / long   steer   <- matched, free-form
# Cross-LOCALIZATION comes from pairing a cell with its counterpart under the
# other LOC_PAIR, exactly as eval_bias.sh / eval_bias_2.sh do for heads.

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
model_id="google/gemma-3-12b-it"
model_name="gemma-3-12b-it"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"

# Layer counts and coefficients. With --steering_scale relative the effective
# coefficient is N * n_scale, i.e. a multiple of the mean residual norm at the
# steered layer: N=1..10 with n_scale=0.1 sweeps 0.1x .. 1.0x. N stays integral
# because the scorer parses it out of the gen filename as an integer.
TOPK_LAYERS="1,2,3,5,7,9,10"
N_VALS="1,2,4,5,6,8,10"
N_SCALE=0.1

echo "=== localization from_${source}_to_${base} ==="

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
    --sdp_backend math \
    --strict_determinism \
    --topk_layers "$TOPK_LAYERS" \
    --n_vals "$N_VALS" \
    --n_scale "$N_SCALE" \
    --steering_scale relative \
    --rank_by cumulative
}

# eval set          test file base    steering set      add            sub
run_cell female-single male-single    female-single     female-single  male-single
run_cell female-single male-single    female-long       female-long    male-long
run_cell female-long   male-long      female-single     female-single  male-single
run_cell female-long   male-long      female-long       female-long    male-long
