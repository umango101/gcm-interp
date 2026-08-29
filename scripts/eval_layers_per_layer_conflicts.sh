#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 48:00:00
#SBATCH -J eval_layers_hier
#SBATCH -o /orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp/logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Per-layer steering sweep for the three hierarchy arms: steer each residual
# stream INDIVIDUALLY and measure every layer, rather than selecting a top k.
#
# Why per_layer. Top-k asks whether ATP's chosen layers beat random -- a few
# coarse points, and it confounds "the ranking is good" with "steering more
# layers does more". Measuring all 24 turns the ATP score into a PREDICTION to
# correlate against the measurement, which is the localization claim itself, and
# it makes a random arm redundant: with every layer measured there is nothing
# left to randomize.
#
# Six cells: 3 arms x {dev-single-test, devNaive-single-test}. The naive file is
# the transfer test -- same conflict with the ICL preamble removed -- so a layer
# that carries the in-context policy and one that carries the trained prior are
# distinguishable by comparing the two sweeps.
#
# Steering vector = mean(user-single-desired-all) - mean(dev-single-desired-all)
# WITHIN AN ARM, read at the final <|message|> position. Those two sets share
# every question turn and differ only in the eight demo answers, so the vector
# is the ICL-preamble contrast itself.
#
#   sbatch scripts/eval_layers_per_layer_conflicts.sh
#   ARMS=sysdev sbatch scripts/eval_layers_per_layer_conflicts.sh   # one arm

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/conflict-syc
set -eu

export RM_INTERP_REPO="/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp"
cd "$RM_INTERP_REPO" || exit 1
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME" logs

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export STRICT_DETERMINISM="1"
export SEED=42

source scripts/_preflight.sh
preflight_gpu

model_id="openai/gpt-oss-20b"
model_name="gpt-oss-20b"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"

source_ds="user-single"
base_ds="dev-single"
IFS=',' read -r -a arms <<< "${ARMS:-devuser,sysuser,sysdev}"
IFS=',' read -r -a tests <<< "${TESTS:-dev-single-test,devNaive-single-test}"

# 14 factors x 24 layers = 336 generation runs per cell. At --n_scale 0.1 the
# alphas span 0.1 to 4.5 x the residual norm, so the tail is well past where
# generation degrades -- which is the point: the high end is what makes
# broken_post/user_net informative. Scoring reads N from the filenames.
N_VALS="${N_VALS:-1,2,4,5,6,8,10,15,20,25,30,35,40,45}"
N_SCALE=0.1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"

# The activation cache holds batch x seq x hidden for every layer at once, so
# prompt length costs what batch size does. The rule-form prompts are roughly
# twice the length of the request-form ones these defaults were tuned on, which
# is what OOM'd the head sweep at the stock batch of 9.
export CACHE_BS="${CACHE_BS:-3}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"

for arm in "${arms[@]}"; do
  for tf in "${tests[@]}"; do
    eval_test="${data}/${arm}/${tf}.jsonl"
    add_path="${data}/${arm}/${source_ds}-desired-all.jsonl"
    sub_path="${data}/${arm}/${base_ds}-desired-all.jsonl"
    for f in "$eval_test" "$add_path" "$sub_path"; do
      [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
    done

    # The map is OPTIONAL in per-layer mode, and this is a warning rather than a
    # gate on purpose: the sweep steers all 24 layers individually and measures
    # each, which is the causal ground truth and needs neither a ranking
    # (nothing is selected) nor the map for its steering vectors (those are
    # diff-in-means over the steering sets). What the map buys is the
    # ATP-vs-measured correlation.
    LOC_MAP="./results_layers/${model_name}/${arm}__from_${source_ds}_to_${base_ds}/atp/numerator_1_layers.pt"
    if [ ! -f "$LOC_MAP" ]; then
      echo "NOTE: no attribution map at ${LOC_MAP}."
      echo "      Sweep runs anyway; only the ATP-vs-measured correlation is lost."
    fi

    echo
    echo "=== per-layer sweep: arm=${arm} eval=${tf} ==="
    python -m layers.run_layers \
      --model_id "$model_id" \
      --batch_size 8 \
      --seed "$SEED" \
      --patch_algo atp \
      --data_dir "$arm" \
      --source "$source_ds" \
      --base "$base_ds" \
      --device "$device" \
      --eval_model \
      --steering \
      --ablation steer \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --eval_test "$eval_test" \
      --steering_add_path "$add_path" \
      --steering_sub_path "$sub_path" \
      --results_root ./results_layers \
      --sweep_mode per_layer \
      --n_vals "$N_VALS" \
      --n_scale "$N_SCALE" \
      --steering_scale relative \
      --rank_by marginal \
      --sdp_backend default \
      --gen_mode prefill \
      --gen_batch_size "$GEN_BATCH_SIZE" || exit 1
  done
done

echo
echo "done. score with:"
echo "  python eval_pipeline_conflict_layers.py --results_dir ./results_layers"
