#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 12:00:00
#SBATCH -J eval_layers_user
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Per-layer steering sweep for the single user-single -> dev-single run: steer
# each residual stream INDIVIDUALLY and measure every layer, rather than
# selecting the top k and steering them together.
#
# Why per_layer. Top-k asks whether ATP's chosen layers beat random -- a handful
# of coarse points, and it confounds "the ranking is good" with "steering more
# layers does more". Per-layer measures all 24 layers, so the ATP score becomes a
# PREDICTION to correlate against the measurement. That correlation IS the
# localization claim, and it makes the random arm redundant: with every layer
# measured there is no selection left to randomize.
#
# One condition, so one tree: this writes into the SAME ./results_layers root the
# localization used. --localization_root exists for the multi-condition case and
# is not needed here.
#
# Steering vector = mean(user-single-desired-all) - mean(dev-single-desired-all),
# read at the final <|message|> position of the question-only prompts. Those two
# sets share every user turn and differ only in the eight demo answers, so the
# vector is the ICL-preamble contrast itself -- which is the manipulation this
# experiment is about.
#
#   sbatch scripts/eval_layers_per_layer_user.sh

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/conflict-syc
set -eu

export RM_INTERP_REPO="/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
cd "$RM_INTERP_REPO" || exit 1

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME" logs

export HARMONY_SYSTEM=minimal

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
# Fragmentation, not just volume: the OOM that killed shard 1 asked for 508 MiB
# with ~900 MiB reserved-but-unallocated. Expandable segments let the allocator
# grow a segment instead of needing one contiguous free block.
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

N_VALS="${N_VALS:-2,5,8,10}"
N_SCALE=0.1
# Single-word answers, so the decode budget is small. The prompt is long (nine
# turns), so prefill dominates the cost either way.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"

eval_test="${data}/${source_ds}/${base_ds}-test.jsonl"
add_path="${data}/${source_ds}/${source_ds}-desired-all.jsonl"
sub_path="${data}/${source_ds}/${base_ds}-desired-all.jsonl"

for f in "$eval_test" "$add_path" "$sub_path"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done

LOC_MAP="./results_layers/${model_name}/from_${source_ds}_to_${base_ds}/atp/numerator_1_layers.pt"
if [ ! -f "$LOC_MAP" ]; then
  echo "No attribution map at ${LOC_MAP}."
  echo "Run scripts/localize_layers_user.sh first."
  exit 1
fi

echo "=== per-layer sweep: layers from ${source_ds}, steering ${source_ds} ==="
python -m layers.run_layers \
  --model_id "$model_id" \
  --batch_size 8 \
  --seed "$SEED" \
  --patch_algo atp \
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
  --gen_batch_size 25 || exit 1

echo
echo "done. score with:"
echo "  python eval_pipeline_conflict_layers.py --results_dir ./results_layers"
