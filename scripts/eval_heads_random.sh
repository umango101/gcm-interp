#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 36:00:00
#SBATCH -J eval_heads_random
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# LAYER-MATCHED RANDOM baseline for the head arm, over the full (N x top_k) grid.
#
#   sbatch scripts/eval_heads_random.sh
#
# For each top_k the control draws the SAME NUMBER OF HEADS PER LAYER as the ATP
# top-k set, choosing which heads within each layer at random. Uniform-random
# selection would differ from the targeted set in two ways at once -- which heads
# and which depths -- so beating it would only show that late-layer heads are
# more steerable. Holding the depth profile fixed isolates the ranking, which is
# the thing ATP is claimed to provide.
#
# Writes to results/{model}/from_.../random/... (config.set_output_prefix uses
# patch_algo), so the targeted arm under atp/ is untouched. Gen files are named
# {N}_random_steer_{topk}_..., which is why the scorer needs apply_reps_param.py.
#
# PREREQUISITES, all of which fail loudly below if missing:
#   python apply_layer_matched_random.py     # the sampler
#   python apply_reps_param.py               # REPS in the scorer
#   the targeted ATP map must already exist -- the control is matched TO it

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubansal/orcd/scratch/conflicts/gcm-interp}"
if [[ ! -f "${REPO_ROOT}/data_handler.py" || ! -f "${REPO_ROOT}/run.py" ]]; then
  echo "REPO_ROOT does not look like the repo root: $REPO_ROOT" >&2
  echo "Override with: sbatch --export=ALL,REPO_ROOT=/path/to/gcm-interp ..." >&2
  exit 1
fi
cd "$REPO_ROOT"
mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME"

export HARMONY_SYSTEM=minimal
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export STRICT_DETERMINISM="1"
export SEED="${SEED:-42}"

# The control itself.
export RANDOM_BASELINE="${RANDOM_BASELINE:-layer_matched}"
export RANDOM_BASELINE_SEED="${RANDOM_BASELINE_SEED:-42}"

model_id="${MODEL_ID:-openai/gpt-oss-20b}"
model_name="${model_id##*/}"
device="${DEVICE:-cuda:0}"
source_ds="${SOURCE:-user-single}"
base_ds="${BASE:-dev-single}"
data="${REPO_ROOT}/data/${model_name}"

ATP_DIR="${REPO_ROOT}/results/${model_name}/from_${source_ds}_to_${base_ds}/atp"
export ATP_MAP="${ATP_MAP:-${ATP_DIR}/numerator_1_heads.pt}"

# --- preflight --------------------------------------------------------------
if ! grep -q "retrieve_layer_matched_random_k" eval/logits_handler.py; then
  echo "eval/logits_handler.py has no layer-matched sampler --" >&2
  echo "  run: python apply_layer_matched_random.py" >&2
  exit 1
fi
if ! grep -q "^REPS = " eval_pipeline_conflict_single.py; then
  echo "eval_pipeline_conflict_single.py has no REPS constant --" >&2
  echo "  run: python apply_reps_param.py" >&2
  echo "  (without it the random gen files cannot be scored)" >&2
  exit 1
fi
if [[ ! -f "$ATP_MAP" ]]; then
  echo "targeted map not found: $ATP_MAP" >&2
  echo "The layer-matched control is matched TO the targeted set, so the map must" >&2
  echo "exist first. If only shards are present, run: python reduce_head_map.py ${ATP_DIR}" >&2
  exit 1
fi

eval_test="${data}/${source_ds}/${base_ds}-test.jsonl"
add_path="${data}/${source_ds}/${source_ds}-desired-all.jsonl"
sub_path="${data}/${source_ds}/${base_ds}-desired-all.jsonl"
for f in "$eval_test" "$add_path" "$sub_path"; do
  [[ -f "$f" ]] || { echo "MISSING $f" >&2; exit 1; }
done

echo "repo root   : $REPO_ROOT"
echo "python      : $(which python)"
echo "contrast    : ${source_ds} -> ${base_ds}"
echo "baseline    : $RANDOM_BASELINE (seed $RANDOM_BASELINE_SEED)"
echo "matched to  : $ATP_MAP"
echo "writes to   : results/${model_name}/from_${source_ds}_to_${base_ds}/random/"
echo

# Same grid as the targeted arm: 14 steering factors x 10 top_k fractions = 140.
python run.py \
  --model_id "$model_id" \
  --batch_size 8 \
  --seed "$SEED" \
  --patch_algo random \
  --source "$source_ds" \
  --base "$base_ds" \
  --device "$device" \
  --eval_model \
  --steering \
  --ablation steer \
  --max_new_tokens "${MAX_NEW_TOKENS:-24}" \
  --eval_test "$eval_test" \
  --steering_add_path "$add_path" \
  --steering_sub_path "$sub_path"

echo
echo "done. score with:"
echo "  python eval_pipeline_conflict_single_random.py --stages merge accuracies plots"
