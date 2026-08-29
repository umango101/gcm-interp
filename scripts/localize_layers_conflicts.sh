#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 08:00:00
#SBATCH -J loc_layers_hier
#SBATCH -o /orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp/logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Layer-level (residual-stream) ATP localization for the three hierarchy arms.
# The layer arm of the head-agnostic question: same estimator, same data, hook
# point moved from o_proj to layer.output.
#
# Writes ./results_layers/gpt-oss-20b/<arm>__from_user-single_to_dev-single/atp/
# The <arm>__ prefix is what keeps the three localizations apart -- all three
# use source=user-single and base=dev-single, so without it they collide.
#
# Notes for the rule-form corpora:
#   * Each row is 9 assistant turns (8 ICL demos + the answer). desired and
#     undesired share identical demos and differ only in the final answer, so
#     the demo terms cancel in L = loglik(undesired) - loglik(desired).
#   * --strict_determinism is NOT passed: gpt-oss MoE routing uses
#     scatter/index_add kernels with no deterministic implementation, so
#     warn_only=False raises. --sdp_backend math still pins the attention
#     backward, which is what lands on the attribution values.
#   * HARMONY_SYSTEM is not exported. These records carry their own system
#     message, so harmony_template never emits its pinned block and the setting
#     would have no effect either way.

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
bash scripts/check_layers_install.sh || exit 1

model_id="openai/gpt-oss-20b"
model_name="gpt-oss-20b"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"

source_ds="user-single"
base_ds="dev-single"
arms=(devuser sysuser sysdev)

for arm in "${arms[@]}"; do
  for f in "${data}/${arm}/${source_ds}-desired-all.jsonl" \
           "${data}/${arm}/${source_ds}-undesired-all.jsonl" \
           "${data}/${arm}/${base_ds}-desired-all.jsonl" \
           "${data}/${arm}/${base_ds}-undesired-all.jsonl" \
           "${data}/${arm}/${base_ds}-test.jsonl" \
           "${data}/${arm}/devNaive-single-test.jsonl"; do
    [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
  done
done
echo "all arms have their five -all/test files"

for arm in "${arms[@]}"; do
  echo
  echo "=== localizing layers: ${arm} (from_${source_ds}_to_${base_ds}) ==="
  python -m layers.run_layers \
    --model_id "$model_id" \
    --batch_size 1 \
    --seed "$SEED" \
    --patch_algo atp \
    --data_dir "$arm" \
    --source "$source_ds" \
    --base "$base_ds" \
    --device "$device" \
    --patch_model \
    --sdp_backend math \
    --results_root ./results_layers || exit 1
done

echo
echo "done. layer profiles:"
for arm in "${arms[@]}"; do
  echo "  ./results_layers/${model_name}/${arm}__from_${source_ds}_to_${base_ds}/atp/layer_effects.csv"
done
