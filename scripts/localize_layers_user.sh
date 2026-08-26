#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 04:00:00
#SBATCH -J loc_layers_user
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Layer-level (residual-stream) ATP localization: user-single -> dev-single.
# The layer arm of the head-agnostic question -- same estimator, same data, hook
# point moved from o_proj to layer.output.
#
# Writes ./results_layers/gpt-oss-20b/from_user-single_to_dev-single/atp/
#   layers_*.pt              per-batch attribution shards
#   numerator_1_layers.pt    reduced [n_layers, n_items] map
#   layer_effects.{png,csv}  cumulative and marginal profiles
#
# Notes specific to the extended-preamble corpus:
#   * Each row is 9 assistant turns (8 ICL demos + the answer). desired and
#     undesired share identical demos and differ only in the final answer, so
#     the demo terms cancel in L = loglik(undesired) - loglik(desired) and the
#     attribution is carried by the final answer position. See README_layers.md.
#   * --strict_determinism is NOT passed: gpt-oss MoE routing uses
#     scatter/index_add kernels with no deterministic implementation, so
#     warn_only=False raises. --sdp_backend math still pins the attention
#     backward, which is what lands on the attribution values.
#   * --head_site is irrelevant here; the layer arm hooks layer.output.

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

# Read at process start; eval/setup.assert_determinism_env() checks them.
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

bash scripts/check_layers_install.sh || exit 1

model_id="openai/gpt-oss-20b"
model_name="gpt-oss-20b"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"

source_ds="user-single"
base_ds="dev-single"

# align_toks and the ATP estimator require source and base to tokenize to the
# SAME length. With the preamble they differ at nine answer positions rather
# than one, so verify_gptoss's "multi-diff" count is expected to be nonzero for
# this corpus -- read len-mismatch=0 as the pass condition, not minimal-pair=OK.
for f in "${data}/${source_ds}/${source_ds}-desired-all.jsonl" \
         "${data}/${source_ds}/${source_ds}-undesired-all.jsonl" \
         "${data}/${source_ds}/${base_ds}-desired-all.jsonl" \
         "${data}/${source_ds}/${base_ds}-undesired-all.jsonl" \
         "${data}/${source_ds}/${base_ds}-test.jsonl"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done
VERIFY=$(find . -maxdepth 2 -name verify_gptoss.py | head -1)
if [ -n "$VERIFY" ]; then
  python "$VERIFY" --stage tokenizer --data_dir "data/${model_name}" || true
fi

echo "=== localizing layers: from_${source_ds}_to_${base_ds} ==="
python -m layers.run_layers \
  --model_id "$model_id" \
  --batch_size 1 \
  --seed "$SEED" \
  --patch_algo atp \
  --source "$source_ds" \
  --base "$base_ds" \
  --device "$device" \
  --patch_model \
  --sdp_backend math \
  --results_root ./results_layers || exit 1

echo
echo "done. layer profile:"
echo "  ./results_layers/${model_name}/from_${source_ds}_to_${base_ds}/atp/layer_effects.csv"
