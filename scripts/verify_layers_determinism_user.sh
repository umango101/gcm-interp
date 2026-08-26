#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 06:00:00
#SBATCH -J verify_layers_det_user
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Runs the localization and one steering sweep twice, into two separate roots,
# then diffs them. RUN THIS FIRST. A nondeterministic backward produces a
# plausible-looking layer ranking, so the failure is invisible in the results
# themselves and only ever shows up as a diff between two runs.
#
# Kept small on purpose (--limit_items 12): nondeterministic kernels diverge
# within a handful of items. Both legs mirror the real run's flags, so a PASS is
# evidence about the run rather than about a special mode.
#
# gpt-oss caveat: --strict_determinism is NOT passed. MoE routing uses
# scatter/index_add kernels with no deterministic implementation, so
# warn_only=False raises on this model. That removes the fail-closed backstop,
# which is exactly why this diff matters more here than it did on Qwen -- it is
# now the only check. If it fails, rerun with STRICT_DETERMINISM=2 to make the
# offending op name itself.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/conflict-syc
set -eu

export RM_INTERP_REPO="/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp"
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
export STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"
export SEED=42

source scripts/_preflight.sh
preflight_gpu

model_id="openai/gpt-oss-20b"
model_name="gpt-oss-20b"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"
source_ds="user-single"
base_ds="dev-single"
LIMIT="${LIMIT:-12}"

ROOT_A="./results_layers_verifyA"
ROOT_B="./results_layers_verifyB"
rm -rf "$ROOT_A" "$ROOT_B"

run_once () {
  local root=$1
  echo "########## run into ${root} ##########"

  python -m layers.run_layers \
    --model_id "$model_id" --batch_size 1 --seed "$SEED" --patch_algo atp \
    --source "$source_ds" --base "$base_ds" --device "$device" \
    --patch_model \
    --results_root "$root" \
    --limit_items "$LIMIT" \
    --sdp_backend math || exit 1

  # Mirrors eval_layers_per_layer_user.sh flag for flag apart from the cut-down
  # sweep. The eval sweep uses --sdp_backend default (flash forward), which is
  # the setting most worth confirming, so pinning math here would test away the
  # risk the harness exists to measure.
  python -m layers.run_layers \
    --model_id "$model_id" --batch_size 8 --seed "$SEED" --patch_algo atp \
    --source "$source_ds" --base "$base_ds" --device "$device" \
    --eval_model --steering --ablation steer \
    --max_new_tokens 24 \
    --eval_test "${data}/${source_ds}/${base_ds}-test.jsonl" \
    --steering_add_path "${data}/${source_ds}/${source_ds}-desired-all.jsonl" \
    --steering_sub_path "${data}/${source_ds}/${base_ds}-desired-all.jsonl" \
    --results_root "$root" \
    --limit_items "$LIMIT" \
    --sweep_mode per_layer \
    --n_vals "5" --n_scale 0.1 \
    --steering_scale relative \
    --rank_by marginal \
    --gen_mode prefill \
    --gen_batch_size 25 \
    --sdp_backend default || exit 1
}

run_once "$ROOT_A"
run_once "$ROOT_B"

echo "########## diff ##########"
python -m layers.verify_determinism "$ROOT_A" "$ROOT_B"
