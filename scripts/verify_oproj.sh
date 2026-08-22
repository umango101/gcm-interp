#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 01:00:00
#SBATCH -J verify_oproj
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH -c 4
#
# ---------------------------------------------------------------------------
# Pre-flight for the o_proj.output -> o_proj.input switch.
#
#     sbatch verify_oproj.sh
#
# Run this on the CURRENT, UNPATCHED code. The verifier only reads o_proj.input
# through nnsight; it does not need any of the switch applied. It reports:
#
#   BLOCKERS  -- nnsight cannot read / write / backprop through o_proj.input,
#                or the per-head slice is not actually a head. These mean the
#                switch cannot proceed as written.
#   ADVISORY  -- model_handler.dim disagrees with the correct input-space head
#                width. EXPECTED on any model where num_heads*head_dim !=
#                hidden_size (Gemma-3-12B: 16*256=4096 vs hidden_size 3840).
#                This is the model_handler.py part of the patch, not a problem
#                with nnsight.
#
# mit_normal_gpu rather than mit_preemptable: this is a short job and there is
# no checkpointing, so a preemption would just waste the model load. Each model
# is a few minutes, dominated by loading weights.
#
# --full_precision is deliberate. Check [1] -- the identity proving that the
# input slice really is head h -- needs o_proj.weight, which 4-bit quantization
# hides. Full bf16: gemma-3-12b ~24GB, Qwen1.5-14B ~28GB, Qwen1.5-32B ~64GB,
# all within an H200's 141GB, one model at a time.
# ---------------------------------------------------------------------------

source ~/.bashrc
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
cd "$RM_INTERP_REPO" || { echo "FATAL: cannot cd to $RM_INTERP_REPO"; exit 1; }
mkdir -p logs

if [[ ! -f verify_oproj_input.py ]]; then
    echo "FATAL: verify_oproj_input.py not found in $RM_INTERP_REPO"
    exit 1
fi

declare -a models=(
  "google/gemma-3-12b-it"
  "Qwen/Qwen1.5-14B-Chat"
  "Qwen/Qwen1.5-32B-Chat"
  "allenai/OLMo-2-1124-13B-DPO"
  "tiiuae/Falcon3-10B-Instruct"
)

# ONLY_MODEL=Qwen/Qwen1.5-14B-Chat sbatch verify_oproj.sh  -> just that one
if [[ -n "${ONLY_MODEL:-}" ]]; then
    models=("$ONLY_MODEL")
fi

declare -a failed=()
declare -a passed=()

for model_id in "${models[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $model_id"
    echo "############################################################"
    python -u verify_oproj_input.py \
        --model_id "$model_id" \
        --device cuda:0 \
        --full_precision
    rc=$?
    if [[ $rc -eq 0 ]]; then
        passed+=("$model_id")
    else
        failed+=("$model_id (exit $rc)")
    fi
    # Free HBM before the next model rather than relying on process teardown
    # ordering when several large checkpoints load back to back. Best-effort:
    # a failure here is not a result, so it must not add noise or change status.
    python - >/dev/null 2>&1 <<'EOF' || true
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
EOF
done

echo ""
echo "==================== summary ===================="
[[ ${#passed[@]} -gt 0 ]] && { echo "safe to switch:"; printf '  %s\n' "${passed[@]}"; }
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "BLOCKED:"
    printf '  %s\n' "${failed[@]}"
    echo ""
    echo "Do not apply the o_proj.input switch for these until the blocker is resolved."
    exit 1
fi
echo ""
echo "No blockers. Re-read the ADVISORY lines above before applying the patch --"
echo "they tell you whether model_handler.dim needs the head_dim fix per model."
