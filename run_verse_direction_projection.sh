#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J vl_proj
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH -c 8
#SBATCH --requeue

# Observational projection onto the verse/prose direction, plus the unembedding decomposition.
# No intervention: two forwards per train item to fit the direction, two per test item to
# measure it. Much cheaper than the steering sweep -- no layer sweep, since one forward yields
# every layer at once.
#
# Keep SOURCE_POSITION and SEED equal to the steering run so both experiments share a direction.

set -euo pipefail
mkdir -p logs
echo "[slurm] job ${SLURM_JOB_ID:-?} on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"

set +eu
source /home/ubansal/miniconda/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate "${GEN_ENV:-syc}"
set -eu
echo "[slurm] env=${CONDA_DEFAULT_ENV:-?} python=$(which python)"

export HF_HOME="${HF_HOME:-$HOME/orcd/scratch/hf_home}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset NVIDIA_TF32_OVERRIDE

REPO="${RM_INTERP_REPO:-$HOME/orcd/scratch/gcm-interp}"
MODEL="${MODEL:-Qwen/Qwen1.5-32B-Chat}"
cd "$REPO"

rc=0
python verse_direction_projection.py \
  --model "$MODEL" \
  --source_position "${SOURCE_POSITION:-last}" \
  --train_frac "${TRAIN_FRAC:-0.5}" \
  --seed "${SEED:-0}" \
  --top_k "${TOP_K:-10}" \
  --max_ref_tokens "${MAX_REF_TOKENS:-128}" \
  || rc=$?

echo "[slurm] python exited rc=$rc $(date)"
exit $rc