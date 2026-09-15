#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 12:00:00
#SBATCH -J vl_abs
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH -c 8
#SBATCH --requeue

# Runs the verse-long causal abstraction experiment against an existing pairs.jsonl.
# Submit scripts/run_build_pairs.sh first.
#
# The cost is one batched forward per (pair, condition, position set, layer chunk), so it scales
# with n_layers / --layer_batch. Records append per (pair, key) with fsync and are skipped on
# restart, so a preemption costs at most one position-set sweep -- hence no signal relay: there
# is nothing to checkpoint that is not already on disk.
#
# EXPERIMENTS=generate runs the free-generation arm at chosen layers, e.g.
#   EXPERIMENTS=generate GEN_LAYERS="28 34 40" sbatch scripts/run_verse_long_abstraction.sh

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
PAIRS="$REPO/data/$(basename "$MODEL")/verse-long-abstraction/pairs.jsonl"

if [[ ! -f "$PAIRS" ]]; then
  echo "[slurm] missing $PAIRS -- run scripts/run_build_pairs.sh first" >&2
  exit 1
fi

cd "$REPO"
rc=0
python verse_long_abstraction.py \
  --model "$MODEL" \
  --experiments ${EXPERIMENTS:-patch transplant} \
  --conditions ${CONDITIONS:-topic format compete} \
  --positions ${POSITIONS:-fmt question suffix last prompt concisely} \
  --transplant_positions ${TRANSPLANT_POSITIONS:-last suffix} \
  --layer_batch "${LAYER_BATCH:-4}" \
  --max_ref_tokens "${MAX_REF_TOKENS:-128}" \
  ${GEN_LAYERS:+--gen_layers $GEN_LAYERS} \
  || rc=$?

echo "[slurm] python exited rc=$rc $(date)"
exit $rc