#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J vl_pairs
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH -c 8
#SBATCH --requeue

# Builds and QCs the (Q1, Q2) pairs for the verse-long causal abstraction experiment.
# Work is three teacher-forced forward passes per candidate pair (~50 pairs), so this is
# minutes of GPU time plus the model load -- 2h is generous.
#
# build_pairs.py has no checkpointing and writes pairs.jsonl at the end, so there is no signal
# relay here: a preempted job just reruns from scratch on requeue. pairs_report.json is written
# last and marks a complete run; the guard below skips the build if it exists, so a resubmit
# cannot silently replace a pair set that experiment records were already computed against.
# FORCE=1 rebuilds.

set -euo pipefail
mkdir -p logs
echo "[slurm] job ${SLURM_JOB_ID:-?} on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"

# set +eu around activate: the conda hook trips `set -u` on unbound vars.
set +eu
source /home/ubansal/miniconda/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate "${GEN_ENV:-syc}"
set -eu
echo "[slurm] env=${CONDA_DEFAULT_ENV:-?} python=$(which python)"

export HF_HOME="${HF_HOME:-$HOME/orcd/scratch/hf_home}"
# Must be set before the first CUDA context; determinism.py also setdefaults it.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset NVIDIA_TF32_OVERRIDE

REPO="${RM_INTERP_REPO:-$HOME/orcd/scratch/gcm-interp}"
MODEL="${MODEL:-Qwen/Qwen1.5-32B-Chat}"
OUT_DIR="$REPO/data/$(basename "$MODEL")/verse-long-abstraction"

if [[ -f "$OUT_DIR/pairs_report.json" && "${FORCE:-0}" != "1" ]]; then
  echo "[slurm] $OUT_DIR/pairs_report.json exists; skipping (FORCE=1 to rebuild)"
  exit 0
fi

cd "$REPO"

# Tokenizer-only sanity pass. Costs seconds, and its output is what you read if the pair count
# or the format-aligned count comes out lower than expected.
python verse_long_utils.py --model "$MODEL" || true

rc=0
python generate_data/verse_long_abstraction/build_pairs.py \
  --model "$MODEL" \
  --max_similarity "${MAX_SIMILARITY:-0.2}" \
  --min_margin "${MIN_MARGIN:-0.0}" \
  --max_ref_tokens "${MAX_REF_TOKENS:-128}" \
  || rc=$?

echo "[slurm] python exited rc=$rc $(date)"
exit $rc