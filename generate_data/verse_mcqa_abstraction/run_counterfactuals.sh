#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J cf_gen
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Builds the three corrupted datasets for the verse-MCQA causal abstraction experiment and
# QCs every clean/corrupted prompt with one next-token forward pass each (~1000 prompts for
# 100 base rows). No generation, so this is minutes of GPU time plus the 32B model load.
#
# build_counterfactuals.py has no checkpointing and writes its outputs at the end, so there
# is no signal relay here: on preemption the requeued job simply reruns from scratch.
# qc_report.json is written last, so it marks a complete run; the guard below skips the
# build if it exists (FORCE=1 to rebuild). This keeps a requeue or resubmit from replacing
# a dataset that patching results were already computed on.

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
# Must be set before the first CUDA context. determinism.py also setdefaults it, but
# exporting here covers anything that initializes CUDA earlier.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset NVIDIA_TF32_OVERRIDE

REPO="${RM_INTERP_REPO:-$HOME/orcd/scratch/gcm-interp}"
MODEL="${MODEL:-Qwen/Qwen1.5-32B-Chat}"
OUT_DIR="$REPO/data/$(basename "$MODEL")/verse-mcqa-abstraction"

if [[ -f "$OUT_DIR/qc_report.json" && "${FORCE:-0}" != "1" ]]; then
  echo "[slurm] $OUT_DIR/qc_report.json exists; skipping (FORCE=1 to rebuild)"
  exit 0
fi

cd "$REPO"
rc=0
python generate_data/verse_mcqa_abstraction/build_counterfactuals.py \
  --model "$MODEL" \
  --batch_size 8 \
  || rc=$?

echo "[slurm] python exited rc=$rc $(date)"
exit $rc