#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 12:00:00
#SBATCH -J gender_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH -c 8

# Five models generate sequentially in one allocation, so 2h is not enough: each
# model is ~2k generations (971 professions x single+story) through eager HF
# generate, and the 32B is the slow one. The job is resumable per model, so a
# preemption costs at most one chunk -- but a wall-clock timeout mid-sweep just
# burns a requeue.

set -euo pipefail
mkdir -p logs

echo "[slurm] job $SLURM_JOB_ID on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The interp env, not `vllm-summ`: this script is plain transformers now, and
# device_map="auto" needs accelerate. `syc` has torch 2.7.0+cu126 / accelerate
# 1.11.0 / transformers 4.53.3, which covers every model in MODELS below (gemma-3
# needs >= 4.50). Note 4.53.3 predates the torch_dtype -> dtype rename; the script
# tries both, so nothing to do here.
#
# set +eu around activate: the conda hook trips `set -u` on unbound vars.
set +eu
source /home/ubansal/miniconda/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate "${GEN_ENV:-syc}"
set -eu
echo "[slurm] env=${CONDA_DEFAULT_ENV:-?} python=$(which python)"

export HF_HOME="${HF_HOME:-$HOME/orcd/scratch/hf_home}"
export TOKENIZERS_PARALLELISM=false

# Determinism. PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG must be exported here:
# the first is read by the interpreter at startup, the second before the first CUDA
# context. Everything else (deterministic algorithms, cudnn flags, TF32) is set by
# enable_determinism() inside the script and read back into the run signature.
export SEED=42
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# NVIDIA_TF32_OVERRIDE=0 disables TF32 at the driver level regardless of the torch
# flags, which would leave the signature recording matmul_allow_tf32=True while the
# hardware quietly did something else. Clear it if a module file set it.
unset NVIDIA_TF32_OVERRIDE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO="$HOME/orcd/scratch/gcm-interp"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/generate_data/bias/output}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO/generate_data/bias/checkpoint}"
export MODELS="allenai/OLMo-2-1124-13B-DPO,tiiuae/Falcon3-10B-Instruct,google/gemma-3-12b-it,Qwen/Qwen1.5-14B-Chat,Qwen/Qwen1.5-32B-Chat"
# One bad model does not cost the other four; the job still exits 1 at the end.
export CONTINUE_ON_ERROR=1

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"

python build_gender_dataset.py &
PY_PID=$!

relay() { echo "[slurm] relaying $1 -> $PY_PID"; kill -"$1" "$PY_PID" 2>/dev/null || true; }
trap 'relay USR1' USR1
trap 'relay TERM' TERM

# Two things this loop fixes:
#  * `wait` returns 128+N as soon as a trapped signal arrives, while python is still
#    writing its checkpoint. A single wait would let the script exit out from under
#    it. Loop until the child is actually gone.
#  * a bare `wait "$PY_PID"; rc=$?` never reaches the assignment under `set -e` --
#    the failing wait exits the script first, so a nonzero rc was silently turned
#    into an unreported exit.
rc=0
while true; do
  # Not `set -e`-triggering: wait is the left side of an && list.
  wait "$PY_PID" && { rc=0; break; }
  rc=$?
  # A status over 128 means either a trapped signal interrupted wait (python still
  # running -- go round again and let it finish checkpointing) or the child itself
  # was killed by a signal (child gone -- report it). kill -0 tells them apart.
  if [ "$rc" -le 128 ] || ! kill -0 "$PY_PID" 2>/dev/null; then
    break
  fi
done

echo "[slurm] python exited rc=$rc $(date)"
exit $rc