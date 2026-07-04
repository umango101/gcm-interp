#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J gen_summary_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=512G
#SBATCH --requeue
#SBATCH -c 8

set -euo pipefail
mkdir -p logs checkpoint output

echo "[slurm] job $SLURM_JOB_ID on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"

# --- environment -----------------------------------------------------------
# Dedicated vLLM env. Do NOT reuse the pinned interp env (`syc`); vLLM ships its
# own torch and will fight your pins. Create once:
#   conda create -n vllm-summ python=3.11 -y && conda activate vllm-summ && pip install vllm transformers
# Engaging compute nodes run a non-interactive shell, so conda's functions are
# not loaded by default -> `conda activate` errors with "Run 'conda init'".
# Load them from whichever conda is on PATH (add `module load miniforge` above
# this line if `conda` itself is not found on a fresh node).
eval "$(conda shell.bash hook)"
conda activate "${SUMM_ENV:-vllm-summ}"

export HF_HOME="${HF_HOME:-$HOME/orcd/scratch/hf_home}"     # cache the 14B weights on scratch
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0   # avoid FlashInfer JIT (needs nvcc); use torch-native sampler

# Single job (no sharding). Use exactly the GPUs SLURM actually granted, so a
# 1-GPU allocation runs TP=1 instead of crashing on "World size (2) > 1 GPU".
# (Qwen1.5-14B fits easily on one H200, so TP=1 is fine and avoids NCCL overhead.)
export SHARD=""                  # full books.json -> checkpoint/summaries.jsonl
export STAGES="generate,verify,build"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"; [ "${NGPU:-0}" -ge 1 ] || NGPU=1
export TENSOR_PARALLEL="$NGPU"
echo "[slurm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} -> TENSOR_PARALLEL=$TENSOR_PARALLEL"
nvidia-smi -L || true

# --- run with preemption-signal relay so python checkpoints before it dies ---
# (mit_normal_gpu can still preempt/requeue; the append-only checkpoint makes
#  completed books safe at chunk granularity regardless.)
run_py() {
  python -u "$@" &
  PY_PID=$!
  relay() { echo "[slurm] relaying $1 -> $PY_PID"; kill -"$1" "$PY_PID" 2>/dev/null || true; }
  trap 'relay USR1' USR1
  trap 'relay TERM' TERM
  wait "$PY_PID"
}

# 1) Full build pipeline: generate -> length-verify -> build 8 jsonl files.
echo "[slurm] === build_summary_dataset.py === $(date)"
run_py build_summary_dataset.py

# 2) MCQA verification: prune rows Qwen answers incorrectly in the two desired files.
echo "[slurm] === verify_mcqa.py === $(date)"
run_py verify_mcqa.py

echo "[slurm] pipeline complete $(date)"
