#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J conflict_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH -c 8

set -euo pipefail
mkdir -p logs checkpoint output

echo "[slurm] job $SLURM_JOB_ID on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"

# Dedicated vLLM env (not the pinned interp env `syc`).
source /home/ubansal/miniconda/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate "${SUMM_ENV:-syc}"

export HF_HOME="${HF_HOME:-/home/ubansal/orcd/scratch/hf_home}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0   # avoid FlashInfer JIT (needs nvcc)
export TENSOR_PARALLEL=1

# Relay preemption signal so python checkpoints the current chunk before exit.
python gen_conflict_polarity.py --out data/gpt-oss-20b --validate --device cuda:0 & 
# python -u gen_conflict_polarity.py &
PY_PID=$!
relay() { echo "[slurm] relaying $1 -> $PY_PID"; kill -"$1" "$PY_PID" 2>/dev/null || true; }
trap 'relay USR1' USR1
trap 'relay TERM' TERM
wait "$PY_PID"; rc=$?
echo "[slurm] python exited rc=$rc $(date)"
exit $rc
