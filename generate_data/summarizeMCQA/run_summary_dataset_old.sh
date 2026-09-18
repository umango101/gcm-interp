#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 01:00:00
#SBATCH -J summary_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH -c 8
set -euo pipefail
mkdir -p logs checkpoint output

echo "[slurm] job ${SLURM_JOB_ID:-none} on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"

# ---- environment (your working vLLM stack) --------------------------------
# System /etc/bashrc and conda activation scripts are NOT nounset-safe, so relax
# errexit+nounset just while sourcing them, then restore strict mode.
set +eu
eval "$(conda shell.bash hook)"
conda activate "${SUMM_ENV:-vllm-summ}"
set -eu

# Avoid FlashInfer JIT sampler (needs nvcc) and the known GLIBCXX ABI mismatch.
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$HOME/orcd/scratch/hf_home}"

# ---- pipeline knobs (all overridable; see CONFIG in the .py) ---------------
export INPUT="${INPUT:-books.json}"
export MODEL="${MODEL:-allenai/OLMo-2-1124-13B-DPO}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"   # one H200 fits 14B in bf16 easily
export LENGTH_RATIO="${LENGTH_RATIO:-2.0}"
export LENGTH_METRIC="${LENGTH_METRIC:-word}"    # word | char | token
export CAP="${CAP:-100}"
# export STRICT=1                                # hard-fail instead of pruning
# export ALLOW_TOKEN_LENGTH_MISMATCH=1           # skip the 'brief'/'detailed' gate

# ---- run: relay SLURM's warning signal so python checkpoints then requeues -
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    requeue_handler() {
        echo "[run] caught USR1 -> forwarding to python for a clean checkpoint"
        kill -USR1 "$PY_PID" 2>/dev/null || true
    }
    trap requeue_handler USR1
    srun --unbuffered python -u build_summary_dataset.py &
    PY_PID=$!
    wait "$PY_PID"
else
    python -u build_summary_dataset.py
fi

echo "[run] finished $(date)"
