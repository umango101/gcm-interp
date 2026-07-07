#!/bin/bash
# Run the narcissism dataset pipeline.
#   Cluster (preemptable, resumes on requeue):  sbatch run_narcissism.sh
#   Interactive on an allocated node:           bash run_narcissism.sh
# The #SBATCH lines below are read by sbatch and ignored by plain bash.
#SBATCH -p mit_normal_gpu
#SBATCH -t 01:00:00
#SBATCH -J gender_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH -c 8
set -euo pipefail
mkdir -p logs checkpoint output

# ---- environment (matches your working vLLM stack) ----
set +eu
source ~/.bashrc
conda activate vllm-summ
set -eu

# Avoid FlashInfer JIT (needs nvcc) and the known GLIBCXX ABI mismatch on Engaging.
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_TOKEN=...   # only if a gated model is selected (defaults are ungated)

# Stages are resumable and skip completed work on requeue; the Python process
# also traps SIGUSR1/SIGTERM to checkpoint the current chunk and exit cleanly.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Under SLURM: forward SIGUSR1 to requeue, and launch via srun.
    requeue_handler() {
        echo "[run] caught SIGUSR1 -> requeueing $SLURM_JOB_ID"
        scontrol requeue "$SLURM_JOB_ID" || true
        exit 0
    }
    trap requeue_handler USR1
    srun --unbuffered python build_narcissism_dataset.py &
    wait $!
else
    # Direct run (e.g. inside salloc or a plain GPU shell).
    python build_narcissism_dataset.py
fi

echo "[run] finished"
