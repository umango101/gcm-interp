#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J fact_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH -c 8

set -euo pipefail
mkdir -p logs checkpoint output

# ---- environment (matches your working vLLM stack) ----
# System /etc/bashrc and conda activation scripts are NOT nounset-safe, so relax
# errexit+nounset just while sourcing them, then restore strict mode.
set +eu
source ~/.bashrc
conda activate vllm-summ
set -eu

# Avoid FlashInfer JIT (needs nvcc) and the known GLIBCXX ABI mismatch on Engaging.
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ---- determinism ----------------------------------------------------------
# Read at process start (by cuBLAS, by the interpreter, by the rust tokenizer),
# so setting them from inside python has no effect -- build_truthfulqa_dataset.py
# asserts them instead of setting them. Same contract as eval/setup.py.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
export VLLM_ENABLE_V1_MULTIPROCESSING="0"
# 0 = warn, 1 = assert the env (default), 2 = also hard-fail on any torch op with
#     no deterministic implementation. Use 2 to find the offending op, not for
#     production runs.
export STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"
# Part of the dataset identity: SEED decides which of A/B holds the truthful
# answer for every row. Existing checkpoints under checkpoint/ are only valid
# for the seed that produced them.
export SEED="${SEED:-42}"

# STAGES is a LIST, and each entry gets its OWN python process below. vLLM does
# not reliably hand back a model's memory, so a single process that loads QC_MODEL
# and then JUDGE_MODEL dies on free memory at the second load; one process per
# stage sidesteps that and keeps each engine's KV cache sized against an empty
# GPU. Override to run a single stage, e.g. STAGES=qcjudge sbatch run_truthfulqa.sh
export STAGES="${STAGES:-qcgen,qcjudge,build}"
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
fi

# One python process per stage, in order. Each exits before the next starts, so
# the GPU is empty when the next engine sizes itself -- and a stage that dies
# leaves the earlier stages' checkpoints intact, so a rerun resumes rather than
# repeating them.
IFS=',' read -r -a _stages <<< "$STAGES"
for stage in "${_stages[@]}"; do
    stage="$(echo "$stage" | tr -d '[:space:]')"
    [[ -z "$stage" ]] && continue
    echo "[run] === stage: $stage ==="
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        STAGES="$stage" srun --unbuffered python build_truthfulqa_dataset.py &
        wait $!
    else
        STAGES="$stage" python build_truthfulqa_dataset.py
    fi
    echo "[run] === stage $stage done ==="
done

echo "[run] finished"
