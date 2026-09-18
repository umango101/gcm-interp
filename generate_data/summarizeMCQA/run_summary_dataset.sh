#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 12:00:00
#SBATCH -J summary_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH -c 8
#
# Usage
# -----
#   sbatch run_summary_dataset.sh                                  # all models, in order
#   MODELS_OVERRIDE="allenai/OLMo-2-1124-13B-DPO" sbatch run_summary_dataset.sh
#   STAGES=build sbatch run_summary_dataset.sh                     # CPU-only re-emit
#
# One job, one GPU, the models run back to back. Each model gets its own
# output/<slug>/ and checkpoint/<slug>/, so finished models are skipped on a
# requeue (cached stages never load weights) and a model that dies doesn't stop
# the ones after it — set CONTINUE_ON_ERROR=0 to make it abort instead.
#
# Walltime: plain `transformers` generation is roughly an order of magnitude
# slower than vLLM was, and this is five models in series, so 12h is a starting
# guess rather than a measured number. The run is checkpointed and requeues
# cleanly, so undershooting costs model reloads, not work.
 
set -euo pipefail
mkdir -p logs checkpoint output
 
echo "[slurm] job ${SLURM_JOB_ID:-none} on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"
 
# ===========================================================================
# THE FIVE MODELS — replace these with yours. They run in this order.
# ===========================================================================
MODELS=(
    "allenai/OLMo-2-1124-13B-DPO",
    "tiiuae/Falcon3-10B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-32B-Chat"
)
 
RUN_MODELS="${MODELS_OVERRIDE:-$(IFS=,; echo "${MODELS[*]}")}"
echo "[run] models: $RUN_MODELS"
 
# ---- environment ----------------------------------------------------------
# System /etc/bashrc and conda activation scripts are NOT nounset-safe, so relax
# errexit+nounset just while sourcing them, then restore strict mode.
set +eu
eval "$(conda shell.bash hook)"
conda activate "${SUMM_ENV:-hf-summ}"
set -eu
 
# Determinism: the .py sets this with os.environ.setdefault before importing
# torch, but exporting it here makes it true for anything else in the process
# tree too, and is the belt-and-braces version of "before the CUDA context".
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
 
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/orcd/scratch/hf_home}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_DISABLE_TELEMETRY=1
 
# ---- pipeline knobs (all overridable; see CONFIG in the .py) ---------------
export INPUT="${INPUT:-books.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-output}"          # -> output/<model-slug>/
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoint}"
export STAGES="${STAGES:-tokcheck,generate,prune,build}"
 
export DEVICE_MAP="${DEVICE_MAP:-auto}"              # one H200 holds 13B bf16 easily
export DTYPE="${DTYPE:-bfloat16}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"                # 'eager' once you hang hooks
export BATCH_SIZE="${BATCH_SIZE:-8}"
# Chunk = checkpoint flush granularity. Smaller chunks mean a USR1 lands on a
# boundary sooner (the python only exits between chunks), at the cost of more
# fsyncs. 16 books x 2 passes is comfortably inside the 600s signal window.
export CHUNK_SIZE="${CHUNK_SIZE:-16}"
 
export LENGTH_RATIO="${LENGTH_RATIO:-2.0}"
export LENGTH_METRIC="${LENGTH_METRIC:-word}"        # word | char | token
export CAP="${CAP:-100}"
export SEED="${SEED:-0}"
# export STRICT_DETERMINISM=1                        # BATCH_SIZE=1, bit-exact across requeues
# export STRICT=1                                    # hard-fail instead of pruning
# export VERIFY_MODE=report                          # keep all, just log MCQA accuracy
# export ALLOW_TOKEN_LENGTH_MISMATCH=1               # skip the contrastive-token gate
# export CONTINUE_ON_ERROR=0                         # abort the sweep on first failure
 
# ---- run: relay SLURM's warning signal so python checkpoints then requeues -
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    requeue_handler() {
        echo "[run] caught USR1 -> forwarding to python for a clean checkpoint"
        kill -USR1 "$PY_PID" 2>/dev/null || true
    }
    trap requeue_handler USR1
 
    srun --unbuffered python -u build_summary_dataset.py --models "$RUN_MODELS" &
    PY_PID=$!
 
    # A trapped signal interrupts `wait`, which returns 128+n *without* reaping
    # the child. Under `set -e` that would kill this script out from under the
    # checkpoint we just asked for, so drop strict mode and wait again until the
    # child is actually gone.
    set +e
    wait "$PY_PID"; rc=$?
    while kill -0 "$PY_PID" 2>/dev/null; do
        wait "$PY_PID"; rc=$?
    done
    set -e
else
    python -u build_summary_dataset.py --models "$RUN_MODELS"
    rc=$?
fi
 
echo "[run] finished rc=$rc $(date)"
exit "$rc"