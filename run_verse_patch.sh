#!/bin/bash
# =============================================================================
# verse_patch.py -- per-layer residual-stream patching, verse run -> prose run.
#
# Resilient by design: verse_patch.py appends an fsync'd record per
# (id, patch_position) to records_accuracy.jsonl and skips completed pairs on
# restart, so a requeued job resumes mid-sweep with no lost work and no
# duplicate compute.  The trap below forwards the signal to Python so the
# in-flight record's file handle is closed cleanly before the requeue.
#
#   sbatch run_verse_patch.sh
#   PATCH_POSITIONS="last prompt answer" sbatch --export=ALL run_verse_patch.sh
#   LAYER_BATCH=4 EXTRA="--lens" sbatch --export=ALL run_verse_patch.sh
#
# mkdir -p logs BEFORE the first submit -- sbatch opens the log files at submit
# time and fails outright if the directory is missing.
# -----------------------------------------------------------------------------
# CHECK THESE FOUR AGAINST run_verse_direction.sh BEFORE THE FIRST SUBMIT.
# Partition and account names are cluster policy, not repo policy, and I have
# guessed them here rather than copied them from a working script.
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --mem=192G
# -----------------------------------------------------------------------------
#SBATCH --job-name=verse-patch
#SBATCH --output=logs/verse_patch_%j.out
#SBATCH --error=logs/verse_patch_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
# =============================================================================

set -euo pipefail

REPO="${REPO:-$HOME/orcd/scratch/gcm-interp}"
MODEL="${MODEL:-Qwen/Qwen1.5-32B-Chat}"
PATCH_POSITIONS="${PATCH_POSITIONS:-last all}"
ALPHA="${ALPHA:-1.0}"
LAYER_BATCH="${LAYER_BATCH:-8}"
MAX_REF_TOKENS="${MAX_REF_TOKENS:-128}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"          # e.g. --lens, --max_items 8, --generate_layers 20 40

MODEL_SHORT="${MODEL##*/}"
OUT_DIR="${OUT_DIR:-$REPO/results/$MODEL_SHORT/verse-patch}"

cd "$REPO"
mkdir -p logs "$OUT_DIR"

# ---------------------------------------------------------------- environment
# set +eu around the conda hook: the activate scripts reference unset variables
# and trip -u before the env is usable.
set +eu
eval "$(conda shell.bash hook)"
conda activate syc
set -eu

export HF_HOME=/home/ubansal/orcd/scratch/hf_home
export HF_HUB_OFFLINE=1        # the weights are already in scratch; no hub call from a compute node
export PYTHONNOUSERSITE=1      # keeps ~/.local out of sys.path ahead of the env
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "[slurm] job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "[slurm] restart count: ${SLURM_RESTART_COUNT:-0}"
echo "[slurm] repo $REPO @ $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') "\
"$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

# ---------------------------------------------------------------- version guard
# Fail fast rather than installing anything. An unpinned `pip install -U nnsight`
# inside a SLURM loop is what pulled 0.7.0 and broke the localization run; and
# `syc` has drifted back to transformers 4.57.1 more than once.
python - <<'PY'
import sys
import nnsight, transformers, torch
print(f"[env] python      {sys.version.split()[0]}")
print(f"[env] torch       {torch.__version__}  cuda {torch.version.cuda}")
print(f"[env] transformers {transformers.__version__}")
print(f"[env] nnsight     {nnsight.__version__}")
bad = []
if not nnsight.__version__.startswith("0.4.11"):
    bad.append(f"nnsight {nnsight.__version__} (need 0.4.11; 0.5+ renamed LanguageModel's "
               f"config= kwarg and leaves model.config None)")
if not transformers.__version__.startswith("4.53.3"):
    bad.append(f"transformers {transformers.__version__} (need 4.53.3)")
if not torch.cuda.is_available():
    bad.append("no CUDA device visible")
if bad:
    sys.exit("[env] refusing to run:\n  - " + "\n  - ".join(bad))
PY

# ---------------------------------------------------------------- signal handling
CHILD=""
on_signal() {
    echo "[slurm] signal at $(date -Is); draining child ${CHILD:-none}"
    if [[ -n "$CHILD" ]]; then
        kill -TERM "$CHILD" 2>/dev/null || true
        wait "$CHILD" 2>/dev/null || true
    fi
    echo "[slurm] requeueing $SLURM_JOB_ID (records_accuracy.jsonl carries the progress)"
    scontrol requeue "$SLURM_JOB_ID" || echo "[slurm] requeue failed; resubmit by hand"
    exit 0
}
trap on_signal USR1 TERM

# ---------------------------------------------------------------- run
# Backgrounded + wait so the trap fires immediately instead of after the
# foreground command returns.
# shellcheck disable=SC2086
python -u verse_patch.py \
    --model "$MODEL" \
    --out_dir "$OUT_DIR" \
    --patch_positions $PATCH_POSITIONS \
    --alpha "$ALPHA" \
    --layer_batch "$LAYER_BATCH" \
    --max_ref_tokens "$MAX_REF_TOKENS" \
    --seed "$SEED" \
    $EXTRA &
CHILD=$!

set +e
wait "$CHILD"
STATUS=$?
set -e

echo "[slurm] python exited $STATUS at $(date -Is)"
if [[ $STATUS -ne 0 ]]; then
    echo "[slurm] NOT requeueing on a nonzero exit -- a crash would requeue forever."
    echo "[slurm] partial records are in $OUT_DIR/records_accuracy.jsonl;"
    echo "[slurm] fix and resubmit, or run with --plot_only to aggregate what landed."
    exit $STATUS
fi

echo "[slurm] done. outputs in $OUT_DIR"
ls -la "$OUT_DIR"