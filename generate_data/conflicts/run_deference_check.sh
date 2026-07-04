#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 01:00:00
#SBATCH -J gen_conflicts
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

set -euo pipefail

# --------------------------------------------------------------------------
# Paths -- adjust to your checkout.
# --------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/orcd/scratch/gcm-interp}"
SCRIPT_DIR="$REPO_DIR/generate_data/conflicts"                       # wherever gen_conflicts.py / run_deference_check.py live
PAIRS_FILE="${PAIRS_FILE:-$REPO_DIR/generate_data/conflicts/conflict_pairs.jsonl}"
OUT_FILE="${OUT_FILE:-$REPO_DIR/generate_data/conflicts/deference_results.jsonl}"
MODEL="${MODEL:-openai/gpt-oss-20b}"

mkdir -p logs "$(dirname "$OUT_FILE")"

# --------------------------------------------------------------------------
# Environment -- separate env for vLLM + openai-harmony (created via
# setup_harmony_env.sh). Deliberately NOT `syc`, to avoid touching that
# pipeline's package set.
# --------------------------------------------------------------------------
HARMONY_ENV="${HARMONY_ENV:-harmony_env}"
set +u   # conda's activate/deactivate hooks reference unset vars; strict mode chokes on them
source ~/miniconda/etc/profile.d/conda.sh
conda activate "$HARMONY_ENV"
set -u

cd "$REPO_DIR"
git checkout cleanup 2>/dev/null || true   # no-op if already on cleanup / not a git concern here

# --------------------------------------------------------------------------
# CUDA toolkit (nvcc) for flashinfer's JIT compilation of some gpt-oss
# kernels. Prefer a system module if Engaging provides one that matches
# torch's CUDA version (matched by prefix, since module names include a
# patch version e.g. cuda/13.0.1); otherwise fall back to a toolkit
# installed inside the conda env by setup_harmony_env.sh.
# --------------------------------------------------------------------------
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
CUDA_MODULE=$(module -t avail cuda 2>&1 | grep -E "^cuda/${TORCH_CUDA}([.).]|$)" | head -n1)
if [[ -n "$CUDA_MODULE" ]]; then
    module load "$CUDA_MODULE"
    echo "[sbatch] loaded system module $CUDA_MODULE"
else
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "[sbatch] no matching system cuda module found for cuda/${TORCH_CUDA}*; using conda-installed toolkit at $CUDA_HOME"
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[sbatch] WARNING: nvcc still not found on PATH. flashinfer JIT compilation will fail."
    echo "[sbatch]   Run setup_harmony_env.sh's cuda-toolkit install step, or 'module avail cuda' to find the right module name."
fi

echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) starting at $(date)"
echo "[sbatch] pairs=$PAIRS_FILE out=$OUT_FILE model=$MODEL"

# --------------------------------------------------------------------------
# SIGTERM/SIGUSR1 relay: SLURM signals this wrapper shell, not the python
# process directly (they're different PIDs once launched in the background).
# Forward the signal so run_deference_check.py's own handler gets a chance
# to stop cleanly after the current batch and leave the --out file in a
# resumable state -- then let --requeue pick it back up.
# --------------------------------------------------------------------------
PYTHON_PID=""

relay_signal() {
    echo "[sbatch] received signal, relaying to python pid=$PYTHON_PID at $(date)"
    if [[ -n "$PYTHON_PID" ]]; then
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
    fi
}
trap relay_signal USR1 TERM

python3 "$SCRIPT_DIR/deference_check.py" \
    --pairs "$PAIRS_FILE" \
    --out "$OUT_FILE" \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --batch-size 32 \
    --use-judge \
    &
PYTHON_PID=$!

wait "$PYTHON_PID"
EXIT_CODE=$?

echo "[sbatch] python exited with code $EXIT_CODE at $(date)"

# If we were preempted/requeued mid-run, SLURM will resubmit this same
# script; run_deference_check.py's resumable --out means it picks up
# from the next un-processed pair_id automatically. Nothing else to do here.
exit $EXIT_CODE
