#!/bin/bash
#SBATCH --job-name=gptoss-conflict-eval
#SBATCH --partition=mit_preemptable          # adjust to your actual preemptable partition name
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120                  # SIGUSR1 120s before time limit / preemption
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -o pipefail   # not -e: a non-zero exit from a graceful stop-signal exit is expected on preemption
                  # not -u: conda's own activate/deactivate scripts reference unset
                  # variables (e.g. _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED) and will
                  # blow up under `set -u`, so we leave nounset off entirely.

source ~/miniconda/etc/profile.d/conda.sh
conda activate harmony_env     # swap for whichever env has vLLM + gpt-oss deps installed

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

python run_conflict_eval.py \
    --input conflict_pairs.jsonl \
    --output_dir results/ \
    --model openai/gpt-oss-20b \
    --chunk_size 16 \
    --max_new_tokens 256 \
    --tp_size 1

echo "[sbatch] run_conflict_eval.py exited with code $?"
echo "[sbatch] if this was a preemption/timeout, --requeue will resubmit and the checkpoint in results/responses.jsonl picks up where it left off"
