#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 12:00:00
#SBATCH -J conflict_pipeline
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 4

# Instruction-conflict eval pipeline: all 9 matrix conditions, all stages, one job.
# merge + build_prompts are CPU-only but run here on the GPU node for simplicity.
# The judge stage is resumable: on preemption + requeue, completed (cell, pass)
# outputs are skipped, so it continues rather than restarting from scratch.

# NO strict mode yet. /etc/bashrc tests BASHRCSOURCED before assigning it, and
# the conda hook has the same habit, so `set -u` here fails on unbound variables
# in code we do not control. Source first, turn on strict mode after.
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc

# Safe from here on: everything below is ours.
set -euo pipefail

# The pipeline and the matrix results both live in the conflicts checkout.
PIPELINE_REPO="/home/ubansal/orcd/scratch/conflicts/gcm-interp"
RESULTS_MATRIX="${PIPELINE_REPO}/results_matrix"
# Unused by this script; exported for anything downstream that expects it.
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"

cd "$PIPELINE_REPO"
mkdir -p logs

# Fail here, with the real path, rather than minutes into a GPU allocation.
if [ ! -d "$RESULTS_MATRIX" ]; then
    echo "results_matrix not found at: $RESULTS_MATRIX" >&2
    find /home/ubansal/orcd/scratch -maxdepth 4 -name results_matrix -type d 2>/dev/null >&2
    exit 1
fi

echo "python:  $(which python)"
echo "results: $RESULTS_MATRIX"

python eval_pipeline_conflict.py \
    --results_dir "$RESULTS_MATRIX" \
    --experiments 01 02 03 04 05 06 07 08 09 \
    --stages merge build_prompts judge accuracies plots \
    --batch_size 32
