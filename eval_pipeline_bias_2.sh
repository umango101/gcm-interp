#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -J bias_pipeline
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 4

# Full harmful->harmless pipeline, all 8 cells, all stages, in one job.
# merge + build_prompts are CPU-only but run here on the GPU node for
# simplicity. The judge stage is resumable: on preemption + requeue,
# completed (cell, pass) outputs are skipped, so it continues where it
# left off rather than restarting from scratch.

source ~/.bashrc
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
cd "$RM_INTERP_REPO"

python eval_pipeline_bias_2.py --stages merge build_prompts judge accuracies plots --batch_size 32
