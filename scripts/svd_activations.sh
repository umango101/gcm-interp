#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J eval_layers
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"

mkdir -p logs

python cache_head_writes.py --model-id Qwen/Qwen1.5-14B-Chat \
    --task female-long --source female-long --base male-long \
    --n-positions 16 --out cache/writes

python cache_head_writes.py --model-id Qwen/Qwen1.5-14B-Chat \
    --task female-single --source female-single --base male-single \
    --n-positions 16 --out cache/writes
