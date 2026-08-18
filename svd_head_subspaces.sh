#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -t 05:30:00
#SBATCH -J svd_subspaces
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 16

source ~/.bashrc
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
cd "$RM_INTERP_REPO"

python svd_head_subspaces.py --model-id Qwen/Qwen1.5-14B-Chat --results-root ./results --task all --topk 0.01 0.03 0.05 0.07 0.09 0.1 0.5 1.0 --out results/svd
