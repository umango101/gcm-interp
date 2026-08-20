#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -t 12:00:00
#SBATCH -J svd_layers
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 16

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"

mkdir -p logs

for POOL in tokens mean last; do
  python svd_head_activations.py --model-id Qwen/Qwen1.5-14B-Chat \
      --long-task female-long --long-base male-long \
      --single-task female-single --single-base male-single \
      --pool $POOL --topk 0.01 0.03 0.05 0.07 0.09 0.1 0.5 1.0 --threads 16 --out results/svd_act
done
