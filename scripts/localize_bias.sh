#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J loc_bias
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH -c 4

source ~/.bashrc
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda info --envs
conda activate /home/ubansal/miniconda/envs/syc
cd /home/ubansal/orcd/scratch/gcm-interp

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"

declare -a pairs=(
  "female-long_male-long"
  "female-single_male-single"
)
declare -A eval_datasets

algos=("atp")
model_id=("google/gemma-3-12b-it")
device="cuda:0"

for model in "${model_id[@]}"; do
    for pair in "${pairs[@]}"; do
        IFS='_' read -r source base <<< "$pair"
    
        for algo in "${algos[@]}"; do
            pip install -U nnsight
            python run.py --model_id "$model" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source "$source" \
                    --base "$base" \
                    --device "$device" \
                    --patch_model
        done
    done
done 
