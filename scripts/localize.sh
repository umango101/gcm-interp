#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 24:00:00
#SBATCH -J q-vp-l
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=48G
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
  "anti-long_pro-long"
  "anti-single_pro-single"
)
declare -A eval_datasets

algos=("atp")
model_id=("Qwen/Qwen1.5-14B-Chat")
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