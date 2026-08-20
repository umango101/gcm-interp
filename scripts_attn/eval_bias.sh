#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 24:00:00
#SBATCH -J long_female
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

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
)

declare -A eval_datasets

algos=("atp")
model_id="Qwen/Qwen1.5-14B-Chat"
model_name="Qwen1.5-14B-Chat"
device="cuda:0"

for pair in "${pairs[@]}"; do
    IFS='_' read -r source base <<< "$pair"
    pip install nnsight==0.4.11
  
    for algo in "${algos[@]}"; do
        python run.py --model_id "$model_id" \
                        --batch_size 8 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-single/male-single-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-single/female-single-steering.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-single/male-single-steering.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 8 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-single/male-single-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-long/female-long-steering.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-long/male-long-steering.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 8 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-long/male-long-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-single/female-single-steering.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-single/male-single-steering.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 8 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-long/male-long-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-long/female-long-steering.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/female-long/male-long-steering.jsonl"
    done
done
