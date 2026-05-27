#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 24:00:00
#SBATCH -J long_para
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=512G
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
  "harmful-long_harmless-long"
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
                        --batch_size 1 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-single/harmless-single-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-single/harmful-single-desired-all.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-single/harmless-single-desired-all.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 1 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-single/harmless-single-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-long/harmful-long-desired-all.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-long/harmless-long-desired-all.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 1 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-long/harmless-long-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-single/harmful-single-desired-all.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-single/harmless-single-desired-all.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 1 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-long/harmless-long-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-long/harmful-long-desired-all.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/Qwen1.5-14B-Chat/harmful-long/harmless-long-desired-all.jsonl"
    done
done