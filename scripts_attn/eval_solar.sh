#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 24:00:00
#SBATCH -J long_jailbreak
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
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
  "anti-long_pro-long"
)

declare -A eval_datasets

algos=("atp")
model_id="upstage/SOLAR-10.7B-Instruct-v1.0"
model_name="SOLAR-10.7B-Instruct-v1.0"
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
                        --eval_test  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-single/pro-single-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-single/anti-single-desired-all.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-single/pro-single-desired-all.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 1 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-single/pro-single-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-long/anti-long-desired-steering.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-long/pro-long-desired-steering.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 1 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-long/pro-long-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-single/anti-single-desired-all.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-single/pro-single-desired-all.jsonl"

        python run.py --model_id "$model_id" \
                        --batch_size 1 \
                        --patch_algo "$algo" \
                        --source $source \
                        --base  $base \
                        --device "$device" \
                        --eval_model \
                        --eval_test  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-long/pro-long-test.jsonl"\
                        --steering \
                        --ablation steer \
                        --steering_add_path  "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-long/anti-long-desired-steering.jsonl" \
                        --steering_sub_path "${RM_INTERP_REPO}/data/SOLAR-10.7B-Instruct-v1.0/anti-long/pro-long-desired-steering.jsonl"
    done
done