#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -J long_persona_experiment
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
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
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubansal/orcd/scratch/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib

declare -a pairs=(
  "extraversion-long_introversion-long"
)
declare -A eval_datasets

algos=("atp")
model_id="google/gemma-3-12b-it"
model_name="gemma-3-12b-it"
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
                    --patch_model
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-long/introversion-long-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-long/extraversion-long-steering.jsonl" \
                    --steering_sub_path "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-long/introversion-long-steering.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-single/introversion-single-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-long/extraversion-long-steering.jsonl" \
                    --steering_sub_path "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-long/introversion-long-steering.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-long/introversion-long-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-single/extraversion-single-steering.jsonl" \
                    --steering_sub_path "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-single/introversion-single-steering.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-single/introversion-single-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-single/extraversion-single-steering.jsonl" \
                    --steering_sub_path "/home/ubansal/orcd/scratch/gcm-interp/data/${model_name}/extraversion-single/introversion-single-steering.jsonl"
  done
done
