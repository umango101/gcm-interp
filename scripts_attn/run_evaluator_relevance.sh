#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 06:00:00
#SBATCH -J relevance
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
python -m pip install vllm

echo "RM_INTERP_REPO is $RM_INTERP_REPO"

python judge-evals/evaluator.py --input_csv eval_para_long/Qwen1.5-14B-Chat/relevance_fluency_prompts.csv --relevance