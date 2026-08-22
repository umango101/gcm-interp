#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J svd_act_gpu
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --array=0-2

# GPU alternative to run_svd_activations.sh. Use this if the mit_normal queue is
# long, or if you want many more null draws than 20.
#
# GPU choice matters: an L40S runs float64 at roughly 1/64 of its float32 rate
# and would be slower than a CPU node, so on L40S you must pass --dtype float32.
# The H200 requested here does float64 well, so either dtype is fine; float32 is
# still ~2x faster and the metrics agree with float64 to ~1e-4, far below the
# spread of the layer-matched null they are compared against.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"
mkdir -p logs results/svd_act_gpu

POOLS=(tokens mean last)
POOL="${POOLS[${SLURM_ARRAY_TASK_ID:-0}]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "pool = ${POOL}"

TOPKS="0.005 0.01 0.015 0.02 0.025 0.03 0.05 0.07 0.09 0.1 0.5 1.0"

python svd_head_activations.py \
    --model-id Qwen/Qwen1.5-14B-Chat \
    --long-task female-long     --long-base   male-long \
    --single-task female-single --single-base male-single \
    --pool "$POOL" \
    --topk $TOPKS \
    --max-columns 512 \
    --device cuda \
    --dtype float32 \
    --n-null 50 \
    --out results/svd_act_gpu

echo "done: pool=${POOL}"
