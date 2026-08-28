#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J gen_scoring
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 4

source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

cd /home/ubansal/orcd/scratch/conflicts/gcm-interp/generate_data/conflict
mkdir -p logs

MODEL="${MODEL:-openai/gpt-oss-20b}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"

echo "env:  ${CONDA_PREFIX:-?}"

python check_generations.py --data_root data/gpt-oss-20b --out generations.json
