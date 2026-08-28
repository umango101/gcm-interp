#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J qc_datasets
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

echo "env:  ${CONDA_PREFIX:-?}"

for ARM in devuser sysuser sysdev; do
  python qc_hierarchy_datasets.py \
    --data_dir data/gpt-oss-20b/hier-$ARM/candidates \
    --out data/gpt-oss-20b/hier-$ARM/pair_qc.json
done

python sweep_demos.py --arms devuser sysuser sysdev --demos 4 6 8 10 --out sweep_demos.json
