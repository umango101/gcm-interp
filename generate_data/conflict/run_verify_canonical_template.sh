#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J verify_harmony
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
  python make_hierarchy_datasets.py --mode candidates --arm $ARM \
    --out_dir data/gpt-oss-20b/hier-$ARM/candidates \
    --tokenizer openai/gpt-oss-20b
done

python verify_canonical_template.py --files \
  data/gpt-oss-20b/hier-devuser/candidates/dev-single-desired-all.jsonl \
  data/gpt-oss-20b/hier-sysuser/candidates/dev-single-desired-all.jsonl \
  data/gpt-oss-20b/hier-sysdev/candidates/dev-single-desired-all.jsonl

python probe_level_compliance.py --out probe_levels.json

python probe_level_compliance.py --variants devuser,sysuser,sysdev_ruleform,norule --out probe_levels_v2.json
