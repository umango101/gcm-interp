#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 12:00:00
#SBATCH -J abstraction
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH -c 8

set -eo pipefail
MODEL=${MODEL:-Qwen/Qwen1.5-32B-Chat}
REPO=${RM_INTERP_REPO:-$HOME/orcd/scratch/gcm-interp}
export HF_HOME=${HF_HOME:-/home/ubansal/orcd/scratch/hf_home}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# No pip installs in here: nnsight stays pinned at 0.4.11 in the env.

eval "$(conda shell.bash hook)"
set +eu
conda activate syc
set -eu

cd "$REPO"
MODEL_NAME=$(basename "$MODEL")
DATA_DIR="$REPO/data/$MODEL_NAME/verse-mcqa-abstraction"

# 1) counterfactual datasets + QC (skipped on requeue once the report exists)
if [[ ! -f "$DATA_DIR/qc_report.json" ]]; then
  python generate_data/verse_mcqa_abstraction/build_counterfactuals.py \
    --model "$MODEL" --batch_size 8
fi

# 2) patching (resumes from records_step*.jsonl on requeue)
python mcqa_abstraction_patching.py \
  --model "$MODEL" \
  --steps 1 2 3 \
  --positions paren final \
  --layer_batch 8