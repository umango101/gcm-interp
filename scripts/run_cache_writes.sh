#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J cache_writes
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"
mkdir -p logs cache/writes

MODEL="Qwen/Qwen1.5-14B-Chat"
MODEL_NAME="${MODEL##*/}"
NPOS=16
CACHE="cache/writes"

# On mit_preemptable a requeue restarts this script from the top, so skip any
# task whose cache is already complete instead of redoing it.
cache_task () {
  local task="$1" src="$2" bse="$3"
  local dir="${CACHE}/${MODEL_NAME}/${task}"
  if [[ -f "${dir}/source_desired.pt" && -f "${dir}/base_desired.pt" \
        && -f "${dir}/meta_desired.json" ]]; then
    echo "[skip] ${task}: cache already present in ${dir}"
    return 0
  fi
  echo "[run] caching ${task} (${src} vs ${bse})"
  python cache_head_writes.py \
      --model-id "$MODEL" \
      --task "$task" --source "$src" --base "$bse" \
      --n-positions "$NPOS" \
      --batch-size 8 \
      --out "$CACHE"
}

cache_task female-long   female-long   male-long
cache_task female-single female-single male-single

echo "done. cache contents:"
du -sh "${CACHE}/${MODEL_NAME}"/*
