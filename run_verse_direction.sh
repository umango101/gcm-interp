#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 00:30:00
#SBATCH -J vl_dir
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH -c 8
#SBATCH --requeue

# Per-layer verse/prose difference-in-means: logit lens readout + steering accuracy sweep.
# Reads data/{model}/verse-long/ directly -- no pair building needed.
#
# Cost is two forwards per train item (direction) plus one batched forward per
# (test item, scaling, steer position, layer chunk) -- so runtime scales with the number of
# entries in SCALINGS x STEER_POSITIONS. Records append with fsync and are skipped per
# (item, scaling, steer position) on restart, so preemption costs at most one sweep.
#
# STEER_POSITIONS="last all prompt answer" sbatch scripts/run_verse_direction.sh
# SOURCE_POSITION=answer sbatch scripts/run_verse_direction.sh    # direction over answer tokens
# GENERATE_LAYERS="30 40" sbatch scripts/run_verse_direction.sh   # dump text for the judge

set -euo pipefail
mkdir -p logs
echo "[slurm] job ${SLURM_JOB_ID:-?} on $(hostname) restart=${SLURM_RESTART_COUNT:-0} $(date)"

set +eu
source /home/ubansal/miniconda/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate "${GEN_ENV:-syc}"
set -eu
echo "[slurm] env=${CONDA_DEFAULT_ENV:-?} python=$(which python)"

export HF_HOME="${HF_HOME:-$HOME/orcd/scratch/hf_home}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset NVIDIA_TF32_OVERRIDE

REPO="${RM_INTERP_REPO:-$HOME/orcd/scratch/gcm-interp}"
MODEL="${MODEL:-Qwen/Qwen1.5-32B-Chat}"
cd "$REPO"

rc=0
python verse_direction_logit_lens.py \
  --model "$MODEL" \
  --source_position "${SOURCE_POSITION:-last}" \
  --steer_positions ${STEER_POSITIONS:-last all prompt answer} \
  --scalings ${SCALINGS:-raw normed} \
  --resid_frac "${RESID_FRAC:-0.1}" \
  --alpha "${ALPHA:-1.0}" \
  --top_k "${TOP_K:-10}" \
  --train_frac "${TRAIN_FRAC:-0.5}" \
  --seed "${SEED:-0}" \
  --layer_batch "${LAYER_BATCH:-16}" \
  ${GENERATE_LAYERS:+--generate_layers $GENERATE_LAYERS} \
  || rc=$?

echo "[slurm] python exited rc=$rc $(date)"
exit $rc