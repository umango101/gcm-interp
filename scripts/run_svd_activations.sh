#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -t 04:00:00
#SBATCH -J svd_act
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 96
#SBATCH --array=0-2

# One array task per pool, so the three run concurrently and a failure in one
# does not take the others down. mit_normal has 192-core nodes (6 of them with
# 1510G), so -c 96 is a single socket and schedules readily.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"
mkdir -p logs results/svd_act

# Keep BLAS from oversubscribing: torch --threads sets the torch pool, but the
# OpenMP/MKL env vars govern the underlying BLAS and default to all 192 cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}

POOLS=(tokens mean last)
POOL="${POOLS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "pool = ${POOL}, cores = ${SLURM_CPUS_PER_TASK:-96}"

# 5120 / head_dim 128 = 40 heads is the rank ceiling for tokens/last, i.e.
# top_k 0.025. Values at or below it are the ones whose overlap is readable;
# the larger ones are kept because they are cheap now, but expect rank_a and
# rank_b to sit near 5120 there and the overlap to be uninformative.
TOPKS="0.005 0.01 0.015 0.02 0.025 0.03 0.05 0.07 0.09 0.1 0.5 1.0"

python svd_head_activations.py \
    --model-id Qwen/Qwen1.5-14B-Chat \
    --long-task female-long     --long-base   male-long \
    --single-task female-single --single-base male-single \
    --pool "$POOL" \
    --topk $TOPKS \
    --max-columns 512 \
    --dtype float32 \
    --threads "${SLURM_CPUS_PER_TASK:-96}" \
    --n-null 20 \
    --out results/svd_act

echo "done: pool=${POOL}"
