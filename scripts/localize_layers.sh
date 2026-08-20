#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J loc_layers
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 4

# Layer-level localization: ATP on residual streams for both task formats.
# Writes ./results_layers/{model}/from_{source}_to_{base}/atp/layers_*.pt plus the
# reduced map numerator_1_layers.pt and a per-layer effect plot.
#
# Deliberately NOT running `pip install -U nnsight` -- that unpinned upgrade pulled
# 0.7.0 mid-run once already. nnsight stays at the 0.4.11 the tracing code targets.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"

mkdir -p logs
source scripts/_preflight.sh
preflight_gpu
echo "RM_INTERP_REPO is $RM_INTERP_REPO"

declare -a pairs=(
  "female-long_male-long"
  "female-single_male-single"
)

algo="atp"
model_id="google/gemma-3-12b-it"
device="cuda:0"

for pair in "${pairs[@]}"; do
  IFS='_' read -r source base <<< "$pair"
  echo "=== localizing layers: from_${source}_to_${base} ==="
  python -m layers.run_layers \
    --model_id "$model_id" \
    --batch_size 1 \
    --patch_algo "$algo" \
    --source "$source" \
    --base "$base" \
    --device "$device" \
    --patch_model \
    --sdp_backend math \
    --strict_determinism \
    --results_root ./results_layers
done
