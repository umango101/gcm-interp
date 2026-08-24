#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 00:40:00
#SBATCH -J icl_test2
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -c 4

# ICL preamble v2: Dev / Neutral / User on 40 balanced held-out color items.
# One forward pass per (condition x item) = 120 passes. Minutes.
#
#   sbatch test_icl_preamble2.sh              # 6 demos
#   N_DEMOS=3 sbatch test_icl_preamble2.sh    # dose check

source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

cd /home/ubansal/orcd/scratch/conflicts/gcm-interp
mkdir -p logs

N_DEMOS="${N_DEMOS:-6}"
echo "env:    ${CONDA_PREFIX:-?}"
echo "demos:  $N_DEMOS"
python -c "import transformers; print('transformers:', transformers.__version__)"

python test_icl_preamble2.py \
    --model "${MODEL:-openai/gpt-oss-20b}" \
    --n_demos "$N_DEMOS" \
    --reasoning "${REASONING:-low}" \
    --out "icl_preamble_test_v2_demos${N_DEMOS}.json"
