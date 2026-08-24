#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 00:30:00
#SBATCH -J icl_test
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -c 4

# Quick check: does the ICL preamble induce user-deference in gpt-oss-20b?
# One forward pass per (condition x test item) -- minutes, not hours.
#
# NOTE: the 'syc' env is for the gemma head-localization work and its
# transformers is too old for gpt-oss (needs >= 4.55). Use whichever env the
# matrix run uses:  grep -n 'conda activate' run_gptoss_matrix.sh
#
#   ENV=/path/to/gptoss-env sbatch test_icl_preamble.sh

source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

cd /home/ubansal/orcd/scratch/conflicts/gcm-interp
mkdir -p logs

MODEL="${MODEL:-openai/gpt-oss-20b}"
echo "env:    ${CONDA_PREFIX:-?}"
echo "python: $(which python)"
echo "model:  $MODEL"
python -c "import transformers; print('transformers:', transformers.__version__)"

python test_icl_preamble.py --model "$MODEL" --n_demos "${N_DEMOS:-6}" --reasoning low --print_prompt
