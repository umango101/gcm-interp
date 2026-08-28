#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J privilege_qc
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -c 4

# Three stages in one job:
#   1. emit candidate files over ~70 color pairs
#   2. QC every pair against gpt-oss-20b (both checks, ~560 forward passes)
#   3. emit the five real files from surviving pairs only, with the test set
#      drawn from color pairs that appear nowhere else
#
# A pair is dropped from ALL files if any of its lines fails either check.

source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

cd /home/ubansal/orcd/scratch/conflicts/gcm-interp/generate_data/conflict
mkdir -p logs

DATA_DIR="${DATA_DIR:-data/gpt-oss-20b/privilege}"
CAND_DIR="$DATA_DIR/candidates"
MODEL="${MODEL:-openai/gpt-oss-20b}"
QC_JSON="$DATA_DIR/pair_qc.json"

echo "env:  ${CONDA_PREFIX:-?}"
echo "data: $DATA_DIR"

echo; echo "=== stage 1: candidates ==="
python make_privilege_datasets.py --mode candidates \
    --out_dir "$CAND_DIR" --tokenizer "$MODEL" \
    --n_conflict_demos "${N_CONFLICT_DEMOS:-6}" \
    --n_candidate_pairs "${N_CANDIDATE_PAIRS:-70}"

echo; echo "=== stage 2: QC ==="
python qc_privilege_datasets.py --data_dir "$CAND_DIR" --model "$MODEL" \
    --reasoning "${REASONING:-low}" --out "$QC_JSON"

echo; echo "=== stage 3: final datasets ==="
python make_privilege_datasets.py --mode final \
    --out_dir "$DATA_DIR" --tokenizer "$MODEL" --qc "$QC_JSON" \
    --meta "$CAND_DIR/candidate_meta.json" \
    --n_loc "${N_LOC:-25}" --n_test "${N_TEST:-25}"
