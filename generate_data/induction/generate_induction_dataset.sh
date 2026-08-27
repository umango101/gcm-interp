#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 04:00:00
#SBATCH -J gen_induction_data
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

# Build the INDUCTION control corpus: match-and-copy available vs not.
#
#   sbatch generate_data/induction/generate_induction_dataset.sh
#   sbatch --export=ALL,STAGES=final generate_data/induction/generate_induction_dataset.sh
#
# Stages: build (CPU + tokenizer), qc (GPU, loads gpt-oss), final.
# Output: data/gpt-oss-20b/induction-single/
#
# WHY. Both the privilege and position corpora present an eight-turn ICL preamble
# and ask for a one-word answer, so induction heads plausibly rank highly in both
# ATP maps for reasons unrelated to either construct. Localizing on a corpus where
# match-and-copy is the ONLY thing that varies gives the shared component a name,
# and gives the privilege-vs-position overlap number the floor it otherwise lacks:
# whatever overlap induction shows with privilege is roughly what two contrasts
# sharing this surface form produce for generic reasons.
#
# KEY TOKEN LENGTHS. --key_tokens / --value_tokens select which slice of the nonce
# pool to use. Stage 1 prints the histogram; if it aborts saying the pool is too
# small, pick a length with more entries rather than raising the pool blindly.
# Keys must be length-matched to each other (the query is the only token that
# differs between conditions, so a length mismatch breaks align_toks) and so must
# values (desired vs undesired share every token but the last).

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

REPO="${REPO:-/home/ubansal/orcd/scratch/conflicts/gcm-interp/generate_data/induction}"
cd "$REPO"
mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="false"
export PYTHONHASHSEED="0"
SEED="${SEED:-42}"

model_id="${MODEL_ID:-openai/gpt-oss-20b}"
model_name="${model_id##*/}"

DATA="${REPO}/data/${model_name}"
CAND="${DATA}/induction/candidates"
FINAL="${DATA}/induction-single"      # data_handler resolves {source}/{base}-*.jsonl
QC_JSON="${REPO}/pair_qc_induction.json"

KEY_TOKENS="${KEY_TOKENS:-2}"
VALUE_TOKENS="${VALUE_TOKENS:-2}"
N_PAIRS="${N_PAIRS:-8}"
N_CANDIDATE_ROWS="${N_CANDIDATE_ROWS:-160}"

STAGES="${STAGES:-build qc final}"
FORCE="${FORCE:-0}"

has_stage () { [[ " $STAGES " == *" $1 "* ]]; }
skip_if () {
  if [[ -e "$1" && "$FORCE" != "1" ]]; then
    echo "[skip] $2: $1 already exists (FORCE=1 to redo)"; return 0
  fi
  return 1
}

echo "repo   : $REPO"
echo "python : $(which python)"
echo "model  : $model_id"
echo "stages : $STAGES"
echo "out    : $FINAL"
echo

if has_stage build && ! skip_if "${CAND}/candidate_meta.json" build; then
  echo "########## build: candidate corpus ##########"
  python "make_induction_datasets.py" \
    --mode candidates \
    --out_dir "$CAND" \
    --tokenizer "$model_id" \
    --key_tokens "$KEY_TOKENS" \
    --value_tokens "$VALUE_TOKENS" \
    --n_pairs "$N_PAIRS" \
    --n_candidate_rows "$N_CANDIDATE_ROWS" \
    --seed "$SEED"
fi

if has_stage qc && ! skip_if "$QC_JSON" qc; then
  echo "########## qc: does the model actually match-and-copy? ##########"
  # Only the induction condition is checked -- noinduction is SUPPOSED to fail.
  # See qc_induction_datasets.py.
  PYTHONPATH="/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp/generate_data/conflict:${PYTHONPATH:-}" \
  python "qc_induction_datasets.py" \
    --data_dir "$CAND" \
    --model "$model_id" \
    --out "$QC_JSON"
fi

if has_stage final && ! skip_if "${FINAL}/noinduction-single-test.jsonl" final; then
  echo "########## final: emit the four -all files + test ##########"
  mkdir -p "$FINAL"
  python "make_induction_datasets.py" \
    --mode final \
    --out_dir "$FINAL" \
    --qc "$QC_JSON" \
    --meta "${CAND}/candidate_meta.json" \
    --tokenizer "$model_id" \
    --n_pairs "$N_PAIRS" \
    --seed "$SEED"
fi

echo
echo "done. next:"
echo "  sbatch --export=ALL,SOURCE=induction-single,BASE=noinduction-single scripts/localize_heads.sh"
echo "  B_SOURCE=induction-single B_BASE=noinduction-single B_LABEL=induction \\"
echo "    bash scripts/compare_privilege_position.sh"
