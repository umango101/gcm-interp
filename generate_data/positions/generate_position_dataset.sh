#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 04:00:00
#SBATCH -J gen_position_data
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

# Build the POSITION control corpus: first-named vs second-named, no privilege.
#
#   sbatch generate_data/positions/generate_position_dataset.sh
#   sbatch --export=ALL,STAGES=final generate_data/positions/generate_position_dataset.sh
#
# Three stages:
#   build  candidate corpus from the word pools           (CPU + tokenizer)
#   qc     does each preamble actually induce position-   (GPU, loads gpt-oss)
#          following? failing pairs are dropped
#   final  emit the four -all files + test into
#          data/gpt-oss-20b/first-single/
#
# The GPU is only needed for qc. Run STAGES="build final" on a login node if you
# already have a QC json.
#
# WHAT THE CORPUS IS. The privilege ATP contrast could be picking out heads that
# implement "answer the first-named option" rather than anything about
# instruction privilege. This corpus deletes the privilege axis entirely -- no
# rule in any system message, no user turn that requests an answer -- so the only
# thing separating the two conditions is which POSITION the demos answer in.
# Localizing on it (scripts/localize_heads.sh) gives a positional-deference head
# map to compare against the privilege one.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

# TWO paths, not one. They are different things, and conflating them is what put
# the last build's data/ tree inside generate_data/positions/:
#   REPO_ROOT   where data/, run.py and data_handler.py live. EVERYTHING the rest
#               of the pipeline reads is relative to this -- localize_heads.sh
#               looks for the corpus at $REPO_ROOT/data/{model}/first-single/.
#   SCRIPT_DIR  where this script and its two .py files live.
REPO_ROOT="${REPO_ROOT:-/home/ubansal/orcd/scratch/conflicts/gcm-interp}"
SCRIPT_DIR="${REPO_ROOT}/generate_data/positions"

if [[ ! -f "${REPO_ROOT}/data_handler.py" ]]; then
  echo "REPO_ROOT does not look like the repo root (no data_handler.py):" >&2
  echo "  $REPO_ROOT" >&2
  echo "Override with: sbatch --export=ALL,REPO_ROOT=/path/to/gcm-interp ..." >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="false"
export PYTHONHASHSEED="0"
SEED="${SEED:-42}"
# Demo count is the main lever on how strongly the preamble overrides the
# model's default. gpt-oss has a strong primacy bias: at 8 demos the
# "answer the second-named option" preamble only lands ~30% of the time,
# which collapses pair survival to ~0.3^2. Raise this before relaxing QC.
N_DEMOS="${N_DEMOS:-8}"
N_CANDIDATE_PAIRS="${N_CANDIDATE_PAIRS:-95}"
# Pairs needed = N_LOC + N_TEST, and the color pool caps out around 82, so
# these are the headroom lever. The ATP localization never reads the test
# file -- it only matters if you later run a steering sweep on this corpus --
# so shrink N_TEST before touching N_LOC.
N_LOC="${N_LOC:-50}"
N_TEST="${N_TEST:-25}"
model_id="${MODEL_ID:-openai/gpt-oss-20b}"
model_name="${model_id##*/}"

# All relative to REPO_ROOT, so data_handler and localize_heads.sh find them.
DATA="${REPO_ROOT}/data/${model_name}"
CAND="${DATA}/position/candidates"
FINAL="${DATA}/first-single"          # data_handler resolves {source}/{base}-*.jsonl
QC_JSON="${REPO_ROOT}/pair_qc_position.json"

STAGES="${STAGES:-build qc final}"
FORCE="${FORCE:-0}"

has_stage () { [[ " $STAGES " == *" $1 "* ]]; }
skip_if () {
  if [[ -e "$1" && "$FORCE" != "1" ]]; then
    echo "[skip] $2: $1 already exists (FORCE=1 to redo)"; return 0
  fi
  return 1
}

echo "repo root  : $REPO_ROOT"
echo "script dir : $SCRIPT_DIR"
echo "python     : $(which python)"
echo "model      : $model_id"
echo "stages     : $STAGES"
echo "out        : $FINAL"
echo

for f in make_position_datasets.py qc_position_datasets.py; do
  if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
    echo "Generator not found: ${SCRIPT_DIR}/${f}" >&2
    echo "Is REPO_ROOT the right checkout? It is currently $REPO_ROOT" >&2
    exit 1
  fi
done

if has_stage build && ! skip_if "${CAND}/candidate_meta.json" build; then
  echo "########## build: candidate corpus ##########"
  python "${SCRIPT_DIR}/make_position_datasets.py" \
    --mode candidates \
    --out_dir "$CAND" \
    --n_demos "$N_DEMOS" \
    --n_candidate_pairs "$N_CANDIDATE_PAIRS"  \
    --tokenizer "$model_id" \
    --seed "$SEED"
fi

if has_stage qc && ! skip_if "$QC_JSON" qc; then
  echo "########## qc: does the preamble induce position-following? ##########"
  echo "  QC_CHECKS=${QC_CHECKS:-both}"
  if [[ "${QC_CHECKS:-both}" != "both" ]] && \
     ! grep -q "QC_CHECKS" "${SCRIPT_DIR}/qc_position_datasets.py"; then
    echo "  ERROR: QC_CHECKS is set but ${SCRIPT_DIR}/qc_position_datasets.py" >&2
    echo "         does not support it -- that copy is stale. Update it, or the" >&2
    echo "         run silently gates on BOTH conditions again." >&2
    exit 1
  fi
  # qc_position_datasets.py imports qc_privilege_datasets rather than duplicating
  # its scoring, so the conflict package has to be importable. Built from
  # REPO_ROOT rather than typed as a literal -- the previous version had a
  # truncated absolute path, which is precisely what this avoids.
  PYTHONPATH="${REPO_ROOT}/generate_data/conflict:${PYTHONPATH:-}" \
  python "${SCRIPT_DIR}/qc_position_datasets.py" \
    --data_dir "$CAND" \
    --model "$model_id" \
    --out "$QC_JSON"
fi

if has_stage final && ! skip_if "${FINAL}/second-single-test.jsonl" final; then
  echo "########## final: emit the four -all files + test ##########"
  mkdir -p "$FINAL"
  python "${SCRIPT_DIR}/make_position_datasets.py" \
    --mode final \
    --out_dir "$FINAL" \
    --n_loc "$N_LOC" \
    --n_test "$N_TEST" \
    --n_demos "$N_DEMOS"  \
    --qc "$QC_JSON" \
    --meta "${CAND}/candidate_meta.json" \
    --tokenizer "$model_id" \
    --seed "$SEED"
fi

echo
echo "done. next:"
echo "  sbatch --export=ALL,SOURCE=first-single,BASE=second-single scripts/localize_heads.sh"
