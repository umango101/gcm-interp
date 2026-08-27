#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 08:00:00
#SBATCH -J localize_heads
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# ATP attention-head localization for ONE contrast. Reads its corpus from
#   data/{model}/{EXPERIMENT}/
# where data_handler resolves {EXPERIMENT}/{SOURCE}-*.jsonl and
# {EXPERIMENT}/{BASE}-*.jsonl. EXPERIMENT defaults to SOURCE, which is how every
# corpus in this repo is laid out.
#
#   sbatch --export=ALL,SOURCE=user-single,BASE=dev-single              scripts/localize_heads.sh
#   sbatch --export=ALL,SOURCE=first-single,BASE=second-single          scripts/localize_heads.sh
#   sbatch --export=ALL,SOURCE=induction-single,BASE=noinduction-single scripts/localize_heads.sh
#
# Writes results/{model}/from_{SOURCE}_to_{BASE}/atp/numerator_1_heads.pt.
#
# WHY ALL THREE CONTRASTS GO THROUGH THIS ONE SCRIPT. Two head maps are
# comparable head-for-head only if they were produced identically -- same batch
# size, same precision, same algo, same seed. Sharing the launcher guarantees
# that; separate launchers are how a batch-size difference quietly becomes part
# of the overlap statistic that compare_head_maps.py reports. If you change a
# flag here, EVERY map has to be rebuilt, not just the next one.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

# The repo root: where data/, run.py and data_handler.py live. Everything below
# is relative to it. Guarded, because pointing this at a subdirectory silently
# sends results/ and data/ somewhere the rest of the pipeline will not look.
REPO_ROOT="${REPO_ROOT:-/home/ubansal/orcd/scratch/conflicts/gcm-interp}"
if [[ ! -f "${REPO_ROOT}/data_handler.py" || ! -f "${REPO_ROOT}/run.py" ]]; then
  echo "REPO_ROOT does not look like the repo root (need run.py and data_handler.py):" >&2
  echo "  $REPO_ROOT" >&2
  echo "Override with: sbatch --export=ALL,REPO_ROOT=/path/to/gcm-interp,SOURCE=...,BASE=... ..." >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME"

export HARMONY_SYSTEM=minimal

# Read at process start; eval/setup.assert_determinism_env() checks them.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export STRICT_DETERMINISM="1"
export SEED="${SEED:-42}"

model_id="${MODEL_ID:-openai/gpt-oss-20b}"
model_name="${model_id##*/}"
device="${DEVICE:-cuda:0}"
batch_size="${BATCH_SIZE:-1}"     # ATP is forced to 1; changing it changes the map
algo="${ALGO:-atp}"

SOURCE="${SOURCE:?set SOURCE, e.g. --export=ALL,SOURCE=first-single,BASE=second-single}"
BASE="${BASE:?set BASE, e.g. --export=ALL,SOURCE=first-single,BASE=second-single}"
EXPERIMENT="${EXPERIMENT:-$SOURCE}"

DATA_DIR="${REPO_ROOT}/data/${model_name}/${EXPERIMENT}"
OUT_DIR="${REPO_ROOT}/results/${model_name}/from_${SOURCE}_to_${BASE}"
MAP="${OUT_DIR}/${algo}/numerator_1_heads.pt"

echo "repo root  : $REPO_ROOT"
echo "python     : $(which python)"
echo "experiment : $EXPERIMENT"
echo "contrast   : ${SOURCE} -> ${BASE}"
echo "data dir   : $DATA_DIR"
echo "map        : $MAP"
echo

# The four files the ATP contrast reads. Checked up front so a typo in SOURCE or
# BASE fails in seconds instead of after 40GB of weights have loaded.
missing=0
for f in "${DATA_DIR}/${SOURCE}-desired-all.jsonl" \
         "${DATA_DIR}/${SOURCE}-undesired-all.jsonl" \
         "${DATA_DIR}/${BASE}-desired-all.jsonl" \
         "${DATA_DIR}/${BASE}-undesired-all.jsonl"; do
  if [[ ! -f "$f" ]]; then echo "MISSING $f"; missing=1; fi
done
if [[ $missing -eq 1 ]]; then
  echo
  echo "Contents of ${DATA_DIR}:"
  ls -1 "$DATA_DIR" 2>/dev/null || echo "  (no such directory)"
  echo
  echo "If the corpus was built under generate_data/*/data/, it landed in the wrong"
  echo "place -- move it to ${REPO_ROOT}/data/ and resubmit."
  exit 1
fi

if [[ -f "$MAP" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] map already exists (FORCE=1 to redo): $MAP"
  exit 0
fi

# run.py resumes by skipping attribution shards that already exist, so a partial
# run against a PREVIOUS version of the corpus would be silently reused and
# blended into the new map. FORCE clears the tree rather than trusting the caller
# to have remembered.
if [[ "${FORCE:-0}" == "1" && -d "$OUT_DIR" ]]; then
  echo "[force] removing $OUT_DIR"
  rm -rf "$OUT_DIR"
fi

echo "########## localize: ${SOURCE} -> ${BASE} ##########"
python run.py \
  --model_id "$model_id" \
  --batch_size "$batch_size" \
  --seed "$SEED" \
  --patch_algo "$algo" \
  --source "$SOURCE" \
  --base "$BASE" \
  --device "$device" \
  --patch_model

echo
echo "done: $MAP"
echo
echo "compare against the privilege map with:"
echo "  B_SOURCE=${SOURCE} B_BASE=${BASE} B_LABEL=<label> \\"
echo "    bash scripts/compare_privilege_position.sh"
