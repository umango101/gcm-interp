#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH -J qc_datasets
#SBATCH -o /home/ubansal/orcd/scratch/conflicts/gcm-interp/generate_data/conflict/logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 4

# The -o path must be ABSOLUTE and its directory must already exist: Slurm opens
# that file before this script runs. Once, by hand:
#   mkdir -p /home/ubansal/orcd/scratch/conflicts/gcm-interp/generate_data/conflict/logs

source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate "${ENV:-/home/ubansal/miniconda/envs/conflict-syc}"
set -euo pipefail

# `set -e` only comes on after activate, so verify the interpreter explicitly.
python -c "import torch, transformers" || {
  echo "conda env did not activate correctly: ${CONDA_PREFIX:-unset}"; exit 1; }

cd /home/ubansal/orcd/scratch/conflicts/gcm-interp/generate_data/conflict

MODEL="${MODEL:-openai/gpt-oss-20b}"
FORM="${FORM:-rule}"
NCAND="${NCAND:-82}"          # the color pool supports ~82 length-matched pairs
MIN_PAIRS="${MIN_PAIRS:-30}"  # abort rather than build an arm this thin
ARMS=(devuser sysuser sysdev)

# Every stage is deterministic and overwrites in place, so a requeue on the
# preemptable partition simply redoes the work. Set SKIP_SYSDEV_QC=1 to reuse an
# existing sysdev pair_qc.json and save the ~10 minutes of GPU it costs.
SKIP_SYSDEV_QC="${SKIP_SYSDEV_QC:-0}"

echo "env:   ${CONDA_PREFIX:-?}"
echo "model: $MODEL | form: $FORM | candidates: $NCAND"

# ---------------------------------------------------------------------------
# 1. candidates
# ---------------------------------------------------------------------------
for ARM in "${ARMS[@]}"; do
  python make_hierarchy_datasets.py --mode candidates --arm "$ARM" --form "$FORM" \
    --n_candidate_pairs "$NCAND" \
    --out_dir "data/gpt-oss-20b/hier-$ARM/candidates" \
    --tokenizer "$MODEL"
done

# ---------------------------------------------------------------------------
# 2. QC -- the gate differs by arm, for a reason worth keeping straight
# ---------------------------------------------------------------------------
# devuser / sysuser: at the user boundary the preamble does not reverse the
# model's preference, only compresses it -- dev ahead by +2.87 under one
# preamble and +0.95 under the other, a swing of ~1.9 that never crosses zero.
# The `forced` gate therefore rejects almost everything (6/82, 2/82), and a
# swing threshold would mean selecting on the strength of the manipulation with
# the pair budget as the criterion. Gate on the privileged check alone and carry
# the whole swing distribution into the corpus, to be reported rather than
# filtered.
#
# sysdev: the manipulation really does flip behaviour under the subordinate
# preamble, so the standard `forced` gate applies there.
for ARM in devuser sysuser; do
  python qc_hierarchy_datasets.py --model "$MODEL" --gate privileged \
    --data_dir "data/gpt-oss-20b/hier-$ARM/candidates" \
    --out "data/gpt-oss-20b/hier-$ARM/pair_qc.json"
done

if [[ "$SKIP_SYSDEV_QC" == "1" && -f data/gpt-oss-20b/hier-sysdev/pair_qc.json ]]; then
  echo "reusing existing sysdev pair_qc.json (SKIP_SYSDEV_QC=1)"
else
  python qc_hierarchy_datasets.py --model "$MODEL" \
    --data_dir "data/gpt-oss-20b/hier-sysdev/candidates" \
    --out "data/gpt-oss-20b/hier-sysdev/pair_qc.json"
fi

# ---------------------------------------------------------------------------
# 3. report survival and choose per-arm splits
# ---------------------------------------------------------------------------
# Splits come from what actually survived rather than a hardcoded 25/25. One arm
# landing at 48 should not kill the build partway through the loop, leaving some
# arms written and others not.
SPLIT_FILE="$(mktemp)"
trap 'rm -f "$SPLIT_FILE"' EXIT
python - "$MIN_PAIRS" "${ARMS[@]}" > "$SPLIT_FILE" <<'PY'
import json, sys
min_pairs, arms = int(sys.argv[1]), sys.argv[2:]
report, short = [], []
for arm in arms:
    q = json.load(open(f"data/gpt-oss-20b/hier-{arm}/pair_qc.json"))
    n = len(q["passing_pairs"])
    # Even split capped at 25 each: more localization pairs buy little past
    # that, and the held-out test set is what the confidence intervals rest on.
    n_loc = min(25, n // 2)
    n_test = min(25, n - n_loc)
    c = q["per_check"].get("contrast")
    swing = (f"  swing {c['mean_swing']:+.2f}+-{c['sd_swing']:.2f}, "
             f"{c['frac_swing_negative']:.0%} negative" if c else "")
    print(f"{arm} {n_loc} {n_test}")                       # machine-readable
    report.append(f"  {arm:<9} {n:>3} pairs -> loc {n_loc}, test {n_test}{swing}")
    if n_loc + n_test < min_pairs:
        short.append((arm, n))
print("\n".join(report), file=sys.stderr)
for a, n in short:
    print(f"  SHORTFALL {a}: only {n} pairs", file=sys.stderr)
if short:
    print("\nToo few pairs for a usable corpus. Raise NCAND if the color pool "
          "allows it, or\nread the QC line-pass rates above. Do not relax the "
          "gate to reach a count --\nthe gate is what makes the privileged "
          "condition trustworthy.", file=sys.stderr)
    sys.exit(1)
PY

mapfile -t SPLITS < <(grep -E '^[a-z_]+ [0-9]+ [0-9]+$' "$SPLIT_FILE")
if [[ ${#SPLITS[@]} -ne ${#ARMS[@]} ]]; then
  echo "expected ${#ARMS[@]} split lines, got ${#SPLITS[@]}"; exit 1
fi

# ---------------------------------------------------------------------------
# 4. build
# ---------------------------------------------------------------------------
for LINE in "${SPLITS[@]}"; do
  read -r ARM N_LOC N_TEST <<< "$LINE"
  D="data/gpt-oss-20b/hier-$ARM"
  echo "building $ARM with n_loc=$N_LOC n_test=$N_TEST"
  python make_hierarchy_datasets.py --mode final --arm "$ARM" --form "$FORM" \
    --n_loc "$N_LOC" --n_test "$N_TEST" \
    --out_dir "$D" --meta "$D/candidates/candidate_meta.json" \
    --qc "$D/pair_qc.json" --tokenizer "$MODEL"
done

# ---------------------------------------------------------------------------
# 5. naive transfer sets
# ---------------------------------------------------------------------------
# --overwrite is required for requeue: this script is not checkpointed, so a
# restart finds the file already there. Every other stage overwrites in place.
for ARM in "${ARMS[@]}"; do
  D="data/gpt-oss-20b/hier-$ARM"
  python make_devnaive_test.py --overwrite \
    --in "$D/dev-single-test.jsonl" \
    --meta "$D/candidates/candidate_meta.json" \
    --out "$D/devNaive-single-test.jsonl"
done

# ---------------------------------------------------------------------------
# 6. line counts, so the log records what was actually written
# ---------------------------------------------------------------------------
echo
for ARM in "${ARMS[@]}"; do
  D="data/gpt-oss-20b/hier-$ARM"
  printf "%-9s" "$ARM"
  for F in dev-single-desired-all dev-single-undesired-all \
           user-single-desired-all user-single-undesired-all \
           dev-single-test devNaive-single-test; do
    printf "  %s=%s" "$F" "$(wc -l < "$D/$F.jsonl")"
  done
  echo
done

echo "done"
