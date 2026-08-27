#!/bin/bash
# Compare two ATP head maps -- by default privilege vs position.
#
#   bash scripts/compare_privilege_position.sh
#   A_SOURCE=user-single A_BASE=dev-single \
#   B_SOURCE=first-single B_BASE=second-single \
#     bash scripts/compare_privilege_position.sh
#
# CPU only and quick -- no model, no GPU, so this runs on a login node. It is a
# plain bash script rather than an sbatch job for that reason; there is nothing
# to queue for.
#
# Prerequisite: both maps exist. Build them with the SAME launcher so their flags
# match:
#   sbatch --export=ALL,SOURCE=user-single,BASE=dev-single      scripts/localize_heads.sh
#   sbatch --export=ALL,SOURCE=first-single,BASE=second-single  scripts/localize_heads.sh
#
# HOW TO READ THE OUTPUT
#   LOW overlap is the strong result: a dissociation, meaning the privilege heads
#   are not the position heads and the privilege localization is not explained by
#   positional deference.
#   HIGH overlap is ambiguous, not a refutation. Both corpora share an eight-demo
#   ICL preamble, a one-word answer and the same instruction frames, so heads
#   doing "attend to the final question and copy a demonstrated answer" rank high
#   in both regardless. Calibrate with a third contrast of the same surface form
#   and neither construct before concluding anything from a high number.

set -euo pipefail

# REPO_ROOT, matching localize_heads.sh and the two generation scripts. This
# script used to read RM_INTERP_REPO, which ~/.bashrc exports pointing at the
# OTHER checkout -- so it silently looked for head maps in the cleanup repo and
# reported both as missing. RM_INTERP_REPO is still honoured as a fallback, but
# the guard below is what stops a wrong value from looking like a missing map.
REPO="${REPO_ROOT:-${RM_INTERP_REPO:-/home/ubansal/orcd/scratch/conflicts/gcm-interp}}"
if [[ ! -f "${REPO}/data_handler.py" || ! -f "${REPO}/run.py" ]]; then
  echo "REPO_ROOT does not look like the repo root (need run.py and data_handler.py):" >&2
  echo "  $REPO" >&2
  echo "Override with: REPO_ROOT=/path/to/gcm-interp bash $0" >&2
  exit 1
fi
cd "$REPO"

model_id="${MODEL_ID:-openai/gpt-oss-20b}"
model_name="${model_id##*/}"
algo="${ALGO:-atp}"

A_SOURCE="${A_SOURCE:-user-single}";  A_BASE="${A_BASE:-dev-single}"
B_SOURCE="${B_SOURCE:-first-single}"; B_BASE="${B_BASE:-second-single}"
A_LABEL="${A_LABEL:-privilege}"
B_LABEL="${B_LABEL:-position}"

A_MAP="${REPO}/results/${model_name}/from_${A_SOURCE}_to_${A_BASE}/${algo}/numerator_1_heads.pt"
B_MAP="${REPO}/results/${model_name}/from_${B_SOURCE}_to_${B_BASE}/${algo}/numerator_1_heads.pt"
OUT="${OUT:-${REPO}/results_head_overlap/${A_LABEL}_vs_${B_LABEL}}"

echo "${A_LABEL}: ${A_MAP}"
echo "${B_LABEL}: ${B_MAP}"
echo "out      : ${OUT}"
echo

missing=0
for m in "$A_MAP" "$B_MAP"; do
  if [[ ! -f "$m" ]]; then echo "MISSING head map: $m"; missing=1; fi
done
if [[ $missing -eq 1 ]]; then
  echo
  echo "Build it with scripts/localize_heads.sh -- and with the SAME flags as the"
  echo "other map, or the overlap statistic partly measures that difference."
  echo
  echo "If the localization HAS run, check two things:"
  echo "  1. repo: this looked under $REPO"
  echo "     -- override with REPO_ROOT=/path/to/gcm-interp"
  for m in "$A_MAP" "$B_MAP"; do
    d="$(dirname "$m")"
    n=$(ls -1 "$d"/heads_*.pt 2>/dev/null | wc -l)
    if [[ "$n" -gt 0 ]]; then
      echo "  2. $d has $n shard(s) but no reduced map --"
      echo "     the map is written lazily; run the eval stage for that contrast."
    fi
  done
  exit 1
fi

mkdir -p "$OUT"

# Both scorings, because they answer different questions: abs asks whether the
# same heads are involved at all, signed asks whether the two contrasts push them
# in the same direction. Reporting only one invites the obvious objection.
for score in abs signed; do
  echo "==================== score=${score} ===================="
  python compare_head_maps.py \
    --a "$A_MAP" --label_a "$A_LABEL" \
    --b "$B_MAP" --label_b "$B_LABEL" \
    --score "$score" \
    --ks ${KS:-10 20 50 100} \
    --out "${OUT}/${score}" | tee "${OUT}/${score}.txt"
  echo
done

echo "results under ${OUT}/"
