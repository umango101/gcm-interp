#!/bin/bash
# Compare the privilege and position ATP head maps, per arm.
#
#   bash scripts/compare_privilege_position_hier.sh
#   ARMS="devuser" bash scripts/compare_privilege_position_hier.sh
#
# CPU only and quick -- no model, no GPU, so this runs on a login node.
#
# WHY THIS REPLACES compare_privilege_position.sh RATHER THAN EXTENDING IT
# ------------------------------------------------------------------------
# The old script builds paths as results/{model}/from_{SOURCE}_to_{BASE}/, which
# has no arm component, and its defaults point at from_first-single_to_second-
# single -- the REQUEST-FORM position corpus. That corpus was a separate set of
# prompts, so its overlap with the privilege map was confounded with the two
# corpora simply having different shapes. Worse, the request form carried a copy
# cue at the answer position, which is the thing the rule form was built to
# remove. Numbers from it should not be reported alongside rule-form results.
#
# The pos-<arm> corpora are different in kind: byte-identical prompts to the
# privilege corpus of the same arm, same pairs in the same order, differing only
# in which policy the eight demos demonstrate (privileged-rule vs first-named).
# Overlap between those two maps is therefore about the two POLICIES and not
# about corpus shape -- which is the whole reason to compute it.
#
# HOW TO READ THE OUTPUT
# ----------------------
# LOW overlap is the strong result: a dissociation, meaning the privilege heads
# are not the position heads and the privilege localization is not explained by
# positional deference.
#
# HIGH overlap is still not a refutation, but for a narrower reason than before.
# Surface form is now controlled by construction, so the shared-surface-form
# objection is gone. What remains is DEPTH: both maps concentrate in mid-to-late
# layers, and any two sets drawn from those layers overlap more than chance.
# Run calibrate_overlap.py on the result to compare against a layer-matched null
# before concluding anything -- the raw number cannot separate "the same heads"
# from "heads at the same depths".
#
# One asymmetry to report either way: the position preambles differ at nine
# assistant messages against the privilege corpus's seven, because the two
# agreement demos are not agreement demos under a position policy. The position
# contrast therefore perturbs slightly more of the preamble.

set -euo pipefail

REPO="${RM_INTERP_REPO:-/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp}"
cd "$REPO"

model_id="${MODEL_ID:-openai/gpt-oss-20b}"
model_name="${model_id##*/}"
algo="${ALGO:-atp}"
pair="${PAIR:-from_user-single_to_dev-single}"
ARMS="${ARMS:-devuser sysuser sysdev}"

A_LABEL="${A_LABEL:-privilege}"
B_LABEL="${B_LABEL:-position}"

fail=0
for arm in $ARMS; do
  A_MAP="${REPO}/results/${model_name}/${arm}__${pair}/${algo}/numerator_1_heads.pt"
  B_MAP="${REPO}/results/${model_name}/pos-${arm}__${pair}/${algo}/numerator_1_heads.pt"
  for m in "$A_MAP" "$B_MAP"; do
    [[ -f "$m" ]] || { echo "MISSING head map: $m"; fail=1; }
  done
done
if [[ $fail -eq 1 ]]; then
  echo
  echo "Build the missing maps with:"
  echo "  python run.py --model_id ${model_id} --patch_algo ${algo} \\"
  echo "    --data_dir <arm or pos-arm> --source user-single --base dev-single \\"
  echo "    --device cuda:0 --patch_model --batch_size 1"
  exit 1
fi

for arm in $ARMS; do
  A_MAP="${REPO}/results/${model_name}/${arm}__${pair}/${algo}/numerator_1_heads.pt"
  B_MAP="${REPO}/results/${model_name}/pos-${arm}__${pair}/${algo}/numerator_1_heads.pt"
  OUT="${OUT_ROOT:-${REPO}/results_head_overlap}/${arm}/${A_LABEL}_vs_${B_LABEL}"

  echo "############################ arm=${arm} ############################"
  echo "${A_LABEL}: ${A_MAP}"
  echo "${B_LABEL}: ${B_MAP}"
  echo "out      : ${OUT}"
  echo

  # The two maps must cover the same items, or the comparison is partly about
  # which items each was computed over. --from_corpus guarantees this by
  # construction; the check is here because a silently mismatched pair would
  # still produce a plausible-looking overlap number.
  python - "$A_MAP" "$B_MAP" "$arm" <<'PY'
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
if a.shape != b.shape:
    raise SystemExit(
        f"{sys.argv[3]}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}. "
        "The privilege and position maps must be computed over the same items; "
        "rebuild the position corpus with --from_corpus.")
print(f"  shapes match: {tuple(a.shape)}")
PY

  mkdir -p "$OUT"

  # Both scorings, because they answer different questions: abs asks whether the
  # same heads are involved at all, signed asks whether the two contrasts push
  # them in the same direction. Reporting only one invites the obvious objection.
  for score in abs signed; do
    echo "==================== ${arm} score=${score} ===================="
    python compare_head_maps.py \
      --a "$A_MAP" --label_a "$A_LABEL" \
      --b "$B_MAP" --label_b "$B_LABEL" \
      --score "$score" \
      --ks ${KS:-10 20 50 100} \
      --out "${OUT}/${score}" | tee "${OUT}/${score}.txt"
    echo
  done
done

echo "results under ${OUT_ROOT:-${REPO}/results_head_overlap}/<arm>/${A_LABEL}_vs_${B_LABEL}/"
echo
echo "NEXT: the raw overlap is not readable on its own. Run"
echo "  python calibrate_overlap.py ..."
echo "to compare it against a layer-matched null, and get the split-half"
echo "ceiling from the per-item shards so a middling number can be told apart"
echo "from 'ATP maps do not reproduce at this sample size'."
