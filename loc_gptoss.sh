#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 04:00:00
#SBATCH -J loc_gptoss
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

source ~/.bashrc
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
cd "$RM_INTERP_REPO"

# Keep the rendered prompt inside gpt-oss's 128-token sliding window.
# 'minimal' trims the pinned Harmony system block; 'default' is the standard one.
export HARMONY_SYSTEM=minimal
# eval/activations.py reads this (it has no ModelHandler in scope).
export HEAD_SITE=o_proj_input

# gpt-oss-20b is MXFP4 on disk and dequantizes to ~40GB bf16.
# bitsandbytes nf4 on top of MXFP4 will error, so the loader skips it.

# --- build the corpus from real model output ---
# --validate does two things: drops word pairs the model mishandles (agreement
# prompts must yield the agreed word, conflict prompts must defer to the
# developer, and the two responses must match in token length), then writes the
# model's ACTUAL responses as the assistant completions -- cross-pasted so that
# source-desired == base-undesired, the same construction as the harmful/
# harmless sets. Without --validate the completions are placeholders.
# Re-run whenever the model, template, or word list changes.
# Skip if the corpus is already built: --validate takes ~10 min and re-running
# it can select a different pair set, which would make results across jobs
# incomparable. Delete data/validated_pairs.json to force a rebuild.
if [ ! -f data/validated_pairs.json ]; then
    GEN=$(find generate_data -name gen_conflict_polarity.py | head -1)
    if [ -z "$GEN" ]; then echo "gen_conflict_polarity.py not found"; exit 1; fi
    python "$GEN" --out data/gpt-oss-20b --validate --device cuda:0 || exit 1
else
    echo "using existing corpus (data/validated_pairs.json present)"
fi

# --- pre-flight: fails loudly rather than producing quiet garbage ---
VERIFY=$(find . -name verify_gptoss.py | head -1)
python "$VERIFY" --stage tokenizer --data_dir data/gpt-oss-20b || exit 1
python "$VERIFY" --stage model --device cuda:0 || exit 1

# source_base pairs: source is the condition of interest, base is the control
declare -a pairs=(
  "roleConflict-single_roleAgree-single"      # MAIN: cross-role conflict vs agreement
  "withinConflict-single_withinAgree-single"  # control: same conflict, one role
)

model_id="openai/gpt-oss-20b"
device="cuda:0"

for pair in "${pairs[@]}"; do
    IFS='_' read -r source base <<< "$pair"
    python run.py --model_id "$model_id" \
        --batch_size 1 \
        --patch_algo atp \
        --source "$source" \
        --base "$base" \
        --device "$device" \
        --head_site o_proj_input \
        --patch_model
done
