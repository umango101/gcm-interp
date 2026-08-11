#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 24:00:00
#SBATCH -J gptoss_conflict_experiment
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Full developer-vs-user instruction-conflict experiment on gpt-oss-20b.
#
# Mirrors the OLMo bias script: localize with ATP, then evaluate the located
# heads by desired-all, crossed over datasets so that localize-here / steer-there
# transfer is measured rather than assumed.
#
# Where this necessarily differs from the OLMo script:
#   * --head_site o_proj_input. gpt-oss has hidden_size 2880 but 64 heads of
#     head_dim 64 (= 4096), so per-head slices only exist on the INPUT side of
#     o_proj. Slicing o_proj.output would carve the residual stream into 45-wide
#     chunks that are not heads. HEAD_SITE is exported for eval/activations.py,
#     which has no ModelHandler in scope.
#   * No --full_precision. For gpt-oss that selects device_map='auto'; the
#     dequantized bf16 weights (~40GB) fit one H200, and pinning the device
#     keeps activations off CPU during ATP's backward pass.
#   * HARMONY_SYSTEM=minimal. gpt-oss alternates 128-token sliding-window
#     attention layers with full-attention layers, so tokens spent on system
#     boilerplate are tokens the sliding-window layers cannot see past.
#   * No `pip install` in the loop. The env is pinned in requirements.txt
#     (transformers 4.57.1 / nnsight 0.4.11); reinstalling mid-job is how it
#     drifted to transformers 5.x last time. Verified once below instead.

source ~/.bashrc
export RM_INTERP_REPO="/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
cd "$RM_INTERP_REPO" || exit 1

# Model weights are ~13GB on disk and must not land in the home quota.
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME"

export HARMONY_SYSTEM=minimal
export HEAD_SITE=o_proj_input

model_id="openai/gpt-oss-20b"
model_name="gpt-oss-20b"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"

# ---------------------------------------------------------------------------
# 0. environment guard
# ---------------------------------------------------------------------------
python - <<'EOF' || exit 1
import sys
import transformers, nnsight, huggingface_hub
want = {'transformers': '4.57.1', 'nnsight': '0.4.11'}
got = {'transformers': transformers.__version__, 'nnsight': nnsight.__version__}
print('versions:', got, 'hub', huggingface_hub.__version__)
bad = {k: (v, want[k]) for k, v in got.items() if v != want[k]}
if bad:
    print('VERSION MISMATCH (got, want):', bad)
    print('nnsight 0.5+ renames the tracing API and transformers 5.x breaks '
          'pyreft, which run.py imports transitively. Run: pip install -r requirements.txt')
    sys.exit(1)
from transformers import Mxfp4Config, GptOssForCausalLM
print('gpt-oss support: OK')
EOF

# ---------------------------------------------------------------------------
# 1. corpus (skipped if already built)
# ---------------------------------------------------------------------------
# --validate filters word pairs against the model and writes its real responses
# as the completions. Rebuilding selects a DIFFERENT pair set, which would make
# results incomparable across jobs -- delete validated_pairs.json to force it.
#if [ ! -f data/validated_pairs.json ]; then
#    GEN=$(find generate_data -name gen_conflict_polarity.py | head -1)
#    [ -z "$GEN" ] && { echo "gen_conflict_polarity.py not found"; exit 1; }
#    python "$GEN" --out "data/${model_name}" --validate --device "$device" || exit 1
#else
#    echo "using existing corpus (data/validated_pairs.json present)"
#fi

# ---------------------------------------------------------------------------
# 2. pre-flight
# ---------------------------------------------------------------------------
VERIFY=$(find . -name verify_gptoss.py | head -1)
python "$VERIFY" --stage tokenizer --data_dir "data/${model_name}" || exit 1
python "$VERIFY" --stage model --device "$device" || exit 1

# ---------------------------------------------------------------------------
# 3. localization + desired-all eval
# ---------------------------------------------------------------------------
# BOTH datasets are localized. roleConflict alone cannot separate the effect of
# interest from shallow cross-span token matching: under agreement the developer
# names the same word the user demands, under conflict a different one, so a head
# that just matches repeated tokens scores high. withinConflict has the same
# repetition structure with both constraints inside one developer turn, so the
# top-k list should be ranked on roleConflict MINUS withinConflict.
declare -a pairs=(
  "roleConflict-single_roleAgree-single"
  "withinConflict-single_withinAgree-single"
)
algos=("atp")

for pair in "${pairs[@]}"; do
  IFS='_' read -r source base <<< "$pair"

  for algo in "${algos[@]}"; do

    # --- localize -----------------------------------------------------------
    python run.py --model_id "$model_id" \
                  --batch_size 1 \
                  --patch_algo "$algo" \
                  --source "$source" \
                  --base "$base" \
                  --device "$device" \
                  --head_site o_proj_input \
                  --patch_model || exit 1

    # --- steer: within-dataset ---------------------------------------------
    # Steering vector = mean(conflict prompts) - mean(agreement prompts), applied
    # to held-out agreement prompts. The desired-all and test halves are disjoint by
    # construction (split on developer phrasing), so this measures a transferable
    # direction rather than memorised items.
    python run.py --model_id "$model_id" \
                  --batch_size 1 \
                  --patch_algo "$algo" \
                  --source "$source" \
                  --base "$base" \
                  --device "$device" \
                  --head_site o_proj_input \
                  --eval_model \
                  --kv_caching \
                  --eval_test "${data}/${source}/${base}-test.jsonl" \
                  --steering \
                  --ablation steer \
                  --steering_add_path "${data}/${source}/${source}-desired-all.jsonl" \
                  --steering_sub_path "${data}/${source}/${base}-desired-all.jsonl" || exit 1

    # --- steer: cross-dataset ----------------------------------------------
    # Vectors from this dataset applied to the OTHER dataset's test prompts.
    # A direction that only works where it was derived is dataset-specific;
    # one that crosses is closer to the thing being claimed.
    if [ "$source" = "roleConflict-single" ]; then
      other_src="withinConflict-single"; other_base="withinAgree-single"
    else
      other_src="roleConflict-single"; other_base="roleAgree-single"
    fi

    python run.py --model_id "$model_id" \
                  --batch_size 1 \
                  --patch_algo "$algo" \
                  --source "$source" \
                  --base "$base" \
                  --device "$device" \
                  --head_site o_proj_input \
                  --eval_model \
                  --kv_caching \
                  --eval_test "${data}/${other_src}/${other_base}-test.jsonl" \
                  --steering \
                  --ablation steer \
                  --steering_add_path "${data}/${source}/${source}-desired-all.jsonl" \
                  --steering_sub_path "${data}/${source}/${base}-desired-all.jsonl" || exit 1

    # --- ablate: mean -------------------------------------------------------
    # Replacing the located heads rather than adding to them. Steering can
    # succeed through a direction the heads merely correlate with; mean ablation
    # tests whether the heads are load-bearing.
    python run.py --model_id "$model_id" \
                  --batch_size 1 \
                  --patch_algo "$algo" \
                  --source "$source" \
                  --base "$base" \
                  --device "$device" \
                  --head_site o_proj_input \
                  --eval_model \
                  --kv_caching \
                  --eval_test "${data}/${source}/${base}-test.jsonl" \
                  --steering \
                  --ablation mean \
                  --steering_add_path "${data}/${source}/${source}-desired-all.jsonl" \
                  --steering_sub_path "${data}/${source}/${base}-desired-all.jsonl" || exit 1
  done
done

echo "done. results under results/${model_name}/"
