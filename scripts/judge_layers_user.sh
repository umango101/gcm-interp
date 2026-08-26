#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 08:00:00
#SBATCH -J layers_judge
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 4

# Judge + accuracies + plots for the layer arm (user-single -> dev-single).
#
# merge and build_prompts are CPU-only and are assumed to have run already:
#     python eval_pipeline_conflict_layers.py --stages merge build_prompts
# This job starts at the stage that actually needs a GPU. Rerunning merge here
# would be harmless but wastes allocation time.
#
# Resumable. _judge_pass_done() skips a (cell, pass) whose output already has the
# expected row count, so a preemption + requeue continues rather than restarting.
# That is the whole reason --requeue is safe on this one.
#
#   sbatch scripts/judge_layers_user.sh

# NO strict mode yet: /etc/bashrc and the conda hook both test unbound variables
# before assigning them, so `set -u` here fails inside code we do not control.
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh

# The JUDGE env, not the gpt-oss env. The judge is Llama-3.1-70B through vLLM;
# eval_pipeline_conflict.sh runs it under `syc`. `conflict-syc` is pinned to
# transformers 4.57.1 / nnsight 0.4.11 for gpt-oss tracing and does not
# necessarily carry vLLM. Override with JUDGE_ENV= if that is not where your
# vLLM install lives.
conda activate "${JUDGE_ENV:-/home/ubansal/miniconda/envs/syc}"

# Safe from here: everything below is ours.
set -euo pipefail

PIPELINE_REPO="${PIPELINE_REPO:-/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp}"
RESULTS_LAYERS="${RESULTS_LAYERS:-${PIPELINE_REPO}/results_layers}"
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"

cd "$PIPELINE_REPO"
mkdir -p logs "$HF_HOME"

echo "python : $(which python)"
echo "repo   : $PIPELINE_REPO"
echo "results: $RESULTS_LAYERS"

# --- preflight -------------------------------------------------------------
# Everything below fails in seconds rather than after the judge has spent
# minutes loading 40GB of weights.

if [ ! -d "$RESULTS_LAYERS" ]; then
  echo "results_layers not found at: $RESULTS_LAYERS" >&2
  find /orcd/scratch/orcd/008/ubansal -maxdepth 4 -name results_layers -type d 2>/dev/null >&2
  exit 1
fi

CELL="${RESULTS_LAYERS}/gpt-oss-20b/from_user-single_to_dev-single/atp-per-layer/user-single_eval/user-single_steer"
PROMPTS="${PIPELINE_REPO}/eval_pipeline_conflict/gpt-oss-20b/from_user-single_to_dev-single/atp-per-layer/user-single_eval/user-single_steer/eval_prompts.csv"
if [ ! -f "$PROMPTS" ]; then
  echo "eval_prompts.csv not found at: $PROMPTS" >&2
  echo "Run the CPU stages first:" >&2
  echo "  python eval_pipeline_conflict_layers.py --stages merge build_prompts" >&2
  exit 1
fi

python - <<'PY' || exit 1
import sys
# `import vllm` alone is NOT enough: vllm's __init__ is lazy, so the
# transformers-compatibility failures (e.g. vllm 0.9.x registering "aimv2" when
# transformers >=4.54 already ships it) only fire when LLM is actually pulled in.
# Import what stage_judge_all imports, or the preflight passes and the real
# import dies minutes later.
try:
    import torch, transformers, vllm
    from vllm import LLM, SamplingParams          # noqa: F401
except ImportError as e:
    sys.exit(f"[preflight] {getattr(e, 'name', e)!r} missing from this env -- "
             f"set JUDGE_ENV to the env that has vLLM.")
except ValueError as e:
    sys.exit(f"[preflight] vLLM/transformers are incompatible in this env: {e}\n"
             f"           vllm {vllm.__version__} + transformers {transformers.__version__}.\n"
             f"           The judge is Llama-3.1-70B and does not need the gpt-oss\n"
             f"           transformers pin, so run it in the judge env instead:\n"
             f"             sbatch --export=ALL,JUDGE_ENV=<env> scripts/judge_layers_user.sh")
if torch.cuda.device_count() == 0:
    sys.exit("[preflight] no GPU visible; the judge stage needs one.")
print(f"[preflight] vllm {vllm.__version__} | transformers {transformers.__version__} | "
      f"torch {torch.__version__} | {torch.cuda.device_count()} GPU(s)")
PY

# Layer count comes from the attribution profile rather than a hardcoded 24, so
# a different model does not silently produce a wrong grid.
# plot_layer_effects writes into {loc_dir}/eval/, not {loc_dir}.
EFFECTS="${RESULTS_LAYERS}/gpt-oss-20b/from_user-single_to_dev-single/atp/eval/layer_effects.csv"
if [ -f "$EFFECTS" ]; then
  N_LAYERS=$(($(wc -l < "$EFFECTS") - 1))     # minus the header row
else
  N_LAYERS="${N_LAYERS:-24}"
  echo "layer_effects.csv missing; falling back to N_LAYERS=${N_LAYERS}" >&2
fi

# Must match --n_vals from the sweep, or _check_grid reports the gaps as missing.
N_VALS="${N_VALS:-2,5,8,10}"

echo "n_layers: $N_LAYERS"
echo "n_vals  : $N_VALS"
echo

python eval_pipeline_conflict_layers.py \
    --results_dir "$RESULTS_LAYERS" \
    --stages judge accuracies plots \
    --n_layers "$N_LAYERS" \
    --n_vals "$N_VALS" \
    --batch_size 32

echo
echo "done. accuracies and plots under:"
echo "  ${PIPELINE_REPO}/results_pipeline_conflict/gpt-oss-20b/from_user-single_to_dev-single/atp-per-layer/"
