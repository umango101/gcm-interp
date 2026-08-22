#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 04:00:00
#SBATCH -J verify_layers_det
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

# Runs the same localization + one steering cell twice, into two separate roots,
# then diffs them. Run this BEFORE the full sweep: a nondeterministic backward
# produces a plausible-looking layer ranking, so the failure is invisible in the
# results themselves and only shows up as a diff between runs.
#
# Kept small on purpose (--limit_items 12): the point is to catch nondeterminism,
# and nondeterministic kernels diverge within a handful of items. Scale it up only
# if a small run passes and you suspect something that accumulates slowly.
#
# Both runs go through the same code path as the real sweep -- same flags, same
# defaults -- so a PASS here is evidence about the sweep, not about a special mode.

set +eu
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
set -eu

export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
cd "$RM_INTERP_REPO"

mkdir -p logs
source scripts/_preflight.sh
preflight_gpu

model_id="Qwen/Qwen1.5-14B-Chat"
model_name="Qwen1.5-14B-Chat"
device="cuda:0"
data="${RM_INTERP_REPO}/data/${model_name}"
source="female-long"
base="male-long"
LIMIT="${LIMIT:-12}"

ROOT_A="./results_layers_verifyA"
ROOT_B="./results_layers_verifyB"
rm -rf "$ROOT_A" "$ROOT_B"

run_once () {
  local root=$1
  echo "########## run into ${root} ##########"

  python -m layers.run_layers \
    --model_id "$model_id" --batch_size 1 --patch_algo atp \
    --source "$source" --base "$base" --device "$device" \
    --patch_model \
    --results_root "$root" \
    --limit_items "$LIMIT" \
    --sdp_backend math \
    --strict_determinism

  # This leg must mirror scripts/eval_layers_per_layer.sh flag for flag, apart
  # from the sweep being cut down. A harness that verifies a configuration nobody
  # runs proves nothing about the runs -- in particular the eval sweep uses
  # --sdp_backend default (flash forward), which is precisely the setting most
  # worth confirming, so pinning math here would test away the risk.
  python -m layers.run_layers \
    --model_id "$model_id" --batch_size 8 --patch_algo atp \
    --source "$source" --base "$base" --device "$device" \
    --eval_model --steering --ablation steer \
    --eval_test "${data}/female-single/male-single-test.jsonl" \
    --steering_add_path "${data}/female-long/female-long-steering.jsonl" \
    --steering_sub_path "${data}/female-long/male-long-steering.jsonl" \
    --results_root "$root" \
    --limit_items "$LIMIT" \
    --sweep_mode per_layer \
    --n_vals "5" --n_scale 0.1 \
    --steering_scale relative \
    --gen_mode prefill \
    --gen_batch_size 25 \
    --sdp_backend default \
    --strict_determinism
}

run_once "$ROOT_A"
run_once "$ROOT_B"

echo "########## diff ##########"
python -m layers.verify_determinism "$ROOT_A" "$ROOT_B"
