#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 04:00:00
#SBATCH -J verify_gen_mode
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8

# Verify that --gen_mode all_steps produces the same generations as the old
# --gen_mode recompute path, and time both.
#
# all_steps keeps the KV cache and applies the intervention on every forward, so
# each decode step computes only the new position; recompute disables the cache
# and re-forwards the whole sequence each step. Every position is steered exactly
# once either way, so the outputs should match -- but "should" is the reason to
# check rather than a reason not to. If they diverge, all_steps is not a free
# speedup and the default should go back.
#
# Small on purpose: divergence, if it exists, shows up immediately. Uses the
# long-form eval, where max_new_tokens=256 makes the cost gap largest.

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
LIMIT="${LIMIT:-8}"

run_mode () {
  local mode=$1 root=$2
  echo "########## gen_mode=${mode} ##########"
  local t0=$SECONDS
  python -m layers.run_layers \
    --model_id "$model_id" --batch_size 8 --patch_algo atp \
    --source "$source" --base "$base" --device "$device" \
    --eval_model --steering --ablation steer \
    --eval_test "${data}/female-long/male-long-test.jsonl" \
    --steering_add_path "${data}/female-long/female-long-steering.jsonl" \
    --steering_sub_path "${data}/female-long/male-long-steering.jsonl" \
    --results_root "$root" \
    --limit_items "$LIMIT" \
    --sweep_mode per_layer \
    --topk_layers "1" \
    --n_vals "5" --n_scale 0.1 \
    --steering_scale relative \
    --gen_mode "$mode" \
    --gen_batch_size 4 \
    --sdp_backend default
  echo "[timing] gen_mode=${mode} took $((SECONDS - t0))s"
}

ROOT_FAST="./results_genmode_all_steps"
ROOT_REF="./results_genmode_recompute"
rm -rf "$ROOT_FAST" "$ROOT_REF"

run_mode all_steps "$ROOT_FAST"
run_mode recompute "$ROOT_REF"

echo "########## diff ##########"
echo "Note: determinism.json and steering_meta.json differ by design here --"
echo "the two runs used different gen_mode. Only the generations must match."
python -m layers.verify_determinism "$ROOT_FAST" "$ROOT_REF" --quiet
