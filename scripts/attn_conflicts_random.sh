#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 24:00:00
#SBATCH -J random_conflicts
#SBATCH -o /home/ubansal/orcd/scratch/conflicts/gcm-interp/logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -c 4

source ~/.bashrc
# Was /home/ubansal/orcd/scratch/gcm-interp -- a DIFFERENT checkout from the one
# cd'd into below. Anything reading this variable was pointed at the wrong repo.
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/conflicts/gcm-interp"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
export ATP_MEM=1
# Fragmentation: the activation cache allocates and frees large blocks per
# batch, and the allocator ends up unable to satisfy a 1.3GB request with
# hundreds of MB nominally free.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Activation-caching batch size. The cache holds batch x seq x hidden for every
# layer at once, so prompt length costs the same as batch size: the rule-form
# corpora are about twice as long as the request-form ones the default of 9 was
# tuned for, which OOMs a 140GB card. Lower it further if a cell still fails.
export CACHE_BS="${CACHE_BS:-3}"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/conflict-syc
cd "$RM_INTERP_REPO"
mkdir -p logs

declare -a models=("openai/gpt-oss-20b")
declare -a pairs=("user-single_dev-single")
algos=("random")

# Several draws, because one random head set is a point with no spread and the
# claim is that the real set beats the random DISTRIBUTION. The seed is in every
# output filename, so the draws do not overwrite one another.
IFS=',' read -r -a seeds <<< "${SEEDS:-1,2,3,4,5}"

# The repo already implements this: eval_runner reads RANDOM_BASELINE
# (layer_matched|uniform) and RANDOM_BASELINE_SEED, and layer-matched mode
# recovers the targeted set from the sibling atp/ map, erroring out if it is
# missing. Running BOTH modes is the strongest form -- the gap between them is
# the contribution of depth alone, separate from which heads within a layer.
RANDOM_BASELINE="${RANDOM_BASELINE:-layer_matched}"

# No commas: `("a", "b")` makes the elements `a,` and `b`, so every path built
# from the first two gains a stray comma and check_data reports the whole sweep
# as missing data.
arms=("devuser" "sysuser" "sysdev")

# Both test files per arm. Evaluating only devNaive cannot distinguish "the
# heads do not transfer to the naive prior" from "the steering vector does
# nothing at all" -- the in-distribution cell is the positive control.
test_files=("dev-single-test" "devNaive-single-test")

device="cuda:0"
batch_size=16
FULL_PRECISION="${FULL_PRECISION:-0}"

# Same-arm only by default (3 cells per test file). CROSS_ARM=1 runs the full
# 3x3 transfer grid.
#
# Cross-arm is safe here without any change to the steering code: run_eval goes
# through generate_with_patches, which reduces the cached tensor to a single
# vector (steering_type='last_token' takes patch_activations[layer][-1]) and
# normalises it, so differing prompt lengths across arms never meet. The
# per-position write in get_attn_tensors is the one that would break, and
# run_eval does not call it.
CROSS_ARM="${CROSS_ARM:-0}"

DATA_ROOT="$RM_INTERP_REPO/data/gpt-oss-20b"
# Separate state dir so this can run alongside the atp sweep without either
# job's sentinels being read by the other.
STATE_DIR="$RM_INTERP_REPO/.run_state_random"
mkdir -p "$STATE_DIR"

# --------------------------------------------------------------------------- #
STOPPED=0
on_signal() {
    STOPPED=1
    echo ""
    echo "[signal] caught $1 -- stopping after the current step."
    echo "[signal] finished steps are recorded in $STATE_DIR; resubmit to resume."
}
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal INT'  INT

requeue_if_slurm() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "[exit] requeueing job $SLURM_JOB_ID to continue the sweep"
        scontrol requeue "$SLURM_JOB_ID" || echo "[exit] requeue failed; resubmit by hand"
    fi
}

declare -a selected=()
if [[ -n "${ONLY_MODEL:-}" ]]; then
    for m in "${models[@]}"; do
        [[ "${m##*/}" == "$ONLY_MODEL" || "$m" == "$ONLY_MODEL" ]] && selected+=("$m")
    done
    [[ ${#selected[@]} -eq 0 ]] && { echo "FATAL: ONLY_MODEL='$ONLY_MODEL' matches nothing."; exit 1; }
else
    selected=("${models[@]}")
fi

echo "[plan] models: ${selected[*]}"
echo "[plan] arms:   ${arms[*]}"
echo "[plan] tests:  ${test_files[*]}"
echo "[plan] seeds: ${seeds[*]}  RANDOM_BASELINE=$RANDOM_BASELINE"
echo "[plan] cross_arm=$CROSS_ARM full_precision=$FULL_PRECISION"
echo "[plan] data:   $DATA_ROOT"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if ! python -c "import nnsight, sys; sys.exit(0 if nnsight.__version__ == '0.4.11' else 1)" 2>/dev/null; then
        echo "[setup] installing nnsight==0.4.11"
        pip install nnsight==0.4.11 || { echo "FATAL: nnsight install failed"; exit 1; }
    fi
fi

declare -a FAILURES=()
declare -a SKIPPED=()
N_DONE=0
N_RAN=0

run_step() {
    local key="$1"; shift
    local desc="$1"; shift
    local sentinel="$STATE_DIR/${key}.done"
    local argv="$*"

    if [[ -f "$sentinel" && "${FORCE:-0}" != "1" ]]; then
        local stored
        stored="$(sed -n '2p' "$sentinel")"
        if [[ "$stored" == "$argv" ]]; then
            echo "[skip] $desc"; N_DONE=$(( N_DONE + 1 )); return 0
        fi
        echo "[stale] $desc -- recorded argv differs; re-running"
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry ] $desc"; N_DONE=$(( N_DONE + 1 )); return 0
    fi

    echo "[run ] $desc"
    # Passed by environment, not argv: this is the repo's own interface for the
    # random baseline. It is deliberately NOT in the sentinel argv either, so
    # changing the mode does not invalidate finished cells of the other mode --
    # they write to different filenames and are separate results.
    export RANDOM_BASELINE RANDOM_BASELINE_SEED
    # Backgrounded with `wait`: bash defers traps until a foreground command
    # returns, so with `python run.py` in the foreground the USR1@300 preemption
    # warning was absorbed entirely and never stopped the sweep in time.
    python run.py "$@" &
    local pid=$!
    wait "$pid"
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        { date -Is; echo "$argv"; } > "$sentinel"
        echo "[ok  ] $desc"; N_DONE=$(( N_DONE + 1 )); N_RAN=$(( N_RAN + 1 )); return 0
    fi
    if [[ $rc -ge 128 || $STOPPED -eq 1 ]]; then
        STOPPED=1
        echo "[stop] $desc interrupted (exit $rc); not marked done, will retry"
        return 1
    fi
    echo "[FAIL] $desc (exit $rc)"
    FAILURES+=("$desc (exit $rc)")
    return 1
}

check_data() {
    local arm="$1"
    local -a missing=()
    local f
    for f in dev-single-desired-all dev-single-undesired-all \
             user-single-desired-all user-single-undesired-all \
             "${test_files[@]}"; do
        [[ -f "$DATA_ROOT/$arm/$f.jsonl" ]] || missing+=("$arm/$f.jsonl")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[skip] $arm: ${#missing[@]} data file(s) missing"
        printf '         %s\n' "${missing[@]:0:4}"
        SKIPPED+=("$arm (${#missing[@]} files missing)")
        return 1
    fi
    return 0
}

if [[ "$CROSS_ARM" == "1" ]]; then
    N_EVAL_CELLS=$(( ${#arms[@]} * ${#arms[@]} * ${#test_files[@]} ))
else
    N_EVAL_CELLS=$(( ${#arms[@]} * ${#test_files[@]} ))
fi
N_TOTAL=$(( ${#selected[@]} * ${#pairs[@]} * N_EVAL_CELLS * ${#seeds[@]} ))

for model_id in "${selected[@]}"; do
    [[ $STOPPED -eq 1 ]] && break
    model_name="${model_id##*/}"
    echo ""
    echo "==================== $model_name ===================="

    for pair in "${pairs[@]}"; do
        [[ $STOPPED -eq 1 ]] && break
        IFS='_' read -r source base <<< "$pair"

        for algo in "${algos[@]}"; do
            [[ $STOPPED -eq 1 ]] && break

            # No localization here. The random arm is a baseline FOR atp: it
            # reads atp's map to learn how many heads to draw and, when
            # layer-matched, from which layers. Run scripts/attn_conflicts.sh
            # first, or these cells fall back to a uniform draw of the default
            # size and stop being a matched comparison.
            for arm in "${arms[@]}"; do
                MAP="results/${model_name}/${arm}__from_${source}_to_${base}/atp/numerator_1_heads.pt"
                if [[ ! -f "$MAP" ]]; then
                    echo "[skip] no atp map at $MAP"
                    echo "       layer_matched mode exits rather than silently"
                    echo "       falling back, so wait for the atp sweep to"
                    echo "       finish this arm's localization first."
                fi
            done

            # ---- eval ------------------------------------------------------ #
            for eval_arm in "${arms[@]}"; do
                [[ $STOPPED -eq 1 ]] && break
                check_data "$eval_arm" || continue

                declare -a steer_arms=()
                if [[ "$CROSS_ARM" == "1" ]]; then steer_arms=("${arms[@]}")
                else steer_arms=("$eval_arm"); fi

                for steer_arm in "${steer_arms[@]}"; do
                    [[ $STOPPED -eq 1 ]] && break
                    for tf in "${test_files[@]}"; do
                    for seed in "${seeds[@]}"; do
                        [[ $STOPPED -eq 1 ]] && break
                        export RANDOM_BASELINE_SEED="$seed"
                        eval_test="$DATA_ROOT/$eval_arm/$tf.jsonl"
                        steer_add="$DATA_ROOT/$steer_arm/user-single-desired-all.jsonl"
                        steer_sub="$DATA_ROOT/$steer_arm/dev-single-desired-all.jsonl"

                        declare -a extra=()
                        [[ "$FULL_PRECISION" == "1" ]] && extra+=(--full_precision)

                        run_step "${model_name}__${eval_arm}__${source}_to_${base}__${algo}_${RANDOM_BASELINE}__s${seed}__eval_${tf}__steer_${steer_arm}" \
                                 "[$(( N_DONE + 1 ))/$N_TOTAL] $model_name | eval:$eval_arm/$tf steer:$steer_arm | ${RANDOM_BASELINE} s${seed}" \
                                 --model_id "$model_id" \
                                 --batch_size "$batch_size" \
                                 --patch_algo "$algo" \
                                 --data_dir "$eval_arm" \
                                 --source "$source" \
                                 --base "$base" \
                                 --device "$device" \
                                 --eval_model \
                                 --kv_caching \
                                 "${extra[@]}" \
                                 --eval_test "$eval_test" \
                                 --steering \
                                 --ablation steer \
                                 --steering_add_path "$steer_add" \
                                 --steering_sub_path "$steer_sub"
                    done
                    done
                done
            done
        done
    done
done

echo ""
echo "==================== summary ===================="
echo "steps complete: $N_DONE/$N_TOTAL   (ran $N_RAN this pass)"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "skipped (missing data):"; printf '  %s\n' "${SKIPPED[@]}"
fi
if [[ $STOPPED -eq 1 ]]; then
    echo "stopped early. The interrupted step has no sentinel and will be retried."
    requeue_if_slurm
    exit 0
fi
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "FAILED steps:"; printf '  %s\n' "${FAILURES[@]}"
    exit 1
fi
echo "all steps completed for: ${selected[*]}"
