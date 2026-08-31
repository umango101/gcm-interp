#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 08:00:00
#SBATCH -J rand_ablate
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
# NECESSITY, not sufficiency. Every result so far comes from ADDING a direction,
# which shows a direction exists that moves the answer. This asks the other
# question: if the heads' outputs are destroyed, does deference break? A
# localization claim needs both, and the steering-only version is the one a
# reviewer will push on.
algos=("random")

# WHAT THIS INTERVENTION ACTUALLY IS. mean_ablations_cache averages over
# source_qs_toks -- with --source user-single that is the USER-preamble corpus --
# and generate_with_patches then does `= N * vec` rather than `+=`. So each head
# is REPLACED by its average value under the opposite policy. That is a mean
# INTERCHANGE ablation, not a neutral mean ablation, and it is the stronger test
# of the two: the replacement is on-distribution rather than an average over a
# mixture the model never sees. Say "mean-interchange" in the paper, not "mean
# ablation".
#
# N=1 and STEER_NORMALIZE=0 are both required. _steering_vector normalises when
# normalize=True (eval_runner hardcodes it), so the default would replace each
# head with a UNIT vector scaled by N -- an arbitrary magnitude in the mean
# direction, not the mean. Requires the one-line eval_runner edit that reads
# STEER_NORMALIZE.
export STEER_NORMALIZE="${STEER_NORMALIZE:-0}"
export N_VALS="${N_VALS:-1}"

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



DATA_ROOT="$RM_INTERP_REPO/data/gpt-oss-20b"
# Separate state dir so this can run alongside the other sweeps.
STATE_DIR="$RM_INTERP_REPO/.run_state_ablate"
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
echo "[plan] ablation=mean (interchange from the source corpus)"
echo "[plan] N=$N_VAL STEER_NORMALIZE=$STEER_NORMALIZE"
echo "[plan] full_precision=$FULL_PRECISION"
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
    export STEER_NORMALIZE N_VALS
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

N_EVAL_CELLS=$(( ${#arms[@]} * ${#test_files[@]} ))
N_TOTAL=$(( ${#selected[@]} * ${#pairs[@]} * N_EVAL_CELLS ))

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

            # No localization here: this reads the map the atp sweep already
            # produced and ablates the heads it selected.
            for arm in "${arms[@]}"; do
                MAP="results/${model_name}/${arm}__from_${source}_to_${base}/atp/numerator_1_heads.pt"
                [[ -f "$MAP" ]] || echo "[warn] no atp map at $MAP -- run the atp sweep first"
            done

            # ---- eval ------------------------------------------------------ #
            for eval_arm in "${arms[@]}"; do
                [[ $STOPPED -eq 1 ]] && break
                check_data "$eval_arm" || continue

                # No cross-arm axis: nothing is transferred here. The
                # replacement values come from the eval arm's own source corpus.
                steer_arm="$eval_arm"

                for tf in "${test_files[@]}"; do
                    [[ $STOPPED -eq 1 ]] && break
                        eval_test="$DATA_ROOT/$eval_arm/$tf.jsonl"
                        steer_add="$DATA_ROOT/$steer_arm/user-single-desired-all.jsonl"
                        steer_sub="$DATA_ROOT/$steer_arm/dev-single-desired-all.jsonl"

                        declare -a extra=()
                        [[ "$FULL_PRECISION" == "1" ]] && extra+=(--full_precision)

                        run_step "${model_name}__${eval_arm}__${source}_to_${base}__${algo}__ablate_mean__eval_${tf}" \
                                 "[$(( N_DONE + 1 ))/$N_TOTAL] $model_name | eval:$eval_arm/$tf | mean-interchange ablation" \
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
                                 --ablation mean \
                                 --steering_add_path "$steer_add" \
                                 --steering_sub_path "$steer_sub"
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
