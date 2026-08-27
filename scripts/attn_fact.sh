#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -J fact_experiment
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -c 4

source ~/.bashrc
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda info --envs
conda activate /home/ubansal/miniconda/envs/syc
cd "$RM_INTERP_REPO" || { echo "FATAL: cannot cd to $RM_INTERP_REPO"; exit 1; }

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubansal/orcd/scratch/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib

# --------------------------------------------------------------------------- #
# What to run                                                                  #
# --------------------------------------------------------------------------- #
# model_name (and so every data/ and results/ path) is the part of the HF id
# after the slash -- config.py derives data_path the same way, so adding a model
# here is the only edit needed.
declare -a models=(
  "tiiuae/Falcon3-10B-Instruct"
)

declare -a pairs=(
  "lying-long_truthful-long"
  "lying-single_truthful-single"
)

algos=("atp")
formats=("long" "single")          # the eval grid is formats x formats
device="cuda:0"
batch_size=1

# The original script passed --full_precision on the FIRST of its four eval
# invocations and not the other three, mixing one full-bf16 cell into a table
# of three 4-bit cells. This applies it uniformly instead. It is part of each
# sentinel's recorded argv, so flipping it re-runs the affected steps rather
# than skipping them as already done.
FULL_PRECISION="${FULL_PRECISION:-0}"

DATA_ROOT="$RM_INTERP_REPO/data"
STATE_DIR="$RM_INTERP_REPO/.run_state"
mkdir -p "$STATE_DIR" logs

# --------------------------------------------------------------------------- #
# Preemption handling                                                          #
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

# Called on the way out when we stopped early under SLURM: ask for a requeue so
# the sweep continues in a fresh allocation. Harmless outside SLURM.
requeue_if_slurm() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "[exit] requeueing job $SLURM_JOB_ID to continue the sweep"
        scontrol requeue "$SLURM_JOB_ID" || echo "[exit] requeue failed; resubmit by hand"
    fi
}

# --------------------------------------------------------------------------- #
# Model selection                                                              #
# --------------------------------------------------------------------------- #
declare -a selected=()
if [[ -n "${ONLY_MODEL:-}" ]]; then
    if [[ "$ONLY_MODEL" =~ ^[0-9]+$ ]]; then
        if [[ "$ONLY_MODEL" -ge "${#models[@]}" ]]; then
            echo "FATAL: ONLY_MODEL index $ONLY_MODEL exceeds the ${#models[@]} models listed."
            exit 1
        fi
        selected=("${models[$ONLY_MODEL]}")
    else
        for m in "${models[@]}"; do
            [[ "${m##*/}" == "$ONLY_MODEL" || "$m" == "$ONLY_MODEL" ]] && selected+=("$m")
        done
        if [[ ${#selected[@]} -eq 0 ]]; then
            echo "FATAL: ONLY_MODEL='$ONLY_MODEL' matches nothing in the models list."
            exit 1
        fi
    fi
else
    selected=("${models[@]}")
fi

echo "[plan] models: ${selected[*]}"
echo "[plan] pairs:  ${pairs[*]}"
echo "[plan] algos:  ${algos[*]}   full_precision=$FULL_PRECISION"
echo "[plan] state:  $STATE_DIR ($(find "$STATE_DIR" -name '*.done' 2>/dev/null | wc -l) steps already recorded)"

# nnsight pin, once per job rather than once per pair as in the original.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if ! python -c "import nnsight, sys; sys.exit(0 if nnsight.__version__ == '0.4.11' else 1)" 2>/dev/null; then
        echo "[setup] installing nnsight==0.4.11"
        pip install nnsight==0.4.11 || { echo "FATAL: nnsight install failed"; exit 1; }
    fi
fi

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
declare -a FAILURES=()
declare -a SKIPPED=()
N_DONE=0
N_RAN=0

# run_step <sentinel-key> <description> <run.py args...>
#
# The sentinel stores the exact argv that produced it. A step is skipped only if
# a sentinel exists AND its argv matches what we are about to run -- so editing
# FULL_PRECISION, batch_size, or a data path re-runs the affected steps instead
# of silently reporting them as complete.
run_step() {
    local key="$1"; shift
    local desc="$1"; shift
    local sentinel="$STATE_DIR/${key}.done"
    local argv="$*"

    if [[ -f "$sentinel" && "${FORCE:-0}" != "1" ]]; then
        local stored
        stored="$(sed -n '2p' "$sentinel")"
        if [[ "$stored" == "$argv" ]]; then
            echo "[skip] $desc"
            N_DONE=$(( N_DONE + 1 ))
            return 0
        fi
        echo "[stale] $desc -- recorded argv differs from current settings; re-running"
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry ] $desc"
        N_DONE=$(( N_DONE + 1 ))
        return 0
    fi

    echo "[run ] $desc"
    python run.py "$@"
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        { date -Is; echo "$argv"; } > "$sentinel"
        echo "[ok  ] $desc"
        N_DONE=$(( N_DONE + 1 ))
        N_RAN=$(( N_RAN + 1 ))
        return 0
    fi

    # 128+N means killed by signal N (143=TERM, 130=INT, 138=USR1). That is
    # preemption, not a broken cell: leave the sentinel unwritten so the step is
    # retried, and stop rather than marching the rest of the sweep into a GPU
    # that is being torn down.
    if [[ $rc -ge 128 || $STOPPED -eq 1 ]]; then
        STOPPED=1
        echo "[stop] $desc interrupted (exit $rc); not marked done, will retry on resume"
        return 1
    fi

    echo "[FAIL] $desc (exit $rc)"
    FAILURES+=("$desc (exit $rc)")
    return 1
}

# Every file the patch + eval steps read for one model. Missing data is a
# planning error, not a crash three hours in: OLMo-2-1124-13B-DPO, for one, has
# no lying-* directories built.
check_data() {
    local model_name="$1" source="$2" base="$3"
    local -a needed=(
        "$DATA_ROOT/$model_name/$source/$base-desired-all.jsonl"
        "$DATA_ROOT/$model_name/$source/$base-undesired-all.jsonl"
        "$DATA_ROOT/$model_name/$source/$source-desired-all.jsonl"
        "$DATA_ROOT/$model_name/$source/$source-undesired-all.jsonl"
    )
    local fmt
    for fmt in "${formats[@]}"; do
        needed+=(
            "$DATA_ROOT/$model_name/lying-$fmt/truthful-$fmt-test.jsonl"
            "$DATA_ROOT/$model_name/lying-$fmt/lying-$fmt-steering.jsonl"
            "$DATA_ROOT/$model_name/lying-$fmt/truthful-$fmt-steering.jsonl"
        )
    done
    local -a missing=()
    local f
    for f in "${needed[@]}"; do
        [[ -f "$f" ]] || missing+=("${f#"$DATA_ROOT"/}")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[skip] $model_name / $source -> $base: ${#missing[@]} data file(s) missing"
        printf '         %s\n' "${missing[@]:0:4}"
        [[ ${#missing[@]} -gt 4 ]] && echo "         ... and $(( ${#missing[@]} - 4 )) more"
        SKIPPED+=("$model_name / $source -> $base (${#missing[@]} files missing)")
        return 1
    fi
    return 0
}

# Total steps in the plan, for progress reporting that stays meaningful across
# requeues (the count includes steps a previous pass already finished).
N_TOTAL=$(( ${#selected[@]} * ${#pairs[@]} * ${#algos[@]} * (1 + ${#formats[@]} * ${#formats[@]}) ))

# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #
for model_id in "${selected[@]}"; do
    [[ $STOPPED -eq 1 ]] && break
    model_name="${model_id##*/}"
    echo ""
    echo "==================== $model_name ===================="

    for pair in "${pairs[@]}"; do
        [[ $STOPPED -eq 1 ]] && break
        IFS='_' read -r source base <<< "$pair"
        check_data "$model_name" "$source" "$base" || continue

        for algo in "${algos[@]}"; do
            [[ $STOPPED -eq 1 ]] && break
            key_prefix="${model_name}__${source}_to_${base}__${algo}"

            # ---- localization -------------------------------------------- #
            # All four eval cells read the attribution map this produces, so if
            # it fails the rest of the pair is unusable: move to the next pair.
            run_step "${key_prefix}__patch" \
                     "[$(( N_DONE + 1 ))/$N_TOTAL] $model_name | $source -> $base | $algo | localize" \
                     --model_id "$model_id" \
                     --batch_size "$batch_size" \
                     --patch_algo "$algo" \
                     --source "$source" \
                     --base "$base" \
                     --device "$device" \
                     --patch_model || continue

            # ---- eval grid: (eval format) x (steering-vector format) ------ #
            for eval_fmt in "${formats[@]}"; do
                [[ $STOPPED -eq 1 ]] && break
                for steer_fmt in "${formats[@]}"; do
                    [[ $STOPPED -eq 1 ]] && break
                    eval_test="$DATA_ROOT/$model_name/lying-$eval_fmt/truthful-$eval_fmt-test.jsonl"
                    steer_add="$DATA_ROOT/$model_name/lying-$steer_fmt/lying-$steer_fmt-steering.jsonl"
                    steer_sub="$DATA_ROOT/$model_name/lying-$steer_fmt/truthful-$steer_fmt-steering.jsonl"

                    declare -a extra=()
                    [[ "$FULL_PRECISION" == "1" ]] && extra+=(--full_precision)

                    run_step "${key_prefix}__eval_${eval_fmt}__steer_${steer_fmt}" \
                             "[$(( N_DONE + 1 ))/$N_TOTAL] $model_name | $source -> $base | $algo | eval:$eval_fmt steer:$steer_fmt" \
                             --model_id "$model_id" \
                             --batch_size "$batch_size" \
                             --patch_algo "$algo" \
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

# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #
echo ""
echo "==================== summary ===================="
echo "steps complete: $N_DONE/$N_TOTAL   (ran $N_RAN this pass)"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "skipped (missing data):"
    printf '  %s\n' "${SKIPPED[@]}"
fi

if [[ $STOPPED -eq 1 ]]; then
    echo "stopped early (preemption or wall clock). Nothing is lost:"
    echo "  the interrupted step has no sentinel and will be retried."
    requeue_if_slurm
    exit 0
fi

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "FAILED steps:"
    printf '  %s\n' "${FAILURES[@]}"
    echo "Resubmitting retries only these -- completed steps are sentinel-skipped."
    exit 1
fi
echo "all steps completed for: ${selected[*]}"
