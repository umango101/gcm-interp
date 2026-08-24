#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 04:00:00
#SBATCH -J atp_arm_comparison
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -c 4
#
# ---------------------------------------------------------------------------
# Run the ATP localization under several settings and compare the head maps.
#
#     sbatch compare_arms.sh
#
# Localization ONLY (--patch_model). No eval, no steering -- compare_head_maps.py
# reads the heads_*.pt shards directly, so a full sweep is not needed to see
# whether a setting changes which heads get selected.
#
# Each arm runs with --patch_algo atp-<label>. That is not a hack for its own
# sake: patching.py:43 and logits_handler.py:27/35 dispatch on `'atp' in
# patch_algo`, and config.set_output_prefix uses patch_algo as a path component,
# so an atp-<label> arm routes through the normal atp code path while writing to
# its own results/.../atp-<label>/ tree. Shards from different arms cannot
# collide, and each arm resumes independently.
#
# Arms are "label|extra run.py args". Add your own -- e.g. if the source-side
# completion change is behind a flag, add an arm for it:
#     EXTRA_ARMS="srccomp|--effect_positions all --source_completion" sbatch compare_arms.sh
#
# Defaults compare the two effect-position settings on the same data:
#   all    -- sum over every sequence position (original behaviour)
#   prompt -- sum only over unpadded positions before the response start
# ---------------------------------------------------------------------------

source ~/.bashrc
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/syc
cd "$RM_INTERP_REPO" || { echo "FATAL: cannot cd to $RM_INTERP_REPO"; exit 1; }
mkdir -p logs comparisons

model_id="${MODEL_ID:-Qwen/Qwen1.5-14B-Chat}"
model_name="${model_id##*/}"
source_cond="${SOURCE:-female-long}"
base_cond="${BASE:-male-long}"
device="cuda:0"
TOPK="${TOPK:-0.05}"

# num_attention_heads, needed by compare_head_maps.py to fold the flat head axis
case "$model_name" in
    Qwen1.5-14B-Chat|Qwen1.5-32B-Chat|OLMo-2-1124-13B-DPO) NUM_HEADS="${NUM_HEADS:-40}" ;;
    gemma-3-12b-it)                                        NUM_HEADS="${NUM_HEADS:-16}" ;;
    *)  NUM_HEADS="${NUM_HEADS:-}" ;;
esac
if [[ -z "$NUM_HEADS" ]]; then
    echo "FATAL: set NUM_HEADS for $model_name (num_attention_heads from its config)."
    exit 1
fi

# The three arms that separate "what the source contains" from "which positions
# are summed". Arm 1 is the original behaviour; arms 2 and 3 are the two
# independent ways of removing the pad-vs-token comparison.
declare -a ARMS=(
  "qonly-all|--source_prompt question --effect_positions all"
  "qonly-prompt|--source_prompt question --effect_positions prompt"
  "srccomp-all|--source_prompt completion --effect_positions all"
)
if [[ -n "${EXTRA_ARMS:-}" ]]; then
    IFS=';' read -ra _extra <<< "$EXTRA_ARMS"
    for a in "${_extra[@]}"; do [[ -n "$a" ]] && ARMS+=("$a"); done
fi

STATE_DIR="$RM_INTERP_REPO/.run_state"
mkdir -p "$STATE_DIR"
STOPPED=0
on_signal() { STOPPED=1; echo ""; echo "[signal] caught $1 -- stopping after the current arm."; }
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal INT'  INT

echo "[plan] $model_name | $source_cond -> $base_cond | num_heads=$NUM_HEADS"
echo "[plan] arms:"
for arm in "${ARMS[@]}"; do echo "         ${arm%%|*}   (${arm#*|})"; done

# --- data present? -------------------------------------------------------- #
D="$RM_INTERP_REPO/data/$model_name/$source_cond"
missing=()
for f in "$base_cond-desired-all.jsonl" "$base_cond-undesired-all.jsonl" \
         "$source_cond-desired-all.jsonl" "$source_cond-undesired-all.jsonl"; do
    [[ -f "$D/$f" ]] || missing+=("$f")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "FATAL: missing localization data in $D:"; printf '   %s\n' "${missing[@]}"; exit 1
fi

if ! python -c "import nnsight, sys; sys.exit(0 if nnsight.__version__ == '0.4.11' else 1)" 2>/dev/null; then
    pip install nnsight==0.4.11 || { echo "FATAL: nnsight install failed"; exit 1; }
fi

# --- localization, one run per arm ---------------------------------------- #
declare -a DONE_LABELS=()
declare -a FAILED=()

for arm in "${ARMS[@]}"; do
    [[ $STOPPED -eq 1 ]] && break
    label="${arm%%|*}"
    extra="${arm#*|}"
    algo="atp-${label}"
    outdir="results/$model_name/from_${source_cond}_to_${base_cond}/$algo"
    key="$STATE_DIR/${model_name}__${source_cond}_to_${base_cond}__${algo}__patch.done"
    argv="$model_id|$source_cond|$base_cond|$extra"

    echo ""
    echo "==================== arm: $label ===================="
    if [[ -f "$key" && "${FORCE:-0}" != "1" && "$(sed -n '2p' "$key")" == "$argv" ]]; then
        echo "[skip] already localized -> $outdir"
        DONE_LABELS+=("$label"); continue
    fi

    # shellcheck disable=SC2086
    python run.py --model_id "$model_id" \
                  --batch_size 1 \
                  --patch_algo "$algo" \
                  --source "$source_cond" \
                  --base "$base_cond" \
                  --device "$device" \
                  --patch_model $extra
    rc=$?
    if [[ $rc -eq 0 ]]; then
        { date -Is; echo "$argv"; } > "$key"
        n=$(ls "$outdir"/heads_*.pt 2>/dev/null | wc -l)
        echo "[ok  ] arm $label -> $n shards in $outdir"
        DONE_LABELS+=("$label")
    elif [[ $rc -ge 128 || $STOPPED -eq 1 ]]; then
        STOPPED=1
        echo "[stop] arm $label interrupted (exit $rc); not marked done, will retry on resume"
    else
        echo "[FAIL] arm $label (exit $rc)"
        FAILED+=("$label (exit $rc)")
    fi
done

if [[ $STOPPED -eq 1 ]]; then
    echo ""
    echo "stopped early; resubmit to resume. Completed arms: ${DONE_LABELS[*]:-none}"
    [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

# --- comparisons ----------------------------------------------------------- #
if [[ ! -f compare_head_maps.py ]]; then
    echo "FATAL: compare_head_maps.py not found in $RM_INTERP_REPO"
    exit 1
fi

root="results/$model_name/from_${source_cond}_to_${base_cond}"
tag="${model_name}_${source_cond}_to_${base_cond}"

echo ""
echo "==================== per-arm maps ===================="
for label in "${DONE_LABELS[@]}"; do
    python compare_head_maps.py "$root/atp-$label" \
        --labels "$label" --num_heads "$NUM_HEADS" --topk "$TOPK" \
        --out "comparisons/${tag}_${label}.png" 2>&1 | sed 's/^/  /'
done

echo ""
echo "==================== pairwise comparisons ===================="
n=${#DONE_LABELS[@]}
for ((i = 0; i < n; i++)); do
    for ((j = i + 1; j < n; j++)); do
        a="${DONE_LABELS[$i]}"; b="${DONE_LABELS[$j]}"
        echo ""
        echo "---------- $a vs $b ----------"
        python compare_head_maps.py "$root/atp-$a" "$root/atp-$b" \
            --labels "$a" "$b" --num_heads "$NUM_HEADS" --topk "$TOPK" \
            --out "comparisons/${tag}_${a}_vs_${b}.png" 2>&1 | sed 's/^/  /'
    done
done

echo ""
echo "==================== summary ===================="
echo "arms localized: ${DONE_LABELS[*]:-none}"
echo "figures + stats: comparisons/${tag}_*.png"
echo ""
echo "Read the top-k overlap line against its chance value:"
echo "  overlap near chance  -> the setting changes WHICH heads are selected"
echo "  overlap near 100%    -> the setting only rescales effect magnitudes"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""; echo "FAILED arms:"; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
