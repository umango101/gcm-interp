#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 36:00:00
#SBATCH -J gptoss_matrix
#SBATCH -o logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8

# Localization x steering matrix on gpt-oss-20b: 9 conditions over 3
# localizations. Replaces run_gptoss_experiment.sh.
#
# THE DESIGN
#   Which HEADS are steered and which PROMPTS they are steered on are varied
#   independently. In every condition the steering vector and the test prompts
#   come from the same dataset (the "steer on" dataset); only the head list
#   comes from the "localize" side. That makes the head set the single varying
#   factor between conditions that share a steer dataset.
#
#   This differs from the old script's cross-dataset arm, which carried the
#   steering VECTOR across datasets as well. That conflated two questions; if
#   you want the old behaviour, set STEER_VEC_FROM_LOC=1 below.
#
#   #  localize                              steer on     role
#   1  roleConflict-roleAgree                role         undifferenced baseline
#   2  withinConflict-withinAgree            within       undifferenced baseline
#   3  roleConflict-roleAgree                within       head transfer
#   4  withinConflict-withinAgree            role         head transfer
#   5  role MINUS roleInverted               role         primary control
#   6  role MINUS within                     role
#   7  role MINUS within                     within
#   8  within MINUS role                     role
#   9  within MINUS role                     within
#
#   1 and 2 are the baselines every differenced condition is read against. Do
#   not drop them: a differenced efficacy number is uninterpretable on its own.
#
# MINUS = set difference over top-k MEMBERSHIP, not subtraction of scores.
#   roleConflict and roleInverted share a base and so score OPPOSITE directions
#   (roleConflict's desired answer flips between conditions, roleInverted's does
#   not); roleConflict and withinConflict have different bases and so different
#   scales. Membership is invariant to both. See localize_diff.py.
#
# WHAT THIS DOES NOT DO
#   No scoring. It writes generations only. Every condition x topk x N cell is
#   ungraded text until you run a scorer that labels each generation three ways:
#   flipped to the developer's word / still the user's word / neither. "Flipped"
#   and "broken" are indistinguishable in any metric that only counts the target
#   token, and at high N the steering is expected to break the output.
#
#   No set-size matching. A differenced head set is SMALLER than the
#   undifferenced one it came from, and steering fewer heads produces less
#   effect regardless of which heads. diff_manifest.json records n_survive per
#   topk; before reading any drop in efficacy as mechanistic, compare against
#   the top-n_survive heads of the undifferenced ranking and against
#   n_survive random heads.

source ~/.bashrc
export RM_INTERP_REPO="/orcd/scratch/orcd/008/ubansal/conflicts/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/conflict-syc
cd "$RM_INTERP_REPO" || exit 1
mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/008/ubansal/hf}"
mkdir -p "$HF_HOME"

export HARMONY_SYSTEM=minimal
export HEAD_SITE=o_proj_input

# --- determinism (read at process start; setting these in Python is too late) --
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
export STRICT_DETERMINISM="1"
export SEED=42

model_id="openai/gpt-oss-20b"
model_name="gpt-oss-20b"
device="cuda:0"
algo="atp"
data="${RM_INTERP_REPO}/data/${model_name}"

# Localization shards and attribution maps are shared across conditions and
# built once. Each eval condition gets its own results root, because the output
# prefix is a function of (source, base, eval_test, steering_add) only -- so
# conditions 1 and 5, which differ ONLY in their head list, would otherwise
# resolve to the same directory and silently overwrite each other's
# generations. --results_root exists for this.
SHARED_ROOT="${RM_INTERP_REPO}/results"
MATRIX_ROOT="${RM_INTERP_REPO}/results_matrix"
mkdir -p "$MATRIX_ROOT"

# Set to 1 to reproduce the old cross-dataset behaviour (steering vector taken
# from the localization dataset rather than the steer dataset).
STEER_VEC_FROM_LOC=0

# dataset key -> "dir source base"
declare -A DS=(
  [role]="roleConflict-single roleConflict-single roleAgree-single"
  [within]="withinConflict-single withinConflict-single withinAgree-single"
  [inverted]="roleInverted-single roleInverted-single roleAgree-single"
)
ds_field() { read -r d s b <<< "${DS[$1]}"; case "$2" in dir) echo "$d";; src) echo "$s";; base) echo "$b";; esac; }

# ---------------------------------------------------------------------------
# 0. pre-flight
# ---------------------------------------------------------------------------
VERIFY=$(find . -name verify_gptoss.py | head -1)
python "$VERIFY" --stage tokenizer --data_dir "data/${model_name}" || exit 1
python "$VERIFY" --stage model --device "$device" || exit 1

for k in role within inverted; do
  d=$(ds_field "$k" dir); s=$(ds_field "$k" src); b=$(ds_field "$k" base)
  for f in "${data}/${d}/${s}-desired-all.jsonl" "${data}/${d}/${b}-desired-all.jsonl" \
           "${data}/${d}/${b}-test.jsonl"; do
    [ -f "$f" ] || { echo "MISSING $f -- regenerate the corpus first"; exit 1; }
  done
done

# ---------------------------------------------------------------------------
# 1. localize (3 runs, shared)
# ---------------------------------------------------------------------------
for k in role within inverted; do
  s=$(ds_field "$k" src); b=$(ds_field "$k" base)
  echo "=== localizing ${s} -> ${b} ==="
  python run.py --model_id "$model_id" \
                --batch_size 1 \
                --seed "$SEED" \
                --patch_algo "$algo" \
                --source "$s" \
                --base "$b" \
                --device "$device" \
                --head_site o_proj_input \
                --results_root "$SHARED_ROOT" \
                --patch_model || exit 1
done

# ---------------------------------------------------------------------------
# 2. differenced head lists
# ---------------------------------------------------------------------------
# localize_diff.py reduces heads_*.pt to numerator_1_heads.pt for both sides if
# that has not happened yet -- run.py --patch_model writes only the shards. The
# three invocations below therefore also build all three attribution maps, which
# conditions 1-4 need.
run_diff() {   # $1=keep key  $2=remove key
  local ks kb rs rb
  ks=$(ds_field "$1" src); kb=$(ds_field "$1" base)
  rs=$(ds_field "$2" src); rb=$(ds_field "$2" base)
  python localize_diff.py \
      --results_root "$SHARED_ROOT" \
      --model "$model_name" \
      --keep   "from_${ks}_to_${kb}" \
      --remove "from_${rs}_to_${rb}" \
      --algo "$algo" \
      --mode setdiff || exit 1
}
echo "=== differenced head lists ==="
run_diff role inverted
run_diff role within
run_diff within role

# ---------------------------------------------------------------------------
# 3. the 9 conditions
# ---------------------------------------------------------------------------
# name | localize-from key | remove key ("-" for undifferenced) | steer-on key
declare -a CONDITIONS=(
  "01_loc-role_steer-role|role|-|role"
  "02_loc-within_steer-within|within|-|within"
  "03_loc-role_steer-within|role|-|within"
  "04_loc-within_steer-role|within|-|role"
  "05_loc-role-minus-inverted_steer-role|role|inverted|role"
  "06_loc-role-minus-within_steer-role|role|within|role"
  "07_loc-role-minus-within_steer-within|role|within|within"
  "08_loc-within-minus-role_steer-role|within|role|role"
  "09_loc-within-minus-role_steer-within|within|role|within"
)

for spec in "${CONDITIONS[@]}"; do
  IFS='|' read -r name loc_k rm_k steer_k <<< "$spec"

  loc_src=$(ds_field "$loc_k" src);   loc_base=$(ds_field "$loc_k" base)
  st_dir=$(ds_field "$steer_k" dir);  st_src=$(ds_field "$steer_k" src)
  st_base=$(ds_field "$steer_k" base)

  root="${MATRIX_ROOT}/${name}"
  task="${root}/${model_name}/from_${loc_src}_to_${loc_base}/${algo}"
  mkdir -p "$task"

  # The map is only consumed to derive the top-k CSV (and to plot). Link rather
  # than copy so all conditions share one artifact and cannot drift apart.
  ln -sf "${SHARED_ROOT}/${model_name}/from_${loc_src}_to_${loc_base}/${algo}/numerator_1_heads.pt" \
         "${task}/numerator_1_heads.pt"

  # Steering vector source: the steer dataset by default.
  if [ "$STEER_VEC_FROM_LOC" = "1" ]; then
    vec_dir=$(ds_field "$loc_k" dir); vec_src="$loc_src"; vec_base="$loc_base"
  else
    vec_dir="$st_dir"; vec_src="$st_src"; vec_base="$st_base"
  fi

  eval_test="${data}/${st_dir}/${st_base}-test.jsonl"
  add_path="${data}/${vec_dir}/${vec_src}-desired-all.jsonl"
  sub_path="${data}/${vec_dir}/${vec_base}-desired-all.jsonl"

  # set_output_prefix builds this from the two directory names; recomputed here
  # so the differenced CSVs can be installed BEFORE run_eval looks for them.
  eval_prefix="${task}/${st_dir}_eval/${vec_dir}_steer"
  mkdir -p "${eval_prefix}/eval"

  if [ "$rm_k" != "-" ]; then
    rm_src=$(ds_field "$rm_k" src); rm_base=$(ds_field "$rm_k" base)
    echo "--- installing differenced head lists for ${name} ---"
    # run_eval reads {prefix}/eval/{metric}_{reps}_{topk}.csv when present and
    # only computes one when absent. Installing here is what makes this
    # condition steer the differenced set instead of the raw top-k.
    python localize_diff.py \
        --results_root "$SHARED_ROOT" \
        --model "$model_name" \
        --keep   "from_${loc_src}_to_${loc_base}" \
        --remove "from_${rm_src}_to_${rm_base}" \
        --algo "$algo" \
        --mode setdiff \
        --install_into "$eval_prefix" \
        --overwrite || exit 1
  fi

  cat > "${root}/CONDITION.txt" <<EOF
condition   : ${name}
localize    : from_${loc_src}_to_${loc_base}
subtract    : ${rm_k}
steer on    : ${st_dir}
eval_test   : ${eval_test}
steering_add: ${add_path}
steering_sub: ${sub_path}
vector from : $([ "$STEER_VEC_FROM_LOC" = 1 ] && echo localization || echo "steer dataset")
EOF

  echo "=== [${name}] heads from ${loc_src}$([ "$rm_k" != "-" ] && echo " minus ${rm_k}"), steering ${st_dir} ==="
  python run.py --model_id "$model_id" \
                --batch_size 1 \
                --seed "$SEED" \
                --patch_algo "$algo" \
                --source "$loc_src" \
                --base "$loc_base" \
                --device "$device" \
                --head_site o_proj_input \
                --results_root "$root" \
                --eval_model \
                --kv_caching \
                --eval_test "$eval_test" \
                --steering \
                --ablation steer \
                --steering_add_path "$add_path" \
                --steering_sub_path "$sub_path" || exit 1
done

echo
echo "done. ${#CONDITIONS[@]} conditions under ${MATRIX_ROOT}/"
echo "each condition's spec is in <condition>/CONDITION.txt"
echo "survivor counts per topk are in"
echo "  ${SHARED_ROOT}/${model_name}/from_*_minus_from_*/${algo}/eval/diff_manifest.json"
echo
echo "NOTHING IS SCORED YET. Grade each generation three ways (developer's word /"
echo "user's word / neither) before comparing conditions."
