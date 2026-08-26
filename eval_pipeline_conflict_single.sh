#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 00:30:00
#SBATCH -J conflict_single_pipeline
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH -c 2

# Instruction-privilege single-token eval: all stages, one job.
#
# Produces, per cell:
#   results_pipeline_conflict_single/.../plots/flip_heatmap.png
#                                            /dev_post_heatmap.png
#                                            /user_post_heatmap.png
#                                            /broken_post_heatmap.png
#   ... plus {metric}_dataset.csv beside each, the per-(N,topk) accuracy JSONs,
#   and accuracy/summary.json with the counts behind every rate.
#
# NO GPU is requested. eval_pipeline_bias.sh needs an h100 for its 70B judge
# pass; here correctness is string identity against each row's dev_word /
# user_word, so every stage is pandas + matplotlib and finishes in seconds.
# Asking for a GPU would only lengthen the queue wait. The full stage list and
# --batch_size are still passed so this is a drop-in for the bias invocation:
# build_prompts and judge are accepted and no-op.
#
# No resume logic: a preempted run re-runs from scratch in under a minute.
#
# Runs fine outside SLURM too -- `bash eval_pipeline_conflict_single.sh`.

# NO strict mode yet. /etc/bashrc tests BASHRCSOURCED before assigning it, and
# the conda hook has the same habit, so `set -u` here fails on unbound variables
# in code we do not control. Source first, turn on strict mode after.
source ~/.bashrc
source /home/ubansal/miniconda/etc/profile.d/conda.sh
# gpt-oss needs transformers >= 4.55, so this is the pinned conflicts env, not `syc`.
conda activate /home/ubansal/miniconda/envs/conflict-syc

# Safe from here on: everything below is ours.
set -euo pipefail

# The pipeline, the gen files and the test data all live in the conflicts
# checkout -- NOT the RM_INTERP_REPO checkout. These are two separate trees and
# the data dirs under them can diverge.
PIPELINE_REPO="${PIPELINE_REPO:-/home/ubansal/orcd/scratch/conflicts/gcm-interp}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-20b}"
LOCALIZATION="${LOCALIZATION:-from_user-single_to_dev-single}"
CELL="${CELL:-atp/user-single_eval/user-single_steer}"

GEN_DIR="${PIPELINE_REPO}/results/${MODEL_NAME}/${LOCALIZATION}/${CELL}/eval"
TEST_JSONL="${PIPELINE_REPO}/data/${MODEL_NAME}/user-single/dev-single-test.jsonl"
OUT_DIR="${PIPELINE_REPO}/results_pipeline_conflict_single/${MODEL_NAME}/${LOCALIZATION}/${CELL}"

# Unused by this script; exported for anything downstream that expects it.
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"

cd "$PIPELINE_REPO"
mkdir -p logs

# ---------------------------------------------------------------------------
# Preflight. Fail here, naming the real path, rather than inside stage_merge.
# ---------------------------------------------------------------------------
if [ ! -f "$TEST_JSONL" ]; then
    echo "test set not found: $TEST_JSONL" >&2
    exit 1
fi

if [ ! -d "$GEN_DIR" ]; then
    echo "gen files not found at: $GEN_DIR" >&2
    echo "run the eval step of scripts/attn_conflicts.sh first." >&2
    find "${PIPELINE_REPO}/results/${MODEL_NAME}" -maxdepth 5 -type d -name eval 2>/dev/null | head >&2
    exit 1
fi

n_gen=$(find "$GEN_DIR" -name '*_user-single_gen.json' | wc -l)
if [ "$n_gen" -eq 0 ]; then
    echo "no *_user-single_gen.json under: $GEN_DIR" >&2
    ls "$GEN_DIR" | head >&2
    exit 1
fi

# Every gen file must carry one item per test row -- stage_merge joins them
# positionally and refuses to proceed on a mismatch. Checking ALL of them, not
# a sample: gen files are written per (N, topk) cell over a long sweep, so a
# short file is exactly the kind of thing a preemption leaves behind in one
# cell and not the others.
n_test=$(grep -c . "$TEST_JSONL")
python - "$GEN_DIR" "$n_test" <<'PYEOF' || exit 1
import glob, json, os, sys
gen_dir, n_test = sys.argv[1], int(sys.argv[2])
bad = []
for p in sorted(glob.glob(os.path.join(gen_dir, "*_user-single_gen.json"))):
    try:
        n = len(json.load(open(p)))
    except Exception as e:
        bad.append((os.path.basename(p), f"unreadable: {e}"))
        continue
    if n != n_test:
        bad.append((os.path.basename(p), f"{n} items"))
if bad:
    print(f"row count mismatch against {n_test} test rows -- the positional "
          f"join would misalign:", file=sys.stderr)
    for name, why in bad[:10]:
        print(f"  {name}: {why}", file=sys.stderr)
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "python:  $(which python)"
echo "gen:     $GEN_DIR ($n_gen files)"
echo "test:    $TEST_JSONL ($n_test rows; all $n_gen gen files match)"
echo "out:     $OUT_DIR"
echo

python eval_pipeline_conflict_single.py \
    --stages merge build_prompts judge accuracies plots \
    --batch_size 32

# ---------------------------------------------------------------------------
# Report what landed, so a silently-empty plots dir is visible in the job log.
# ---------------------------------------------------------------------------
echo
echo "==================== output ===================="
for png in flip dev_post user_post broken_post; do
    f="${OUT_DIR}/plots/${png}_heatmap.png"
    if [ -f "$f" ]; then
        echo "  ok      ${png}_heatmap.png"
    else
        echo "  MISSING ${png}_heatmap.png" >&2
    fi
done
echo "  summary ${OUT_DIR}/accuracy/summary.json"
