# Sourced by the layer-pipeline SLURM scripts, after conda activate and before
# any `python -m` invocation.
#
# Motivation: CUDA error 802 ("system not yet initialized") is the NVSwitch
# fabric manager / IMEX layer not being up on the allocated node. It is a node
# fault, not a job fault -- retrying on the same node fails identically, and on a
# preemptable partition with --requeue that can burn a long series of allocations
# each dying ~90s in, at model load, having produced nothing.
#
# So: probe CUDA before doing any real work, and if it is broken, record the node
# and requeue rather than proceeding.

preflight_gpu () {
  local node="${SLURMD_NODENAME:-$(hostname)}"
  local badlist="${RM_INTERP_REPO:-.}/logs/bad_gpu_nodes.txt"
  local restarts="${SLURM_RESTART_COUNT:-0}"
  # Declared before the probe on purpose: `local rc=$?` assigns the exit status of
  # `local` itself, not of the preceding command, which would swallow the
  # environment-vs-node distinction below.
  local probe_rc=0

  # A real allocation, not just torch.cuda.is_available(): is_available() returns
  # False on a broken fabric rather than raising, which is indistinguishable from
  # a CPU-only node and hides which of the two you are looking at.
  if python - <<'PY'
import sys
try:
    import torch
except ImportError as e:
    # Not a node fault -- requeueing would loop forever on a wrong environment.
    print(f"[preflight] torch not importable: {e}. Wrong conda env, not a bad node.",
          file=sys.stderr)
    sys.exit(70)
try:
    torch.zeros(1, device='cuda')
except Exception as e:
    print(f"[preflight] CUDA unusable: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(75)
print(f"[preflight] CUDA OK: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB free)")
PY
  then
    return 0
  else
    # $? must be read here, as the first thing in the else branch. Reading it after
    # `fi` yields the exit status of the `if` compound (0 when no branch is taken),
    # not of the failed condition -- which would collapse the env/node distinction.
    probe_rc=$?
  fi

  if [ "$probe_rc" -eq 70 ]; then
    echo "[preflight] environment problem, not a node problem. Not requeueing."
    exit 70
  fi

  echo "[preflight] node ${node} cannot initialise CUDA (restart #${restarts})."
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 | head -20 || true
    nvidia-smi -q 2>/dev/null | grep -i -A2 -e fabric -e imex || true
  else
    echo "[preflight] nvidia-smi not on PATH -- no GPU allocated, or a broken node image."
  fi

  mkdir -p "$(dirname "$badlist")"
  echo "$node" >> "$badlist"
  local excludes
  excludes="$(sort -u "$badlist" | paste -sd, -)"
  echo "[preflight] nodes seen bad so far: ${excludes}"
  echo "[preflight] resubmit with --exclude=${excludes} to stop landing on them."

  # Requeueing is only useful while there is some chance of a different node.
  # Past a few restarts the likelier explanation is a partition-wide problem or a
  # bad --gres request, and looping just hides it.
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$restarts" -lt 3 ]; then
    echo "[preflight] requeueing job ${SLURM_JOB_ID}."
    scontrol requeue "$SLURM_JOB_ID" || true
  else
    echo "[preflight] not requeueing (restart #${restarts}). If several distinct "
    echo "[preflight] nodes have now failed, this is not a per-node fault -- check "
    echo "[preflight] the partition and the --gres=gpu:h200:1 request."
  fi
  exit 75
}
