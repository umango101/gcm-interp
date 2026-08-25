#!/bin/bash
# Check that the deployed layers/ package matches what was shipped.
#
# An ImportError naming a function that plainly exists in the source is almost
# always a deployment problem rather than a code problem, and it has two causes
# that look identical from the traceback:
#
#   1. A partial copy -- some files updated, others left at an older revision.
#      The checksum comparison below catches this.
#   2. Stale bytecode. Python reuses a cached .pyc when the source's recorded
#      mtime and size still match, so copying with `cp -p`, `scp -p`, or
#      `rsync -t` can preserve a timestamp that makes a NEW source look like the
#      OLD one the .pyc was built from. The source then reads correctly while
#      Python keeps executing the previous version. Clearing __pycache__ rules
#      this out, which is why it runs before the import check.
#
# Usage:  bash scripts/check_layers_install.sh
# Exit 0 = deployment matches the manifest and every expected symbol imports.

set -u
cd "${RM_INTERP_REPO:-$(dirname "$0")/..}"

echo "=== 1. clearing stale bytecode ==="
find layers -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find layers -name '*.pyc' -delete 2>/dev/null || true
echo "cleared."

echo
echo "=== 2. checksums vs manifest ==="
if [ ! -f layers/MANIFEST.sha256 ]; then
  echo "MANIFEST.sha256 missing -- copy it across too, then rerun."
  exit 2
fi
mismatch=0
while read -r want path; do
  case "$want" in \#*) continue ;; esac
  [ -z "${path:-}" ] && continue
  if [ ! -f "$path" ]; then
    echo "  MISSING  $path"
    mismatch=1
    continue
  fi
  got="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$got" = "$want" ]; then
    echo "  ok       $path"
  else
    echo "  STALE    $path  (got ${got:0:12}, want ${want:0:12}, $(wc -l < "$path") lines)"
    mismatch=1
  fi
done < layers/MANIFEST.sha256

echo
echo "=== 3. import check ==="
python - <<'PY'
import importlib
import sys

# Exactly the names run_layers.py imports at module scope. If a file is stale,
# this reports which symbol is missing from which module rather than failing on
# whichever import happens to come first.
expected = {
    'layers.layer_utils': ['get_layers', 'detect_tuple_output', 'resid_out',
                           'resid_add', 'resid_set', 'val', 'compute_resid_norms'],
    'layers.layer_config': ['LayerConfig'],
    'layers.layer_patching': ['LayerPatching'],
    'layers.layer_localize': ['run_localization', 'reduce_layer_effects',
                              'get_top_k_layers', 'retrieve_random_k_layers',
                              'plot_layer_effects', 'layer_scores'],
    'layers.layer_steering': ['layer_steering_cache', 'build_layer_vectors',
                              'generate_with_layer_patches'],
    'layers.plot_layer_sweep': ['load_dataset_csv'],
    'layers.layer_determinism': ['harden', 'write_fingerprint', 'env_fingerprint',
                                 'fingerprint_tensor', 'fingerprint_json'],
}

bad = 0
env_problem = 0
for mod_name, names in expected.items():
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        # A missing third-party dependency is an environment problem, not a stale
        # copy. Reporting it as staleness would send you recopying files that are
        # already correct.
        if not (e.name or '').startswith('layers'):
            print(f"  ENV      {mod_name}: dependency {e.name!r} not installed "
                  f"(wrong conda env?)")
            env_problem = 1
        else:
            print(f"  FAIL     {mod_name}: {type(e).__name__}: {e}")
            bad = 1
        continue
    except Exception as e:
        print(f"  FAIL     {mod_name}: {type(e).__name__}: {e}")
        bad = 1
        continue
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        print(f"  STALE    {mod_name} ({mod.__file__}) missing: {', '.join(missing)}")
        bad = 1
    else:
        print(f"  ok       {mod_name}")

if env_problem and not bad:
    print("\n  Imports could not be checked: dependencies are missing from this "
          "environment.\n  Activate the syc env and rerun -- this says nothing about "
          "whether the files are current.")
    sys.exit(3)
sys.exit(bad)
PY
import_rc=$?

echo
if [ "$import_rc" -eq 3 ]; then
  if [ "$mismatch" -eq 0 ]; then
    echo "CHECKSUMS PASS, imports unverified (missing dependencies -- wrong env?)."
    exit 3
  fi
  echo "FAIL: checksum mismatches above, and imports could not be verified."
  exit 1
fi

if [ "$mismatch" -eq 0 ] && [ "$import_rc" -eq 0 ]; then
  echo "PASS: deployment matches the manifest and every expected symbol imports."
  exit 0
fi
echo "FAIL: recopy the files listed above (all of layers/ and scripts/ is safest),"
echo "      then rerun this check before resubmitting."
exit 1
