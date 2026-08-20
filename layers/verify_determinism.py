"""Diff two independent runs of the layer pipeline.

Determinism claims are worth what they can be demonstrated to be worth, so this
compares two results trees produced by the same command and reports where they
diverge. Run it after ``scripts/verify_layers_determinism.sh``, or against any
two roots:

    python -m layers.verify_determinism ./results_layers_runA ./results_layers_runB

Exit code 0 means every compared artifact is bit-identical. Non-zero means it is
not, and the report names the first place the two runs parted company --- which
is the useful information, because attribution maps, steering caches, and
generations fail for different reasons:

  attribution map differs  -> the backward pass is nondeterministic. Almost always
                              a fused SDPA backward; check determinism.json for
                              flash_sdp / mem_efficient_sdp and rerun with
                              --sdp_backend math --strict_determinism.
  map matches, topk differs -> ties in the ranking being broken inconsistently.
                              Should be impossible given the explicit sort, so
                              treat it as a real bug rather than noise.
  topk matches, gens differ -> the generation path, not the localization. Check
                              that batch sizes match in determinism.json, and
                              that --no_deterministic was not set on one side.
"""

import argparse
import glob
import json
import os
import sys

import torch


def _rel(root, path):
    return os.path.relpath(path, root)


def _compare_tensors(a_path, b_path):
    a = torch.load(a_path, map_location='cpu')
    b = torch.load(b_path, map_location='cpu')
    if isinstance(a, dict):
        a = a.get('cache', a)
    if isinstance(b, dict):
        b = b.get('cache', b)
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return None, "not tensors"
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    if torch.equal(a32, b32):
        return True, "bit-identical"
    diff = (a32 - b32).abs()
    denom = a32.abs().clamp(min=1e-12)
    return False, (f"max abs diff {diff.max():.3e}, "
                   f"max rel diff {(diff / denom).max():.3e}, "
                   f"{int((diff > 0).sum())}/{diff.numel()} elements differ")


def _compare_text(a_path, b_path):
    with open(a_path) as f:
        a = f.read()
    with open(b_path) as f:
        b = f.read()
    if a == b:
        return True, "identical"
    if a_path.endswith('.json'):
        try:
            ja, jb = json.loads(a), json.loads(b)
            if isinstance(ja, list) and isinstance(jb, list) and len(ja) == len(jb):
                n = sum(1 for x, y in zip(ja, jb) if x != y)
                return False, f"{n}/{len(ja)} entries differ"
        except Exception:
            pass
    return False, "content differs"


def compare_roots(root_a, root_b, patterns):
    results = []
    for pattern in patterns:
        a_files = sorted(glob.glob(os.path.join(root_a, pattern), recursive=True))
        for a_path in a_files:
            rel = _rel(root_a, a_path)
            b_path = os.path.join(root_b, rel)
            if not os.path.exists(b_path):
                results.append((rel, None, "missing in second run"))
                continue
            if a_path.endswith('.pt'):
                ok, detail = _compare_tensors(a_path, b_path)
            else:
                ok, detail = _compare_text(a_path, b_path)
            results.append((rel, ok, detail))
    return results


def main():
    ap = argparse.ArgumentParser(description="Diff two layer-pipeline results trees.")
    ap.add_argument('root_a')
    ap.add_argument('root_b')
    ap.add_argument('--quiet', action='store_true',
                    help='Only print artifacts that differ.')
    args = ap.parse_args()

    # Ordered so the report reads causally: inputs first, then the ranking derived
    # from them, then the generations derived from that. The first failure in this
    # order is the one to debug; later ones are usually downstream of it.
    stages = [
        ("attribution shards", ["**/layers_*.pt"]),
        ("attribution map", ["**/numerator_1_layers.pt"]),
        ("steering cache", ["**/_steering_cache/*.pt"]),
        ("layer selections", ["**/eval/numerator_1_targeted_*.csv",
                              "**/eval/random_random_*.csv"]),
        ("steering metadata", ["**/eval/steering_meta.json"]),
        ("unsteered baseline", ["**/eval/unsteered_*.json"]),
        ("steered generations", ["**/eval/*_gen.json"]),
    ]

    any_diff = False
    any_compared = False
    for label, patterns in stages:
        results = compare_roots(args.root_a, args.root_b, patterns)
        if not results:
            continue
        any_compared = True
        n_ok = sum(1 for _, ok, _ in results if ok is True)
        n_bad = len(results) - n_ok
        status = "MATCH" if n_bad == 0 else "DIFFER"
        print(f"\n[{status}] {label}: {n_ok}/{len(results)} identical")
        for rel, ok, detail in results:
            if ok is True and args.quiet:
                continue
            if ok is not True:
                any_diff = True
                print(f"    DIFF {rel}: {detail}")
            else:
                print(f"    ok   {rel}")
        if n_bad:
            any_diff = True

    # A fingerprint mismatch explains a diff; it is not itself a determinism
    # failure, so it is reported separately and does not set the exit code.
    for rel in ["determinism.json"]:
        for a_path in sorted(glob.glob(os.path.join(args.root_a, "**", rel), recursive=True)):
            b_path = os.path.join(args.root_b, _rel(args.root_a, a_path))
            if not os.path.exists(b_path):
                continue
            with open(a_path) as f:
                fa = json.load(f)
            with open(b_path) as f:
                fb = json.load(f)
            drift = {k: (fa[k], fb.get(k)) for k in fa if fa[k] != fb.get(k)}
            if drift:
                print(f"\n[ENV DRIFT] {_rel(args.root_a, a_path)}")
                for k, (va, vb) in drift.items():
                    print(f"    {k}: {va!r} vs {vb!r}")
                print("    The two runs did not execute under the same conditions, so "
                      "any diff above is explained rather than mysterious.")

    if not any_compared:
        print("No comparable artifacts found. Check the roots are correct and that "
              "both runs actually produced output.")
        sys.exit(2)

    print("\n" + ("FAIL: the two runs diverged." if any_diff
                  else "PASS: every compared artifact is bit-identical."))
    sys.exit(1 if any_diff else 0)


if __name__ == "__main__":
    main()
