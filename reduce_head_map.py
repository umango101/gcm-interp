#!/usr/bin/env python3
"""Reduce ATP head shards into numerator_1_heads.pt without running an eval pass.

    python reduce_head_map.py \\
        results/gpt-oss-20b/from_user-single_to_dev-single/atp \\
        results/gpt-oss-20b/from_first-single_to_second-single/atp

`--patch_model` writes one shard per item (heads_{i}.pt); the reduced map is only
built when eval/logits_handler.load_logits is called, which normally happens
during an eval run. If all you want is the map -- e.g. to feed
compare_head_maps.py -- a full eval pass is a GPU job to produce a file that
needs no GPU.

THIS MIRRORS logits_handler.load_logits EXACTLY, and that matters: two maps are
only comparable head-for-head if they were reduced identically, so this
reimplements those lines verbatim rather than approximating them.

    cols = [torch.load(shard).squeeze().unsqueeze(-1) for shard in sorted_shards]
    all_logits = torch.cat(cols, dim=-1)                      # [l, n*m, items]
    all_logits = einops.reduce(..., 'l (n m) b -> l n b', 'sum', n=num_heads)

einops is replaced by an equivalent reshape+sum so this has no extra dependency;
the two are verified equal in the einops-available path below.

If a map already exists it is left alone unless --force: silently rewriting one
that an eval run produced would break the provenance the comparison depends on.
"""

import argparse
import glob
import os
import re
import sys


def shard_indices(d, which="heads"):
    """Shard indices present, sorted numerically -- NOT lexicographically.

    heads_10.pt sorts before heads_2.pt as a string, and the reduction
    concatenates along the item axis, so a lexicographic order would silently
    permute items between two maps reduced from different-sized runs.
    """
    out = []
    for path in glob.glob(os.path.join(d, f"{which}_*.pt")):
        m = re.fullmatch(rf"{which}_(\d+)\.pt", os.path.basename(path))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def reduce_dir(d, num_heads, which="heads", force=False, verify_einops=True):
    import torch

    map_path = os.path.join(d, f"numerator_1_{which}.pt")
    if os.path.exists(map_path) and not force:
        print(f"[skip] {map_path} exists (--force to rebuild)")
        return map_path

    idx = shard_indices(d, which)
    if not idx:
        print(f"[fail] no {which}_*.pt shards in {d}", file=sys.stderr)
        return None
    print(f"{d}\n  {len(idx)} shards (indices {idx[0]}..{idx[-1]})")

    cols = []
    for i in idx:
        t = torch.load(os.path.join(d, f"{which}_{i}.pt"), map_location="cpu")
        cols.append(t.squeeze().unsqueeze(-1))
    all_logits = torch.cat(cols, dim=-1)
    print(f"  concatenated: {tuple(all_logits.shape)}  [layers, heads*head_dim, items]")

    l, nm, b = all_logits.shape
    if nm % num_heads:
        print(f"[fail] dim1={nm} is not divisible by --num_heads {num_heads}. "
              f"Pass the model's real head count.", file=sys.stderr)
        return None
    head_dim = nm // num_heads
    print(f"  {l} layers x {num_heads} heads x head_dim {head_dim}")

    reduced = all_logits.reshape(l, num_heads, head_dim, b).sum(dim=2)

    if verify_einops:
        try:
            import einops
            ref = einops.reduce(all_logits, "l (n m) b -> l n b", "sum", n=num_heads)
            assert torch.equal(reduced, ref), "reshape+sum disagrees with einops"
            print("  reduction matches einops reference")
        except ImportError:
            print("  (einops not installed; reshape+sum used, unverified)")

    print(f"  reduced: {tuple(reduced.shape)}  [layers, heads, items]")
    torch.save(reduced, map_path)
    print(f"  wrote {map_path}\n")
    return map_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+",
                    help="directories holding heads_*.pt (the .../atp dirs)")
    ap.add_argument("--num_heads", type=int, default=64,
                    help="attention heads per layer. gpt-oss-20b is 64. The printed "
                         "head_dim is your check that this is right.")
    ap.add_argument("--which", default="heads")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a map exists. Off by default: silently "
                         "replacing a map an eval run produced would break the "
                         "provenance the head comparison relies on.")
    args = ap.parse_args()

    ok = True
    for d in args.dirs:
        if reduce_dir(d, args.num_heads, args.which, args.force) is None:
            ok = False
    if not ok:
        return 1
    print("Both maps must come from the same reducer to be comparable -- if one was "
          "built by an eval run and the other by this script, rebuild both here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
