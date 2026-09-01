#!/usr/bin/env python3
"""Ceiling, observed, floor -- the three numbers an overlap needs to be readable.

    python map_calibration.py

CPU only. Reads the attribution maps and their per-item shards; no model.

THE PROBLEM THIS SOLVES
-----------------------
"privilege and position share 4 of the top 10 heads" is not interpretable. Two
bounds are missing:

  CEILING  how much two maps of the SAME construct agree. Built by reducing the
           per-item shards in halves -- items 0..n/2 and n/2..n -- and comparing
           the two resulting maps. If a map only agrees with itself at 6/10,
           then 4/10 against a different construct is high, not low.

  FLOOR    how much two sets drawn from the same LAYERS agree. Both maps
           concentrate in mid-to-late layers, so any two sets from that band
           overlap above chance. This is the null that matters; uniform chance
           (0.1 heads at k=10) is not.

The observed number is only meaningful positioned between them. Reporting it
alone is what made the earlier 5-of-10 result uninterpretable.

SIGNED VS ABS, AND WHY BOTH
---------------------------
get_top_k_layer_and_head selects on the SIGNED mean over items, so the signed
sets are the ones actually steered and ablated -- those are reproduced here
exactly, including the tie-breaking. The abs sets answer a different question
(is this head involved at all, in either direction) and are reported alongside,
because the position comparison's whole result is that abs agreement is high
while signed agreement is not.
"""

import os
import re
import glob
import json
import random
import argparse
import statistics

import torch


def reduce_shards(shard_files, n_heads=64):
    """Shards -> (layers, heads) map, matching load_logits' reduction.

    Each shard is one item at o_proj-input width; the head dimension is the sum
    over each head's head_dim components. Getting this wrong produces a
    plausible-looking tensor of the wrong shape, so it is done in one place.
    """
    import einops
    cols = [torch.load(f).squeeze().unsqueeze(-1) for f in shard_files]
    stacked = torch.cat(cols, dim=-1)
    return einops.reduce(stacked, "l (n m) b -> l n b", "sum", n=n_heads)


def shard_files(map_dir):
    fs = glob.glob(os.path.join(map_dir, "heads_*.pt"))
    return sorted(fs, key=lambda f: int(re.search(r"heads_(\d+)\.pt$", f).group(1)))


def top_k_set(patches, k, score="signed"):
    """The top-k heads, reproducing get_top_k_layer_and_head exactly.

    Selection is on the SIGNED mean over items -- not the magnitude -- so these
    are the heads that were actually steered. Ties resolve to the lower flat
    index on any device, as in the original.
    """
    p = patches.to(torch.float32)
    if p.dim() == 3:
        p = p.mean(dim=-1)
    p = p.cpu()
    flat = p.reshape(-1)
    vals = flat.abs() if score == "abs" else flat
    order = sorted(range(flat.numel()), key=lambda i: (-vals[i].item(), i))[:k]
    n_heads = p.shape[1]
    return {(i // n_heads, i % n_heads) for i in order}


def layer_hist(head_set):
    h = {}
    for l, _ in head_set:
        h[l] = h.get(l, 0) + 1
    return h


def matched_null(hist, other_set, n_heads, rng, draws):
    """Overlap distribution for sets drawn from the same layer histogram."""
    out = []
    for _ in range(draws):
        s = set()
        for layer, k in hist.items():
            s.update((layer, h) for h in rng.sample(range(n_heads), min(k, n_heads)))
        out.append(len(s & other_set))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/gpt-oss-20b")
    ap.add_argument("--pair", default="from_user-single_to_dev-single")
    ap.add_argument("--arms", nargs="+", default=["devuser", "sysuser", "sysdev"])
    ap.add_argument("--induction", default="induction_heads.json")
    ap.add_argument("--k", type=int, default=15,
                    help="head-set size; 15 is top-1%% of 1536, the operating "
                         "point the steering and ablation results use")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="map_calibration.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ind_top = {}
    if os.path.exists(args.induction):
        ind = json.load(open(args.induction))
        ind_top = {(r["layer"], r["head"]) for r in ind["ranked"][:args.k]}
    else:
        print(f"note: {args.induction} not found; skipping induction column")

    results = {}
    hdr = (f"{'arm':<9}{'score':<8}{'ceiling':>9}{'vs pos':>8}{'vs ind':>8}"
           f"{'floor':>14}{'p(pos)':>8}")
    print(f"k = {args.k} heads\n")
    print(hdr)
    print("-" * len(hdr))

    for arm in args.arms:
        priv_dir = os.path.join(args.results, f"{arm}__{args.pair}", "atp")
        pos_dir = os.path.join(args.results, f"pos-{arm}__{args.pair}", "atp")
        priv_map = os.path.join(priv_dir, "numerator_1_heads.pt")
        pos_map = os.path.join(pos_dir, "numerator_1_heads.pt")
        if not (os.path.exists(priv_map) and os.path.exists(pos_map)):
            print(f"  {arm}: missing map(s); skipped")
            continue

        priv = torch.load(priv_map)
        pos = torch.load(pos_map)
        if priv.shape != pos.shape:
            print(f"  {arm}: shape mismatch {tuple(priv.shape)} vs "
                  f"{tuple(pos.shape)}; the maps cover different items")
            continue
        n_heads = priv.shape[1]

        # Ceiling: two halves of the SAME contrast, same reduction, same
        # selection. Free -- the per-item shards are already on disk.
        fs = shard_files(priv_dir)
        ceiling = {}
        if len(fs) >= 4:
            mid = len(fs) // 2
            h1 = reduce_shards(fs[:mid], n_heads)
            h2 = reduce_shards(fs[mid:], n_heads)
            for score in ("signed", "abs"):
                ceiling[score] = len(top_k_set(h1, args.k, score)
                                     & top_k_set(h2, args.k, score))
        else:
            print(f"  {arm}: only {len(fs)} shards; cannot split-half")

        for score in ("signed", "abs"):
            a = top_k_set(priv, args.k, score)
            b = top_k_set(pos, args.k, score)
            obs_pos = len(a & b)
            obs_ind = len(a & ind_top) if ind_top else float("nan")

            null = matched_null(layer_hist(a), b, n_heads, rng, args.draws)
            fl = statistics.mean(null)
            fl_hi = sorted(null)[int(0.95 * len(null))]
            p = sum(x >= obs_pos for x in null) / len(null)

            c = ceiling.get(score, float("nan"))
            print(f"{arm:<9}{score:<8}{c:>7}/{args.k}{obs_pos:>6}/{args.k}"
                  f"{obs_ind:>6.0f}/{args.k}"
                  f"{fl:>8.1f} (p95 {fl_hi}){p:>8.3f}")
            results[f"{arm}:{score}"] = {
                "k": args.k, "ceiling_split_half": c,
                "observed_vs_position": obs_pos,
                "observed_vs_induction": obs_ind,
                "matched_null_mean": fl, "matched_null_p95": fl_hi,
                "p_vs_matched_null": p,
                "privilege_layer_hist": layer_hist(a),
            }

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nHOW TO READ\n"
          "  Position the observed number between ceiling and floor. Close to\n"
          "  the ceiling means the two contrasts localize the same thing; close\n"
          "  to the floor means the agreement is explained by depth alone.\n"
          "  A LOW ceiling is itself a finding: it means ATP maps do not\n"
          "  reproduce across halves of this corpus, and no overlap computed\n"
          "  from them can be read as evidence about mechanisms.\n"
          "  signed is the operating set -- what was steered and ablated. abs\n"
          "  asks only whether a head is involved in either direction.")


if __name__ == "__main__":
    main()
