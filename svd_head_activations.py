#!/usr/bin/env python
"""
svd_head_activations.py -- the activation-weighted counterpart to
svd_head_subspaces.py.

The static version asks whether the long-form and single-token head sets *can*
write into the same subspace, using col(W_O^h).  This one asks whether they
actually *do*, by replacing each head's weight block with the contributions it
produced on your minimally contrastive data.

Construction
------------
For head (l, h), cache_head_writes.py gives z_h(x) for each example x on both
sides of the contrast.  The head's contribution to the residual stream is

    c_h(x) = z_h(x) @ W_O^{l,h}.T          in R^{d_model}

and the quantity ATP linearizes is the difference between the two sides:

    dc_h(x) = (z_h^src(x) - z_h^base(x)) @ W_O^{l,h}.T

Stacking those over examples gives a d_model x n_examples block per head, and
concatenating blocks over a head set gives M -- the same object the static
script builds, but with columns that are realised writes instead of weight
columns.  Everything downstream (spectra, principal angles, Grassmann overlap,
rank sharing, nulls) is imported from svd_head_subspaces and unchanged.

Why this is the stronger test
-----------------------------
col(W_O^h) has dimension head_dim regardless of whether the head ever uses all
of it.  Two heads can share a subspace neither exercises, and the static metrics
would score that as sharing.  Here a direction only enters M if some example
actually pushed along it, so rank reflects use rather than capacity -- and the
spectra are genuinely low-rank rather than saturating.

--contrast controls what the columns are:
    diff  (default)  z_src - z_base, matched pairwise by example index.  This is
                     the behavior-relevant signal, and the object ATP scores.
    src / base       raw writes on one side, for a sanity comparison.  Expect
                     these to be dominated by task-general structure shared by
                     every head, which is why diff is the default.

Usage
-----
  # 1. cache both formats first (GPU)
  python cache_head_writes.py --model-id Qwen/Qwen1.5-14B-Chat \\
      --task female-long   --source female-long   --base male-long   --out cache/writes
  python cache_head_writes.py --model-id Qwen/Qwen1.5-14B-Chat \\
      --task female-single --source female-single --base male-single --out cache/writes

  # 2. compare (CPU)
  python svd_head_activations.py --model-id Qwen/Qwen1.5-14B-Chat \\
      --long-task female-long --single-task female-single \\
      --cache-root cache/writes --topk 0.005 0.01 0.02 --out results/svd_act
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import svd_head_subspaces as base


POOL_HELP = """
--pool selects how the cached [n_layers, B, S, H] tensor becomes the rows of M.
S is the number of trailing prompt positions kept by cache_head_writes.py
(--n-positions); B is the number of prompt pairs.

  last      z[:, :, -1, :]  ->  (B, H).  One row per prompt at the final token,
            the position steering intervenes at. Rank ceiling head_dim.
  tokens    reshape         ->  (B*S, H). Every kept position from every prompt
            pooled. The widest view of what the head touches. Rank ceiling
            head_dim.
  mean      mean over B     ->  (S, H).  The component consistent across the
            dataset; prompt-specific variation averages out like 1/sqrt(B).
            Rank ceiling S. At S=1 with --contrast diff this IS the steering
            vector, so the comparison degenerates to a cosine.
  residual  x - mean_B(x)   ->  (B*S, H). What 'tokens' has that 'mean' does
            not. Scatter(tokens) = B*Scatter(mean) + Scatter(residual), so
            running mean and residual together splits the pooled result into
            dataset-consistent and prompt-specific parts.
"""


def pool_writes(z: torch.Tensor, pool: str) -> torch.Tensor:
    """[n_layers, B, S, H] -> [n_layers, N, H] under the chosen pooling."""
    if z.dim() == 3:
        if pool not in ("last", "tokens"):
            raise SystemExit(
                f"--pool {pool} needs a cache with a position axis. Re-run "
                f"cache_head_writes.py with --n-positions K (e.g. 64).")
        return z
    L, B, S, H = z.shape
    if pool == "last":
        return z[:, :, -1, :]
    if pool == "tokens":
        return z.reshape(L, B * S, H)
    if pool == "mean":
        return z.mean(dim=1)
    if pool == "residual":
        return (z - z.mean(dim=1, keepdim=True)).reshape(L, B * S, H)
    raise ValueError(pool)


def load_writes(cache_root: str, model_name: str, task: str, split: str,
                contrast: str) -> torch.Tensor:
    """Return per-head activations as [n_layers, B, S, H] (or [n_layers, N, H])."""
    d = os.path.join(cache_root, model_name, task)
    src_p = os.path.join(d, f"source_{split}.pt")
    base_p = os.path.join(d, f"base_{split}.pt")
    for p in (src_p, base_p):
        if not os.path.exists(p):
            raise SystemExit(f"Missing cache {p}. Run cache_head_writes.py for {task} first.")
    if contrast == "src":
        return torch.load(src_p, map_location="cpu").to(torch.float32)
    if contrast == "base":
        return torch.load(base_p, map_location="cpu").to(torch.float32)
    zs = torch.load(src_p, map_location="cpu").to(torch.float32)
    zb = torch.load(base_p, map_location="cpu").to(torch.float32)
    if zs.shape != zb.shape:
        n = min(zs.shape[1], zb.shape[1])
        # (axis 1 is the prompt axis in both the 3-D and 4-D cache layouts)
        print(f"[warn] {task}: source has {zs.shape[1]} rows, base has {zb.shape[1]}; "
              f"truncating both to {n}. Pairwise matching assumes equal-length, "
              f"index-aligned datasets.", file=sys.stderr)
        zs, zb = zs[:, :n], zb[:, :n]
    return zs - zb


def build_matrix_act(z: torch.Tensor, o_proj: torch.Tensor,
                     heads: List[Tuple[int, int]], n_heads: int,
                     normalize: bool = True, dtype=None) -> torch.Tensor:
    """Concatenate realised per-head writes into M of shape [d_model, n * n_examples].

    z         [n_layers, n_examples, n_heads*head_dim]  cached activations
    o_proj    [n_layers, d_model, n_heads*head_dim]     weights
    """
    dtype = dtype or base.DTYPE
    cols = []
    for (l, h) in heads:
        hd = o_proj.shape[2] // n_heads
        sl = slice(h * hd, (h + 1) * hd)
        W = o_proj[l][:, sl].to(dtype)          # [d_model, head_dim]
        dz = z[l][:, sl].to(dtype)              # [n_examples, head_dim]
        C = (dz @ W.T).T                        # [d_model, n_examples]
        if normalize:
            nrm = torch.linalg.norm(C)
            if nrm > 0:
                C = C / nrm
        cols.append(C)
    return torch.cat(cols, dim=1)


def run_task(long_task: str, single_task: str, long_attr: str, single_attr: str,
             z_long: torch.Tensor, z_single: torch.Tensor, o_proj: torch.Tensor,
             n_heads: int, args) -> List[Dict]:
    attr_l = torch.load(long_attr, map_location="cpu")
    attr_s = torch.load(single_attr, map_location="cpu")
    n_layers, d_model, in_dim = o_proj.shape
    ambient = d_model
    rows = []

    for name, z, task in (("long", z_long, long_task), ("single", z_single, single_task)):
        if z.shape[0] != n_layers or z.shape[2] != in_dim:
            raise SystemExit(
                f"{task}: cached activations are {tuple(z.shape)}, incompatible with "
                f"o_proj {tuple(o_proj.shape)}. Cache and weights must be the same model.")

    for topk in args.topk:
        df_l = base.get_top_k_layer_and_head(attr_l, topk, args.algo, args.abs)
        df_s = base.get_top_k_layer_and_head(attr_s, topk, args.algo, args.abs)
        heads_l = [(int(r.layer), int(r.head)) for r in df_l.itertuples()]
        heads_s = [(int(r.layer), int(r.head)) for r in df_s.itertuples()]
        shared = set(heads_l) & set(heads_s)
        jac = len(shared) / max(len(set(heads_l) | set(heads_s)), 1)

        cols_per_head_l, cols_per_head_s = z_long.shape[1], z_single.shape[1]
        if len(heads_l) * cols_per_head_l >= ambient:
            print(f"[note] {long_task} top_k={topk}: {len(heads_l)} heads x "
                  f"{cols_per_head_l} examples >= d_model {ambient}. Unlike the static "
                  f"version this is not automatic saturation -- each block has rank at "
                  f"most head_dim and usually far less -- but check rank_a/rank_b "
                  f"against the ambient dim before reading the overlap.", file=sys.stderr)

        variants = {"all": (heads_l, heads_s)}
        if args.disjoint and shared:
            variants["disjoint"] = ([h for h in heads_l if h not in shared],
                                    [h for h in heads_s if h not in shared])

        for variant, (hl, hs) in variants.items():
            if not hl or not hs:
                continue
            Ml = build_matrix_act(z_long, o_proj, hl, n_heads, args.normalize_heads)
            Ms = build_matrix_act(z_single, o_proj, hs, n_heads, args.normalize_heads)
            res = base.compare(Ml, Ms, args.energy, args.fixed_rank)

            null_rows = {"uniform": [], "layer_matched": []}
            for seed in range(args.n_null):
                r = np.random.default_rng(args.seed + seed)
                rl = base.random_heads(n_layers, n_heads, len(hl), r)
                rs = base.random_heads(n_layers, n_heads, len(hs), r)
                null_rows["uniform"].append(base.compare(
                    build_matrix_act(z_long, o_proj, rl, n_heads, args.normalize_heads),
                    build_matrix_act(z_single, o_proj, rs, n_heads, args.normalize_heads),
                    args.energy, args.fixed_rank))
                ll = base.layer_matched_heads(hl, n_heads, r)
                ls = base.layer_matched_heads(hs, n_heads, r)
                null_rows["layer_matched"].append(base.compare(
                    build_matrix_act(z_long, o_proj, ll, n_heads, args.normalize_heads),
                    build_matrix_act(z_single, o_proj, ls, n_heads, args.normalize_heads),
                    args.energy, args.fixed_rank))

            row = {
                "task": f"{long_task}|{single_task}", "contrast": args.contrast,
                "pool": args.pool, "variant": variant, "top_k": topk,
                "n_heads_long": len(hl), "n_heads_single": len(hs),
                "head_jaccard": jac, "n_shared_heads": len(shared),
                "ambient_dim": ambient,
                "n_examples_long": cols_per_head_l, "n_examples_single": cols_per_head_s,
                "eff_rank_long": res["_stats_a"]["effective_rank"],
                "eff_rank_single": res["_stats_b"]["effective_rank"],
                "stable_rank_long": res["_stats_a"]["stable_rank"],
                "stable_rank_single": res["_stats_b"]["stable_rank"],
            }
            for k in ("rank_a", "rank_b", "rank_joint", "rank_sharing",
                      "grassmann_overlap", "energy_b_in_a", "energy_a_in_b",
                      "mean_principal_angle_deg", "n_angles_cos_gt_0.9",
                      "n_angles_cos_gt_0.99"):
                row[k] = res[k]
            for null_name, entries in null_rows.items():
                if not entries:
                    continue
                for metric in ("grassmann_overlap", "rank_sharing", "energy_b_in_a"):
                    vals = np.array([e[metric] for e in entries], dtype=float)
                    sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    row[f"null_{null_name}_{metric}_mean"] = float(vals.mean())
                    row[f"null_{null_name}_{metric}_std"] = float(sd)
                    row[f"null_{null_name}_{metric}_z"] = (
                        float((row[metric] - vals.mean()) / sd) if sd > 0 else float("nan"))
                    row[f"null_{null_name}_{metric}_pct"] = float((vals < row[metric]).mean())
            rows.append(row)

            if args.plots and variant == "all":
                nulls = {k: v[0] for k, v in null_rows.items() if v}
                base.make_plots(os.path.join(args.out, "plots"),
                                f"{long_task}_{args.pool}", topk, res, nulls, "head")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--o-proj-file", default=None)
    p.add_argument("--cache-root", default="./cache/writes")
    p.add_argument("--results-root", default="./results")
    p.add_argument("--algo", default="atp")
    p.add_argument("--long-task", required=True, help="e.g. female-long")
    p.add_argument("--single-task", required=True, help="e.g. female-single")
    p.add_argument("--long-source", default=None,
                   help="Defaults to --long-task (results dir from_{src}_to_{base}).")
    p.add_argument("--long-base", required=True, help="e.g. male-long")
    p.add_argument("--single-source", default=None)
    p.add_argument("--single-base", required=True, help="e.g. male-single")
    p.add_argument("--split", default="desired", choices=["desired", "undesired"])
    p.add_argument("--contrast", default="diff", choices=["diff", "src", "base"])
    p.add_argument("--pool", default="last",
                   choices=["last", "tokens", "mean", "residual"],
                   help=POOL_HELP)
    p.add_argument("--topk", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.03])
    p.add_argument("--energy", type=float, default=0.99)
    p.add_argument("--fixed-rank", type=int, default=None)
    p.add_argument("--normalize-heads", action="store_true", default=True,
                   help="Unit-Frobenius each head's write block. Turn off to let "
                        "heads that write more count for more.")
    p.add_argument("--no-normalize-heads", dest="normalize_heads", action="store_false")
    p.add_argument("--abs", action="store_true")
    p.add_argument("--disjoint", action="store_true", default=True)
    p.add_argument("--no-disjoint", dest="disjoint", action="store_false")
    p.add_argument("--n-null", type=int, default=5)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", action="store_true", default=True)
    p.add_argument("--no-plots", dest="plots", action="store_false")
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--out", default="./results/svd_activations")
    args = p.parse_args()

    base.DTYPE = torch.float64 if args.dtype == "float64" else torch.float32
    if args.threads:
        torch.set_num_threads(args.threads)
    args.long_source = args.long_source or args.long_task
    args.single_source = args.single_source or args.single_task

    model_name = args.model_id.split("/")[-1]
    os.makedirs(args.out, exist_ok=True)

    if args.o_proj_file:
        o_proj = torch.load(args.o_proj_file, map_location="cpu").to(torch.float32)
    else:
        print(f"Reading o_proj weights for {args.model_id} ...")
        o_proj = base.load_o_proj(args.model_id, local_dir=args.model_dir)

    n_heads = args.num_heads
    if n_heads is None:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(args.model_id)
            cfg = getattr(cfg, "text_config", cfg)
            n_heads = cfg.num_attention_heads
        except Exception as e:
            raise SystemExit(f"Could not read num_attention_heads ({e}); pass --num-heads.")

    n_layers, d_model, in_dim = o_proj.shape
    print(f"model: {n_layers} layers, d_model={d_model}, n_heads={n_heads}, "
          f"head_dim={in_dim // n_heads}")

    long_attr = os.path.join(args.results_root, model_name,
                             f"from_{args.long_source}_to_{args.long_base}",
                             args.algo, "numerator_1_heads.pt")
    single_attr = os.path.join(args.results_root, model_name,
                               f"from_{args.single_source}_to_{args.single_base}",
                               args.algo, "numerator_1_heads.pt")
    for pth in (long_attr, single_attr):
        if not os.path.exists(pth):
            raise SystemExit(f"Missing attribution map: {pth}")

    z_long_raw = load_writes(args.cache_root, model_name, args.long_task, args.split, args.contrast)
    z_single_raw = load_writes(args.cache_root, model_name, args.single_task, args.split, args.contrast)
    print(f"cached: long {tuple(z_long_raw.shape)}, single {tuple(z_single_raw.shape)} "
          f"(contrast={args.contrast})")
    z_long = pool_writes(z_long_raw, args.pool)
    z_single = pool_writes(z_single_raw, args.pool)
    del z_long_raw, z_single_raw
    print(f"pool={args.pool}: long {tuple(z_long.shape)}, single {tuple(z_single.shape)} "
          f"-> {z_long.shape[1]} columns per head")
    if args.pool == "mean":
        print(f"note: --pool mean caps each head's block at rank {z_long.shape[1]}; "
              f"at 1 this is a steering-vector cosine, not a subspace comparison.")

    rows = run_task(args.long_task, args.single_task, long_attr, single_attr,
                    z_long, z_single, o_proj, n_heads, args)

    df = pd.DataFrame(rows)
    csv = os.path.join(args.out,
                       f"svd_activations_{model_name}_{args.long_task}_{args.pool}.csv")
    df.to_csv(csv, index=False)
    print(f"\nWrote {csv}")

    cols = ["pool", "variant", "top_k", "n_heads_long", "head_jaccard",
            "eff_rank_long", "eff_rank_single", "rank_a", "rank_b", "rank_joint",
            "rank_sharing", "grassmann_overlap",
            "null_layer_matched_grassmann_overlap_mean",
            "null_layer_matched_grassmann_overlap_z"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
