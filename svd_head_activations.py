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


def _column_selection(n_cols: int, max_columns: int, seed: int):
    if not max_columns or n_cols <= max_columns:
        return None
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_cols, generator=g)[:max_columns].sort().values


def load_and_pool(cache_root: str, model_name: str, task: str, split: str,
                  contrast: str, pool: str, max_columns: int, seed: int) -> torch.Tensor:
    """Read the cache, difference it, pool it, and subsample -- one layer at a time.

    The cache is [n_layers, B, S, H] in float16 (legacy 3-D caches are read as
    S=1).  At B=1600, S=64, H=5120 that is ~42 GB per side on disk, so it is
    memory-mapped and each layer is converted, differenced, pooled and
    subsampled before the next is touched.  Peak host memory is two layers in
    float32, not two whole caches.
    """
    d = os.path.join(cache_root, model_name, task)
    src_p = os.path.join(d, f"source_{split}.pt")
    base_p = os.path.join(d, f"base_{split}.pt")
    for pth in (src_p, base_p):
        if not os.path.exists(pth):
            raise SystemExit(f"Missing cache {pth}. Run cache_head_writes.py for {task} first.")
    try:
        zs = torch.load(src_p, map_location="cpu", mmap=True)
        zb = torch.load(base_p, map_location="cpu", mmap=True)
    except Exception:
        zs = torch.load(src_p, map_location="cpu")
        zb = torch.load(base_p, map_location="cpu")

    if zs.dim() == 3:
        zs, zb = zs.unsqueeze(2), zb.unsqueeze(2)
    L, B, S, H = zs.shape
    if zb.shape[1] != B:
        B = min(B, zb.shape[1])
        print(f"[warn] {task}: prompt counts differ between sides; truncating to {B} "
              f"to keep the pairing.", file=sys.stderr)
    if pool in ("last", "mean") and S == 1 and pool == "mean":
        print(f"[warn] {task}: cache has one position per prompt, so --pool mean "
              f"yields a single column per head -- that is the steering vector, not a "
              f"subspace. Re-cache with --n-positions 64 for a meaningful mean view.",
              file=sys.stderr)

    n_cols = {"last": B, "tokens": B * S, "mean": S, "residual": B * S}[pool]
    sel = _column_selection(n_cols, max_columns, seed)

    out = []
    for l in range(L):
        a = zs[l, :B].to(torch.float32)
        b = zb[l, :B].to(torch.float32)
        x = a - b if contrast == "diff" else (a if contrast == "src" else b)
        del a, b
        if pool == "last":
            p = x[:, -1, :]
        elif pool == "tokens":
            p = x.reshape(B * S, H)
        elif pool == "mean":
            p = x.mean(dim=0)
        elif pool == "residual":
            p = (x - x.mean(dim=0, keepdim=True)).reshape(B * S, H)
        else:
            raise ValueError(pool)
        del x
        out.append(p[sel] if sel is not None else p.clone())
    return torch.stack(out)


def compress_head_blocks(z: torch.Tensor, o_proj: torch.Tensor, n_heads: int,
                         energy: float = 1.0, device: str = "cpu",
                         dtype=torch.float32) -> torch.Tensor:
    """Replace each head's write block by an exactly equivalent narrower one.

    Head (l, h) contributes C = (dz @ W.T).T, shape [d_model, n_cols].  Its rank
    is at most head_dim no matter how many examples n_cols holds, so most of
    those columns are redundant: with the small Gram K = C.T C = V diag(s^2) V.T,
    the matrix B = C V has the same Gram (B B.T = C C.T), hence the same
    singular values and the same column space, but only rank(C) <= head_dim
    columns.  Every metric downstream depends on C only through C C.T, so this
    is exact, not an approximation.

    With --max-columns 512 and head_dim 128 that is a 4x reduction in every
    matmul afterwards; the whole precompute costs one 512x512 eigh per head.

    Returns [n_layers, n_heads, d_model, r_max], zero-padded (zero columns
    change neither the Gram nor the column space).
    """
    n_layers, d_model, in_dim = o_proj.shape
    hd = in_dim // n_heads
    n_cols = z.shape[1]
    r_max = min(hd, n_cols)
    out = torch.zeros(n_layers, n_heads, d_model, r_max, dtype=torch.float32)
    dev = torch.device(device)
    ranks = []
    for l in range(n_layers):
        W_l = o_proj[l].to(device=dev, dtype=dtype)          # [d_model, in_dim]
        z_l = z[l].to(device=dev, dtype=dtype)               # [n_cols, in_dim]
        for h in range(n_heads):
            sl = slice(h * hd, (h + 1) * hd)
            C = (z_l[:, sl] @ W_l[:, sl].T).T                # [d_model, n_cols]
            K = C.T @ C                                      # [n_cols, n_cols]
            K = 0.5 * (K + K.T)
            evals, evecs = torch.linalg.eigh(K)
            evals = torch.flip(evals, dims=[0]).clamp_min(0)
            evecs = torch.flip(evecs, dims=[1])
            if energy < 1.0:
                cum = torch.cumsum(evals, 0) / evals.sum().clamp_min(1e-30)
                r = int(torch.searchsorted(
                    cum, torch.tensor(energy, dtype=cum.dtype, device=cum.device)
                ).item()) + 1
            else:
                tol = evals[0] * n_cols * 1e-12
                r = int((evals > tol).sum().item())
            r = max(1, min(r, r_max))
            ranks.append(r)
            out[l, h, :, :r] = (C @ evecs[:, :r]).to("cpu", torch.float32)
        del W_l, z_l
    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    print(f"compressed head blocks: {n_cols} -> {r_max} columns "
          f"(actual rank median {int(ranks_t.median())}, max {int(ranks_t.max())})")
    return out


def build_matrix_compressed(blocks: torch.Tensor, heads: List[Tuple[int, int]],
                            normalize: bool = True, dtype=None) -> torch.Tensor:
    dtype = dtype or base.DTYPE
    cols = []
    for (l, h) in heads:
        B = blocks[l, h].to(device=base.DEVICE, dtype=dtype)
        if normalize:
            nrm = torch.linalg.norm(B)
            if nrm > 0:
                B = B / nrm
        cols.append(B)
    return torch.cat(cols, dim=1)


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
             n_heads: int, args, blocks_long: torch.Tensor,
             blocks_single: torch.Tensor) -> List[Dict]:
    attr_l = torch.load(long_attr, map_location="cpu")
    attr_s = torch.load(single_attr, map_location="cpu")
    n_layers, d_model, in_dim = o_proj.shape
    ambient = d_model
    rows = []

    for topk in args.topk:
        df_l = base.get_top_k_layer_and_head(attr_l, topk, args.algo, args.abs)
        df_s = base.get_top_k_layer_and_head(attr_s, topk, args.algo, args.abs)
        heads_l = [(int(r.layer), int(r.head)) for r in df_l.itertuples()]
        heads_s = [(int(r.layer), int(r.head)) for r in df_s.itertuples()]
        shared = set(heads_l) & set(heads_s)
        jac = len(shared) / max(len(set(heads_l) | set(heads_s)), 1)

        cols_per_head_l = blocks_long.shape[3]
        cols_per_head_s = blocks_single.shape[3]
        head_dim = in_dim // n_heads
        # A head block is [d_model, n_cols] but has rank at most head_dim, so the
        # column count is NOT the bound that matters -- min(head_dim, n_cols) is.
        rank_cap_l = len(heads_l) * min(head_dim, cols_per_head_l)
        rank_cap_s = len(heads_s) * min(head_dim, cols_per_head_s)
        if min(rank_cap_l, rank_cap_s) >= ambient:
            print(f"[warn] {long_task} top_k={topk}: {len(heads_l)} heads x "
                  f"min(head_dim {head_dim}, {cols_per_head_l} cols) = {rank_cap_l} "
                  f">= d_model {ambient}; the head sets can span the whole residual "
                  f"stream, same ceiling as the static version ({ambient // head_dim} "
                  f"heads). Whether they actually do is an empirical question -- read "
                  f"rank_a/rank_b: near {ambient} means saturated and the overlap is "
                  f"uninformative; well below means the note is moot.", file=sys.stderr)
        if len(heads_l) * cols_per_head_l >= 50000:
            print(f"[note] {long_task} top_k={topk}: {len(heads_l) * cols_per_head_l} "
                  f"columns per side. Cost and memory grow linearly in this; "
                  f"--max-columns subsamples examples without changing the subspace "
                  f"(each block is at most {head_dim}-dimensional).", file=sys.stderr)

        variants = {"all": (heads_l, heads_s)}
        if args.disjoint and shared:
            variants["disjoint"] = ([h for h in heads_l if h not in shared],
                                    [h for h in heads_s if h not in shared])

        for variant, (hl, hs) in variants.items():
            if not hl or not hs:
                continue
            Ml = build_matrix_compressed(blocks_long, hl, args.normalize_heads)
            Ms = build_matrix_compressed(blocks_single, hs, args.normalize_heads)
            res = base.compare(Ml, Ms, args.energy, args.fixed_rank)

            null_rows = {"uniform": [], "layer_matched": []}
            for seed in range(args.n_null):
                r = np.random.default_rng(args.seed + seed)
                rl = base.random_heads(n_layers, n_heads, len(hl), r)
                rs = base.random_heads(n_layers, n_heads, len(hs), r)
                null_rows["uniform"].append(base.compare(
                    build_matrix_compressed(blocks_long, rl, args.normalize_heads),
                    build_matrix_compressed(blocks_single, rs, args.normalize_heads),
                    args.energy, args.fixed_rank))
                ll = base.layer_matched_heads(hl, n_heads, r)
                ls = base.layer_matched_heads(hs, n_heads, r)
                null_rows["layer_matched"].append(base.compare(
                    build_matrix_compressed(blocks_long, ll, args.normalize_heads),
                    build_matrix_compressed(blocks_single, ls, args.normalize_heads),
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
    p.add_argument("--max-columns", type=int, default=None,
                   help="Randomly subsample this many columns (examples/positions) per "
                        "head. Each head block has rank at most head_dim, so a few "
                        "hundred columns already determine the subspace; the rest only "
                        "cost time and memory. 4x head_dim is a reasonable setting.")
    p.add_argument("--n-null", type=int, default=5)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda. On an L40S pair with --dtype float32; its "
                        "float64 rate is ~1/64 of float32. H100/H200/A100 do float64 "
                        "well.")
    p.add_argument("--compress-energy", type=float, default=1.0,
                   help="1.0 keeps every nonzero direction (exact). Lower (e.g. 0.999) "
                        "truncates each head block further, cheaper but lossy.")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", action="store_true", default=True)
    p.add_argument("--no-plots", dest="plots", action="store_false")
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--out", default="./results/svd_activations")
    args = p.parse_args()

    base.DTYPE = torch.float64 if args.dtype == "float64" else torch.float32
    base.DEVICE = torch.device(args.device)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but no CUDA device is visible.")
        name = torch.cuda.get_device_name(0)
        print(f"device: {name}")
        if "L40S" in name and args.dtype == "float64":
            print("warn: L40S float64 is ~1/64 of its float32 rate and will be slower "
                  "than a CPU node. Use --dtype float32, or request -G h100:1.")
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

    z_long = load_and_pool(args.cache_root, model_name, args.long_task, args.split,
                           args.contrast, args.pool, args.max_columns, args.seed)
    z_single = load_and_pool(args.cache_root, model_name, args.single_task, args.split,
                             args.contrast, args.pool, args.max_columns, args.seed + 1)
    print(f"pool={args.pool}, contrast={args.contrast}: long {tuple(z_long.shape)}, "
          f"single {tuple(z_single.shape)} -> {z_long.shape[1]} columns per head")

    print("compressing per-head blocks (exact; see compress_head_blocks docstring)")
    blocks_long = compress_head_blocks(z_long, o_proj, n_heads, args.compress_energy,
                                       args.device)
    blocks_single = compress_head_blocks(z_single, o_proj, n_heads, args.compress_energy,
                                         args.device)
    del z_long, z_single

    rows = run_task(args.long_task, args.single_task, long_attr, single_attr,
                    None, None, o_proj, n_heads, args, blocks_long, blocks_single)

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
