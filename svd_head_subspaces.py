#!/usr/bin/env python
"""
svd_head_subspaces.py -- do the long-form and single-token localized heads write
into the same subspace of the residual stream?

For a task (e.g. female), this loads the two ATP attribution maps produced by
run.py --patch_model:

    {results_root}/{model}/from_{src}-long_to_{base}-long/{algo}/numerator_1_heads.pt
    {results_root}/{model}/from_{src}-single_to_{base}-single/{algo}/numerator_1_heads.pt

selects the top-k heads from each exactly the way eval/logits_handler.py does,
pulls the corresponding blocks out of each layer's o_proj weight, and asks how
much the two sets of blocks overlap spectrally.

Head -> weight block
--------------------
o_proj.weight has shape [d_model, n_heads * head_dim] (nn.Linear stores
[out, in]).  Head h's write matrix is the COLUMN block

    W_O^{l,h} = o_proj.weight[:, h*head_dim : (h+1)*head_dim]      (--space head)

and col(W_O^{l,h}) is the <=head_dim-dimensional subspace of the residual stream
that head h can write into.  This is the default and is what "the o_proj of head
h" normally means.

Your localization pipeline, however, indexes o_proj.OUTPUT
(patching_utils.get_activations -> layer.self_attn.o_proj.output, then
einops.reduce 'l (n m) b -> l n b'), so its index i selects residual-stream
coordinates [i*head_dim, (i+1)*head_dim) rather than attention head i.  If you
want the analysis to match that indexing literally, pass --space block, which
uses the ROW block o_proj.weight[i*head_dim : (i+1)*head_dim, :] instead.  Run
both; if they disagree, the head-indexing question has to be settled first.

Metrics
-------
Per set S of heads, M_S = [ W^{l,h} for (l,h) in S ] concatenated along columns,
so col(M_S) is the joint write subspace.  Left singular values come from an
eigendecomposition of the d_model x d_model Gram matrix, which is much cheaper
than a full SVD when |S| * head_dim > d_model.

  effective_rank    exp(H(s / sum s))            -- Roy & Vetterli
  stable_rank       ||M||_F^2 / ||M||_2^2
  rank_99           smallest r with cumulative s^2 >= 99% of total
  grassmann_overlap ||Q_a^T Q_b||_F^2 / min(r_a, r_b)  in [0, 1]
  energy_b_in_a     ||Q_a Q_a^T M_b||_F^2 / ||M_b||_F^2, and the reverse
  rank_sharing      (r_a + r_b - r_joint) / min(r_a, r_b), 1 = nested, 0 = orthogonal
  principal angles  singular values of Q_a^T Q_b

Controls (all on by default -- the raw numbers are close to meaningless alone)
-----------------------------------------------------------------------------
  1. Head-identity overlap.  If the two top-k sets literally contain the same
     heads, subspace overlap is trivial.  Jaccard is reported, and every metric
     is recomputed on the SHARED-HEADS-REMOVED sets ("disjoint" rows).
  2. Random-head null, matched on set size, and a second null matched on the
     per-layer head counts of the real sets (heads in one layer are column
     blocks of one matrix, so layer concentration alone inflates overlap).
  3. Saturation guard.  |S| * head_dim >= d_model means the set spans the whole
     residual stream and every overlap is 1 by construction.  The script warns
     and reports the largest top_k that stays below saturation.

Usage
-----
  python svd_head_subspaces.py --model-id google/gemma-3-12b-it --task female \
      --topk 0.005 0.01 0.02 0.03 --out results/svd

  # cache the o_proj weights once, then iterate without touching HF
  python svd_head_subspaces.py --model-id google/gemma-3-12b-it \
      --dump-o-proj cache/gemma-3-12b-it_oproj.pt
  python svd_head_subspaces.py --model-id google/gemma-3-12b-it \
      --o-proj-file cache/gemma-3-12b-it_oproj.pt --task all
"""

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

DTYPE = torch.float64  # set from --dtype in main()

# ---------------------------------------------------------------------------
# head selection -- kept bit-identical to eval/logits_handler.py
# ---------------------------------------------------------------------------

def get_top_k_layer_and_head(patches, top_k, patch_algo, use_abs=False):
    """Reimplementation of eval.logits_handler.get_top_k_layer_and_head.

    Kept local so this script runs without importing the repo (which pulls in
    nnsight).  `use_abs` is an addition: the pipeline ranks on signed ATP
    effects, but for a subspace question the sign of the effect is often not
    what you care about.  Default False == pipeline behaviour.
    """
    if isinstance(patches, str):
        patches = torch.load(patches, map_location="cpu")
    patches = patches.to(torch.float32)
    if patch_algo != "probes":
        patches = patches.mean(dim=-1)
    flat = patches.abs().view(-1) if use_abs else patches.view(-1)
    k = int(top_k * flat.numel())
    k = max(k, 1)
    top_values, top_indices = flat.topk(k=k)
    layer_indices = top_indices // patches.shape[1]
    head_indices = top_indices % patches.shape[1]
    df = pd.DataFrame({
        "layer": layer_indices.numpy(),
        "head": head_indices.numpy(),
        "value": top_values.numpy(),
    })
    return df.sort_values(by=["layer", "head"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# o_proj weights
# ---------------------------------------------------------------------------

_OPROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj\.weight$")


def load_o_proj(model_id: str, revision: Optional[str] = None,
                local_dir: Optional[str] = None) -> torch.Tensor:
    """Return o_proj weights stacked as [n_layers, d_model, n_heads*head_dim].

    Reads tensors straight out of the safetensors shards, so nothing is
    instantiated on GPU and a 12B model costs a few hundred MB of host RAM.
    Works for nested text configs (Gemma-3) because the key regex only anchors
    on `layers.{i}.self_attn.o_proj.weight`.
    """
    from safetensors import safe_open

    if local_dir and os.path.isdir(local_dir):
        root = local_dir
        files = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
        shard_paths = {f: os.path.join(root, f) for f in files}
        key_to_file = {}
        for f, p in shard_paths.items():
            with safe_open(p, framework="pt") as sf:
                for k in sf.keys():
                    key_to_file[k] = p
    else:
        from huggingface_hub import hf_hub_download
        try:
            idx_path = hf_hub_download(model_id, "model.safetensors.index.json",
                                       revision=revision)
            with open(idx_path) as fh:
                weight_map = json.load(fh)["weight_map"]
            key_to_file = {}
            wanted_files = {v for k, v in weight_map.items() if _OPROJ_RE.search(k)}
            local = {f: hf_hub_download(model_id, f, revision=revision)
                     for f in sorted(wanted_files)}
            for k, v in weight_map.items():
                if v in local:
                    key_to_file[k] = local[v]
        except Exception:
            p = hf_hub_download(model_id, "model.safetensors", revision=revision)
            with safe_open(p, framework="pt") as sf:
                key_to_file = {k: p for k in sf.keys()}

    layers: Dict[int, torch.Tensor] = {}
    open_files: Dict[str, object] = {}
    try:
        for key, path in key_to_file.items():
            m = _OPROJ_RE.search(key)
            if not m or "vision" in key:
                continue
            if path not in open_files:
                open_files[path] = safe_open(path, framework="pt").__enter__()
            layers[int(m.group(1))] = open_files[path].get_tensor(key).to(torch.float32)
    finally:
        for f in open_files.values():
            try:
                f.__exit__(None, None, None)
            except Exception:
                pass

    if not layers:
        raise RuntimeError(
            f"No `layers.N.self_attn.o_proj.weight` keys found for {model_id}. "
            "If this is an unusual architecture, dump the weights yourself and "
            "pass --o-proj-file."
        )
    ordered = [layers[i] for i in sorted(layers)]
    return torch.stack(ordered)  # [L, d_model, n_heads*head_dim]


# ---------------------------------------------------------------------------
# matrix construction
# ---------------------------------------------------------------------------

def head_block(W: torch.Tensor, head: int, n_heads: int, space: str) -> torch.Tensor:
    """Slice one head's block out of a single layer's o_proj weight.

    space='head'  -> columns W[:, h*hd:(h+1)*hd] with hd = in_features/n_heads.
                     This is attention head h's write matrix; its column space is
                     the subspace of the residual stream head h can write into.
    space='block' -> rows W[h*rd:(h+1)*rd, :].T with rd = out_features/n_heads.
                     These are the residual-stream coordinates that the
                     pipeline's index h actually refers to, since the
                     localization reduces o_proj.OUTPUT with
                     einops.reduce('l (n m) b -> l n b', n=num_heads).
    Both come back as [rows, cols] with `cols` the dimension being spanned, so
    everything downstream is identical.
    """
    if space == "head":
        hd = W.shape[1] // n_heads
        return W[:, head * hd:(head + 1) * hd]
    elif space == "block":
        rd = W.shape[0] // n_heads
        return W[head * rd:(head + 1) * rd, :].T
    raise ValueError(space)


def build_matrix(o_proj: torch.Tensor, heads: List[Tuple[int, int]], n_heads: int,
                 space: str, normalize: bool = True, dtype=None) -> torch.Tensor:
    """Concatenate the per-head blocks into M of shape [ambient_dim, n*block_cols]."""
    dtype = dtype or DTYPE
    cols = []
    for (l, h) in heads:
        B = head_block(o_proj[l], h, n_heads, space).to(dtype)
        if normalize:
            n = torch.linalg.norm(B)
            if n > 0:
                B = B / n
        cols.append(B)
    return torch.cat(cols, dim=1)


# ---------------------------------------------------------------------------
# spectra and subspace comparison
# ---------------------------------------------------------------------------

def left_spectrum(M: torch.Tensor, gram: Optional[torch.Tensor] = None):
    """(U, s) for the left singular system.

    Two routes, picked on shape.  A thin SVD of M costs O(d * n^2) and a Gram
    eigendecomposition costs O(d^3), so the thin SVD wins whenever the head set
    does not already saturate the residual stream (n_cols <= d_model) -- which
    is the only regime the comparison is meaningful in anyway.  Above that the
    Gram route is cheaper and is used instead.
    """
    if gram is None and M.shape[1] <= M.shape[0]:
        U, s, _ = torch.linalg.svd(M, full_matrices=False)
        return U, s
    G = M @ M.T if gram is None else gram
    G = 0.5 * (G + G.T)
    evals, evecs = torch.linalg.eigh(G)
    evals = torch.flip(evals, dims=[0]).clamp_min(0)
    evecs = torch.flip(evecs, dims=[1])
    return evecs, torch.sqrt(evals)


def effective_rank(s: torch.Tensor, eps: float = 1e-12) -> float:
    p = s / s.sum().clamp_min(eps)
    p = p[p > eps]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def stable_rank(s: torch.Tensor, eps: float = 1e-12) -> float:
    return float((s.pow(2).sum() / s[0].pow(2).clamp_min(eps)))


def rank_at_energy(s: torch.Tensor, thresh: float) -> int:
    e = s.pow(2)
    c = torch.cumsum(e, 0) / e.sum().clamp_min(1e-12)
    r = int(torch.searchsorted(c, torch.tensor(thresh, dtype=c.dtype)).item()) + 1
    return max(1, min(r, int(s.numel())))


@dataclass
class SetStats:
    n_heads: int
    ambient_dim: int
    n_cols: int
    saturated: bool
    effective_rank: float
    stable_rank: float
    rank_90: int
    rank_99: int
    numeric_rank: int


def set_stats(M: torch.Tensor, s: torch.Tensor, tol_scale: float = 1e-10) -> SetStats:
    tol = s[0] * max(M.shape) * tol_scale
    return SetStats(
        n_heads=-1,
        ambient_dim=M.shape[0],
        n_cols=M.shape[1],
        saturated=bool(M.shape[1] >= M.shape[0]),
        effective_rank=effective_rank(s),
        stable_rank=stable_rank(s),
        rank_90=rank_at_energy(s, 0.90),
        rank_99=rank_at_energy(s, 0.99),
        numeric_rank=int((s > tol).sum().item()),
    )


def compare(Ma: torch.Tensor, Mb: torch.Tensor, energy: float = 0.99,
            fixed_rank: Optional[int] = None) -> Dict:
    """All cross-set metrics for two head-set matrices sharing an ambient dim."""
    Ua, sa = left_spectrum(Ma)
    Ub, sb = left_spectrum(Mb)
    ra = fixed_rank or rank_at_energy(sa, energy)
    rb = fixed_rank or rank_at_energy(sb, energy)
    ra = min(ra, Ma.shape[0])
    rb = min(rb, Mb.shape[0])

    Qa, Qb = Ua[:, :ra], Ub[:, :rb]
    C = Qa.T @ Qb
    cosines = torch.linalg.svdvals(C).clamp(0, 1)

    grassmann = float((C.pow(2).sum() / min(ra, rb)))
    e_b_in_a = float((Qa.T @ Mb).pow(2).sum() / Mb.pow(2).sum().clamp_min(1e-12))
    e_a_in_b = float((Qb.T @ Ma).pow(2).sum() / Ma.pow(2).sum().clamp_min(1e-12))

    Mj = torch.cat([Ma, Mb], dim=1)
    if Mj.shape[1] <= Mj.shape[0]:
        sj = torch.linalg.svdvals(Mj)
    else:
        _, sj = left_spectrum(None, gram=(Ma @ Ma.T) + (Mb @ Mb.T))
    rj = rank_at_energy(sj, energy)
    sharing = (ra + rb - rj) / max(min(ra, rb), 1)

    return {
        "rank_a": ra,
        "rank_b": rb,
        "rank_joint": rj,
        "rank_sharing": float(sharing),
        "grassmann_overlap": grassmann,
        "energy_b_in_a": e_b_in_a,
        "energy_a_in_b": e_a_in_b,
        "mean_principal_angle_deg": float(
            torch.rad2deg(torch.arccos(cosines.clamp(-1, 1))).mean()),
        "n_angles_cos_gt_0.9": int((cosines > 0.9).sum().item()),
        "n_angles_cos_gt_0.99": int((cosines > 0.99).sum().item()),
        "_cosines": cosines.cpu().numpy(),
        "_sa": sa.cpu().numpy(),
        "_sb": sb.cpu().numpy(),
        "_sj": sj.cpu().numpy(),
        "_stats_a": asdict(set_stats(Ma, sa)),
        "_stats_b": asdict(set_stats(Mb, sb)),
    }


# ---------------------------------------------------------------------------
# nulls
# ---------------------------------------------------------------------------

def random_heads(n_layers: int, n_heads: int, k: int, rng: np.random.Generator,
                 exclude: Optional[set] = None) -> List[Tuple[int, int]]:
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    if exclude:
        pool = [p for p in pool if p not in exclude]
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[i] for i in idx]


def layer_matched_heads(reference: List[Tuple[int, int]], n_heads: int,
                        rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Same number of heads per layer as `reference`, but random head indices."""
    out = []
    counts: Dict[int, int] = {}
    for l, _ in reference:
        counts[l] = counts.get(l, 0) + 1
    for l, c in counts.items():
        picks = rng.choice(n_heads, size=min(c, n_heads), replace=False)
        out.extend((l, int(h)) for h in picks)
    return out


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_tasks(results_root: str, model: str, algo: str) -> List[Tuple[str, str, str]]:
    """Find (task, long_dir, single_dir) triples under results_root/model."""
    base = os.path.join(results_root, model)
    if not os.path.isdir(base):
        raise SystemExit(f"No results directory at {base}")
    out = []
    pat = re.compile(r"^from_(?P<src>.+)-long_to_(?P<base>.+)-long$")
    for d in sorted(os.listdir(base)):
        m = pat.match(d)
        if not m:
            continue
        src, bse = m.group("src"), m.group("base")
        single = f"from_{src}-single_to_{bse}-single"
        long_p = os.path.join(base, d, algo, "numerator_1_heads.pt")
        sing_p = os.path.join(base, single, algo, "numerator_1_heads.pt")
        if os.path.exists(long_p) and os.path.exists(sing_p):
            out.append((src, long_p, sing_p))
        else:
            missing = [p for p in (long_p, sing_p) if not os.path.exists(p)]
            print(f"[skip] {src}: missing {missing}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def make_plots(outdir: str, task: str, topk: float, res: Dict, nulls: Dict, space: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    tag = f"{task}_{space}_topk{topk}"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in (("long", res["_sa"]), ("single", res["_sb"]), ("joint", res["_sj"])):
        e = s ** 2
        ax[0].semilogy(np.maximum(s / s[0], 1e-16), label=name)
        ax[1].plot(np.cumsum(e) / e.sum(), label=name)
    ax[0].set_xlabel("index"); ax[0].set_ylabel("$\\sigma_i/\\sigma_0$")
    ax[0].set_title(f"{task} top_k={topk}: o_proj spectra")
    ax[1].axhline(0.99, ls="--", c="k", lw=0.8)
    ax[1].set_xlabel("rank"); ax[1].set_ylabel("cumulative energy")
    ax[1].set_title("energy captured")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"spectra_{tag}.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cos = res["_cosines"]
    ax.plot(np.rad2deg(np.arccos(np.clip(cos, -1, 1))), label="long vs single")
    for key, lab in (("uniform", "random heads"), ("layer_matched", "layer-matched")):
        if key in nulls and nulls[key].get("_cosines") is not None:
            c = nulls[key]["_cosines"]
            ax.plot(np.rad2deg(np.arccos(np.clip(c, -1, 1))), ls="--", alpha=0.7, label=lab)
    ax.set_xlabel("index"); ax.set_ylabel("principal angle (deg)")
    ax.set_title(f"{task} top_k={topk}: principal angles")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"angles_{tag}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_one(task: str, long_path: str, single_path: str, o_proj: torch.Tensor,
            n_heads: int, args, rng) -> List[Dict]:
    attr_long = torch.load(long_path, map_location="cpu")
    attr_single = torch.load(single_path, map_location="cpu")
    n_layers = o_proj.shape[0]

    for name, a in (("long", attr_long), ("single", attr_single)):
        if a.shape[0] != n_layers or a.shape[1] != n_heads:
            warnings.warn(
                f"{task}/{name}: attribution map is {tuple(a.shape[:2])} but the model "
                f"has {n_layers} layers x {n_heads} heads. Check that the map came from "
                f"this model."
            )

    if args.space == "head":
        ambient, block_cols = o_proj.shape[1], o_proj.shape[2] // n_heads
    else:
        ambient, block_cols = o_proj.shape[2], o_proj.shape[1] // n_heads
    rows = []

    for topk in args.topk:
        df_l = get_top_k_layer_and_head(attr_long, topk, args.algo, args.abs)
        df_s = get_top_k_layer_and_head(attr_single, topk, args.algo, args.abs)
        heads_l = [(int(r.layer), int(r.head)) for r in df_l.itertuples()]
        heads_s = [(int(r.layer), int(r.head)) for r in df_s.itertuples()]
        set_l, set_s = set(heads_l), set(heads_s)
        shared = set_l & set_s
        jac = len(shared) / max(len(set_l | set_s), 1)

        n_cols_l = len(heads_l) * block_cols
        if n_cols_l >= ambient:
            print(f"[warn] {task} top_k={topk}: {len(heads_l)} heads x {block_cols} cols "
                  f"= {n_cols_l} >= ambient {ambient}; subspaces saturate the space and "
                  f"overlap is ~1 by construction. Interpret with care.", file=sys.stderr)

        variants = {"all": (heads_l, heads_s)}
        if args.disjoint and shared:
            variants["disjoint"] = ([h for h in heads_l if h not in shared],
                                    [h for h in heads_s if h not in shared])

        for variant, (hl, hs) in variants.items():
            if not hl or not hs:
                continue
            Ml = build_matrix(o_proj, hl, n_heads, args.space, args.normalize_heads)
            Ms = build_matrix(o_proj, hs, n_heads, args.space, args.normalize_heads)
            res = compare(Ml, Ms, args.energy, args.fixed_rank)

            nulls = {}
            null_rows = {"uniform": [], "layer_matched": []}
            for seed in range(args.n_null):
                r = np.random.default_rng(args.seed + seed)
                rand_l = random_heads(n_layers, n_heads, len(hl), r)
                rand_s = random_heads(n_layers, n_heads, len(hs), r)
                nr = compare(build_matrix(o_proj, rand_l, n_heads, args.space, args.normalize_heads),
                             build_matrix(o_proj, rand_s, n_heads, args.space, args.normalize_heads),
                             args.energy, args.fixed_rank)
                null_rows["uniform"].append(nr)
                lm_l = layer_matched_heads(hl, n_heads, r)
                lm_s = layer_matched_heads(hs, n_heads, r)
                nr2 = compare(build_matrix(o_proj, lm_l, n_heads, args.space, args.normalize_heads),
                              build_matrix(o_proj, lm_s, n_heads, args.space, args.normalize_heads),
                              args.energy, args.fixed_rank)
                null_rows["layer_matched"].append(nr2)
            for k, v in null_rows.items():
                if v:
                    nulls[k] = v[0]

            row = {
                "task": task, "space": args.space, "variant": variant, "top_k": topk,
                "n_heads_long": len(hl), "n_heads_single": len(hs),
                "head_jaccard": jac, "n_shared_heads": len(shared),
                "ambient_dim": ambient, "cols_per_head": block_cols,
                "saturated": bool(len(hl) * block_cols >= ambient),
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
                    row[f"null_{null_name}_{metric}_mean"] = float(vals.mean())
                    row[f"null_{null_name}_{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    row[f"null_{null_name}_{metric}_z"] = (
                        float((row[metric] - vals.mean()) / sd) if sd > 0 else float("nan"))
                    row[f"null_{null_name}_{metric}_pct"] = float((vals < row[metric]).mean())
            rows.append(row)

            if args.plots and variant == "all":
                make_plots(os.path.join(args.out, "plots"), task, topk, res, nulls, args.space)

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True,
                   help="HF id, e.g. google/gemma-3-12b-it. Also names the results subdir.")
    p.add_argument("--model-dir", default=None,
                   help="Local snapshot dir to read safetensors from instead of the hub.")
    p.add_argument("--results-root", default="./results")
    p.add_argument("--algo", default="atp")
    p.add_argument("--task", nargs="*", default=["all"],
                   help="Task prefixes (female, lying, extraversion) or 'all'.")
    p.add_argument("--topk", nargs="+", type=float,
                   default=[0.005, 0.01, 0.02, 0.03, 0.05],
                   help="Same fractions the eval pipeline uses.")
    p.add_argument("--space", choices=["head", "block"], default="head",
                   help="'head' = W_O column block (true attention head); "
                        "'block' = row block, matching how the pipeline indexes "
                        "o_proj.output. See module docstring.")
    p.add_argument("--energy", type=float, default=0.99,
                   help="Cumulative-energy threshold defining each subspace's rank.")
    p.add_argument("--fixed-rank", type=int, default=None,
                   help="Override: use the top-r left singular vectors for both sets.")
    p.add_argument("--normalize-heads", action="store_true", default=True,
                   help="Unit-Frobenius each head block so one high-norm head "
                        "cannot dominate the energy metrics.")
    p.add_argument("--no-normalize-heads", dest="normalize_heads", action="store_false")
    p.add_argument("--abs", action="store_true",
                   help="Rank heads by |ATP effect| instead of signed effect.")
    p.add_argument("--disjoint", action="store_true", default=True,
                   help="Also report metrics with heads shared by both sets removed.")
    p.add_argument("--no-disjoint", dest="disjoint", action="store_false")
    p.add_argument("--n-null", type=int, default=5,
                   help="Null draws per condition. Each costs one full compare(), "
                        "so this is the main runtime knob.")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                   help="float32 roughly halves runtime; metrics agree to ~1e-4.")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads (defaults to whatever torch picks).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", action="store_true", default=True)
    p.add_argument("--no-plots", dest="plots", action="store_false")
    p.add_argument("--out", default="./results/svd_subspaces")
    p.add_argument("--o-proj-file", default=None,
                   help="Load stacked o_proj weights [L, d_model, n*hd] from a .pt "
                        "instead of reading the model.")
    p.add_argument("--dump-o-proj", default=None,
                   help="Write the stacked o_proj weights here and exit.")
    p.add_argument("--num-heads", type=int, default=None,
                   help="Override num_attention_heads (otherwise read from config).")
    args = p.parse_args()

    global DTYPE
    DTYPE = torch.float64 if args.dtype == "float64" else torch.float32
    if args.threads:
        torch.set_num_threads(args.threads)

    model_name = args.model_id.split("/")[-1]
    os.makedirs(args.out, exist_ok=True)

    if args.o_proj_file:
        o_proj = torch.load(args.o_proj_file, map_location="cpu").to(torch.float32)
    else:
        print(f"Reading o_proj weights for {args.model_id} ...")
        o_proj = load_o_proj(args.model_id, local_dir=args.model_dir)
    if args.dump_o_proj:
        os.makedirs(os.path.dirname(args.dump_o_proj) or ".", exist_ok=True)
        torch.save(o_proj, args.dump_o_proj)
        print(f"Wrote {tuple(o_proj.shape)} to {args.dump_o_proj}")
        return

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
    if in_dim % n_heads:
        raise SystemExit(f"o_proj in_features {in_dim} not divisible by {n_heads} heads.")
    head_dim = in_dim // n_heads
    print(f"model: {n_layers} layers, d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")
    if in_dim != d_model:
        print(f"note: n_heads*head_dim ({in_dim}) != d_model ({d_model}) -- this is "
              f"normal for Gemma-3. With --space block the row-block width is "
              f"d_model//n_heads = {d_model // n_heads}, which is what the "
              f"localization's einops.reduce actually groups.")
    if d_model % n_heads:
        print(f"warn: d_model {d_model} not divisible by {n_heads}; --space block "
              f"will drop the remainder.")

    triples = discover_tasks(args.results_root, model_name, args.algo)
    if args.task and "all" not in args.task:
        triples = [t for t in triples if t[0].split("-")[0] in args.task or t[0] in args.task]
    if not triples:
        raise SystemExit("No (long, single) attribution-map pairs found.")

    rng = np.random.default_rng(args.seed)
    all_rows = []
    for task, lp, sp in triples:
        print(f"\n=== {task} ===\n  long:   {lp}\n  single: {sp}")
        all_rows += run_one(task, lp, sp, o_proj, n_heads, args, rng)

    df = pd.DataFrame(all_rows)
    csv = os.path.join(args.out, f"svd_subspaces_{model_name}_{args.space}.csv")
    df.to_csv(csv, index=False)
    print(f"\nWrote {csv}")

    cols = ["task", "variant", "top_k", "n_heads_long", "head_jaccard", "saturated",
            "rank_a", "rank_b", "rank_joint", "rank_sharing", "grassmann_overlap",
            "null_layer_matched_grassmann_overlap_mean",
            "null_layer_matched_grassmann_overlap_z"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
