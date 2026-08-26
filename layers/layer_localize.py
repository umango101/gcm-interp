"""Layer localization: run the ATP shards, reduce them, rank the layers.

Two deliberate departures from the head-level equivalents:

1. The reduction enumerates shards from DISK rather than iterating
   ``range(data_handler.LEN)``. LEN is 50 under ``--eval_model`` and 100 under
   ``--patch_model``, so a range-based reduction silently builds the map from
   half the shards and then caches that. A sidecar records the shard count and
   the map is rebuilt whenever it disagrees with what is on disk.

2. Top-k is a COUNT of layers, not a fraction of the search space. There are
   only ~40 layers, so fractions are the wrong unit.
"""

import gc
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from batch_handler import BatchHandler
from layers.layer_patching import LayerPatching

SHARD_PREFIX = 'layers'
MAP_NAME = 'numerator_1_layers.pt'
META_NAME = 'numerator_1_layers.meta.json'


def run_localization(config, data_handler, model_handler, is_tuple):
    """Write one attribution shard per batch of contrastive pairs."""
    out_dir = config.get_output_prefix().rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    batch_size = config.args.batch_size
    batch_handler = BatchHandler(config, data_handler, 0, min(batch_size, data_handler.LEN))
    patching = LayerPatching(model_handler, batch_handler, config, is_tuple)

    for idx in tqdm(range(0, data_handler.LEN, batch_size), desc="ATP shards"):
        shard = f'{out_dir}/{SHARD_PREFIX}_{idx}.pt'
        if os.path.exists(shard):
            continue
        stop = min(idx + batch_size, data_handler.LEN)
        batch_handler.update(idx, stop)
        effects = patching.apply_patching()
        torch.save(effects, shard)
        # One LayerPatching instance serves every shard, so anything still bound
        # here survives into the next backward. apply_patching drops its own
        # locals; this drops the caller's.
        del effects
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[layers] localization shards written to {out_dir}")


def _shard_paths(loc_dir):
    paths = glob.glob(os.path.join(loc_dir, f'{SHARD_PREFIX}_*.pt'))
    keyed = []
    for p in paths:
        m = re.search(rf'{SHARD_PREFIX}_(\d+)\.pt$', os.path.basename(p))
        if m:
            keyed.append((int(m.group(1)), p))
    return [p for _, p in sorted(keyed)]


def reduce_layer_effects(config, force=False):
    """Collapse the shards to a [n_layers, n_items] attribution matrix.

    Each shard is [n_layers, batch, hidden]; hidden is summed here, so the score
    for a layer is the total first-order effect of swapping that whole residual
    stream. Summing (rather than taking a norm) preserves the sign convention the
    head pipeline ranks on.
    """
    loc_dir = config.localization_dir().rstrip('/')
    cache = os.path.join(loc_dir, MAP_NAME)
    meta_path = os.path.join(loc_dir, META_NAME)
    shards = _shard_paths(loc_dir)

    if not shards:
        raise FileNotFoundError(
            f"No attribution shards under {loc_dir}. Run the localization pass "
            f"(--patch_model) for this source/base pair first.")

    if os.path.exists(cache) and os.path.exists(meta_path) and not force:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('n_shards') == len(shards):
            effects = torch.load(cache)
            print(f"[layers] loaded cached map {tuple(effects.shape)} from {cache}")
            return effects
        print(f"[layers] cached map was built from {meta.get('n_shards')} shards but "
              f"{len(shards)} are on disk -- rebuilding.")

    per_shard = []
    for path in shards:
        t = torch.load(path, map_location='cpu').to(torch.float32)
        if t.dim() == 2:                     # [n_layers, hidden] -- batch of 1 already squeezed
            t = t.unsqueeze(1)
        per_shard.append(t.sum(dim=-1))      # [n_layers, batch]
    effects = torch.cat(per_shard, dim=1)    # [n_layers, n_items]

    torch.save(effects, cache)
    with open(meta_path, 'w') as f:
        json.dump({'n_shards': len(shards),
                   'n_items': int(effects.shape[1]),
                   'n_layers': int(effects.shape[0])}, f)
    print(f"[layers] built map {tuple(effects.shape)} from {len(shards)} shards -> {cache}")
    return effects


def layer_scores(effects, rank_by='cumulative'):
    """Per-layer score and the value ranking is performed on.

    cumulative -- the raw attribution at the layer output, which includes every
                  upstream contribution, so late layers dominate by construction.
    marginal   -- effect[l] - effect[l-1], the increment this layer contributes.
                  Layer 0 keeps its cumulative value (nothing precedes it).
    """
    mean = effects.mean(dim=-1)                       # [n_layers]
    marginal = mean.clone()
    marginal[1:] = mean[1:] - mean[:-1]
    if rank_by == 'cumulative':
        rank_score = mean
    elif rank_by == 'cumulative_abs':
        rank_score = mean.abs()
    elif rank_by == 'marginal':
        rank_score = marginal
    elif rank_by == 'marginal_abs':
        rank_score = marginal.abs()
    else:
        raise ValueError(f"Unknown rank_by: {rank_by!r}")
    return mean, marginal, rank_score


def get_top_k_layers(effects, k, rank_by='cumulative'):
    mean, marginal, rank_score = layer_scores(effects, rank_by)
    k = min(int(k), rank_score.numel())
    # Explicit sort rather than torch.topk: topk's tie-breaking is not documented
    # as stable, and ties are exactly what a head/layer-agnostic attribution map
    # would produce. Breaking ties by layer index keeps the selection reproducible
    # even in the degenerate case this experiment is testing for.
    scores = rank_score.tolist()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    idx = torch.tensor(order, dtype=torch.long)
    df = pd.DataFrame({
        'layer': idx.numpy(),
        'value': mean[idx].numpy(),
        'marginal': marginal[idx].numpy(),
        'rank_score': rank_score[idx].numpy(),
    })
    return df.sort_values(by=['layer']).reset_index(drop=True)


def retrieve_random_k_layers(num_layers, k, seed=42):
    """Random baseline arm: k layers drawn without replacement, seeded.

    Seeded on (seed, k) so that the k=3 draw is not a prefix of the k=5 draw --
    otherwise the random arm's own k-curve is autocorrelated and cannot be read
    against the targeted arm's.
    """
    import random as _random
    rng = _random.Random(int(seed) * 1000 + int(k))
    k = min(int(k), num_layers)
    selected = sorted(rng.sample(range(num_layers), k))
    return pd.DataFrame({'layer': selected,
                         'value': [float('nan')] * k,
                         'marginal': [float('nan')] * k,
                         'rank_score': [float('nan')] * k})


def plot_layer_effects(config, effects, out_dir):
    mean, marginal, _ = layer_scores(effects, 'cumulative')
    layers = list(range(len(mean)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(layers, mean.numpy())
    axes[0].set_ylabel('cumulative ATP effect')
    axes[0].set_title(f"Layer attribution: from_{config.args.source}_to_{config.args.base}")
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(layers, marginal.numpy(), color='tab:orange')
    axes[1].set_ylabel('marginal (effect[l] - effect[l-1])')
    axes[1].set_xlabel('layer')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'layer_effects.png')
    plt.savefig(path, dpi=150)
    plt.close()

    pd.DataFrame({'layer': layers,
                  'cumulative': mean.numpy(),
                  'marginal': marginal.numpy()}).to_csv(
        os.path.join(out_dir, 'layer_effects.csv'), index=False)
    print(f"[layers] wrote {path} and layer_effects.csv")
