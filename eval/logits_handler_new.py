import os
import re
import glob
import json
import torch
import einops
import pandas as pd
import matplotlib.pyplot as plt
import random

def discover_shards(logits_path):
    """Attribution shards written by Experiment.run(), in numeric index order.

    Discovered from disk rather than iterated as range(data_handler.LEN). At eval
    time LEN is the TEST-set size (50, data_handler.py:182) while localization
    wrote one shard per LOCALIZATION example (100, line 184), so a range()-based
    loop silently averaged only the first half of the shards. Globbing also
    handles the acp case, where shard indices are strided by batch_size rather
    than consecutive.
    """
    stem = os.path.basename(logits_path)
    rx = re.compile(rf"^{re.escape(stem)}_(\d+)\.pt$")
    found = []
    for p in glob.glob(f"{logits_path}_*.pt"):
        m = rx.match(os.path.basename(p))
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def _meta_path(cache_path):
    return cache_path.replace(".pt", ".meta.json")


def load_logits(config, data_handler, which_patch, model_handler):
    root = '/'.join(config.get_output_prefix().split('/')[:-3])
    logits_path = f"{root}/{which_patch}"
    all_logits = None

    name = 'numerator_1' if config.args.patch_algo != 'probes' else 'probes'
    cache_path = f"{root}/{name}_{which_patch}.pt"

    if os.path.exists(cache_path):
        # A cached map is only reusable if it was built from the shards that are
        # on disk now. Without this check, a map computed from a partial set of
        # shards is reused forever, including by every later eval cell.
        shards = discover_shards(logits_path)
        meta = None
        if os.path.exists(_meta_path(cache_path)):
            try:
                with open(_meta_path(cache_path)) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                meta = None
        if config.args.patch_algo == 'probes' or (meta and meta.get('shard_indices') == [i for i, _ in shards]):
            print(f"[logits] reusing cached {name} from {cache_path}"
                  + (f" (built from {meta['n_examples']} examples)" if meta else ""))
            return torch.load(cache_path)
        if meta is None:
            print(f"[logits] cached {name} at {cache_path} has no provenance sidecar "
                  f"(written by an older revision); recomputing from {len(shards)} shards.")
        else:
            print(f"[logits] cached {name} was built from {len(meta.get('shard_indices', []))} "
                  f"shards but {len(shards)} are on disk now; recomputing.")

    if config.args.patch_algo != 'probes':
        shards = discover_shards(logits_path)
        if not shards:
            raise FileNotFoundError(
                f"No attribution shards matching {logits_path}_*.pt. "
                "Run the --patch_model step for this source/base pair first."
            )
        n_examples = 0
        for i, path in shards:
            # No try/except: a shard that will not load means the attribution map
            # would be averaged over fewer examples than the run computed, which
            # is a silently wrong result rather than a recoverable one. Delete the
            # bad shard and re-run --patch_model to regenerate just that index.
            try:
                logits = torch.load(path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load attribution shard {path}: {e}. Delete it and "
                    "re-run --patch_model (Experiment.run skips shards that already "
                    "exist, so only this index is recomputed)."
                ) from e
            logits = logits.squeeze().unsqueeze(-1) if 'atp' in config.args.patch_algo else logits
            n_examples += logits.shape[-1]
            all_logits = logits if all_logits is None else torch.cat([all_logits, logits], dim=-1)

        if all_logits.shape[-1] != n_examples:
            raise RuntimeError(
                f"Assembled {all_logits.shape[-1]} examples from shards totalling "
                f"{n_examples}; concatenation is inconsistent."
            )
        print(f"[logits] {name}: {len(shards)} shards -> {n_examples} examples "
              f"(indices {shards[0][0]}..{shards[-1][0]})")
        if 'atp' in config.args.patch_algo and n_examples != len(shards):
            print(f"[logits] NOTE: {n_examples} examples from {len(shards)} shards "
                  "(expected 1 example per shard for atp).")

        name = 'numerator_1'
        if 'atp' in config.args.patch_algo:
            all_logits = einops.reduce(all_logits, 'l (n m) b -> l n b', 'sum', n=model_handler.num_heads)
        if config.args.patch_algo == 'acp':
            all_logits = all_logits.squeeze()
            base_des_post_patch = all_logits[0, ...]
            base_undes_post_patch = all_logits[1, ...]
            all_logits = (base_undes_post_patch - base_des_post_patch)
    else:
        name = 'probes'
        shards = []
        n_examples = None
        with open(f'{logits_path}.json', 'r') as f:
            raw_logits = json.load(f)
        logits = [[float(head_val) for head_val in layer_dict.values()] for layer_dict in raw_logits.values()]
        all_logits = torch.tensor(logits)

    plot_logit_metrics(config, model_handler, all_logits, name, which_patch)
    cache_path = f"{root}/{name}_{which_patch}.pt"
    torch.save(all_logits, cache_path)
    with open(_meta_path(cache_path), 'w') as f:
        json.dump({'n_examples': n_examples,
                   'shard_indices': [i for i, _ in shards],
                   'patch_algo': config.args.patch_algo,
                   'source': config.args.source,
                   'base': config.args.base}, f, indent=2)
    return all_logits

def get_top_k_layer_and_head(patches, top_k, patch_algo):
    if isinstance(patches, str):
        patches = torch.load(patches)
    patches = patches.to(torch.float32)
    if patch_algo != 'probes':
        patches = patches.mean(dim=-1)
    flat = patches.view(-1)
    top_values, top_indices = flat.topk(k=int(top_k * flat.numel()))
    layer_indices = top_indices // patches.shape[1]
    neuron_indices = top_indices % patches.shape[1]
    df = pd.DataFrame({
        'layer': layer_indices.numpy(),
        'neuron': neuron_indices.numpy(),
        'value': top_values.numpy()
    })
    return df.sort_values(by=['layer', 'neuron'])

def retrieve_random_k(num_layers, num_heads, k, seed=42):
    rng = random.Random(seed)
    total = num_layers * num_heads
    num_samples = int(k * total)
    all_combinations = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    selected = rng.sample(all_combinations, num_samples)
    df = pd.DataFrame(selected, columns=['layer', 'neuron'])
    return df.sort_values(by=['layer', 'neuron'])

def plot_logit_metrics(config, model_handler, metric, name, which_patch):
    metric = metric.to(torch.float32)
    if config.args.patch_algo != 'probes':
        metric = metric.mean(dim=-1)

    plt.imshow(metric, cmap="viridis")
    plt.colorbar(label='Indirect Effect size')
    plt.ylabel("Layers")
    plt.xlabel("Heads")
    plt.grid(True)
    titles = {
        "numerator_1": f"Post-patch logit difference: {config.args.base}",
        "probes": f"Probes accuracy between desired and undesired responses {config.args.base}"
    }
    plt.title(titles.get(name, name))
    plt.xticks(ticks=range(model_handler.num_heads))
    plt.yticks(ticks=range(model_handler.model.config.num_hidden_layers))
    plt.tight_layout()
    os.makedirs(f'{config.get_output_prefix()}/eval/', exist_ok=True)
    plt.savefig(f"{config.get_output_prefix()}/eval/{name}_heatmap.png")
    plt.close()
