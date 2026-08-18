import os
import json
import torch
import einops
import pandas as pd
import matplotlib.pyplot as plt
import random

def shard_indices(logits_path):
    """Indices of every {which_patch}_{i}.pt shard Experiment.run left on disk.

    Enumerated from the filesystem, never from a count. load_logits used to
    range over data_handler.LEN, which is min(len(base_desired), 50) under
    --eval_model but the full corpus under --patch_model -- so a 100-item
    localization was reduced using shards 0..49 and the other half was dropped
    with no error.
    """
    shard_dir = os.path.dirname(logits_path)
    stem = os.path.basename(logits_path)
    if not os.path.isdir(shard_dir):
        return []
    return sorted(int(f[len(stem) + 1:-3]) for f in os.listdir(shard_dir)
                  if f.startswith(stem + "_") and f.endswith(".pt")
                  and f[len(stem) + 1:-3].isdigit())


def load_logits(config, data_handler, which_patch, model_handler):
    logits_path = f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{which_patch}"
    # print('Loading logits from:', logits_path)
    all_logits = None

    name = 'numerator_1' if config.args.patch_algo != 'probes' else 'probes'
    print('/'.join(config.get_output_prefix().split('/')[:-3]))
    map_path = f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{name}_{which_patch}.pt"
    shard_idx = shard_indices(logits_path) if config.args.patch_algo != 'probes' else []

    if os.path.exists(map_path):
        all_logits = torch.load(map_path)
        # A map cached by the OLD reducer was built from a subset of the shards
        # and is indistinguishable from a correct one by name alone. Its last
        # dimension is one column per item consumed, so compare it against what
        # is actually on disk and rebuild when they disagree -- otherwise the
        # shard fix is inert for every results tree that already has a map,
        # which is every tree produced before it landed.
        if 'atp' in config.args.patch_algo and shard_idx and \
                all_logits.shape[-1] != len(shard_idx):
            stale = f'{map_path}.stale-{all_logits.shape[-1]}-of-{len(shard_idx)}-shards'
            os.replace(map_path, stale)
            print(f'STALE ATTRIBUTION MAP: {map_path} has '
                  f'{all_logits.shape[-1]} items but {len(shard_idx)} shards are '
                  f'on disk. It was reduced from a subset. Moved to {stale} and '
                  f'rebuilding from all shards.\n  Any top-k CSV already written '
                  f'next to it came from the stale map -- delete those to force '
                  f'them to be recomputed.')
            all_logits = None

    if all_logits is None:
        print('Computing attribution map afresh -> {}'.format(map_path))
        if config.args.patch_algo != 'probes':
            if not shard_idx:
                raise FileNotFoundError(
                    f"no {os.path.basename(logits_path)}_*.pt shards under "
                    f"{os.path.dirname(logits_path)}; run --patch_model first")
            print(f"loading {len(shard_idx)} shards from "
                  f"{os.path.dirname(logits_path)} "
                  f"(indices {shard_idx[0]}..{shard_idx[-1]})")
            cols = []
            for i in shard_idx:
                with open(f"{logits_path}_{i}.pt", 'rb') as f:
                    logits = torch.load(f)
                    cols.append(logits.squeeze().unsqueeze(-1)
                                if 'atp' in config.args.patch_algo else logits)
            # One cat, not one per shard: repeated torch.cat reallocates the
            # whole tensor every iteration.
            all_logits = torch.cat(cols, dim=-1)

            # print('patcher ', all_logits.shape)
            name = 'numerator_1'
            if 'atp' in config.args.patch_algo:
                # print('all_logits ', all_logits.shape)
                # all_logits = all_logits.sum(dim=1)
                all_logits = einops.reduce(all_logits, 'l (n m) b -> l n b', 'sum', n=model_handler.num_heads)
            if config.args.patch_algo == 'acp':
                # print('all_logits before squeeze ', all_logits.shape)
                all_logits = all_logits.squeeze()
                # print('all_logits after squeeze ', all_logits.shape)
                base_des_post_patch = all_logits[0,...]
                base_undes_post_patch = all_logits[1,...]
                all_logits = (base_undes_post_patch - base_des_post_patch)
                # print('all_logits final ', all_logits.shape)
        else:
            name = 'probes'
            with open(f'{logits_path}.json', 'r') as f:
                raw_logits = json.load(f)
            logits = [[float(head_val) for head_val in layer_dict.values()] for layer_dict in raw_logits.values()]
            all_logits = torch.tensor(logits)
        plot_logit_metrics(config, model_handler, all_logits, name, which_patch)
        torch.save(all_logits, f"{'/'.join(config.get_output_prefix().split('/')[:-3])}/{name}_{which_patch}.pt")
    return all_logits

def get_top_k_layer_and_head(patches, top_k, patch_algo):
    if isinstance(patches, str):
        patches = torch.load(patches)
    patches = patches.to(torch.float32)
    if patch_algo != 'probes':
        patches = patches.mean(dim=-1)
    patches = patches.cpu()  # CUDA topk/sort tie-breaking is not specified
    flat = patches.view(-1)
    k = int(round(top_k * flat.numel()))
    k = max(1, min(k, flat.numel()))
    # topk does not specify how it orders equal values. Sort by (-value, index)
    # so ties always resolve to the lower flat index, on any device or version.
    order = sorted(range(flat.numel()), key=lambda i: (-flat[i].item(), i))[:k]
    top_indices = torch.tensor(order, dtype=torch.long)
    top_values = flat[top_indices]
    layer_indices = top_indices // patches.shape[1]
    neuron_indices = top_indices % patches.shape[1]
    df = pd.DataFrame({
        'layer': layer_indices.numpy(),
        'neuron': neuron_indices.numpy(),
        'value': top_values.numpy()
    })
    return df.sort_values(by=['layer', 'neuron'], kind='mergesort').reset_index(drop=True)

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
