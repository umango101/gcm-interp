"""Differential head localization: heads that surface in one contrast but not another.

Runs AFTER two `run.py --patch_model` localizations have written their
attribution maps. Loads both, takes the top-k head set of each, and writes the
set difference in the SAME CSV format the eval stage already consumes, so the
result drops into the existing pipeline with no other change.

    keep    from_roleConflict-single_to_roleAgree-single   (the contrast of interest)
    remove  from_roleInverted-single_to_roleAgree-single   (the control)

WHY SET DIFFERENCE RATHER THAN SCORE SUBTRACTION
------------------------------------------------
The two ATP maps are not on a common scale, and they do not even point the same
way. Under roleConflict the desired answer flips between conditions, so ATP's
d(logP(undesired) - logP(desired)) contrast scores heads that push toward the
DEVELOPER's word. Under roleInverted the developer turn is byte-identical across
conditions and the desired answer does not flip, so the same metric scores heads
that push toward the USER's word. Subtracting raw scores would therefore be
combining two quantities with different units and different signs.

Set difference only uses rank MEMBERSHIP -- "did this head make the top k" -- so
it is invariant to both problems. That is what --mode setdiff does, and it is the
default.

--mode rank_diff is offered for the case where you want a fixed-size output. It
converts each map to within-map ranks before subtracting, which fixes the scale
mismatch but NOT the direction mismatch; read its output as "heads ranked higher
by conflict than by inverted", not as a purified conflict score.

OUTPUT
------
    {results}/{model}/{keep}_minus_{remove}/{algo}/eval/numerator_1_targeted_{topk}.csv

with columns layer, neuron, value sorted by (layer, neuron) -- byte-compatible
with what save_top_k() writes, because it calls the same function to build it.

`--install_into <eval_prefix>` additionally copies the CSVs into an eval output
directory. run_eval() reads {prefix}/eval/{metric}_{reps}_{topk}.csv when it
exists and only computes one when it does not, so installing the differenced
CSVs there is what makes the eval steer the differenced head set.

USAGE
-----
    python localize_diff.py \
        --keep   from_roleConflict-single_to_roleAgree-single \
        --remove from_roleInverted-single_to_roleAgree-single
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Imported rather than reimplemented: identical tie-breaking, identical column
# order, identical sort. A local copy would drift from the eval stage silently.
from eval.logits_handler import get_top_k_layer_and_head, shard_indices

# The default sweep in eval/eval_runner.py:run_eval. The filenames must match
# character for character (f"...{topk}.csv" on the raw float), so this list is
# copied rather than reformatted.
DEFAULT_TOPK = [1.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5]


def build_map_from_shards(task_dir, which_patch, num_heads, algo):
    """Reduce heads_*.pt shards to the [layers, heads, items] attribution map.

    `run.py --patch_model` writes one shard per item and stops; the reduction to
    numerator_1_{which_patch}.pt is done later, by
    eval/logits_handler.py:load_logits, on the first eval. Doing it here means
    the differential localization runs straight after --patch_model without a
    throwaway eval pass, and the file it writes is the one load_logits would
    have written -- so the normal pipeline picks it up unchanged.

    Every shard on disk is consumed, in index order. load_logits used to range
    over data_handler.LEN, which is 50 under --eval_model and 100 under
    --patch_model, so it silently averaged half the corpus.
    """
    idx = shard_indices(os.path.join(task_dir, which_patch))
    if not idx:
        return None
    stem = which_patch
    cols = []
    for i in idx:
        t = torch.load(os.path.join(task_dir, f'{stem}_{i}.pt'))
        cols.append(t.squeeze().unsqueeze(-1) if 'atp' in algo else t)
    all_logits = torch.cat(cols, dim=-1)
    if 'atp' in algo:
        L, width, B = all_logits.shape
        if width % num_heads:
            raise ValueError(
                f'shard width {width} is not divisible by --num_heads {num_heads}. '
                f'width is num_heads * head_dim at the o_proj_input site; pass the '
                f'value ModelHandler printed as "[head geometry] n_heads=".')
        all_logits = all_logits.view(L, num_heads, width // num_heads, B).sum(dim=2)
    print(f'  built map from {len(idx)} shards (indices {idx[0]}..{idx[-1]})')
    return all_logits


def load_map(results_root, model, task, algo, which_patch='heads', num_heads=64):
    """Load one attribution map, building it from shards if it is not cached."""
    task_dir = os.path.join(results_root, model, task, algo)
    path = os.path.join(task_dir, f'numerator_1_{which_patch}.pt')
    if os.path.exists(path):
        cached = torch.load(path)
        n_shards = len(shard_indices(os.path.join(task_dir, which_patch)))
        # Same staleness check load_logits does: a map reduced from a subset of
        # the shards is indistinguishable by filename. Its last dimension is one
        # column per item consumed, so compare against what is on disk.
        if 'atp' in algo and n_shards and cached.shape[-1] != n_shards:
            stale = f'{path}.stale-{cached.shape[-1]}-of-{n_shards}-shards'
            os.replace(path, stale)
            print(f'  STALE: {path} had {cached.shape[-1]} items but {n_shards} '
                  f'shards exist. Moved to {stale}; rebuilding from all shards.')
        else:
            return cached, path
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(
            f'{task_dir} does not exist. Run the localization for {task} first:\n'
            f'  python run.py --patch_model --patch_algo {algo} '
            f'--source <src> --base <base> ...')
    built = build_map_from_shards(task_dir, which_patch, num_heads, algo)
    if built is None:
        raise FileNotFoundError(
            f'no {which_patch}_*.pt shards and no numerator_1_{which_patch}.pt '
            f'under {task_dir}. Run the localization for {task} first.')
    torch.save(built, path)
    print(f'  cached -> {path}')
    return built, path


def _key(df):
    return set(zip(df['layer'].tolist(), df['neuron'].tolist()))


def setdiff(keep_map, remove_map, topk, algo):
    """Heads in keep's top-k that are NOT in remove's top-k, at the same k."""
    keep_df = get_top_k_layer_and_head(keep_map, topk, algo)
    remove_df = get_top_k_layer_and_head(remove_map, topk, algo)
    drop = _key(remove_df)
    mask = [(l, n) not in drop for l, n in zip(keep_df['layer'], keep_df['neuron'])]
    out = keep_df[mask].reset_index(drop=True)
    return out, len(keep_df), len(remove_df)


def rank_diff(keep_map, remove_map, topk, algo):
    """Top-k of (rank in keep) - (rank in remove), ranks taken within each map.

    Fixed output size, but see the direction caveat in the module docstring.
    """
    def to_rank(m):
        m = m.to(torch.float32)
        if algo != 'probes':
            m = m.mean(dim=-1)
        flat = m.cpu().reshape(-1)
        # rank 0 = highest score. Ties resolve to the lower flat index, matching
        # get_top_k_layer_and_head.
        order = sorted(range(flat.numel()), key=lambda i: (-flat[i].item(), i))
        r = torch.empty(flat.numel(), dtype=torch.float32)
        for rank, idx in enumerate(order):
            r[idx] = rank
        return r.reshape(m.shape)

    # Higher = ranked better by keep than by remove, so negate the difference to
    # keep "large value = more interesting", the convention every other map here
    # uses.
    diff = (to_rank(remove_map) - to_rank(keep_map)).unsqueeze(-1)
    out = get_top_k_layer_and_head(diff, topk, algo)
    n = int(round(topk * diff[..., 0].numel()))
    return out, n, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results_root', default='./results')
    ap.add_argument('--model', default='gpt-oss-20b')
    ap.add_argument('--keep', default='from_roleConflict-single_to_roleAgree-single',
                    help='Task dir whose heads you want to keep.')
    ap.add_argument('--remove', default='from_roleInverted-single_to_roleAgree-single',
                    help='Task dir whose heads you want to subtract out.')
    ap.add_argument('--algo', default='atp')
    ap.add_argument('--which_patch', default='heads')
    ap.add_argument('--num_heads', type=int, default=64,
                    help='Only used when a map has to be rebuilt from shards. '
                         'gpt-oss-20b is 64. This is num_attention_heads, the '
                         'value ModelHandler prints as "[head geometry] n_heads=".')
    ap.add_argument('--mode', default='setdiff', choices=['setdiff', 'rank_diff'])
    ap.add_argument('--topk', type=float, nargs='*', default=None,
                    help=f'Defaults to run_eval\'s sweep: {DEFAULT_TOPK}')
    ap.add_argument('--install_into', default=None,
                    help='Eval output prefix to copy the CSVs into, e.g. '
                         './results/gpt-oss-20b/from_roleConflict-single_to_'
                         'roleAgree-single/atp/roleConflict-single_eval/'
                         'roleConflict-single_steer/ . The CSVs land in '
                         '<prefix>/eval/ where run_eval looks for them.')
    ap.add_argument('--overwrite', action='store_true',
                    help='Replace CSVs already present at the install target. '
                         'Without this an existing file is left alone and '
                         'reported, since run_eval may already have used it.')
    args = ap.parse_args()

    topk_vals = args.topk if args.topk else DEFAULT_TOPK

    keep_map, keep_path = load_map(args.results_root, args.model, args.keep,
                                   args.algo, args.which_patch, args.num_heads)
    remove_map, remove_path = load_map(args.results_root, args.model, args.remove,
                                       args.algo, args.which_patch, args.num_heads)
    print(f'keep   {keep_path}  shape {tuple(keep_map.shape)}')
    print(f'remove {remove_path}  shape {tuple(remove_map.shape)}')
    if keep_map.shape[:2] != remove_map.shape[:2]:
        raise ValueError(
            f'layer/head geometry differs: {tuple(keep_map.shape[:2])} vs '
            f'{tuple(remove_map.shape[:2])}. Both maps must come from the same '
            f'model and the same --head_site.')
    if keep_map.shape[-1] != remove_map.shape[-1]:
        # Not fatal -- each column is one item and the maps are averaged over
        # items -- but a mismatch usually means one localization consumed fewer
        # shards than the other, which is worth seeing.
        print(f'  NOTE: item counts differ ({keep_map.shape[-1]} vs '
              f'{remove_map.shape[-1]}); both are averaged over items, but check '
              f'that neither localization dropped shards.')

    out_task = f'{args.keep}_minus_{args.remove}'
    out_dir = os.path.join(args.results_root, args.model, out_task, args.algo, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    fn = setdiff if args.mode == 'setdiff' else rank_diff
    manifest = {'keep': args.keep, 'remove': args.remove, 'mode': args.mode,
                'algo': args.algo, 'keep_map': keep_path,
                'remove_map': remove_path, 'per_topk': {}}
    n_heads = keep_map.shape[0] * keep_map.shape[1]

    print(f'\n{"topk":>6}  {"keep":>6}  {"remove":>6}  {"survive":>8}  {"%kept":>6}')
    for topk in topk_vals:
        df, n_keep, n_remove = fn(keep_map, remove_map, topk, args.algo)
        name = f'numerator_1_targeted_{topk}.csv'
        # Empty results are written, not skipped. run_eval computes a top-k CSV
        # only when the file is ABSENT -- so a missing file here does not mean
        # "no heads", it means the eval silently falls back to the undifferenced
        # keep-map top-k, which is the opposite of what this script is for.
        df.to_csv(os.path.join(out_dir, name), index=False)
        pct = 100.0 * len(df) / max(n_keep, 1)
        flag = '   <-- EMPTY, eval would steer nothing' if len(df) == 0 else ''
        print(f'{topk:>6}  {n_keep:>6}  {n_remove:>6}  {len(df):>8}  {pct:>5.1f}%{flag}')
        manifest['per_topk'][str(topk)] = {
            'n_keep': n_keep, 'n_remove': n_remove, 'n_survive': len(df),
            'frac_of_keep': len(df) / max(n_keep, 1), 'file': name}

    if args.mode == 'setdiff' and any(
            v['n_survive'] == 0 for v in manifest['per_topk'].values()):
        print('\nNOTE: at least one topk left no surviving heads. At topk=1.0 both '
              'sets are all {} heads, so an empty difference there is arithmetic, '
              'not a result.'.format(n_heads))

    with open(os.path.join(out_dir, 'diff_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nwrote {len(topk_vals)} CSVs + diff_manifest.json to {out_dir}')

    if args.install_into:
        dest = os.path.join(args.install_into, 'eval')
        os.makedirs(dest, exist_ok=True)
        copied, skipped = 0, []
        for topk in topk_vals:
            name = f'numerator_1_targeted_{topk}.csv'
            target = os.path.join(dest, name)
            if os.path.exists(target) and not args.overwrite:
                skipped.append(name)
                continue
            shutil.copyfile(os.path.join(out_dir, name), target)
            copied += 1
        print(f'installed {copied} CSVs into {dest}')
        if skipped:
            print(f'  left {len(skipped)} existing files alone (pass --overwrite '
                  f'to replace): {skipped[:3]}')


if __name__ == '__main__':
    main()
