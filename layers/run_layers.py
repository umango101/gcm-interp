"""Entry point for the layer-level residual-stream experiment.

Run from the repo root as a module so the top-level packages resolve:

    python -m layers.run_layers --patch_model ...      # localize layers
    python -m layers.run_layers --eval_model ...       # steer those layers

The (localization x eval x steer) matrix is expressed exactly as in the head
pipeline: ``--source``/``--base`` pick the localization, ``--eval_test`` picks
the test set, ``--steering_add_path``/``--steering_sub_path`` pick the steering
vector. Cross-steering is therefore just a different combination of the same
flags -- no separate code path, and no risk of the cross cells diverging from
the matched ones.

Outputs land under ``--results_root`` (default ``./results_layers``) in the same
directory shape and the same gen filenames the head pipeline uses, so
``eval_pipeline_bias.py`` scores these runs after repointing its ``RESULTS_DIR``
constant and setting ``TOP_KS`` to the layer counts.
"""

import gc
import json
import os
import sys
import tempfile

# Determinism on this branch is owned by eval/setup.py, not by cleanup's
# determinism.py (which does not exist here, and is weaker -- it leaves TF32 on).
# CUBLAS_WORKSPACE_CONFIG / PYTHONHASHSEED / TOKENIZERS_PARALLELISM must be
# exported by the LAUNCHER before the interpreter starts; Config.setup_environment()
# asserts that via assert_determinism_env() and then calls set_seed(), which pins
# cuDNN, disables TF32, and enables use_deterministic_algorithms. Setting any of it
# from here would be too late to take effect while looking like it worked.
# harden() below adds the two things set_seed() does not cover: the SDPA backend
# and fail-closed strictness.


import pandas as pd
import torch
from tqdm import tqdm

from layers.layer_config import LayerConfig
from model_handler import ModelHandler
from data_handler import DataHandler
from batch_handler import BatchHandler
from eval.setup import set_seed
from eval.generation import select_gen_qs_toks, decode_responses
from layers.layer_utils import detect_tuple_output, compute_resid_norms, get_layers
from layers.layer_determinism import (
    harden, write_fingerprint, fingerprint_tensor, fingerprint_json,
)
from layers.layer_localize import (
    run_localization, reduce_layer_effects, get_top_k_layers,
    retrieve_random_k_layers, plot_layer_effects, layer_scores,
)
from layers.layer_steering import (
    layer_steering_cache, build_layer_vectors, generate_with_layer_patches,
)

import logging
logging.basicConfig(level=logging.WARNING)


def save_prompt_responses(responses, path):
    """Byte-identical output format to eval_runner's version.

    Copied rather than imported: ``eval.eval_runner`` does ``import pyreft`` at
    module top, and pyreft hard-pins transformers==4.45.1, so importing it here
    would couple this pipeline to a dependency it never uses.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for entry in responses:
            for _, v in entry.items():
                f.write(f"{v}\n")
            f.write('-' * 40 + '\n')
    with open(path.replace('.txt', '.json'), 'w') as jf:
        json.dump(responses, jf)
    print(f"[layers] saved responses to {path} and {path.replace('.txt', '.json')}")


def _atomic_torch_save(obj, path):
    """Write via a temp file in the same directory, then rename.

    The steering cache is deliberately shared across localizations -- that is what
    makes it reusable -- which means two concurrently running eval jobs
    (LOC_PAIR=...-long and LOC_PAIR=...-single) target the SAME file with the same
    steering set. torch.save is not atomic, so a plain save lets one job read a
    half-written file the other is still producing. os.replace is atomic on POSIX
    within a filesystem, so a reader sees either the old file or the complete new
    one, never a partial.

    Two jobs racing to build it is harmless beyond the wasted compute: the
    computation is deterministic, so whichever write lands last is byte-identical
    to the one it replaced.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_json_dump(obj, path):
    """Same atomicity as _atomic_torch_save; the baseline is now shared too."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
    os.close(fd)
    try:
        with open(tmp, 'w') as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _tolerant_torch_load(path):
    """Load, treating a corrupt file as absent rather than fatal.

    Guards against caches left behind by a non-atomic writer -- an interrupted or
    preempted job, or a run from before _atomic_torch_save existed. Rebuilding
    costs one pass; crashing costs the queue slot.
    """
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"[layers] cache at {path} is unreadable ({type(e).__name__}: {e}); "
              f"rebuilding.")
        return None


def _steering_cache_path(config):
    """Shared across cells so the same steering set is not re-traced per cell.

    Keyed on the steering directory alone for ``last_token`` (left padding makes
    the last position pad-independent). ``all_tokens`` averages over the padded
    span, so its cache genuinely depends on the eval set that set max_len and the
    key carries the eval directory too.
    """
    model = config.args.model_id.split('/')[-1]
    # localization_root, not results_root: when eval cells write into
    # per-experiment roots (see LayerConfig), results_root differs per cell and the
    # cache would be rebuilt for each one. localization_root is the shared root.
    root = config.args.localization_root.rstrip('/')
    steer_dir = config.args.steering_add_path.split('/')[-2]
    stype = config.args.steering_type
    name = f"{steer_dir}__{stype}"
    if stype != 'last_token':
        eval_dir = (config.args.eval_test.split('/')[-2]
                    if isinstance(config.args.eval_test, str) else config.args.source)
        name += f"__{eval_dir}"
    return os.path.join(root, model, '_steering_cache', f"{name}.pt")


def _load_or_build_steering_cache(config, model, data_handler, is_tuple):
    """Build the steering cache, or reuse one built under identical conditions.

    The cache is shared across cells, which makes its provenance a determinism
    hazard: it is accumulated batch by batch, so a cache built at batch 8 differs
    in the last bits from one built at batch 16, and reusing it across runs with
    different batch sizes would quietly mix the two. The build conditions are
    stored alongside it and a mismatch forces a rebuild rather than a silent reuse.
    """
    path = _steering_cache_path(config)
    build_key = {
        'batch_size': config.args.steering_norm_batch,
        'n_pairs': int(data_handler.steering_qs_toks['add']['input_ids'].shape[0]),
        'steering_type': config.args.steering_type,
        'model_id': config.args.model_id,
        # The cache is produced by forward passes, so the attention backend is part
        # of its provenance: a cache built under math SDPA is not the same tensor as
        # one built under flash, and reusing one under the other mixes numerics
        # across cells without any visible symptom.
        'sdp_backend': config.args.sdp_backend,
    }

    if os.path.exists(path):
        blob = _tolerant_torch_load(path)
        if blob is not None and blob.get('build_key') == build_key:
            print(f"[layers] reusing steering cache {path}")
            return blob['cache'], blob['spread']
        if blob is not None:
            print(f"[layers] steering cache at {path} was built under "
                  f"{blob.get('build_key')} but this run needs {build_key} -- rebuilding.")

    cache, spread = layer_steering_cache(
        model, data_handler, is_tuple, batch_size=config.args.steering_norm_batch)
    _atomic_torch_save({'cache': cache, 'spread': spread, 'build_key': build_key}, path)
    return cache, spread


def _unsteered_outputs(config, data_handler, model):
    """Baseline generations, cached and SHARED across cells.

    The baseline depends only on the eval set and the generation settings -- not on
    the localization or the steering vector. Keying it per cell meant regenerating
    it once per cell: eight times for two distinct eval sets. It now lives beside
    the steering cache, keyed on what it actually depends on.

    Those keys are load-bearing rather than bookkeeping. With left padding, batched
    generation is batch-size dependent, and the steered and unsteered arms are
    compared item by item -- so a baseline reused from a run with a different batch
    size or gen_mode yields a difference that is an artefact of the settings rather
    than of steering. A mismatch rebuilds.
    """
    model_name = config.args.model_id.split('/')[-1]
    # Shared root, same reason as the steering cache: the unsteered baseline
    # depends on the eval set and generation settings, not on which experiment
    # directory this cell's generations happen to land in.
    root = config.args.localization_root.rstrip('/')
    build_key = {
        'batch_size': config.args.batch_size,
        'max_new_tokens': config.args.max_new_tokens,
        'n_items': int(data_handler.LEN),
        'gen_mode': config.args.gen_mode,
        'model_id': config.args.model_id,
        'sdp_backend': config.args.sdp_backend,
    }
    path = os.path.join(
        root, model_name, '_baselines',
        f"{config.args.test_dataset}__b{config.args.batch_size}"
        f"__t{config.args.max_new_tokens}__{config.args.gen_mode}.json")

    if os.path.exists(path):
        try:
            with open(path) as f:
                blob = json.load(f)
            if isinstance(blob, dict) and blob.get('build_key') == build_key:
                print(f"[layers] reusing unsteered baseline {path}")
                return blob['outputs']
        except (json.JSONDecodeError, OSError) as e:
            print(f"[layers] baseline at {path} unreadable ({type(e).__name__}); "
                  f"regenerating.")

    outputs = []
    batch_handler = BatchHandler(config, data_handler)
    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
    for _ in tqdm(range(0, min(data_handler.LEN, len_gen_qs), config.args.batch_size),
                  desc="Unsteered baseline"):
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        with model.generate(gen_qs_toks,
                            pad_token_id=model.tokenizer.eos_token_id,
                            use_cache=(config.args.gen_mode != 'recompute'),
                            do_sample=False, top_p=None, top_k=None,
                            temperature=None,
                            max_new_tokens=config.args.max_new_tokens) as _:
            op = model.generator.output.save()
        outputs += op.cpu().numpy().tolist()
        batch_handler.update()

    _atomic_json_dump({'outputs': outputs, 'build_key': build_key}, path)
    return outputs


def run_layer_eval(config, data_handler, model_handler, is_tuple):
    set_seed(config.args.seed)
    model = model_handler.model
    model.eval()
    prefix = config.get_output_prefix().rstrip('/')
    eval_dir = f"{prefix}/eval"
    os.makedirs(eval_dir, exist_ok=True)

    is_random = config.args.patch_algo == 'random'
    per_layer = config.args.sweep_mode == 'per_layer'
    reps_type = 'random' if is_random else 'targeted'
    logit_metric = 'random' if is_random else 'numerator_1'
    ablation = config.args.ablation
    num_layers = len(get_layers(model))

    if per_layer and is_random:
        sys.exit(
            "[layers] --sweep_mode per_layer with --patch_algo random is not meaningful: "
            "the per-layer sweep already measures every layer, so there is no selection "
            "left to randomize. The random arm exists to give top-k selection something "
            "to beat; the per-layer sweep replaces that comparison with a direct one "
            "between predicted (ATP) and measured effect at each layer.")

    effects = None
    if not is_random:
        # In per-layer mode the map is a PREDICTION to be checked against the sweep,
        # not an input that chooses layers -- so a missing map is a reason to warn,
        # not to abort. The sweep is the ground truth and stands on its own.
        try:
            effects = reduce_layer_effects(config, force=config.args.force_reduce)
            plot_layer_effects(config, effects, eval_dir)
        except FileNotFoundError:
            if not per_layer:
                raise
            print("[layers] no attribution map found; running the per-layer sweep "
                  "without it. The ATP-vs-measured comparison will be unavailable "
                  "until the localization pass has run.")

    cache, spread = _load_or_build_steering_cache(config, model, data_handler, is_tuple)

    norms = None
    if config.args.steering_scale == 'relative':
        norms = compute_resid_norms(
            model, select_gen_qs_toks(config, data_handler), is_tuple,
            batch_size=config.args.steering_norm_batch)

    write_fingerprint(config, eval_dir)
    with open(f"{eval_dir}/steering_meta.json", 'w') as f:
        json.dump({
            'steering_scale': config.args.steering_scale,
            'n_scale': config.args.n_scale,
            'n_vals': config.args.n_vals,
            'alphas': [n * config.args.n_scale for n in config.args.n_vals],
            'sweep_mode': config.args.sweep_mode,
            # What the third numeric field of each gen filename means. In topk mode
            # it is a COUNT of layers steered together; in per_layer mode it is a
            # single layer INDEX. Same filename shape, different meaning -- recorded
            # here so a directory can be read correctly without guessing.
            'sweep_axis': 'layer_index' if config.args.sweep_mode == 'per_layer' else 'top_k',
            'topk_layers': (list(range(len(get_layers(model_handler.model))))
                            if config.args.sweep_mode == 'per_layer'
                            else config.args.topk_layers),
            'rank_by': config.args.rank_by,
            'steering_type': config.args.steering_type,
            'gen_mode': config.args.gen_mode,
            'gen_batch_size': config.args.batch_size,
            'resid_norms': None if norms is None else norms.tolist(),
            'pair_spread': spread.tolist(),
            # Content hashes of every input to the sweep. Two runs that agree here
            # but disagree downstream have a nondeterministic generation path;
            # two runs that disagree here never had a chance to match.
            'fingerprints': {
                'attribution_map': None if effects is None else fingerprint_tensor(effects),
                'steering_cache': fingerprint_tensor(cache),
                'resid_norms': None if norms is None else fingerprint_tensor(norms),
            },
        }, f, indent=2)

    original_outputs = _unsteered_outputs(config, data_handler, model)

    # Both modes reduce to the same thing: a list of (label, layers) pairs. The label
    # is what lands in the gen filename's third numeric field -- a layer count in topk
    # mode, a layer index in per_layer mode -- so the sweep loop below is shared and
    # the two modes cannot drift apart.
    if per_layer:
        sweep = [(layer, [layer]) for layer in range(num_layers)]
        sweep_desc = "Layers"
    else:
        sweep = [(k, None) for k in config.args.topk_layers]
        sweep_desc = "Layer counts"

    for label, preselected in tqdm(sweep, desc=sweep_desc):
        topk_csv = f"{eval_dir}/{logit_metric}_{reps_type}_{label}.csv"
        if os.path.exists(topk_csv):
            topk_df = pd.read_csv(topk_csv)
        elif preselected is not None:
            # Per-layer: the layer is given. Carry its attribution score alongside so
            # the predicted and measured effects sit in one file per layer.
            score = marginal = float('nan')
            if effects is not None:
                mean, marg, _ = layer_scores(effects, config.args.rank_by)
                score, marginal = float(mean[label]), float(marg[label])
            topk_df = pd.DataFrame({'layer': [label], 'value': [score],
                                    'marginal': [marginal], 'rank_score': [score]})
            topk_df.to_csv(topk_csv, index=False)
        else:
            if is_random:
                topk_df = retrieve_random_k_layers(num_layers, label, seed=config.args.seed)
            else:
                topk_df = get_top_k_layers(effects, label, rank_by=config.args.rank_by)
            topk_df.to_csv(topk_csv, index=False)
        selected = [int(x) for x in topk_df['layer'].tolist()]
        axis = 'layer' if per_layer else 'k'
        print(f"[layers] {axis}={label} -> steering layers {selected}")

        vectors = build_layer_vectors(
            cache, selected,
            scale=config.args.steering_scale,
            norms=norms,
            steering_type=config.args.steering_type,
            attention_mask=None)

        for N in tqdm(config.args.n_vals, desc=f"N sweep ({axis}={label})", leave=False):
            gen_txt = (f"{eval_dir}/{N}_{reps_type}_{ablation}_{label}_"
                       f"{config.args.test_dataset}_gen.txt")
            gen_json = gen_txt.replace('.txt', '.json')
            if os.path.exists(gen_txt) and os.path.exists(gen_json):
                print(f"[layers] skipping {axis}={label} N={N}; gen files present.")
                continue

            alpha = N * config.args.n_scale
            print(f"[layers] steer {axis}={label} N={N} alpha={alpha:g} "
                  f"scale={config.args.steering_scale} "
                  f"eval={config.args.test_dataset} "
                  f"loc=from_{config.args.source}_to_{config.args.base}")

            decoded_all = []
            batch_handler = BatchHandler(config, data_handler)
            len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
            for idx in tqdm(range(0, min(data_handler.LEN, len_gen_qs), config.args.batch_size),
                            desc="Batches", leave=False):
                gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                edited = generate_with_layer_patches(
                    model, gen_qs_toks, vectors, alpha, is_tuple,
                    ablation=ablation,
                    max_new_tokens=config.args.max_new_tokens,
                    gen_mode=config.args.gen_mode)
                decoded_all += decode_responses(
                    model, gen_qs_toks,
                    original_outputs[idx:idx + config.args.batch_size],
                    edited, config.args.base)
                batch_handler.update()
                gc.collect()
                torch.cuda.empty_cache()

            save_prompt_responses(decoded_all, gen_txt)

    print("[layers] evaluation complete.")


def main():
    print('[layers] parsing config...')
    config = LayerConfig()
    # Config.setup_environment() -- assert_determinism_env() + set_seed() -- has
    # already run by this point, via LayerConfig.__init__ -> Config.__init__.
    # Must precede the first forward/backward: SDPA dispatch is read at kernel
    # launch, and the fused backward kernels are the dominant source of
    # run-to-run drift in the attribution map.
    if config.args.strict_determinism and 'gpt-oss' in config.args.model_id.lower():
        print("[layers] WARNING: --strict_determinism on a MoE model. gpt-oss routing "
              "uses scatter/index_add kernels with no deterministic implementation, so "
              "warn_only=False will RAISE rather than warn. Drop the flag (SDPA is "
              "still pinned by --sdp_backend math) or expect the run to abort and name "
              "the offending op.")
    harden(sdp_backend=config.args.sdp_backend,
           strict=config.args.strict_determinism)


    print('[layers] loading model...')
    model_handler = ModelHandler(config)
    config.args.batch_size = 5
    data_handler = DataHandler(config, model_handler)

    if config.args.limit_items:
        data_handler.truncate_to_len(min(config.args.limit_items, data_handler.LEN))
        print(f"[layers] limited to {data_handler.LEN} items")

    is_tuple = detect_tuple_output(
        model_handler.model, model_handler.tokenizer, config.args.device)

    if config.args.patch_model:
        # ATP holds four full activation sets plus their graph; batch of 1, as in run.py.
        config.args.batch_size = 1
        loc_dir = config.get_output_prefix().rstrip('/')
        write_fingerprint(config, loc_dir)
        run_localization(config, data_handler, model_handler, is_tuple)
        effects = reduce_layer_effects(config, force=True)
        print(f"[layers] attribution map fingerprint: {fingerprint_tensor(effects)}")
        plot_layer_effects(config, effects, f"{loc_dir}/eval")

    if config.args.eval_model:
        # The original guard avoids a trailing batch of exactly 1, which the
        # padding path handles badly. A user-supplied size is respected unless it
        # would produce that remainder.
        auto = max(x for x in range(16, 0, -1) if data_handler.LEN % x != 1)
        requested = config.args.gen_batch_size
        if requested is None:
            config.args.batch_size = auto
        elif data_handler.LEN % requested == 1:
            print(f"[layers] --gen_batch_size {requested} leaves a trailing batch of 1 "
                  f"for {data_handler.LEN} items; using {auto} instead.")
            config.args.batch_size = auto
        else:
            config.args.batch_size = requested
        print(f"[layers] generation batch size {config.args.batch_size}, "
              f"gen_mode={config.args.gen_mode}")
        if not config.args.steering:
            sys.exit("[layers] --eval_model requires --steering for the layer pipeline.")
        run_layer_eval(config, data_handler, model_handler, is_tuple)


if __name__ == "__main__":
    main()
