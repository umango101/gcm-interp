"""Diff-in-means steering vectors on the residual stream, and steered generation.

The vector at layer l is

    v[l] = mean_i h[l] (steering_add_i) - mean_i h[l] (steering_sub_i)

over the row-aligned minimally contrastive pairs in ``{task}-steering.jsonl``.
Because the pairs are aligned, this is identical to the mean of the per-pair
differences -- the diff-in-diff estimate -- and it is computed that way here so
the per-pair spread can be reported alongside it. A steering vector whose mean
is small relative to its across-pair spread is not a direction; it is noise, and
that is worth seeing before a whole sweep is read as a null result.

Which vector is applied to which eval set is controlled entirely by
``--steering_add_path``/``--steering_sub_path`` versus ``--eval_test``, exactly
as in the head pipeline, so the cross-steer cells need no special casing here.
"""

import gc

import torch
from tqdm import tqdm

from layers.layer_utils import get_layers, resid_out, resid_add, resid_set, val


@torch.no_grad()
def layer_steering_cache(model, data_handler, is_tuple, batch_size=8):
    """Return (cache, spread).

    cache  -- [n_layers, seq, hidden] mean per-pair residual difference
    spread -- [n_layers] mean across-pair std of the last-token difference,
              a sanity check on whether the direction is actually consistent
    """
    add_toks = data_handler.steering_qs_toks['add']
    sub_toks = data_handler.steering_qs_toks['sub']
    n_add = add_toks['input_ids'].shape[0]
    n_sub = sub_toks['input_ids'].shape[0]
    if n_add != n_sub:
        raise ValueError(
            f"Steering add/sub sets must be row-aligned minimal pairs, got "
            f"{n_add} and {n_sub} rows. A mean-of-means over unequal sets is not "
            f"the paired difference this experiment assumes.")

    num_layers = len(get_layers(model))
    running = None          # [n_layers, seq, hidden] running sum of per-pair diffs
    last_tok_diffs = [[] for _ in range(num_layers)]
    count = 0

    for i in tqdm(range(0, n_add, batch_size), desc="Steering cache"):
        a = {'input_ids': add_toks['input_ids'][i:i + batch_size].to(model.device),
             'attention_mask': add_toks['attention_mask'][i:i + batch_size].to(model.device)}
        b = {'input_ids': sub_toks['input_ids'][i:i + batch_size].to(model.device),
             'attention_mask': sub_toks['attention_mask'][i:i + batch_size].to(model.device)}

        with model.trace(a) as _:
            a_h = [resid_out(layer, is_tuple).detach().cpu().save() for layer in get_layers(model)]
        with model.trace(b) as _:
            b_h = [resid_out(layer, is_tuple).detach().cpu().save() for layer in get_layers(model)]

        diffs = [(val(x).to(torch.float32) - val(y).to(torch.float32))
                 for x, y in zip(a_h, b_h)]                    # each [b, seq, hidden]
        batch_sum = torch.stack([d.sum(dim=0) for d in diffs], dim=0)   # [n_layers, seq, hidden]
        running = batch_sum if running is None else running + batch_sum
        for li, d in enumerate(diffs):
            last_tok_diffs[li].append(d[:, -1, :])
        count += a['input_ids'].shape[0]
        gc.collect()
        torch.cuda.empty_cache()

    cache = running / count
    spread = torch.stack([torch.cat(d, dim=0).std(dim=0).mean() for d in last_tok_diffs])
    last_norms = torch.stack([cache[l, -1, :].norm() for l in range(num_layers)])
    print(f"[layers] steering cache {tuple(cache.shape)} over {count} pairs; "
          f"last-token ||mean diff|| min={last_norms.min():.2f} max={last_norms.max():.2f}")
    return cache, spread


def build_layer_vectors(cache, layers, scale='relative', norms=None,
                        steering_type='last_token', attention_mask=None):
    """Turn the cache into one vector per selected layer, at the requested scale.

    relative -- unit direction rescaled to ``norms[l]``, the mean residual norm of
                the prompts being steered. The eval-time coefficient alpha then
                reads as a multiple of a typical residual, which is comparable
                across layers and across the cross-steer cells.
    raw      -- the diff-in-means vector untouched (CAA convention).
    unit     -- unit norm, matching the head pipeline's convention. Negligible
                against a residual stream; useful only as a parity check.
    """
    vectors = {}
    for l in layers:
        l = int(l)
        if steering_type == 'last_token':
            v = cache[l, -1, :]
        elif steering_type == 'all_tokens':
            if attention_mask is None:
                v = cache[l].mean(dim=0)
            else:
                # Left padding means a plain mean averages in pad positions, which
                # drags the direction toward whatever the pad embedding produces.
                m = attention_mask.to(cache.dtype).unsqueeze(-1)
                v = (cache[l] * m).sum(dim=0) / m.sum().clamp(min=1)
        else:
            raise ValueError(f"Unknown steering_type: {steering_type!r}")

        v = v.to(torch.float32)
        if scale == 'raw':
            pass
        elif scale == 'unit':
            v = v / (v.norm() + 1e-12)
        elif scale == 'relative':
            if norms is None:
                raise ValueError("steering_scale=relative needs the residual-norm reference.")
            v = v / (v.norm() + 1e-12) * float(norms[l])
        else:
            raise ValueError(f"Unknown steering_scale: {scale!r}")
        vectors[l] = v
    return vectors


def generate_with_layer_patches(model, gen_toks, vectors, alpha, is_tuple,
                                ablation='steer', max_new_tokens=256,
                                gen_mode='all_steps'):
    """Generate with ``alpha * vectors[l]`` written into each selected residual stream.

    The head pipeline offered two options, and they are a false dichotomy:
    ``kv_caching=True`` steered the prefill only (fast, but generated tokens are
    never steered), and ``kv_caching=False`` steered every position by disabling
    the cache and recomputing the whole sequence at every decode step. The second
    is what you want semantically and is quadratic in the number of generated
    tokens -- at max_new_tokens=256 it does roughly two orders of magnitude more
    work than it needs to.

    The third option is ``use_cache=True`` combined with ``model.all()``: the
    intervention runs on every forward, but with the cache intact each decode
    forward computes only the new position. Every position is still steered
    exactly once -- prompt positions during prefill, generated positions as they
    are produced -- so this is semantically identical to the uncached path at
    linear cost.

    gen_mode:
      all_steps -- use_cache=True + model.all(). Steers prompt and every generated
                   token. The default, and equivalent to `recompute`.
      prefill   -- use_cache=True, prefill only. Generated tokens unsteered. This
                   is also what "steer prompt positions only" means, so it
                   subsumes the old steer_positions='prompt'.
      recompute -- use_cache=False + model.all(). The old default. Kept as the
                   numerical reference `all_steps` is checked against, not because
                   it does anything the default does not.
    """
    gen_kwargs = dict(
        pad_token_id=model.tokenizer.eos_token_id,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        max_new_tokens=max_new_tokens,
    )

    def _write():
        for l, v in vectors.items():
            layer = get_layers(model)[int(l)]
            value = (alpha * v).to(model.device)
            if ablation == 'steer':
                resid_add(layer, is_tuple, value)
            elif ablation == 'mean':
                resid_set(layer, is_tuple, value)
            else:
                raise ValueError(
                    f"Layer steering implements 'steer' (add) and 'mean' (replace); "
                    f"got {ablation!r}.")

    if gen_mode == 'prefill':
        with model.generate(gen_toks, use_cache=True, **gen_kwargs) as _:
            _write()
            generated = model.generator.output.save()
    elif gen_mode == 'all_steps':
        with model.generate(gen_toks, use_cache=True, **gen_kwargs) as _:
            with model.all():
                _write()
            generated = model.generator.output.save()
    elif gen_mode == 'recompute':
        with model.generate(gen_toks, use_cache=False, **gen_kwargs) as _:
            with model.all():
                _write()
            generated = model.generator.output.save()
    else:
        raise ValueError(f"Unknown gen_mode: {gen_mode!r}")
    return generated
