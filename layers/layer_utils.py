"""Hook-point helpers for the layer-level (residual-stream) pipeline.

The head-level code hooks ``layer.self_attn.o_proj.output``, which is always a
bare tensor. The residual stream is ``layer.output``, whose type is NOT stable
across transformers versions: through 4.53.x the decoder layers of Llama/Qwen2/
OLMo2/Gemma3 return ``(hidden_states, ...)``; 4.54+ simplified several of them
to return ``hidden_states`` directly. Since `syc` has drifted between 4.53.3 and
4.57.1 more than once, every read/write here goes through an accessor that is
told which shape it is dealing with, and the shape is detected once at startup
with a real 2-token trace rather than inferred from a version string.
"""

import torch


def get_layers(model):
    """The decoder layer list, for both plain and nnsight-wrapped models."""
    return model.model.layers


def detect_tuple_output(model, tokenizer, device):
    """Return True if decoder layers emit a tuple, False if a bare tensor.

    One forward pass on a 2-token input. Cheap, and far safer than branching on
    transformers.__version__.
    """
    toks = tokenizer("a b", return_tensors="pt")
    inputs = {
        "input_ids": toks["input_ids"].to(device),
        "attention_mask": toks["attention_mask"].to(device),
    }
    with model.trace(inputs) as _:
        out = get_layers(model)[0].output.save()
    val = getattr(out, "value", out)
    is_tuple = isinstance(val, (tuple, list))
    print(f"[layers] decoder layer output is {'a tuple' if is_tuple else 'a bare tensor'}")
    return is_tuple


def resid_out(layer, is_tuple):
    """Read proxy for the residual stream leaving ``layer``. Call inside a trace."""
    return layer.output[0] if is_tuple else layer.output


def resid_add(layer, is_tuple, value, upto=None):
    """In-place add ``value`` to the residual stream. Call inside a trace.

    ``upto`` bounds the write to the first ``upto`` sequence positions; None
    writes every position.
    """
    if is_tuple:
        if upto is None:
            layer.output[0][:] += value
        else:
            layer.output[0][:, :upto, :] += value
    else:
        if upto is None:
            layer.output[:] += value
        else:
            layer.output[:, :upto, :] += value


def resid_set(layer, is_tuple, value, upto=None):
    """In-place overwrite of the residual stream (the 'mean'/replace ablation)."""
    if is_tuple:
        if upto is None:
            layer.output[0][:] = value
        else:
            layer.output[0][:, :upto, :] = value
    else:
        if upto is None:
            layer.output[:] = value
        else:
            layer.output[:, :upto, :] = value


def val(p):
    """Unwrap an nnsight proxy to its tensor; no-op on real tensors.

    Same helper as patching.py's ``_val``. ``.grad`` is intercepted on the proxy
    and creates an empty node when read after the trace has closed, while
    ``requires_grad``/``grad_fn`` fall through to the real tensor -- which is why
    a broken proxy looks healthy right up until ``.grad`` is dereferenced.
    """
    return getattr(p, "value", p)


@torch.no_grad()
def compute_resid_norms(model, toks, is_tuple, batch_size=8, position=-1):
    """Mean L2 norm of the residual stream at each layer, over a set of prompts.

    Used as the reference scale for ``--steering_scale relative``: a coefficient
    of alpha means "add a vector alpha times as long as the residual stream
    typically is at this layer, at this position".

    The norms are measured on the prompts being STEERED (the eval/test set), not
    on the prompts the steering vector was derived from. That matters for the
    cross-steer cells: it keeps alpha meaning the same thing whether the vector
    came from the free-form or the single-token dataset.
    """
    num_layers = len(get_layers(model))
    sums = torch.zeros(num_layers, dtype=torch.float64)
    count = 0
    n = toks["input_ids"].shape[0]
    for i in range(0, n, batch_size):
        sl = {
            "input_ids": toks["input_ids"][i:i + batch_size].to(model.device),
            "attention_mask": toks["attention_mask"][i:i + batch_size].to(model.device),
        }
        with model.trace(sl) as _:
            saved = [resid_out(layer, is_tuple).detach().cpu().save()
                     for layer in get_layers(model)]
        for li, h in enumerate(saved):
            h = val(h).to(torch.float32)          # [b, seq, hidden]
            sums[li] += h[:, position, :].norm(dim=-1).sum().item()
        count += sl["input_ids"].shape[0]
    norms = (sums / max(count, 1)).to(torch.float32)
    print(f"[layers] residual norms (mean over {count} prompts) "
          f"min={norms.min():.1f} max={norms.max():.1f}")
    return norms
