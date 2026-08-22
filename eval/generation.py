import os
from tqdm import tqdm
import torch
from pathlib import Path

def select_gen_qs_toks(config, batch_handler):
    if config.args.eval_train:
        print("Evaluating on training set.")
        return batch_handler.base_qs_toks['desired']
    elif config.args.eval_test:
        print("Evaluating on test set. ", batch_handler.base_qs_toks['test']['input_ids'].shape[0])
        return batch_handler.base_qs_toks['test']
    elif config.args.eval_transfer:
        print("Evaluating on eval_test dataset.")
        return batch_handler.eval_transfer['queries']
    else:
        raise ValueError("Either eval_train or eval_test must be True.")
def generate_with_patches(model, gen_toks, patch_activations, topk_df, N, ablation_type, DIM, max_new_tokens=256, normalize=True, steering_type='last_token', kv_caching=False):
    patch_activations = patch_activations['desired'].to(model.device)
    layer_ids = topk_df['layer'].unique()
    print(f"Generating for ", gen_toks['input_ids'].shape, " with normalization set to ", normalize, " steering type ", steering_type, " kv_caching ", kv_caching)

    gen_kwargs = dict(
        pad_token_id=model.tokenizer.eos_token_id,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        max_new_tokens=max_new_tokens,
    )

    def _steering_vector(layer_idx, sl):
        if steering_type == 'last_token':
            vec = patch_activations[layer_idx][-1, sl]
        elif steering_type == 'all_tokens':
            vec = patch_activations[layer_idx][:, sl].mean(dim=0)
        else:
            raise ValueError(f"Unknown steering_type: {steering_type!r}")
        if normalize:
            vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-12)
        return vec

    if kv_caching:
        # KV caching ON: no model.all(). The interventions apply during prefill only;
        # decoding steps read from the KV cache and are not re-steered. The write targets
        # the full o_proj input ([..., sl]) because prefill covers every prompt position
        # in a single forward pass. (Using model.all() here would reapply the write on each
        # 1-token decode step and index positions that don't exist -> corrupted output.)
        with model.generate(gen_toks, use_cache=True, **gen_kwargs) as tracer:
            for layer_idx in layer_ids:
                head_ids = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
                layer = model.model.layers[layer_idx]
                for head_idx in head_ids:
                    sl = slice(DIM * head_idx, DIM * (head_idx + 1))
                    steering_vector = _steering_vector(layer_idx, sl)
                    if ablation_type == 'mean':
                        layer.self_attn.o_proj.input[..., sl] = N * steering_vector
                    elif ablation_type == 'steer':
                        layer.self_attn.o_proj.input[..., sl] += N * steering_vector
            generated = model.generator.output.save()
    else:
        # KV caching OFF: model.all() reapplies the intervention on every decoding step,
        # so use_cache=False is required (each step recomputes all positions). The write is
        # bounded to the steered prompt positions ([..., :patch_activations.shape[1], sl]).
        with model.generate(gen_toks, use_cache=False, **gen_kwargs) as tracer:
            with model.all():
                for layer_idx in layer_ids:
                    head_ids = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
                    layer = model.model.layers[layer_idx]
                    for head_idx in head_ids:
                        sl = slice(DIM * head_idx, DIM * (head_idx + 1))
                        steering_vector = _steering_vector(layer_idx, sl)
                        if ablation_type == 'mean':
                            layer.self_attn.o_proj.input[..., :patch_activations.shape[1], sl] = N * steering_vector
                        elif ablation_type == 'steer':
                            layer.self_attn.o_proj.input[..., :patch_activations.shape[1], sl] += N * steering_vector
            generated = model.generator.output.save()

    return generated

def decode_responses(model, inputs, originals, edited, base, answers=None):
    decoded = []
    for i in tqdm(range(len(originals)), desc="Decoding Responses"):
        query = model.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
        orig = model.tokenizer.decode(originals[i], skip_special_tokens=True).split(query)[-1]
        edit = model.tokenizer.decode(edited[i], skip_special_tokens=True).split(query)[-1]
        to_append = {
            'query': query,
            f'old_{base}': orig,
            f'edit_{base}': edit
        }
        if answers is not None:
            to_append['answer'] = answers[i]
        decoded.append(to_append)
    assert len(decoded) > 0, "No responses decoded. Check the generation process."
    return decoded
