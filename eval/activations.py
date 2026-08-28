import os


def _head_site(layer, head_site=None):
    """Per-head activation site.

    head_site comes from ModelHandler.head_site (i.e. from --head_site) so that
    caching and steering cannot disagree. The HEAD_SITE env var is only a
    fallback for callers with no ModelHandler in scope; if neither is set on a
    model where num_heads*head_dim != hidden_size, the cache would be read at
    o_proj.output (width hidden_size) while generation writes at o_proj.input
    (width num_heads*head_dim), silently mismatching every head slice.
    """
    site = head_site or os.environ.get('HEAD_SITE')
    if site is None:
        raise ValueError(
            "head_site is unset; pass model_handler.head_site or export HEAD_SITE")
    if site == 'o_proj_input':
        return layer.self_attn.o_proj.input
    return layer.self_attn.o_proj.output


import torch

def mean_ablations_cache(model, data_handler, batch_size=9, key='desired', head_site=None):
    toks = data_handler.source_qs_toks[key]
    attn_layer_cache = [[] for _ in range(len(model.model.layers))]
    for i in range(0, toks['input_ids'].shape[0], batch_size):
        input_slice = {
            'input_ids': toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': toks['attention_mask'][i:i+batch_size].to(model.device)
        }
        with model.trace(input_slice) as _:
            for idx, layer in enumerate(model.model.layers):
                attn_layer_cache[idx].append(_head_site(layer, head_site).detach().cpu().save())
    attn_cache = [torch.cat(attns_in_layer, dim=0).mean(dim=0).to(model.device) for attns_in_layer in attn_layer_cache]
    return torch.stack(attn_cache)

def steering_reps_cache(model, data_handler, batch_size=9, key='desired', mean=True, head_site=None):
    source_toks = data_handler.steering_qs_toks['add']
    base_toks = data_handler.steering_qs_toks['sub']
    num_layers = len(model.model.layers)
    steer = [[] for _ in range(num_layers)]
    base = [[] for _ in range(num_layers)]

    # print(base_toks['input_ids'].shape, source_toks['input_ids'].shape)
    # print('BASE TOKS INPUT IDS ', base_toks['input_ids'][0])
    # print('BASE TOKS TOKENS ', model.tokenizer.convert_ids_to_tokens(base_toks['input_ids'][0]))
    # print('source TOKS INPUT IDS ', source_toks['input_ids'][0])
    # print('source TOKS TOKENS ', model.tokenizer.convert_ids_to_tokens(source_toks['input_ids'][0]))

    # No gradients are needed to cache activations. Without this the
    # autograd graph for every traced layer is held for the whole loop,
    # which is what exhausts the card on the longer rule-form prompts.
    # Numerically identical; memory only.
    with torch.no_grad():
        for i in range(0, source_toks['input_ids'].shape[0], batch_size):
            s_slice = {
                'input_ids': source_toks['input_ids'][i:i+batch_size].to(model.device),
                'attention_mask': source_toks['attention_mask'][i:i+batch_size].to(model.device)
            }
            b_slice = {
                'input_ids': base_toks['input_ids'][i:i+batch_size].to(model.device),
                'attention_mask': base_toks['attention_mask'][i:i+batch_size].to(model.device)
            }

            with model.trace(s_slice) as _:
                for idx, layer in enumerate(model.model.layers):
                    steer[idx].append(_head_site(layer, head_site).detach().cpu().save())
            with model.trace(b_slice) as _:
                for idx, layer in enumerate(model.model.layers):
                    base[idx].append(_head_site(layer, head_site).detach().cpu().save())

    if mean:
        print('########### Mean steering cache ########### ', source_toks['input_ids'].shape[0], steer[0][0].shape, base[0][0].shape, len(steer[0]), len(base))
        cache = [torch.cat(steer[i], dim=0).mean(0) - torch.cat(base[i], dim=0).mean(0) for i in range(num_layers)]
        print('########### Mean steering cache after ########### ', cache[0].shape)
    else:
        cache = [torch.cat(steer[i], dim=0) - torch.cat(base[i], dim=0) for i in range(num_layers)]
        print('########### Steering cache after ########### ', cache[0].shape)
    print('Stacked steering cache ', torch.stack(cache).shape, model.config)
    if key == 'desired':
        # Was: f'{model_id.split("/")[0]}_..._steer.pt' in the CWD -- that is the
        # ORG ("openai"), and the name omits the sub set and the eval set, so
        # concurrent runs of the same source overwrite each other's cache.
        cfg = data_handler.config
        out_dir = os.path.join(cfg.get_output_prefix(), 'eval')
        os.makedirs(out_dir, exist_ok=True)
        add = os.path.basename(cfg.args.steering_add_path).replace('.jsonl', '')
        sub = os.path.basename(cfg.args.steering_sub_path).replace('.jsonl', '')
        torch.save(torch.stack(cache),
                   os.path.join(out_dir, f'steering_cache_{add}_minus_{sub}.pt'))
    return torch.stack(cache)