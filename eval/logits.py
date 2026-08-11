import torch
import gc
from tqdm import tqdm

def patch_heads_and_get_logit(model, DIM, patching_reps, top_indices, base_toks, resp_start, N, ablation_type, get_response_logits, top_tokens=False, head_site=None):
    # head_site must match the site localization read from, or the vectors are
    # written into a space they were never measured in.  Defaults to the
    # historical o_proj.output for backwards compatibility.
    patching_reps = patching_reps.to(model.device)

    def _site(layer_module):
        if head_site == 'o_proj_input':
            return layer_module.self_attn.o_proj.input
        return layer_module.self_attn.o_proj.output

    with model.trace(base_toks) as _:
        for _, row in top_indices.iterrows():
            layer = int(row['layer'])
            head = int(row['neuron'])
            head_slice = slice(DIM * head, DIM * (head + 1))
            site = _site(model.model.layers[layer])
            if ablation_type == 'mean':
                site[:, :patching_reps.shape[1], head_slice] = N * patching_reps[layer][:, head_slice]
            elif ablation_type == 'steer':
                site[:, :patching_reps.shape[1], head_slice] += N * patching_reps[layer][:, head_slice]
        logits = model.lm_head.output.save()
    if top_tokens:
        last_logits = logits[:, -1:, :].detach().cpu()
        top_tokens = torch.argmax(last_logits, dim=-1)
        gc.collect()
        torch.cuda.empty_cache()
        return top_tokens
    
    logits = get_response_logits(base_toks, resp_start, logits)
    gc.collect()
    torch.cuda.empty_cache()
    return logits

def get_logits_before_patch(model, batch_handler, get_response_logits):
    base_toks = batch_handler.base_toks
    response_start_positions = batch_handler.response_start_positions
    scores = {}
    for key in ['desired', 'undesired']:
        with model.trace(base_toks[key]) as _:
            logits = model.lm_head.output.save()
        logits = get_response_logits(base_toks[key], response_start_positions['base'][key], logits)
        scores[key] = logits
    gc.collect()
    torch.cuda.empty_cache()
    stacked = torch.stack([scores['desired'], scores['undesired']])
    return stacked

def compute_logit_scores(batch_handler, topk_df, patching_reps, model_handler, ablation_type, get_response_logits, N):
    base_toks = batch_handler.base_toks
    response_start_positions = batch_handler.response_start_positions
    scores = {}
    for key in ['desired', 'undesired']:
        logits = patch_heads_and_get_logit(
            model_handler.model,
            model_handler.head_dim,
            patching_reps[key],
            topk_df,
            base_toks[key],
            response_start_positions['base'][key],
            N,
            ablation_type,
            get_response_logits,
            head_site=model_handler.head_site,
        )
        scores[key] = logits
    stacked = torch.stack([scores['desired'], scores['undesired']])
    return stacked

def get_heads(model, DIM, patching_reps, toks, N, ablation_type, patch=True, head_site=None):
    heads_by_layer = []

    for layer in tqdm(range(len(model.model.layers)), desc="Collecting heads for layer"):
        heads = []
        for head in range(model.config.num_attention_heads):
            head_slice = slice(DIM * head, DIM * (head + 1))
            with model.trace(toks) as _:
                if patch == True:
                    heads.append(N * patching_reps[layer][:, head_slice].detach().cpu())
                else:
                    site = (model.model.layers[layer].self_attn.o_proj.input
                            if head_site == 'o_proj_input'
                            else model.model.layers[layer].self_attn.o_proj.output)
                    heads.append(site[:, :, head_slice].detach().cpu().save())
        heads_by_layer.append(torch.stack(heads, dim =1))

    heads_by_layer = torch.stack(heads_by_layer)
    gc.collect()
    torch.cuda.empty_cache()
    return heads_by_layer