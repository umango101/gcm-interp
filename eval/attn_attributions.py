import torch
import math
import matplotlib.pyplot as plt
import re
import numpy as np
import torch.nn.functional as F
import nnsight
from tqdm import tqdm
def reshape_for_heads(x, num_heads):
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    x = x.view(batch_size, seq_len, num_heads, head_dim)
    return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

def compute_attention_weights(Q, K):
    # Q: [layers, batch, seq_len, q_heads, d_head]
    # K: [layers, batch, seq_len, k_heads, d_head]

    Q = Q.permute(0, 3, 1, 2, 4)  # [layers, q_heads, batch, seq_len, d_head]
    K = K.permute(0, 3, 1, 2, 4)  # [layers, k_heads, batch, seq_len, d_head]

    n_layers, q_heads, batch_size, seq_len, d_head = Q.shape
    k_heads = K.shape[1]
    gqa_factor = q_heads // k_heads

    attention_weights = torch.zeros((n_layers, q_heads, batch_size, seq_len, seq_len), device=Q.device)

    for l in range(n_layers):
        for h in range(q_heads):
            k_idx = h // gqa_factor  # map q head to shared k head
            for b in range(batch_size):
                Q_b = Q[l, h, b]  # [seq_len, d_head]
                K_b = K[l, k_idx, b]  # [seq_len, d_head]
                scores = torch.matmul(Q_b, K_b.transpose(0, 1)) / math.sqrt(d_head)
                attn = F.softmax(scores, dim=-1)
                attention_weights[l, h, b] = attn

    return attention_weights  # [layers, q_heads, batch, seq_len, seq_len]

def clean_token(t):
    return re.sub(r"^[^a-zA-Z0-9]+", "", t)  # remove non-alphanumeric prefix

import pickle

def plot_layer_heads(pre_attn, post_attn, layer_head_list, seq_len_in, original_tokens, edited_tokens, op_path, avg=False, pkl_path=None):
    original_tokens = [clean_token(t) if t not in ['<pad>', '<s>', '</s>', '<unk>'] else t for t in original_tokens]
    edited_tokens = [clean_token(t) if t not in ['<pad>', '<s>', '</s>', '<unk>'] else t for t in edited_tokens]
    num_pad_tokens = original_tokens.count('endoftext|>') + original_tokens.count('<pad>') + original_tokens.count('<s>') + original_tokens.count('</s>')
    num_plots = len(layer_head_list)
    cols = min(num_plots, 3)
    rows = (num_plots + cols - 1) // cols
    fig_width = 24 * cols
    fig_height = 6 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    ### dictionary to store matrices
    combined_dict = {}

    for idx, (layer, head) in tqdm(enumerate(layer_head_list)):
        ax = axes[idx]
        layer = int(layer)
        head = int(head)
        pre_attn_matrix = pre_attn[layer, head].detach().cpu().numpy()
        post_attn_matrix = post_attn[layer, head].detach().cpu().numpy()

        # Focus on region: input original_tokens (query) × output original_tokens (key)
        pre_crop = pre_attn_matrix[num_pad_tokens:seq_len_in, seq_len_in:]
        post_crop = post_attn_matrix[num_pad_tokens:seq_len_in, seq_len_in:]

        # Concatenate side-by-side
        combined_matrix = np.concatenate([pre_crop, post_crop], axis=1)

        ### store in dict
        combined_dict[(layer, head)] = combined_matrix

        # Labels
        x_labels_pre = original_tokens[seq_len_in:]
        x_labels_post = edited_tokens[seq_len_in:]
        y_labels = original_tokens[num_pad_tokens:seq_len_in]

        if avg:
            x_labels_pre = range(len(x_labels_pre))
            x_labels_post = range(len(x_labels_post))
            y_labels = range(len(y_labels))

        combined_x_labels = list(x_labels_pre) + list(x_labels_post)

        # Plot
        im = ax.imshow(combined_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
        ax.set_title(f'Layer {layer}, Head {head}', fontsize=14)
        ax.set_xlabel("Pre | Post (Key tokens)", fontsize=12)
        ax.set_ylabel("Query (input tokens)", fontsize=12)
        ax.set_xticks(range(len(combined_x_labels)))
        ax.set_xticklabels(combined_x_labels, rotation=90, fontsize=9)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=10)

        # Separator line between pre and post
        ax.axvline(x=len(x_labels_pre)-0.5, color='white', linestyle='--', linewidth=1)

        fig.colorbar(im, ax=ax, shrink=0.7)

    # Turn off unused axes
    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.suptitle("Pre vs Post Attention Heads (Side-by-Side)", fontsize=18)
    plt.savefig(op_path, bbox_inches='tight')
    plt.close(fig)

    ### Save pickle if path is given
    pkl_path = pkl_path or op_path.replace('.png', '.pkl')
    with open(pkl_path, "wb") as f:
        pickle.dump(combined_dict, f)



def get_attn_tensors(model, gen_toks, patch_activations, topk_df, N, ablation_type, DIM, edit=True, head_site=None):
    # Must match ModelHandler.head_site, or the steering vectors are written
    # into a space they were never measured in.
    def _head_site_proxy(layer_module):
        if head_site == 'o_proj_input':
            return layer_module.self_attn.o_proj.input
        return layer_module.self_attn.o_proj.output

    with torch.no_grad():
        if edit:
            patch_activations = patch_activations['desired']
            layer_ids = topk_df['layer'].unique()
            with model.generate(
                gen_toks,
                pad_token_id=model.tokenizer.eos_token_id,
                use_cache=False, 
                do_sample=False,  
                top_p=None, 
                top_k=None, 
                temperature=None, 
                max_new_tokens=256
            ):
                q_hidden_states = [nnsight.list().save() for _ in range(len(model.model.layers))]
                k_hidden_states = [nnsight.list().save() for _ in range(len(model.model.layers))]
                with model.all():
                    for layer_idx in layer_ids:
                        head_ids = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
                        layer = model.model.layers[layer_idx]
                        for head_idx in head_ids:
                            sl = slice(DIM * head_idx, DIM * (head_idx + 1))
                            if ablation_type == 'steer':
                                _head_site_proxy(layer)[..., :patch_activations.shape[1], sl] += N * patch_activations[layer_idx][:, sl]
                    for i, layer in enumerate(model.model.layers):
                        q_hidden_states[i].append(layer.self_attn.q_proj.output)
                        k_hidden_states[i].append(layer.self_attn.k_proj.output)

                generated = model.generator.output.save()
        else:
            with model.generate(
                gen_toks,
                pad_token_id=model.tokenizer.eos_token_id,
                use_cache=False, 
                do_sample=False,  
                top_p=None, 
                top_k=None, 
                temperature=None, 
                max_new_tokens=256
            ):
                q_hidden_states = [nnsight.list().save() for _ in range(len(model.model.layers))]
                k_hidden_states = [nnsight.list().save() for _ in range(len(model.model.layers))]
                with model.all():
                    for i, layer in enumerate(model.model.layers):
                        q_hidden_states[i].append(layer.self_attn.q_proj.output)
                        k_hidden_states[i].append(layer.self_attn.k_proj.output)

                generated = model.generator.output.save()

        q_tensor = torch.stack([
            q_hidden_states[layer][-1]
            for layer in range(len(model.model.layers))
        ], dim=0)

        k_tensor = torch.stack([
            k_hidden_states[layer][-1]
            for layer in range(len(model.model.layers))
        ], dim=0)

        q_num_heads = model.config.num_attention_heads  # usually 32
        q_head_dim = q_tensor.shape[-1] // q_num_heads
        q_tensor = q_tensor.view(len(model.model.layers), q_hidden_states[0][-1].shape[0], q_hidden_states[0][-1].shape[1], q_num_heads, q_head_dim)

        k_head_dim = q_head_dim  # usually the same head_dim
        k_num_heads = k_tensor.shape[-1] // k_head_dim
        k_tensor = k_tensor.view(len(model.model.layers), k_hidden_states[0][-1].shape[0], k_hidden_states[0][-1].shape[1], k_num_heads, k_head_dim)
        return generated, q_tensor.detach().cpu(), k_tensor.detach().cpu()