import os
import torch
import torch.nn.functional as F
import gc
from index_utils import *
from tqdm import tqdm
import json

class PatchingUtils:
    def __init__(self, patching_handler):
        self.patching_handler = patching_handler
        self.config = patching_handler.config
        self.model_handler = patching_handler.model_handler
        self.batch_handler = patching_handler.batch_handler
        self.data_handler = patching_handler.batch_handler.data_handler
        self.align_toks = self.data_handler.align_toks
        self.index_utils = IndexUtils(self.model_handler, self.batch_handler.data_handler)
        
    def get_response_logits(self, toks, resp_start_positions, logits, retain_grad=False):
        # DEBUG_RSP=1 prints, once per process, exactly which tokens the
        # log-likelihood is summed over. "scored tokens" must START with the
        # answer word; if it starts with <|return|> the window is off by one.
        if os.environ.get("DEBUG_RSP") == "1" and not getattr(self, "_rsp_debugged", False):
            self._rsp_debugged = True
            tk = self.model_handler.tokenizer
            ids = toks['input_ids'][0]
            r = int(resp_start_positions[0])
            print("[rsp] marker tokens :", self.model_handler.alignment_tokens.tolist(), flush=True)
            print("[rsp] rsp index     :", r, "of", ids.shape[0], flush=True)
            print("[rsp] token AT rsp  :", repr(tk.decode([int(ids[r])])), flush=True)
            print("[rsp] scored tokens :", repr(tk.decode(ids[r:].tolist())), flush=True)
        if retain_grad:
            logits = getattr(logits, "value", logits)
            toks = {'input_ids': toks['input_ids'], 'attention_mask': toks['attention_mask']}
            # Slice to the scored window BEFORE the softmax. log_softmax over the
            # full sequence materialises a [seq, vocab] tensor -- ~1.2 GB at seq
            # 1500 with gpt-oss's 201k vocab -- and autograd saves its output for
            # the backward pass, so it stays resident. Every row but the response
            # window is then discarded by the gather. Doing the reduction on the
            # slice keeps the whole thing to a few rows, which also makes the
            # float32 cast affordable (better numerics than bf16 log_softmax).
            log_likelihoods = []
            for i, response_start_position in enumerate(resp_start_positions):
                r = int(response_start_position)
                sel = logits[i, r-1:-1, :].float()
                tgt = toks['input_ids'][i, r:].unsqueeze(-1).to(sel.device)
                log_likelihoods.append(
                    (sel.gather(-1, tgt).squeeze(-1) - sel.logsumexp(dim=-1)).sum()
                )
            log_likelihoods = torch.stack(log_likelihoods)
        else:
            logits = logits.detach().cpu()
            log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
            toks = {'input_ids': toks['input_ids'].detach().cpu(), 'attention_mask': toks['attention_mask'].detach().cpu()}
            log_likelihoods = torch.stack([
                log_probs[i, response_start_position-1:-1, :].gather(-1, toks['input_ids'][i, response_start_position:].unsqueeze(-1)).squeeze(-1).sum()
                for i, response_start_position in enumerate(resp_start_positions)
            ]).detach().cpu()
        toks = {'input_ids': toks['input_ids'].to(self.model_handler.device), 'attention_mask': toks['attention_mask'].to(self.model_handler.device)}
        return log_likelihoods

    def patch_heads(self, base_toks, source_toks, resp_start_positions):
        source_toks = self.align_toks(source_toks, base_toks)
        model = self.model_handler.model
        # Geometry comes from ModelHandler, not from hidden_size // num_heads:
        # those differ whenever num_heads * head_dim != hidden_size (gpt-oss,
        # gemma-3), and the per-head slice then has to live on o_proj.input.
        num_heads = self.model_handler.num_heads
        source_heads = self.get_activations(source_toks, which_patch='heads', base_toks=base_toks, align=True, logit=False)
        with torch.no_grad():
            layer_results = []
            for layer_idx in tqdm(range(len(model.model.layers))):
                head_results = []
                for head_idx in range(num_heads):
                    head = self.model_handler.head_slice(head_idx)
                    with model.trace(base_toks) as invoker:
                        self.model_handler.head_site_proxy(
                            model.model.layers[layer_idx])[:, :, head] = \
                            source_heads[layer_idx][:, :, head].to(model.device)
                        logits = model.lm_head.output.detach().cpu().save()
                    head_results.append(self.get_response_logits(base_toks, resp_start_positions, logits)) # Shape: [batch_size]
                layer_results.append(torch.stack(head_results).detach().cpu()) # Shape: [num_heads, batch_size]
        layer_results = torch.stack(layer_results).detach().cpu() # Shape: [num_layers, num_heads, batch_size]
        return layer_results
    
    def get_activations(self, toks, which_patch='heads', resp_start_positions=None, retain_grad=False, align=False, base_toks=None, logit=True):
        model = self.model_handler.model
        attn_effects = []
        if align:
            toks = self.align_toks(toks, base_toks)
        with model.trace(toks) as _:
            for layer in model.model.layers:
                if which_patch =='heads':
                    self_attn = self.model_handler.head_site_proxy(layer)
                if retain_grad:
                    self_attn.retain_grad()
                    attn_effects.append(self_attn.save())
                else:
                    attn_effects.append(self_attn.detach().cpu().save())
            model_logits = model.lm_head.output.save()
        if logit:
            model_logits = self.get_response_logits(toks, resp_start_positions, model_logits, retain_grad=True)
            return model_logits, attn_effects
        else:
            return attn_effects
