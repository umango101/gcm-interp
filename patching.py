import torch
import gc
from patching_utils import PatchingUtils
import einops
import gc

def _val(p):
    """Unwrap an nnsight proxy to its underlying tensor; no-op on real tensors."""
    return getattr(p, "value", p)

class Patching:
    def __init__(self, model_handler, batch_handler, config):
        self.model_handler = model_handler
        self.batch_handler = batch_handler
        self.config = config
        self.print_now = False
        self.patching_utils = PatchingUtils(self)
        self.align_toks = self.patching_utils.align_toks
    
    def apply_patching(self):
        base_toks = self.batch_handler.base_toks
        source_qs_toks = self.batch_handler.source_qs_toks
        response_start_positions = self.batch_handler.response_start_positions
        model = self.model_handler.model
        if self.config.args.patch_algo == 'acp':
            base_desired_logits_post_patch = self.patching_utils.patch_heads(base_toks['desired'], source_qs_toks['desired'], response_start_positions['base']['desired'])

            base_undesired_logits_post_patch = self.patching_utils.patch_heads(base_toks['undesired'], source_qs_toks['undesired'], response_start_positions['base']['undesired'])
            
            base_desired_logits_pre_patch, _ = self.patching_utils.get_activations(base_toks['desired'], which_patch='heads', resp_start_positions=response_start_positions['base']['desired'])
            
            base_undesired_logits_pre_patch, _ = self.patching_utils.get_activations(base_toks['undesired'], which_patch='heads', resp_start_positions=response_start_positions['base']['undesired'])
            
            logits = torch.stack([
                base_desired_logits_post_patch, 
                base_undesired_logits_post_patch,
                base_desired_logits_pre_patch.expand(base_desired_logits_post_patch.shape).to(base_desired_logits_post_patch.device),
                base_undesired_logits_pre_patch.expand(base_undesired_logits_post_patch.shape).to(base_undesired_logits_post_patch.device)
            ], dim=0).detach().cpu()
            torch.cuda.empty_cache()
            gc.collect()
            return logits
        elif 'atp' in self.config.args.patch_algo:
            base_desired_logits, base_desired_attn = self.patching_utils.get_activations(base_toks['desired'], which_patch='heads', resp_start_positions=response_start_positions['base']['desired'], retain_grad=True, logit=True)
            
            base_undesired_logits, base_undesired_attn = self.patching_utils.get_activations(base_toks['undesired'], which_patch='heads', resp_start_positions=response_start_positions['base']['undesired'], retain_grad=True, logit=True)

            if self.config.args.patch_algo == 'atp-zero':
                source_q_des_attn = [torch.zeros_like(bda) for bda in base_desired_attn]
                source_q_undes_attn = [torch.zeros_like(bua) for bua in base_undesired_attn]
            else:
                source_q_des_attn = self.patching_utils.get_activations(source_qs_toks['desired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=True, base_toks=base_toks['desired'])
                source_q_undes_attn = self.patching_utils.get_activations(source_qs_toks['undesired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=True, base_toks=base_toks['undesired'])

            L = base_undesired_logits - base_desired_logits
            L.backward(retain_graph=True)
            attn_desired_effects = []
            attn_undesired_effects = []
            net_effects = []
            for idx in range(len(model.model.layers)):
                p = base_desired_attn[idx]
                t = getattr(p, "value", p)
                print(type(p).__name__, type(t).__name__, "grad is None:", t.grad is None)
                # attn_desired_effects.append(
                #     base_desired_attn[idx].grad * 
                #     (source_q_des_attn[idx] - base_desired_attn[idx])
                # )
                # attn_undesired_effects.append(
                #     base_undesired_attn[idx].grad * 
                #     (source_q_undes_attn[idx] - base_undesired_attn[idx])
                # )
                bd = _val(base_desired_attn[idx])
                bu = _val(base_undesired_attn[idx])
                sd = _val(source_q_des_attn[idx])
                su = _val(source_q_undes_attn[idx])
                attn_desired_effects.append(bd.grad * (sd - bd))
                attn_undesired_effects.append(bu.grad * (su - bu))

                net_effects.append(attn_desired_effects[idx].sum(dim=1) + attn_undesired_effects[idx].sum(dim=1))
            net_effects = torch.stack([h for h in net_effects], dim=0).detach().cpu()
            gc.collect()
            torch.cuda.empty_cache()
            print('net_effects', net_effects.shape)
            return net_effects
