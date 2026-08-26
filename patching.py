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
            # The underlying HF module; nnsight's LanguageModel wrapper does not
            # forward .parameters().
            hf_model = getattr(model, "_model", model)

            base_desired_logits, base_desired_attn = self.patching_utils.get_activations(base_toks['desired'], which_patch='heads', resp_start_positions=response_start_positions['base']['desired'], retain_grad=True, logit=True)
            
            base_undesired_logits, base_undesired_attn = self.patching_utils.get_activations(base_toks['undesired'], which_patch='heads', resp_start_positions=response_start_positions['base']['undesired'], retain_grad=True, logit=True)

            # Unwrap once, up front: every use below wants the real tensor.
            base_desired_attn = [_val(p) for p in base_desired_attn]
            base_undesired_attn = [_val(p) for p in base_undesired_attn]

            if self.config.args.patch_algo == 'atp-zero':
                source_q_des_attn = [torch.zeros_like(bda) for bda in base_desired_attn]
                source_q_undes_attn = [torch.zeros_like(bua) for bua in base_undesired_attn]
            else:
                # retain_grad=False on the SOURCE passes. These activations only
                # ever appear as values inside (source - base); nothing is
                # differentiated with respect to them. Keeping retain_grad=True
                # pinned two entire forward graphs -- including gpt-oss's eager
                # attention matrices, O(seq^2 * n_heads) per layer -- on the GPU
                # for the whole backward pass, for nothing. The no-grad path in
                # get_activations returns them detached on CPU instead.
                source_q_des_attn = [_val(p) for p in self.patching_utils.get_activations(source_qs_toks['desired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=False, base_toks=base_toks['desired'])]
                source_q_undes_attn = [_val(p) for p in self.patching_utils.get_activations(source_qs_toks['undesired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=False, base_toks=base_toks['undesired'])]

            L = base_undesired_logits - base_desired_logits

            # inputs=... confines gradient accumulation to the head sites.
            # A bare .backward() also fills .grad for all ~20.9B parameters
            # (~42 GB in bf16), which nothing here ever reads and which survives
            # into the next batch -- that, not fragmentation, is what put batch 1
            # over the H200's 140 GB. Passing inputs= also prunes the backward
            # graph to the paths that actually reach these tensors.
            # retain_graph is gone: there is only one backward, and retaining
            # blocked autograd from freeing activations as it walked down.
            grad_targets = base_desired_attn + base_undesired_attn
            L.backward(inputs=grad_targets)

            assert base_desired_attn[0].grad is not None, (
                "no gradient on the head site -- ATP would return zeros.")

            net_effects = []
            with torch.no_grad():
                for idx in range(len(model.model.layers)):
                    bd = base_desired_attn[idx]
                    bu = base_undesired_attn[idx]
                    # Source activations now live on CPU (see above).
                    sd = source_q_des_attn[idx].to(bd.device, dtype=bd.dtype)
                    su = source_q_undes_attn[idx].to(bu.device, dtype=bu.dtype)
                    effect = (bd.grad * (sd - bd)).sum(dim=1) + (bu.grad * (su - bu)).sum(dim=1)
                    net_effects.append(effect.detach().cpu())
                    del sd, su, effect
            net_effects = torch.stack([h for h in net_effects], dim=0)

            # Drop every GPU reference this batch created before the next one
            # builds its forward graph.
            del L, grad_targets
            del base_desired_logits, base_undesired_logits
            del base_desired_attn, base_undesired_attn
            del source_q_des_attn, source_q_undes_attn
            for p in hf_model.parameters():
                p.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            print('net_effects', net_effects.shape)
            return net_effects
