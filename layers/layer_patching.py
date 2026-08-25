"""Attribution patching over layer residual streams.

Same estimator as ``patching.py``'s atp branch, with the hook point moved from
``layer.self_attn.o_proj.output`` (per-head) to ``layer.output`` (the residual
stream leaving the block):

    L                = loglik(undesired) - loglik(desired)     on the base prompts
    effect[l]        = dL/dh[l] * (h_source[l] - h_base[l])
    net_effect[l]    = effect_desired[l].sum(seq) + effect_undesired[l].sum(seq)

so ``net_effect[l]`` is the first-order estimate of what happens to L if the
whole residual stream at layer l were swapped for the source's. That is exactly
the intervention the layer-level steering arm performs, which is the point of
matching them.

Note this quantity is CUMULATIVE: the residual stream at layer l carries every
upstream contribution, so effect[l] is not "what layer l does". The reduction
step offers a marginal ranking (effect[l] - effect[l-1]) for that reading.
"""

import gc

import torch

from patching_utils import PatchingUtils
from layers.layer_utils import get_layers, resid_out, val


class LayerPatching:
    """Drop-in analogue of ``Patching`` for residual streams.

    Exposes ``config``/``model_handler``/``batch_handler`` so ``PatchingUtils``
    can be constructed against it and its ``get_response_logits`` (the
    log-likelihood metric) and ``align_toks`` reused unchanged.
    """

    def __init__(self, model_handler, batch_handler, config, is_tuple):
        self.model_handler = model_handler
        self.batch_handler = batch_handler
        self.config = config
        self.is_tuple = is_tuple
        self.patching_utils = PatchingUtils(self)
        self.align_toks = self.patching_utils.align_toks

    def get_resid_activations(self, toks, resp_start_positions=None, retain_grad=False,
                              align=False, base_toks=None, logit=True):
        model = self.model_handler.model
        resid = []
        if align:
            toks = self.align_toks(toks, base_toks)
        with model.trace(toks) as _:
            for layer in get_layers(model):
                h = resid_out(layer, self.is_tuple)
                if retain_grad:
                    h.retain_grad()
                    resid.append(h.save())
                else:
                    resid.append(h.detach().cpu().save())
            model_logits = model.lm_head.output.save()
        if logit:
            model_logits = self.patching_utils.get_response_logits(
                toks, resp_start_positions, model_logits, retain_grad=True)
            return model_logits, resid
        return resid

    def apply_patching(self):
        base_toks = self.batch_handler.base_toks
        source_qs_toks = self.batch_handler.source_qs_toks
        resp = self.batch_handler.response_start_positions
        model = self.model_handler.model

        if 'atp' not in self.config.args.patch_algo:
            raise ValueError(
                f"Layer localization supports atp / atp-zero only, got "
                f"{self.config.args.patch_algo!r}. (Random layer selection is chosen at "
                f"eval time with --patch_algo random and needs no localization pass.)")

        base_des_ll, base_des_h = self.get_resid_activations(
            base_toks['desired'], resp_start_positions=resp['base']['desired'],
            retain_grad=True, logit=True)
        base_undes_ll, base_undes_h = self.get_resid_activations(
            base_toks['undesired'], resp_start_positions=resp['base']['undesired'],
            retain_grad=True, logit=True)

        if self.config.args.patch_algo == 'atp-zero':
            src_des_h = [torch.zeros_like(val(h)) for h in base_des_h]
            src_undes_h = [torch.zeros_like(val(h)) for h in base_undes_h]
        else:
            src_des_h = self.get_resid_activations(
                source_qs_toks['desired'], resp_start_positions=None, logit=False,
                align=True, retain_grad=True, base_toks=base_toks['desired'])
            src_undes_h = self.get_resid_activations(
                source_qs_toks['undesired'], resp_start_positions=None, logit=False,
                align=True, retain_grad=True, base_toks=base_toks['undesired'])

        L = base_undes_ll - base_des_ll
        # get_response_logits returns one log-likelihood per item, so L is a vector.
        # patching.py calls .backward() on it directly, which only works because atp
        # is forced to batch_size=1. .sum() is identical at batch 1 (items are
        # independent, so their gradients simply accumulate) and does not blow up if
        # the batch size is ever raised.
        L.sum().backward(retain_graph=True)

        net_effects = []
        for idx in range(len(get_layers(model))):
            bd, bu = val(base_des_h[idx]), val(base_undes_h[idx])
            sd, su = val(src_des_h[idx]), val(src_undes_h[idx])
            if bd.grad is None or bu.grad is None:
                raise RuntimeError(
                    f"No gradient reached the residual stream at layer {idx}. The proxy was "
                    f"unwrapped before reading .grad, so this is not the proxy-shadowing "
                    f"failure -- check that the traces were built with retain_grad=True and "
                    f"that L.backward() ran before this loop.")
            des_effect = bd.grad * (sd.to(bd.device) - bd)
            undes_effect = bu.grad * (su.to(bu.device) - bu)
            # sum over sequence -> [batch, hidden]; hidden is reduced later so the
            # shards stay small and the reduction can be re-run without re-tracing.
            net_effects.append(des_effect.sum(dim=1) + undes_effect.sum(dim=1))

        net_effects = torch.stack(net_effects, dim=0).detach().cpu()  # [n_layers, batch, hidden]
        gc.collect()
        torch.cuda.empty_cache()
        print('[layers] net_effects', tuple(net_effects.shape))
        return net_effects
