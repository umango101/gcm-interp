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

import contextlib
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
                              align=False, base_toks=None, logit=True, build_graph=None):
        """Trace the residual streams.

        retain_grad=True  -- keep the proxies live on GPU so .grad can be read after
                             backward. Costs the full autograd graph of this pass.
        retain_grad=False -- detach to CPU. build_graph defaults to False here, which
                             runs the pass under torch.no_grad(): nothing downstream
                             differentiates through it, and the graph is the single
                             largest thing in this function.
        """
        model = self.model_handler.model
        resid = []
        if align:
            toks = self.align_toks(toks, base_toks)
        if build_graph is None:
            build_graph = retain_grad
        ctx = contextlib.nullcontext() if build_graph else torch.no_grad()
        with ctx:
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
            # retain_grad=False, and therefore no_grad, is deliberate and is the
            # difference between fitting on an H200 and not. Nothing below reads
            # sd.grad or su.grad -- the source activations enter only as VALUES in
            # (sd - bd). Tracing them with retain_grad=True kept two extra full
            # autograd graphs alive across the backward for gradients that were
            # never used, which on gpt-oss-20b is what pushed the second shard to
            # 138 GiB. Detaching to CPU also keeps them off the device until the
            # per-layer loop pulls each one back.
            src_des_h = self.get_resid_activations(
                source_qs_toks['desired'], resp_start_positions=None, logit=False,
                align=True, retain_grad=False, base_toks=base_toks['desired'])
            src_undes_h = self.get_resid_activations(
                source_qs_toks['undesired'], resp_start_positions=None, logit=False,
                align=True, retain_grad=False, base_toks=base_toks['undesired'])

        L = base_undes_ll - base_des_ll
        # get_response_logits returns one log-likelihood per item, so L is a vector.
        # patching.py calls .backward() on it directly, which only works because atp
        # is forced to batch_size=1. .sum() is identical at batch 1 (items are
        # independent, so their gradients simply accumulate) and does not blow up if
        # the batch size is ever raised.
        #
        # No retain_graph: there is exactly one backward here, and retaining pinned
        # both base graphs for the whole per-layer loop below.
        L.sum().backward()

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
            net_effects.append((des_effect.sum(dim=1) + undes_effect.sum(dim=1)).detach().cpu())
            del bd, bu, sd, su, des_effect, undes_effect

        net_effects = torch.stack(net_effects, dim=0).detach().cpu()  # [n_layers, batch, hidden]
        # Drop every reference to the retained graphs before releasing the cache;
        # gc.collect() with the proxies still bound frees nothing.
        del base_des_h, base_undes_h, src_des_h, src_undes_h
        del base_des_ll, base_undes_ll, L
        gc.collect()
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            print(f'[layers] net_effects {tuple(net_effects.shape)}  peak {peak:.1f} GiB',
                  flush=True)
        else:
            torch.cuda.empty_cache()
            print('[layers] net_effects', tuple(net_effects.shape))
        return net_effects
