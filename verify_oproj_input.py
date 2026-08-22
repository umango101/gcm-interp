#!/usr/bin/env python
"""Pre-flight check for switching localization/steering from o_proj.OUTPUT to
o_proj.INPUT. Run against the CURRENT, UNPATCHED code.

Proxy discipline (this is what the previous version got wrong): inside
`with model.trace(...)` everything is an nnsight proxy. Build the graph and
.save() there; do NOT call torch.as_tensor / .backward / arithmetic-that-needs-
a-real-tensor until the context has exited. Then unwrap with _val(). This
mirrors patching.py / patching_utils.get_activations, which trace, exit, and
only then call get_response_logits and L.backward().

  BLOCKERS  -- nnsight cannot write to or backprop through o_proj.input, or the
               per-head slice is not really a head. The switch cannot proceed.
  ADVISORY  -- model_handler.dim disagrees with the correct input-space head
               width. Expected wherever num_heads*head_dim != hidden_size
               (Gemma-3-12B: 16*256=4096 vs hidden_size 3840).

Every write is validated by its effect on o_proj.OUTPUT against the analytic
identity, rather than by reading the input proxy back -- a read-back in the same
trace can return the pre-intervention value depending on graph ordering, which
would tell us nothing.

    python verify_oproj_input.py --model_id google/gemma-3-12b-it --device cuda:0 --full_precision
"""
import argparse
import sys
import traceback

import torch


def _val(p):
    """Unwrap an nnsight proxy to its tensor; no-op on real tensors."""
    return getattr(p, "value", p)


class _Args:
    """Minimal stand-in for Config.args: what ModelHandler actually reads."""
    def __init__(self, model_id, device, full_precision):
        self.model_id = model_id
        self.device = device
        self.full_precision = full_precision
        self.pyreft = False
        self.source = "lying-single"   # any non-'harmful' value -> the \n-marker


class _Config:
    def __init__(self, args):
        self.args = args


def _write_slice(model, oproj, toks, sl, mode, value):
    """Apply one intervention to o_proj's input and return the resulting output.

    Tries `oproj.input[..., sl]` first, then the `oproj.inputs[0][0]` args-tuple
    form, since which one is settable varies across nnsight versions. Returns
    (output_tensor, which_form_worked) or raises the last error.
    """
    errors = []
    for form in ("input", "inputs"):
        try:
            with model.trace(toks) as _:
                target = oproj.input if form == "input" else oproj.inputs[0][0]
                if mode == "set":
                    target[..., sl] = value
                else:
                    target[..., sl] += value
                out_p = oproj.output.save()
            return _val(out_p).detach().float().cpu(), form
        except Exception as e:               # noqa: BLE001 - we report and try the next form
            errors.append((form, repr(e)))
    raise RuntimeError("no settable form of o_proj input: " + "; ".join(f"{f}: {e}" for f, e in errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full_precision", action="store_true",
                    help="Needed for checks 1-2: 4-bit quantization hides o_proj.weight.")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    try:
        import nnsight
        print(f"nnsight {nnsight.__version__}")
    except Exception:
        pass
    from model_handler import ModelHandler

    print(f"=== {args.model_id} ===", flush=True)
    mh = ModelHandler(_Config(_Args(args.model_id, args.device, args.full_precision)))
    model, tok = mh.model, mh.tokenizer
    oproj = model.model.layers[args.layer].self_attn.o_proj

    blockers, advisories = [], []
    toks = tok(["The capital of France is", "The capital of Japan is"],
               return_tensors="pt", padding=True)
    toks = {k: v.to(mh.device) for k, v in toks.items()}

    # ---- 0. shapes, and the correct per-head width -------------------------
    try:
        with model.trace(toks) as _:
            z_p = oproj.input.detach().cpu().save()
            o_p = oproj.output.detach().cpu().save()
        z = _val(z_p).float()
        out = _val(o_p).float()
    except Exception:
        traceback.print_exc()
        print("\nBLOCKER: cannot even read o_proj.input.")
        return 1

    in_dim, out_dim = z.shape[-1], out.shape[-1]
    print(f"[0] o_proj.input  {tuple(z.shape)}")
    print(f"    o_proj.output {tuple(out.shape)}")
    print(f"    num_attention_heads = {mh.num_heads}")

    if in_dim % mh.num_heads != 0:
        blockers.append(f"o_proj.in_features {in_dim} not divisible by {mh.num_heads} heads")
        head_dim = None
    else:
        head_dim = in_dim // mh.num_heads
        print(f"    correct per-head width (input space) = {head_dim}")
        print(f"    model_handler.dim (current)          = {mh.dim}")
        if mh.dim != head_dim:
            advisories.append(
                f"model_handler.dim is {mh.dim} but the input space needs {head_dim} "
                f"({in_dim}/{mh.num_heads}). Apply the model_handler.py part of the patch."
            )
        if in_dim != out_dim:
            print(f"    NOTE: input {in_dim} != output {out_dim} -- "
                  f"num_heads*head_dim != hidden_size for this model.")

    W = getattr(oproj, "weight", None)
    have_W = W is not None and getattr(W, "dtype", None) not in (torch.uint8,)
    if not have_W:
        advisories.append("o_proj.weight unavailable (quantized); re-run with --full_precision "
                          "to verify the head-slice identity")
    Wc = W.detach().float().cpu() if have_W else None
    h = min(args.head, mh.num_heads - 1) if head_dim else 0
    sl = slice(h * head_dim, (h + 1) * head_dim) if head_dim else None
    scale = out.abs().max().item()

    # ---- 1. zeroing head h's input slice == removing z_h W_O^h -------------
    if head_dim and have_W:
        try:
            out_zeroed, form = _write_slice(model, oproj, toks, sl, "set", 0)
            contrib = z[..., sl] @ Wc[:, sl].T
            err = (out - contrib - out_zeroed).abs().max().item()
            print(f"[1] zero head {h} (via .{form}): "
                  f"max|out - z_h W_O^h - out_zeroed| = {err:.3e}  (scale {scale:.3e})")
            if err > max(0.01 * scale, 1e-3):
                blockers.append(f"head-slice identity failed (err {err:.3e} vs scale {scale:.3e})")
            else:
                print(f"    OK: input[..., {h*head_dim}:{(h+1)*head_dim}] is exactly head {h}, "
                      f"and writes via .{form} take effect")
        except Exception as e:
            print(f"[1] FAILED: {e}")
            blockers.append(f"cannot assign to o_proj input: {e}")
    else:
        print("[1] SKIPPED (need head width and o_proj.weight)")

    # ---- 2. in-place += , which is what steering actually does -------------
    if head_dim and have_W:
        try:
            bump = 0.1
            out_bumped, form = _write_slice(model, oproj, toks, sl, "add", bump)
            expected = out + (torch.full((head_dim,), bump) @ Wc[:, sl].T)
            err = (out_bumped - expected).abs().max().item()
            print(f"[2] += {bump} on head {h} (via .{form}): "
                  f"max|observed - predicted| = {err:.3e}  (scale {scale:.3e})")
            if err > max(0.01 * scale, 1e-3):
                blockers.append(f"in-place += did not apply as expected (err {err:.3e})")
            else:
                print("    OK: += on the input slice propagates exactly")
        except Exception as e:
            print(f"[2] FAILED: {e}")
            blockers.append(f"cannot += into o_proj input: {e}")
    else:
        print("[2] SKIPPED (need head width and o_proj.weight)")

    # ---- 3. gradients, which is what ATP needs -----------------------------
    # Trace and save inside; unwrap and call backward OUTSIDE, exactly as
    # patching.py does with L.backward().
    try:
        with model.trace(toks) as _:
            zin = oproj.input
            zin.retain_grad()
            saved = zin.save()
            logits_p = model.lm_head.output.save()
        logits = _val(logits_p)
        logits[:, -1, :].sum().backward()
        g = getattr(_val(saved), "grad", None)
        if g is None:
            print("[3] FAIL: retain_grad on o_proj.input produced no .grad")
            blockers.append("no gradient on o_proj.input -- ATP cannot use the input proxy as-is")
        else:
            gm = g.abs().mean().item()
            print(f"[3] grad {tuple(g.shape)}, abs mean {gm:.3e}")
            if gm == 0:
                blockers.append("gradient w.r.t. o_proj.input is identically zero")
            else:
                print("    OK: gradients reach o_proj.input")
    except Exception:
        traceback.print_exc()
        blockers.append("retain_grad/backward through o_proj.input raised")

    print()
    for a in advisories:
        print(f"ADVISORY: {a}")
    if blockers:
        print("\nBLOCKERS:")
        for b in blockers:
            print(f"  - {b}")
        print(f"\n{args.model_id}: NOT safe to switch as written.")
        return 1
    print(f"\n{args.model_id}: nnsight handles o_proj.input correctly. Safe to switch"
          + (" once the advisories above are addressed." if advisories else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
