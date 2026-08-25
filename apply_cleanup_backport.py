#!/usr/bin/env python3
"""Backport the `cleanup`-branch changes that `conflicts` is still missing.

Run from the repo root of a `conflicts` checkout:

    python apply_cleanup_backport.py --check    # report only, change nothing
    python apply_cleanup_backport.py            # apply

Idempotent: re-running after a successful apply reports "already applied" for
every hunk and writes nothing. All-or-nothing: if any hunk's anchor fails to
match exactly once, no file is written.

WHAT THIS DOES AND DOES NOT TOUCH
---------------------------------
Of the three `cleanup` changes, only one is genuinely absent here:

  1. patching_utils.get_response_logits scoring window -- APPLIED BELOW.
     `cleanup` moved the scored span from `logits[rsp:-1] -> ids[rsp+1:]` to
     `logits[rsp-1:-1] -> ids[rsp:]`.  DataHandler.get_resp_start_pos returns
     `j + marker_len`, i.e. the index of the FIRST answer token, so the old
     window predicted the answer's *second* token onward.  On the single-token
     conflict datasets the assistant turn renders as
     `<|message|>{word}<|return|>`, so the old window scored `<|return|>` and
     nothing else -- the answer token never entered the log-likelihood, and
     every ATP attribution and logit-diff built on it is measuring the wrong
     thing.

  2. o_proj.output -> o_proj.input.  ALREADY PRESENT on this branch, in a more
     general form: ModelHandler.head_site / head_site_proxy / head_slice plus
     the `--head_site` flag, which every launcher pins to `o_proj_input`.
     One leftover hardcoded `.output` remains, in the currently-uncalled
     eval/attn_attributions.get_attn_tensors -- routed through a site helper
     below so it cannot come back wrong when someone calls it.

  3. Determinism.  ALREADY PRESENT on this branch and STRICTER than
     `cleanup`'s determinism.py: eval/setup.assert_determinism_env() +
     set_seed() turn TF32 off and require the env to be exported before
     interpreter start, whereas determinism.py leaves TF32 on and sets
     CUBLAS_WORKSPACE_CONFIG from inside Python.  Deliberately NOT backported:
     it would be a regression.

Also applied: the o_proj.in_features cross-check from `cleanup`'s
ModelHandler, which catches a head_dim/geometry mismatch at load time instead
of silently mis-slicing every head.
"""

import argparse
import sys
from pathlib import Path

# (path, description, old_anchor, new_text)
# old_anchor must appear exactly once in the unmodified file.
EDITS = [
    (
        "patching_utils.py",
        "scoring window, retain_grad branch: rsp+1 -> rsp",
        """                log_probs[i, response_start_position:-1, :].gather(-1, toks['input_ids'][i, 1+response_start_position:].unsqueeze(-1)).squeeze(-1).sum()
                for i, response_start_position in enumerate(resp_start_positions)
            ])
        else:""",
        """                log_probs[i, response_start_position-1:-1, :].gather(-1, toks['input_ids'][i, response_start_position:].unsqueeze(-1)).squeeze(-1).sum()
                for i, response_start_position in enumerate(resp_start_positions)
            ])
        else:""",
    ),
    (
        "patching_utils.py",
        "scoring window, no-grad branch: rsp+1 -> rsp",
        """                log_probs[i, response_start_position:-1, :].gather(-1, toks['input_ids'][i, 1+response_start_position:].unsqueeze(-1)).squeeze(-1).sum()
                for i, response_start_position in enumerate(resp_start_positions)
            ]).detach().cpu()""",
        """                log_probs[i, response_start_position-1:-1, :].gather(-1, toks['input_ids'][i, response_start_position:].unsqueeze(-1)).squeeze(-1).sum()
                for i, response_start_position in enumerate(resp_start_positions)
            ]).detach().cpu()""",
    ),
    (
        "patching_utils.py",
        "DEBUG_RSP=1 one-shot dump of the scored span",
        """    def get_response_logits(self, toks, resp_start_positions, logits, retain_grad=False):
        if retain_grad:""",
        '''    def get_response_logits(self, toks, resp_start_positions, logits, retain_grad=False):
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
        if retain_grad:''',
    ),
    (
        "eval/attn_attributions.py",
        "get_attn_tensors: head_site-aware write site",
        """def get_attn_tensors(model, gen_toks, patch_activations, topk_df, N, ablation_type, DIM, edit=True):""",
        """def get_attn_tensors(model, gen_toks, patch_activations, topk_df, N, ablation_type, DIM, edit=True, head_site=None):
    # Must match ModelHandler.head_site, or the steering vectors are written
    # into a space they were never measured in.
    def _head_site_proxy(layer_module):
        if head_site == 'o_proj_input':
            return layer_module.self_attn.o_proj.input
        return layer_module.self_attn.o_proj.output
""",
    ),
    (
        "eval/attn_attributions.py",
        "get_attn_tensors: steer write routed through the site helper",
        """                                layer.self_attn.o_proj.output[..., :patch_activations.shape[1], sl] += N * patch_activations[layer_idx][:, sl]""",
        """                                _head_site_proxy(layer)[..., :patch_activations.shape[1], sl] += N * patch_activations[layer_idx][:, sl]""",
    ),
    (
        "model_handler.py",
        "cross-check head geometry against o_proj.in_features",
        """        if self.head_site == 'o_proj_input' and self.num_heads * self.head_dim != hidden_size:
            print("[head geometry] NOTE: n_heads*head_dim != hidden_size for this model; "
                  "o_proj.output slicing would not correspond to heads.")""",
        """        if self.head_site == 'o_proj_input' and self.num_heads * self.head_dim != hidden_size:
            print("[head geometry] NOTE: n_heads*head_dim != hidden_size for this model; "
                  "o_proj.output slicing would not correspond to heads.")

        # Cross-check the config-derived geometry against the real module: a
        # wrong head_dim mis-slices every head rather than raising.
        if self.head_site == 'o_proj_input':
            try:
                _oproj = getattr(self.model, "_model", self.model).model.layers[0].self_attn.o_proj
                _in_features = int(_oproj.in_features)
                if _in_features % self.num_heads != 0:
                    raise ValueError(
                        f"o_proj.in_features={_in_features} is not divisible by "
                        f"num_attention_heads={self.num_heads}; per-head slicing "
                        f"would be meaningless.")
                if _in_features // self.num_heads != self.head_dim:
                    print(f"[head geometry] config head_dim {self.head_dim} disagrees with "
                          f"o_proj.in_features//heads = {_in_features // self.num_heads}; "
                          f"using the module's value.", flush=True)
                    self.head_dim = _in_features // self.num_heads
                    self.dim = self.head_dim
                    self.head_width = self.num_heads * self.head_dim
            except (AttributeError, IndexError) as _e:
                print(f"[head geometry] could not read o_proj.in_features ({_e}); "
                      f"keeping head_dim={self.head_dim} from config.", flush=True)""",
    ),
]


def sentinel_for(old, new):
    """A string present in the file only AFTER this edit is applied.

    For a replacement, `new` itself works and keeps the surrounding context, so
    two hunks that happen to change the same line (the two branches of
    get_response_logits) don't alias onto one sentinel. For a pure insertion,
    `new` contains `old` and would match the unapplied file forever, so fall
    back to the longest added line.
    """
    if old not in new:
        return new
    old_lines = set(old.splitlines())
    added = [ln for ln in new.splitlines() if ln.strip() and ln not in old_lines]
    if not added:
        raise AssertionError("edit adds no new line; cannot detect idempotence")
    return max(added, key=len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    pending, already, failed = 0, 0, 0
    texts = {}

    for rel, desc, old, new in EDITS:
        path = root / rel
        if not path.exists():
            print(f"MISSING  {rel}: file not found under {root}")
            failed += 1
            continue
        if rel not in texts:
            texts[rel] = path.read_text()
        text = texts[rel]

        if sentinel_for(old, new) in text:
            print(f"SKIP     {rel}: {desc} (already applied)")
            already += 1
            continue
        n = text.count(old)
        if n != 1:
            print(f"FAIL     {rel}: {desc} -- anchor matched {n} times, expected 1")
            failed += 1
            continue
        texts[rel] = text.replace(old, new, 1)
        print(f"{'WOULD':<8} {rel}: {desc}" if args.check else f"{'APPLY':<8} {rel}: {desc}")
        pending += 1

    if failed:
        print(f"\n{failed} hunk(s) failed -- nothing written. Resolve those by hand.")
        return 1
    if args.check:
        print(f"\ncheck: {pending} to apply, {already} already applied")
        return 0

    for rel, text in texts.items():
        (root / rel).write_text(text)
    print(f"\ndone: {pending} applied, {already} already applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
