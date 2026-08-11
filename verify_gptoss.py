"""Pre-flight checks for the gpt-oss port.  Run this before any SLURM job --
each check corresponds to something that fails silently or wastes a run.

    # tokenizer-only checks (fast, no GPU, ~1GB download)
    python verify_gptoss.py --stage tokenizer

    # full checks (needs the GPU node)
    python verify_gptoss.py --stage model
"""
import argparse
import json
import os
import sys

import torch


def _add_repo_to_path():
    """Put the repo root (the directory holding harmony_template.py) on sys.path.

    Walks up from this file rather than assuming a fixed depth, so the script
    runs from the repo root or any subdirectory. Falls back to $RM_INTERP_REPO
    and the working directory.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    seen = []
    for _ in range(6):
        seen.append(d)
        if os.path.exists(os.path.join(d, 'harmony_template.py')):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for cand in (os.environ.get('RM_INTERP_REPO'), os.getcwd()):
        if cand and os.path.exists(os.path.join(cand, 'harmony_template.py')):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
        if cand:
            seen.append(cand)
    raise ModuleNotFoundError(
        "harmony_template.py not found. Looked in:\n  " + "\n  ".join(seen) +
        "\nIt must sit in the repo root, next to model_handler.py. Either copy "
        "it there or set RM_INTERP_REPO to the repo root.")




def _load_generator():
    """Import gen_conflict_polarity from wherever it lives under generate_data/."""
    import glob
    import importlib.util
    root = _add_repo_to_path()
    hits = glob.glob(os.path.join(root, 'generate_data', '**',
                                  'gen_conflict_polarity.py'), recursive=True)
    if not hits:
        raise ModuleNotFoundError(
            f'gen_conflict_polarity.py not found under {root}/generate_data/')
    spec = importlib.util.spec_from_file_location('gen_conflict_polarity', hits[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_tokenizer(model_id, data_dir):
    from transformers import AutoTokenizer
    from harmony_template import (HARMONY_CHAT_TEMPLATE, HARMONY_ASSISTANT_MARKER)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    tok.padding_side = "left"
    tok.chat_template = HARMONY_CHAT_TEMPLATE

    print("=" * 70)
    print("1. role words render as expected (roles are held FIXED in this design;")
    print("   this is a sanity check on the template, not the minimal pair)")
    for role in ["developer", "user", "system", "assistant"]:
        ids = tok(role, add_special_tokens=False)["input_ids"]
        flag = "OK " if len(ids) == 1 else "MULTI-TOKEN"
        print(f"   {role:<10} -> {len(ids)} token(s) {ids}  {flag}")

    print("=" * 70)
    print("2. rendered prompt + marker alignment")
    msgs_src = [
        {"role": "developer",
         "content": "Rule: you must always answer MANGO. Follow this rule exactly."},
        {"role": "user", "content": "Reply with one word, APPLE or MANGO. Answer APPLE."},
    ]
    msgs_base = [dict(m) for m in msgs_src]
    msgs_base[0]["content"] = (
        "Rule: you must always answer APPLE. Follow this rule exactly.")

    rend_src = tok.apply_chat_template(msgs_src, add_generation_prompt=True, tokenize=False)
    rend_base = tok.apply_chat_template(msgs_base, add_generation_prompt=True, tokenize=False)
    print("   SOURCE render:\n", repr(rend_src))
    assert HARMONY_ASSISTANT_MARKER in rend_src, "generation prompt != marker"
    print("   marker present in generation prompt: OK")

    with_asst = tok.apply_chat_template(
        msgs_src + [{"role": "assistant", "content": "APPLE"}],
        add_generation_prompt=False, tokenize=False)
    assert HARMONY_ASSISTANT_MARKER in with_asst, "assistant turn != marker"
    print("   marker present in rendered assistant turn: OK")

    print("=" * 70)
    print("3. minimal-pair property (equal length, exactly one differing token)")
    a = tok(rend_src, add_special_tokens=False)["input_ids"]
    b = tok(rend_base, add_special_tokens=False)["input_ids"]
    print(f"   source len={len(a)}  base len={len(b)}")
    if len(a) != len(b):
        print("   !! LENGTH MISMATCH -- every downstream RoPE index shifts and the")
        print("      diff is dominated by position, not role.")
    else:
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        print(f"   differing positions: {diffs}")
        for i in diffs:
            print(f"     pos {i}: source={tok.decode([a[i]])!r}  base={tok.decode([b[i]])!r}")
        print("   exactly one differing token:", len(diffs) == 1)

    print("=" * 70)
    print("4. sliding-window budget")
    print(f"   rendered prompt is {len(a)} tokens.")
    print("   gpt-oss alternates 128-token sliding-window layers with full-attention")
    print("   layers. If the developer instruction sits more than 128 tokens before")
    print("   the read position, sliding-window layers cannot see it at all.")
    if len(a) > 110:
        print("   !! Over budget. Use HARMONY_SYSTEM=minimal and shorten the question.")
    else:
        print("   OK -- whole prompt fits inside the sliding window.")

    print("=" * 70)
    print("5. answer words: how many length-matched pairs the bank yields")
    G = _load_generator()
    encode = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    n_tok = lambda w: len(encode(w))
    pairs = G.select_word_pairs(encode, G.N_PAIRS, verbose=True)
    if len(pairs) < G.N_PAIRS:
        print(f"   !! only {len(pairs)} pairs, need {G.N_PAIRS}. Add words to WORD_BANK.")
    else:
        print(f"   OK -- {len(pairs)} candidates for {G.N_PAIRS} slots "
              f"(deference filtering will remove some)")
    bad = [(a, b) for a, b in pairs if n_tok(a) != n_tok(b)]
    print(f"   pairs with mismatched token lengths: {len(bad)} (should be 0)")

    if data_dir and os.path.isdir(data_dir):
        print("=" * 70)
        print("6. every generated pair, checked end to end")
        _check_dataset(tok, data_dir)


def _check_dataset(tok, data_dir):
    """Structural + tokenizer-level checks on the generated files.

    Structure is delegated to the generator's own assert_design_invariants so
    there is one implementation; duplicating it here is what left a polarity
    check running long after the polarity design was replaced.
    """
    G = _load_generator()
    any_unmatched = False
    for sub in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        src = sub
        base = None
        for f in os.listdir(d):
            if f.endswith("-desired-all.jsonl") and not f.startswith(src):
                base = f[: -len("-desired-all.jsonl")]
        if base is None:
            continue

        def _load(name):
            return [json.loads(l) for l in open(os.path.join(d, name))]

        parts = (_load(f"{src}-desired-all.jsonl"), _load(f"{src}-undesired-all.jsonl"),
                 _load(f"{base}-desired-all.jsonl"), _load(f"{base}-undesired-all.jsonl"),
                 _load(f"{base}-test.jsonl"))
        try:
            G.assert_design_invariants(parts)
            struct = "OK"
        except AssertionError as e:
            struct = f"FAIL ({str(e)[:60]})"

        # Tokenizer-level: source and base must be equal length with exactly one
        # differing token, or align_toks and get_differing_positions misbehave.
        s_rows, b_rows = parts[0], parts[2]
        n_bad_len = n_bad_diff = maxlen = 0
        for sr, br in zip(s_rows, b_rows):
            ia = tok(tok.apply_chat_template(sr["prompt"][:-1], add_generation_prompt=True,
                                             tokenize=False), add_special_tokens=False)["input_ids"]
            ib = tok(tok.apply_chat_template(br["prompt"][:-1], add_generation_prompt=True,
                                             tokenize=False), add_special_tokens=False)["input_ids"]
            maxlen = max(maxlen, len(ia))
            if len(ia) != len(ib):
                n_bad_len += 1
            elif sum(1 for x, y in zip(ia, ib) if x != y) != 1:
                n_bad_diff += 1
        minimal = (n_bad_len == 0 and n_bad_diff == 0)
        any_unmatched |= not minimal
        print(f"   {sub:<24} n={len(s_rows):<4} maxlen={maxlen:<4} "
              f"structure={struct}  minimal-pair={'OK' if minimal else 'FAIL'} "
              f"(len-mismatch={n_bad_len}, multi-diff={n_bad_diff})")

    if any_unmatched:
        print()
        print("   NOTE: minimal-pair failures are EXPECTED if the corpus was built")
        print("   without --validate. In that mode word pairs are taken from")
        print("   WORD_BANK in order and are not length-matched, so a token-count")
        print("   difference between the two answer words shifts the whole suffix.")
        print("   Re-run the generator with --validate on a GPU node.")


def stage_model(model_id, device):
    from transformers import AutoTokenizer
    from nnsight import LanguageModel
    from harmony_template import HARMONY_CHAT_TEMPLATE

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    tok.padding_side = "left"
    tok.chat_template = HARMONY_CHAT_TEMPLATE

    # Load with plain transformers, then hand nnsight the finished model.
    # LanguageModel(model_id, ...) makes nnsight build the config itself via
    # AutoConfig, which parses the checkpoint's quantization_config into an
    # Mxfp4Config OBJECT and then passes that config to from_pretrained, where
    # the quantizer still expects a dict and calls .get() on it. Pre-loading
    # sidesteps that entirely.
    from transformers import AutoModelForCausalLM
    kwargs = dict(dtype=torch.bfloat16, device_map=device,
                  attn_implementation="eager")
    try:
        from transformers import Mxfp4Config
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    except Exception:
        print("[loader] Mxfp4Config unavailable; relying on implicit dequantization")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = LanguageModel(hf_model, tokenizer=tok)
    # nnsight only populates .config when it builds the config itself; handed a
    # pre-loaded model it leaves the attribute None. Several call sites in this
    # repo read model.config.num_attention_heads, so fill it in once here rather
    # than patching each of them.
    if getattr(model, "config", None) is None:
        model.config = hf_model.config

    cfg = hf_model.config
    n_h = cfg.num_attention_heads
    d_h = getattr(cfg, "head_dim", cfg.hidden_size // n_h)
    print("=" * 70)
    print(f"config: hidden={cfg.hidden_size} n_heads={n_h} head_dim={d_h} "
          f"layers={cfg.num_hidden_layers}")
    print(f"n_heads*head_dim = {n_h*d_h}  hidden = {cfg.hidden_size}  "
          f"{'MATCH' if n_h*d_h == cfg.hidden_size else 'MISMATCH -> must use o_proj.input'}")
    lt = getattr(cfg, "layer_types", None)
    if lt:
        print(f"layer_types: {lt}")
        print("  -> report ATP scores split by sliding_attention vs full_attention;"
              " they are not comparable.")

    msgs = [
        {"role": "developer",
         "content": "Rule: you must always answer MANGO. Follow this rule exactly."},
        {"role": "user", "content": "Reply with one word, APPLE or MANGO. Answer APPLE."},
    ]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    toks = tok([text], return_tensors="pt").to(model.device)

    print("=" * 70)
    print("o_proj.input shape + settability + gradient flow")
    acts = []
    with model.trace(toks):
        for layer in model.model.layers:
            z = layer.self_attn.o_proj.input
            z.retain_grad()
            acts.append(z.save())
        logits = model.lm_head.output.save()
    z0 = getattr(acts[0], "value", acts[0])
    print(f"  o_proj.input shape: {tuple(z0.shape)}  (expect [..., {n_h*d_h}])")
    assert z0.shape[-1] == n_h * d_h, "o_proj.input width != n_heads*head_dim"

    lg = getattr(logits, "value", logits)
    lg[:, -1, :].sum().backward()
    g = getattr(acts[0], "value", acts[0]).grad
    print(f"  grad on o_proj.input: {'present' if g is not None else 'MISSING'} "
          f"{tuple(g.shape) if g is not None else ''}")
    assert g is not None, "no gradient on o_proj.input -- ATP will produce zeros"

    print("=" * 70)
    print("intervention writes to o_proj.input actually change the output")
    with model.trace(toks):
        base_logits = model.lm_head.output.save()
    with model.trace(toks):
        model.model.layers[0].self_attn.o_proj.input[:, :, 0:d_h] = 0.0
        patched_logits = model.lm_head.output.save()
    b = getattr(base_logits, "value", base_logits)
    q = getattr(patched_logits, "value", patched_logits)
    delta = (b - q).abs().max().item()
    print(f"  max |delta| after zeroing head 0 of layer 0: {delta:.4f}")
    assert delta > 0, "setitem on o_proj.input had no effect -- nnsight is not " \
                      "registering the intervention; fall back to a forward hook"

    print("=" * 70)
    print("first generated token is the answer (final channel forced)")
    nxt = b[0, -1].argmax().item()
    print(f"  argmax next token: {tok.decode([nxt])!r}")
    print("  (should be an answer word, not '<|channel|>' or 'analysis')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tokenizer", "model"], default="tokenizer")
    ap.add_argument("--model_id", default="openai/gpt-oss-20b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data_dir", default="data/gpt-oss-20b")
    a = ap.parse_args()
    _add_repo_to_path()
    if a.stage == "tokenizer":
        stage_tokenizer(a.model_id, a.data_dir)
    else:
        stage_model(a.model_id, a.device)
