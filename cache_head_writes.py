#!/usr/bin/env python
"""
cache_head_writes.py -- cache what each attention head actually writes, on your
minimally contrastive data.

The static analysis in svd_head_subspaces.py looks at col(W_O^h): the subspace a
head *can* write into.  This script collects the other half -- the coefficients
the head actually produces -- so the companion script can restrict each head's
subspace to the part it genuinely uses.

What gets captured
------------------
The input to o_proj, i.e. the concatenated per-head attention outputs z, shape
[batch, seq, n_heads * head_dim].  Head h occupies columns
[h*head_dim, (h+1)*head_dim), and its contribution to the residual stream is
exactly z_h @ W_O^h.T.  Note this is o_proj's INPUT, not its output -- the
output is already summed over heads and cannot be decomposed per head.
(Your localization pipeline hooks o_proj.output; see the head-indexing note in
svd_head_subspaces.py.)

Capture uses a plain forward pre-hook rather than nnsight, so it does not
depend on nnsight's proxy semantics and needs no gradients.

Which prompts
-------------
For a task directory like data/{model}/female-single/, this caches both sides of
the contrast ATP linearizes:

    {source}-desired-all.jsonl   e.g. female-single-desired-all.jsonl
    {base}-desired-all.jsonl     e.g. male-single-desired-all.jsonl

Both are templated question-only with a generation prompt -- the same form
DataHandler builds as source_qs / base_qs -- so example i on one side pairs with
example i on the other.  With left padding (which model_handler sets) the final
position is index -1 for every example, so the pairing needs no alignment.

Usage
-----
  python cache_head_writes.py --model-id Qwen/Qwen1.5-14B-Chat \\
      --task female-single --source female-single --base male-single \\
      --out cache/writes

  # long-form version of the same task
  python cache_head_writes.py --model-id Qwen/Qwen1.5-14B-Chat \\
      --task female-long --source female-long --base male-long \\
      --out cache/writes

Writes cache/writes/{model}/{task}/{split}.pt holding
[n_layers, n_examples, n_heads*head_dim] in float16, plus a meta.json.
Typical size: 40 layers x 200 examples x 5120 x 2B = 82 MB.
"""

import argparse
import ast
import json
import os
import re
from typing import Dict, List

import torch

_OPROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")


def load_jsonl(path: str) -> List[dict]:
    """Mirror DataHandler.load_from_jsonl -- these files are Python-literal lines."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(json.dumps(ast.literal_eval(line))))
            except Exception:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def templated_questions(records: List[dict], tokenizer) -> List[str]:
    """Question-only prompts with a generation prompt appended.

    Same construction as DataHandler.get_templated_prompts(only_q=True,
    add_generation_prompt=True): if the record already carries an assistant
    turn, drop it so the model is left at the point of answering.
    """
    prompts = []
    for r in records:
        msgs = r["prompt"]
        if any(m.get("role") == "assistant" for m in msgs):
            msgs = msgs[:-1]
        prompts.append(tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False))
    return prompts


def pick_positions(attn_mask: torch.Tensor, n_positions: int) -> torch.Tensor:
    """Indices of the last `n_positions` real tokens, one row per example.

    Counting from the END is what makes positions comparable across prompts of
    different lengths: index -1 is the final token in every prompt, -2 the one
    before it, and so on.  Counting from the front would not be comparable,
    which matters for the batch-mean view where position s is averaged across
    prompts.  Returns [batch, n_positions].
    """
    rows = []
    for row in attn_mask:
        real = row.nonzero(as_tuple=True)[0]
        if len(real) < n_positions:
            raise RuntimeError(
                f"prompt has {len(real)} real tokens but {n_positions} were requested; "
                f"short prompts should have been filtered before this point")
        rows.append(real[-n_positions:].clone())
    return torch.stack(rows)


def real_lengths(prompts: List[str], tokenizer, batch_size: int = 64) -> torch.Tensor:
    """Token counts per prompt, without running the model."""
    out = []
    for i in range(0, len(prompts), batch_size):
        toks = tokenizer(prompts[i:i + batch_size], padding=True, truncation=False,
                         return_tensors="pt")
        out.append(toks["attention_mask"].sum(dim=1))
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--task", required=True,
                   help="Data subdirectory, e.g. female-single.")
    p.add_argument("--source", required=True, help="e.g. female-single")
    p.add_argument("--base", required=True, help="e.g. male-single")
    p.add_argument("--split", default="desired", choices=["desired", "undesired"],
                   help="Which -all file to read ({name}-{split}-all.jsonl).")
    p.add_argument("--n-positions", type=int, default=1,
                   help="Keep the last K real tokens per prompt. K=1 reproduces "
                        "steering_type='last_token'; K=64 supports the batch-mean and "
                        "pooled-token views in svd_head_activations.py --pool.")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--full-precision", action="store_true",
                   help="bfloat16 with device_map=auto instead of 4-bit nf4.")
    p.add_argument("--out", default="./cache/writes")
    args = p.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_name = args.model_id.split("/")[-1]
    data_dir = os.path.join(args.data_root, model_name, args.task)
    outdir = os.path.join(args.out, model_name, args.task)
    os.makedirs(outdir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN") or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = dict(token=os.environ.get("HF_TOKEN") or None)
    if args.full_precision:
        kwargs.update(dtype=torch.bfloat16, device_map="auto")
    else:
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16), device_map=args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.eval()

    # Hook every o_proj and stash its input (the per-head z, pre-projection).
    oproj = {}
    for name, mod in model.named_modules():
        m = _OPROJ_RE.search(name)
        if m and "vision" not in name:
            oproj[int(m.group(1))] = mod
    if not oproj:
        raise SystemExit("Found no self_attn.o_proj modules; check the architecture.")
    n_layers = max(oproj) + 1
    print(f"hooking {len(oproj)} o_proj modules across {n_layers} layers")

    captured: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook(_module, inputs):
            captured[layer_idx] = inputs[0].detach()
        return hook

    handles = [mod.register_forward_pre_hook(make_hook(i)) for i, mod in oproj.items()]

    meta = {"model_id": args.model_id, "task": args.task, "split": args.split,
            "n_positions": args.n_positions, "n_layers": n_layers}

    # Pre-pass: both sides must be long enough, and must stay index-aligned, so
    # drop the union of prompts that are too short on either side.
    sides = {}
    for role, name in (("source", args.source), ("base", args.base)):
        path = os.path.join(data_dir, f"{name}-{args.split}-all.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"Missing dataset: {path}")
        records = load_jsonl(path)
        if args.max_examples:
            records = records[:args.max_examples]
        sides[role] = templated_questions(records, tokenizer)
    if len(sides["source"]) != len(sides["base"]):
        n = min(len(sides["source"]), len(sides["base"]))
        print(f"[warn] source has {len(sides['source'])} prompts, base has "
              f"{len(sides['base'])}; truncating both to {n} to keep the pairing.")
        sides = {k: v[:n] for k, v in sides.items()}

    len_src = real_lengths(sides["source"], tokenizer)
    len_base = real_lengths(sides["base"], tokenizer)
    keep_mask = (len_src >= args.n_positions) & (len_base >= args.n_positions)
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        print(f"[warn] dropping {n_dropped}/{len(keep_mask)} prompt pairs shorter than "
              f"{args.n_positions} tokens on one side or the other "
              f"(shortest kept: {int(torch.minimum(len_src, len_base)[keep_mask].min())}).")
    keep_idx = keep_mask.nonzero(as_tuple=True)[0].tolist()
    if not keep_idx:
        raise SystemExit(f"No prompt pairs have {args.n_positions} tokens on both sides.")
    sides = {k: [v[i] for i in keep_idx] for k, v in sides.items()}
    meta["kept_indices"] = keep_idx

    est_gb = (n_layers * len(keep_idx) * args.n_positions
              * oproj[0].weight.shape[1] * 2) / 1e9
    print(f"cache size per side: ~{est_gb:.2f} GB "
          f"({n_layers} layers x {len(keep_idx)} prompts x {args.n_positions} positions)")

    try:
        for role, name in (("source", args.source), ("base", args.base)):
            prompts = sides[role]
            print(f"{role} ({name}): {len(prompts)} prompts")
            print(f"  example: {prompts[0][:200]!r}")

            per_layer: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
            for start in range(0, len(prompts), args.batch_size):
                batch = prompts[start:start + args.batch_size]
                toks = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
                toks = {k: v.to(model.device) for k, v in toks.items()}
                captured.clear()
                with torch.no_grad():
                    model(**toks)
                keep = pick_positions(toks["attention_mask"].cpu(), args.n_positions)
                for l in range(n_layers):
                    z = captured[l].to(torch.float32).cpu()   # [B, S, n_heads*head_dim]
                    gathered = torch.stack([z[b, keep[b], :] for b in range(z.shape[0])])
                    per_layer[l].append(gathered)             # [B, K, n_heads*head_dim]
                print(f"  batch {start // args.batch_size + 1}/"
                      f"{(len(prompts) + args.batch_size - 1) // args.batch_size}", flush=True)

            # [n_layers, n_prompts, n_positions, n_heads*head_dim]
            stacked = torch.stack([torch.cat(rows, dim=0) for rows in per_layer])
            torch.save(stacked.to(torch.float16), os.path.join(outdir, f"{role}_{args.split}.pt"))
            meta[f"{role}_dataset"] = name
            meta[f"{role}_shape"] = list(stacked.shape)
            print(f"  wrote {tuple(stacked.shape)} -> {outdir}/{role}_{args.split}.pt")
    finally:
        for h in handles:
            h.remove()

    with open(os.path.join(outdir, f"meta_{args.split}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {outdir}/meta_{args.split}.json")


if __name__ == "__main__":
    main()
