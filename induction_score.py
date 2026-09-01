#!/usr/bin/env python3
"""Per-head induction score by prefix matching. No corpus, no ATP.

    python induction_score.py --out induction_heads.json

WHY NOT AN INDUCTION CONTRAST CORPUS
------------------------------------
The obvious move is a third ATP contrast on induction-shaped prompts. It is the
wrong instrument here for two reasons.

First, matching it to the privilege corpus is impossible in principle. Induction
requires REPEATED token structure, and introducing repetition is exactly the
surface-form change you would need to hold fixed for the comparison to mean
anything. The position control can be matched because mention order is already
a latent variable of the existing prompts; repetition is not.

Second, two ATP maps computed with the same reducer on similarly shaped prompts
share failure modes. If ATP mislocalizes on this model -- and the per-layer
sweep suggests it does -- both maps inherit the same error and their agreement
says nothing about the mechanisms.

Prefix matching sidesteps both. It is the standard definition (Elhage et al.,
"A Mathematical Framework for Transformer Circuits"; Olsson et al., "In-context
Learning and Induction Heads"): on a repeated random sequence, an induction head
attends from the current token back to the token that FOLLOWED the previous
occurrence of that token. The score is that attention mass, averaged. It is a
property of the head measured directly, computed in one batch of forward passes,
with no gradients and no contrast to confound.

WHAT COMES OUT
--------------
{layer: {head: score}} plus a ranked list, comparable to your ATP maps by head
identity. Random token sequences are used deliberately: real text lets a head
score well through ordinary bigram statistics rather than through copying.

READING IT AGAINST THE PRIVILEGE MAP
------------------------------------
Overlap between the top induction heads and the top privilege heads needs a
calibration floor, or "5 of 10 shared" means nothing. Two sources:

  * the split-half overlap of the privilege map with itself -- the ceiling, i.e.
    the most agreement two maps of the same thing could show;
  * the overlap of a random head set of the same size -- the floor.

Report the induction overlap between those two numbers, not on its own.
"""

import json
import argparse

import determinism


def build_sequences(tok, n_seq, seq_len, seed, vocab_lo=1000, vocab_hi=20000):
    """Random token sequences, each repeated twice.

    The repeat is what makes induction measurable: in the second copy every
    token has appeared before, so a prefix-matching head has somewhere to look.
    Sampling from a mid-vocabulary band avoids special tokens and the very
    high-frequency ids whose behaviour is dominated by unigram statistics.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    half = torch.randint(vocab_lo, vocab_hi, (n_seq, seq_len), generator=g)
    # A BOS-like anchor keeps the first position from acting as an attention
    # sink for the measured span; gpt-oss has strong sink behaviour at position 0.
    bos = torch.full((n_seq, 1), tok.eos_token_id or vocab_lo)
    return torch.cat([bos, half, half], dim=1), seq_len


def induction_scores(model, ids, seq_len, batch_size=4):
    """Mean attention from each position in the repeat to its induction target.

    For a position i in the second copy holding token t, the induction target is
    the position immediately AFTER the earlier occurrence of t. With an exactly
    repeated sequence the earlier occurrence is at i - seq_len, so the target is
    i - seq_len + 1. Reading the diagonal at that offset is equivalent to the
    usual prefix-matching definition and avoids searching for matches token by
    token.

    The +1 is the whole distinction between an induction head and a
    previous-token / duplicate-token head: offset i - seq_len attends to the
    repeat of the CURRENT token, which is a different circuit and would score
    the wrong heads.
    """
    import torch
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    total = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    n_batches = 0

    for start in range(0, ids.shape[0], batch_size):
        batch = ids[start:start + batch_size].to(model.device)
        with torch.no_grad():
            out = model(batch, output_attentions=True)
        for layer, attn in enumerate(out.attentions):
            # attn: [batch, heads, seq, seq]
            a = attn.float()
            # Positions in the second copy, excluding its first token (which has
            # no induction target inside the window).
            q = torch.arange(seq_len + 1, 2 * seq_len + 1, device=a.device)
            k = q - seq_len + 1
            total[layer] += a[:, :, q, k].mean(dim=(0, 2)).double().cpu()
        n_batches += 1
        del out
        torch.cuda.empty_cache()
    return (total / n_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--n_seq", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_nondeterministic", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="induction_heads.json")
    args = ap.parse_args()

    fingerprint = determinism.enforce(
        args.seed, allow_nondeterministic=args.allow_nondeterministic)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"loading {args.model} ...", flush=True)
    # eager attention: output_attentions is silently ignored or unsupported
    # under the fused SDPA/flash paths, and a silently empty attention tensor
    # would give every head a score of zero.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", attn_implementation="eager")
    model.eval()

    ids, seq_len = build_sequences(tok, args.n_seq, args.seq_len, args.seed)
    print(f"{args.n_seq} sequences of {seq_len} tokens, repeated "
          f"(prompt length {ids.shape[1]})")

    scores = induction_scores(model, ids, seq_len, args.batch_size)

    flat = [(float(scores[l, h]), int(l), int(h))
            for l in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(reverse=True)

    print(f"\ntop {args.top} induction heads (layer.head: score)")
    for s, l, h in flat[:args.top]:
        print(f"  {l:>2}.{h:<3} {s:.4f}")

    # A rough scale check: with an exactly repeated sequence a strong induction
    # head puts most of its attention on the target, so scores near 0.5+ are
    # real and scores near 1/seq_len are chance.
    chance = 1.0 / seq_len
    strong = sum(1 for s, _, _ in flat if s > 10 * chance)
    print(f"\nchance level ~{chance:.4f}; {strong} heads above 10x chance")

    with open(args.out, "w") as f:
        json.dump({
            "args": vars(args),
            "fingerprint": fingerprint,
            "chance": chance,
            "scores": {str(l): {str(h): float(scores[l, h])
                                for h in range(scores.shape[1])}
                       for l in range(scores.shape[0])},
            "ranked": [{"layer": l, "head": h, "score": s} for s, l, h in flat],
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
