#!/usr/bin/env python3
"""Add a LAYER-MATCHED random-head baseline to the head eval.

    python apply_layer_matched_random.py --check
    python apply_layer_matched_random.py

Then run the random arm exactly as the targeted arm, with --patch_algo random.

WHY THE EXISTING BASELINE IS TOO WEAK
-------------------------------------
`retrieve_random_k` samples k heads uniformly from all num_layers x num_heads.
But ATP's top-k concentrates heavily in late layers, and steering effectiveness
itself varies with depth, so a uniform-random set differs from the targeted set
in TWO ways at once: which heads, and which depths. Beating it therefore shows
"late-layer heads beat average-layer heads", which is not the claim -- the claim
is that ATP's *ranking within the available heads* is informative.

A layer-matched control fixes the depth profile and randomizes only the choice
within each layer: if ATP selected 7 heads from layer 17, the control also draws
7 heads from layer 17, uniformly at random. Beating THAT is evidence about the
ranking.

Both are kept. RANDOM_BASELINE=uniform reproduces the old behaviour exactly, so
previously generated random arms remain reproducible; layer_matched is the
default because it is the control a reviewer will ask for.

WHAT THIS DOES NOT DO
---------------------
It does not make the random arm cheap. The random arm is a full eval sweep over
the same (N, top_k) grid as the targeted arm, so budget the same GPU time again.
For a short paper, one or two (N, top_k) cells in the effective region is enough
to make the point -- pass a reduced grid rather than skipping the control.
"""

import argparse
import sys
from pathlib import Path

PATH = "eval/logits_handler.py"
RUNNER = "eval/eval_runner.py"

FN_OLD = '''def retrieve_random_k(num_layers, num_heads, k, seed=42):
    rng = random.Random(seed)
    total = num_layers * num_heads
    num_samples = int(k * total)
    all_combinations = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    selected = rng.sample(all_combinations, num_samples)
    df = pd.DataFrame(selected, columns=['layer', 'neuron'])
    return df.sort_values(by=['layer', 'neuron'])'''

FN_NEW = '''def retrieve_random_k(num_layers, num_heads, k, seed=42):
    rng = random.Random(seed)
    total = num_layers * num_heads
    num_samples = int(k * total)
    all_combinations = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    selected = rng.sample(all_combinations, num_samples)
    df = pd.DataFrame(selected, columns=['layer', 'neuron'])
    return df.sort_values(by=['layer', 'neuron'])


def retrieve_layer_matched_random_k(targeted_df, num_heads, seed=42):
    """Random heads with the SAME per-layer counts as `targeted_df`.

    Uniform-random selection differs from an ATP-selected set in two ways at
    once -- which heads, and which layers -- because ATP concentrates in late
    layers and steering effectiveness varies with depth. Matching the layer
    histogram holds depth fixed so the comparison isolates the ranking.

    Draws WITHOUT replacement within each layer, so the control cannot select
    the same head twice; it may overlap the targeted set by chance, which is
    correct -- the null is "any k heads at these depths", not "any k heads at
    these depths that ATP did not pick".
    """
    rng = random.Random(seed)
    rows = []
    for layer, count in targeted_df['layer'].value_counts().items():
        count = int(count)
        if count > num_heads:
            raise ValueError(
                f"layer {layer} needs {count} heads but the model has {num_heads}")
        for head in rng.sample(range(num_heads), count):
            rows.append((int(layer), int(head)))
    df = pd.DataFrame(rows, columns=['layer', 'neuron'])
    return df.sort_values(by=['layer', 'neuron']).reset_index(drop=True)'''

IMPORT_OLD = "from eval.logits_handler import load_logits, get_top_k_layer_and_head, retrieve_random_k"
IMPORT_NEW = ("from eval.logits_handler import (load_logits, get_top_k_layer_and_head,\n"
              "                                 retrieve_random_k,\n"
              "                                 retrieve_layer_matched_random_k)")

CALL_OLD = '''    if reps_type == 'random':
        topk_df = retrieve_random_k(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            topk
        )'''

CALL_NEW = '''    if reps_type == 'random':
        # RANDOM_BASELINE=layer_matched (default) holds the depth profile of the
        # targeted set fixed and randomizes only which heads within each layer,
        # so beating it is evidence about ATP's ranking rather than about late
        # layers being more steerable. =uniform restores the original sampling.
        mode = os.environ.get('RANDOM_BASELINE', 'layer_matched')
        seed = int(os.environ.get('RANDOM_BASELINE_SEED', '42'))
        if mode == 'layer_matched':
            # run_eval passes logits=None when patch_algo == 'random', so the
            # targeted set has to be recovered from the ATP map that lives in the
            # SIBLING atp/ directory -- the random arm writes under random/ and
            # never builds a map of its own.
            _logits = logits
            if _logits is None:
                _algo_dir = '/'.join(config.get_output_prefix().rstrip('/').split('/')[:-2])
                _atp_map = os.environ.get(
                    'ATP_MAP',
                    os.path.join(os.path.dirname(_algo_dir), 'atp', 'numerator_1_heads.pt'))
                if not os.path.exists(_atp_map):
                    raise SystemExit(
                        f"layer-matched random needs the targeted map, not "
                        f"found at\\n  {_atp_map}\\n"
                        f"Point ATP_MAP at it, or use RANDOM_BASELINE=uniform.")
                print(f"[random baseline] layer-matching against {_atp_map}")
                _logits = torch.load(_atp_map, map_location='cpu')
            targeted_df = get_top_k_layer_and_head(_logits, topk, 'atp')
            topk_df = retrieve_layer_matched_random_k(
                targeted_df, model.config.num_attention_heads, seed=seed)
            print(f"[random baseline] layer-matched to the targeted top-{topk} set: "
                  f"{len(topk_df)} heads, layer counts "
                  f"{dict(sorted(targeted_df['layer'].value_counts().items()))}")
        elif mode == 'uniform':
            topk_df = retrieve_random_k(
                model.config.num_hidden_layers,
                model.config.num_attention_heads,
                topk,
                seed=seed
            )
            print(f"[random baseline] uniform over all heads: {len(topk_df)} heads")
        else:
            raise SystemExit(
                f"RANDOM_BASELINE must be layer_matched|uniform, got {mode!r}")'''

EDITS = [
    (PATH, "add retrieve_layer_matched_random_k", FN_OLD, FN_NEW),
    (RUNNER, "import the layer-matched sampler", IMPORT_OLD, IMPORT_NEW),
    (RUNNER, "save_top_k selects layer-matched by default", CALL_OLD, CALL_NEW),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    texts, pending, already, failed = {}, 0, 0, 0
    for rel, desc, old, new in EDITS:
        p = root / rel
        if not p.exists():
            print(f"MISSING  {rel}")
            failed += 1
            continue
        if rel not in texts:
            texts[rel] = p.read_text()
        t = texts[rel]
        if new in t:
            print(f"SKIP     {rel}: {desc} (already applied)")
            already += 1
            continue
        n = t.count(old)
        if n != 1:
            print(f"FAIL     {rel}: {desc} -- anchor matched {n} times, expected 1")
            failed += 1
            continue
        texts[rel] = t.replace(old, new, 1)
        print(f"{'WOULD' if args.check else 'APPLY':<8} {rel}: {desc}")
        pending += 1

    # `os` must be importable in eval_runner for the env lookup.
    if RUNNER in texts and "\nimport os" not in texts[RUNNER]:
        texts[RUNNER] = "import os\n" + texts[RUNNER]
        print(f"{'WOULD' if args.check else 'APPLY':<8} {RUNNER}: add `import os`")
        pending += 1

    if failed:
        print(f"\n{failed} hunk(s) failed -- nothing written.")
        return 1
    if args.check:
        print(f"\ncheck: {pending} to apply, {already} already applied")
        return 0
    for rel, t in texts.items():
        (root / rel).write_text(t)
    print(f"\ndone: {pending} applied, {already} already applied")
    if pending:
        print("\nRun the random arm with the SAME grid as the targeted arm:")
        print("  sbatch --export=ALL,PATCH_ALGO=random,RANDOM_BASELINE=layer_matched ...")
        print("Then score it and plot targeted vs random at matched (N, top_k).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
