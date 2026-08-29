#!/usr/bin/env python3
"""Apply the random-head-baseline changes by anchored replacement.

    python apply_random_baseline.py            # apply
    python apply_random_baseline.py --check    # report only, change nothing

Unified diffs keep failing against this repo because they carry line numbers and
several lines of surrounding context, and the working tree has drifted from the
copy the diff was generated against. Anchored replacement only needs the exact
snippet being changed, so unrelated edits elsewhere in the file are irrelevant.
Every edit is idempotent -- if its result is already present it is reported as
'already applied' rather than duplicated -- so a partial previous attempt is
safe to rerun.

WHAT IT CHANGES, AND WHY
------------------------
The stock --patch_algo random is too weak to serve as the baseline for a
top-k selection claim, in three separate ways:

  1. It samples uniformly over all layers. The ATP top-1% sits in a narrow band
     of layers, so a uniform draw can fail because it picked the wrong DEPTHS
     rather than the wrong heads within them -- which would support "depth
     matters" instead of "these heads do". --random_layer_matched draws from
     the real top-k's per-layer histogram.

  2. The seed was hardcoded. One draw is a point with no spread, and the claim
     is that the real set beats the random DISTRIBUTION. --random_seed exposes
     it, and the seed is embedded in reps_type because output files are named
     {N}_{reps_type}_{ablation}_{topk}_{test}_gen.txt -- without it a second
     seed silently overwrites the first.

  3. load_logits derives the map path from the eval prefix, so with
     --patch_algo random it looked under random/ for a map that only exists
     under atp/. It found nothing and fell back to a default-sized uniform
     draw, which looks identical in the output to a matched one.
"""

import os
import sys
import ast
import argparse

EDITS = [
    ("eval/logits_handler.py", "retrieve_random_k: seed + layer-matched draw",
     "def retrieve_random_k(num_layers, num_heads, k, seed=42):\n"
     "    rng = random.Random(seed)\n"
     "    total = num_layers * num_heads\n"
     "    num_samples = int(k * total)\n"
     "    all_combinations = [(l, h) for l in range(num_layers) for h in range(num_heads)]\n"
     "    selected = rng.sample(all_combinations, num_samples)\n"
     "    df = pd.DataFrame(selected, columns=['layer', 'neuron'])\n"
     "    return df.sort_values(by=['layer', 'neuron'])",
     "def retrieve_random_k(num_layers, num_heads, k, seed=42, layer_counts=None):\n"
     "    \"\"\"A random head set of the same size as the selected one.\n"
     "\n"
     "    layer_counts, when given, is {layer: n_heads_to_draw} -- the per-layer\n"
     "    histogram of the real top-k. Sampling within it makes the baseline\n"
     "    LAYER-MATCHED, which is the version that isolates the claim. Uniform\n"
     "    sampling is weaker: the ATP top-1% sits in a narrow band of layers, so a\n"
     "    uniform set can fail because it drew the wrong layers rather than the\n"
     "    wrong heads inside them.\n"
     "\n"
     "    Run several seeds. One draw gives a point with no spread.\n"
     "    \"\"\"\n"
     "    rng = random.Random(seed)\n"
     "    if layer_counts:\n"
     "        selected = []\n"
     "        for layer, n in sorted(layer_counts.items()):\n"
     "            n = min(int(n), num_heads)\n"
     "            selected.extend((int(layer), h)\n"
     "                            for h in rng.sample(range(num_heads), n))\n"
     "    else:\n"
     "        total = num_layers * num_heads\n"
     "        num_samples = int(k * total)\n"
     "        all_combinations = [(l, h) for l in range(num_layers) for h in range(num_heads)]\n"
     "        selected = rng.sample(all_combinations, num_samples)\n"
     "    df = pd.DataFrame(selected, columns=['layer', 'neuron'])\n"
     "    return df.sort_values(by=['layer', 'neuron'])\n"
     "\n"
     "\n"
     "def layer_histogram_of_topk(logits, topk, patch_algo):\n"
     "    \"\"\"{layer: count} for the real top-k, for layer-matched sampling.\"\"\"\n"
     "    real = get_top_k_layer_and_head(logits, topk, patch_algo)\n"
     "    return real.groupby('layer').size().to_dict()"),

    ("eval/eval_runner.py", "import layer_histogram_of_topk",
     "from eval.logits_handler import load_logits, get_top_k_layer_and_head, retrieve_random_k",
     "from eval.logits_handler import (load_logits, get_top_k_layer_and_head,\n"
     "                                 retrieve_random_k, layer_histogram_of_topk)"),

    ("eval/eval_runner.py", "save_top_k: pass seed and histogram",
     "    if reps_type == 'random':\n"
     "        topk_df = retrieve_random_k(\n"
     "            model.config.num_hidden_layers,\n"
     "            model.config.num_attention_heads,\n"
     "            topk\n"
     "        )",
     "    if reps_type.startswith('random'):\n"
     "        # Layer-matched when the real attribution map is available: the same\n"
     "        # number of heads from each layer the real top-k used. Falls back to\n"
     "        # uniform when there is no map to match against.\n"
     "        layer_counts = None\n"
     "        if getattr(config.args, 'random_layer_matched', False) and logits is not None:\n"
     "            layer_counts = layer_histogram_of_topk(logits, topk, 'atp')\n"
     "            print(f\"[random] layer-matched draw: {layer_counts}\")\n"
     "        topk_df = retrieve_random_k(\n"
     "            model.config.num_hidden_layers,\n"
     "            model.config.num_attention_heads,\n"
     "            topk,\n"
     "            seed=getattr(config.args, 'random_seed', 42),\n"
     "            layer_counts=layer_counts,\n"
     "        )"),

    ("eval/eval_runner.py", "reps_type carries the seed",
     "    reps_types = ['random'] if config.args.patch_algo == 'random' else ['targeted']",
     "    # The seed is in the label because every generated file is named\n"
     "    # {N}_{reps_type}_{ablation}_{topk}_{test}_gen.txt: without it a second\n"
     "    # seed silently overwrites the first, and a baseline of one draw is not\n"
     "    # a baseline.\n"
     "    reps_types = ([f\"random-s{getattr(config.args, 'random_seed', 42)}\"]\n"
     "                  if config.args.patch_algo == 'random' else ['targeted'])"),

    ("config.py", "--random_seed and --random_layer_matched",
     "        parser.add_argument('-source', '--source', type=str, help='Patch from source')",
     "        parser.add_argument('--random_seed', type=int, default=42,\n"
     "                            help='Seed for --patch_algo random. Vary it: one '\n"
     "                                 'draw is a point with no spread.')\n"
     "        parser.add_argument('--random_layer_matched', action='store_true',\n"
     "                            help='Draw random heads from the same per-layer '\n"
     "                                 'histogram as the real top-k, rather than '\n"
     "                                 'uniformly over all layers.')\n"
     "        parser.add_argument('-source', '--source', type=str, help='Patch from source')"),
]

# The map redirect depends on whether data_paths.patch is in already, so it is
# tried in two forms: the patched line first, then the upstream one.
MAP_REDIRECT = [
    ("config.py", "random arm reads the atp map (post data_paths)",
     '            self.output_prefix = f"{root}/{model}/{pair}/{self.args.patch_algo}/{eval_test_dir}_eval/{steering_dir}_steer/"',
     '            # The random arm is a baseline FOR atp: it needs atp\'s map to\n'
     '            # know how many heads to draw and, when layer-matched, from\n'
     '            # which layers. Only the map lookup is redirected; results stay\n'
     '            # distinct because reps_type is in every output filename.\n'
     '            _map_algo = "atp" if self.args.patch_algo == "random" else self.args.patch_algo\n'
     '            self.output_prefix = f"{root}/{model}/{pair}/{_map_algo}/{eval_test_dir}_eval/{steering_dir}_steer/"'),
    ("config.py", "random arm reads the atp map (upstream)",
     '            self.output_prefix = f"{root}/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/{eval_test_dir}_eval/{steering_dir}_steer/"',
     '            _map_algo = "atp" if self.args.patch_algo == "random" else self.args.patch_algo\n'
     '            self.output_prefix = f"{root}/{model}/from_{self.args.source}_to_{self.args.base}/{_map_algo}/{eval_test_dir}_eval/{steering_dir}_steer/"'),
]


def apply_edits(edits, check, applied, failed, skipped, first_match_only=False):
    for path, desc, old, new in edits:
        if not os.path.exists(path):
            failed.append((path, desc, "file not found"))
            continue
        s = open(path).read()
        if new.strip() and new in s:
            skipped.append((path, desc))
            if first_match_only:
                return True
            continue
        n = s.count(old)
        if n == 0:
            failed.append((path, desc, "anchor not found"))
            continue
        if n > 1:
            failed.append((path, desc, f"anchor appears {n} times; not unique"))
            continue
        if not check:
            open(path, "w").write(s.replace(old, new, 1))
        applied.append((path, desc))
        if first_match_only:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    applied, failed, skipped = [], [], []
    apply_edits(EDITS, args.check, applied, failed, skipped)

    # Exactly one of the two map-redirect forms should match.
    pre = len(applied) + len(skipped)
    apply_edits(MAP_REDIRECT, args.check, applied, failed, skipped,
                first_match_only=True)
    if len(applied) + len(skipped) == pre:
        failed.append(("config.py", "random arm reads the atp map",
                       "neither the patched nor the upstream output_prefix line "
                       "matched; paste your set_output_prefix and it can be "
                       "anchored to what you actually have"))
    else:
        failed[:] = [f for f in failed if "reads the atp map" not in f[1]]

    for path, desc in applied:
        print(f"  {'would apply' if args.check else 'applied':<12} {path}: {desc}")
    for path, desc in skipped:
        print(f"  {'already':<12} {path}: {desc}")
    for path, desc, why in failed:
        print(f"  {'FAILED':<12} {path}: {desc}\n               -> {why}")

    if not args.check:
        for path in sorted({p for p, *_ in applied}):
            try:
                ast.parse(open(path).read())
            except SyntaxError as e:
                print(f"  SYNTAX ERROR in {path}: {e}")
                return 2

    if failed:
        print("\nSome edits did not apply. Nothing here is partially written: "
              "each edit is\nall-or-nothing and idempotent, so rerunning after "
              "a fix is safe.")
        return 1
    print("\nOK. Verify with:  python run.py --help | grep random")
    return 0


if __name__ == "__main__":
    sys.exit(main())
