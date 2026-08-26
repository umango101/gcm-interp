#!/usr/bin/env python3
"""Fix `only_q` stripping the final USER turn on ICL-preamble datasets.

Run from the repo root of the `conflicts` checkout:

    python apply_onlyq_fix.py --check
    python apply_onlyq_fix.py

Idempotent; re-running reports "already applied" and writes nothing.

THE BUG
-------
`DataHandler.get_templated_prompts(..., only_q=True)` builds a generation prompt
by dropping the last message. Whether to drop was decided by:

    assistant_exists = any(p['role'] == 'assistant' for p in prompts[0]['prompt'])

`any()` was a correct proxy on the single-turn corpora, where a *-test.jsonl row
is [developer, user] with no assistant turn anywhere, so nothing was stripped.

With the extended-preamble corpus every test row carries eight ICL demo answers,
so `any()` is True and the branch strips the last message -- which in a test row
is the final USER question, not an answer. The model then receives eight demos
ending at "seven or two -> seven" plus a bare generation prompt, and continues
the pattern: every item answers "seven" regardless of steering.

    user-single/dev-single-test                n=18 last=user      any_assistant=True
    roleConflict-single/roleAgree-single-test  n=2  last=user      any_assistant=False

THE FIX
-------
Decide per row, on the LAST message's role. That is the property the code
actually wants, and it agrees with `any()` on every pre-preamble file:

    *-desired-all / *-undesired-all  end in assistant -> strip (unchanged)
    *-test                           end in user      -> keep  (fixed)

SCOPE
-----
Affects the eval/generation path only:
  * base_qs['test']  -- the generation prompts. THIS is what was truncated.
  * base_qs / source_qs / steering add+sub -- all end in an assistant turn, so
    `any()` and the last-message test agree and nothing changes for them. The
    steering vectors and the localization are NOT affected.

Consequence: every gen file produced from a preamble test set is invalid and must
be regenerated. The attribution shards and numerator_1_layers.pt are fine -- the
localization reads base_toks, which are templated in full and never go through
this branch.
"""

import argparse
import sys
from pathlib import Path

PATH = "data_handler.py"

OLD = """            assistant_exists = any([p['role'] == 'assistant' for p in prompts[0]['prompt']])
            if assistant_exists:
                prompt_lengths = [len(p['prompt']) - 1 for p in prompts]
            else:
                prompt_lengths = [max(len(p['prompt']), 1) for p in prompts]"""

NEW = """            # Decide per row on the LAST message's role, not on whether an
            # assistant turn appears anywhere. only_q wants "drop the answer if
            # this row ends with one"; any() is only a proxy for that, and it
            # breaks on ICL-preamble rows, where a *-test row ends with the USER
            # question but contains eight demo answers earlier. Under any() that
            # question was stripped and the model was asked to continue the demo
            # pattern -- which is why every item answered with the last demo's
            # word regardless of steering.
            prompt_lengths = [
                len(p['prompt']) - 1
                if p['prompt'] and p['prompt'][-1]['role'] == 'assistant'
                else max(len(p['prompt']), 1)
                for p in prompts
            ]"""

SENTINEL = "if p['prompt'] and p['prompt'][-1]['role'] == 'assistant'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()

    path = Path(args.root).resolve() / PATH
    if not path.exists():
        print(f"MISSING  {PATH} not found under {args.root}")
        return 1

    text = path.read_text()
    if SENTINEL in text:
        print(f"SKIP     {PATH}: already applied")
        return 0
    n = text.count(OLD)
    if n != 1:
        print(f"FAIL     {PATH}: anchor matched {n} times, expected 1 -- "
              f"apply by hand around 'assistant_exists'.")
        return 1

    if args.check:
        print(f"WOULD    {PATH}: per-row last-message test for only_q")
        return 0

    path.write_text(text.replace(OLD, NEW, 1))
    print(f"APPLY    {PATH}: per-row last-message test for only_q")
    print()
    print("Regenerate the eval sweep -- existing gen files used truncated prompts:")
    print("  rm -rf results_layers/gpt-oss-20b/from_user-single_to_dev-single/atp-per-layer")
    print("  rm -rf eval_pipeline_conflict/gpt-oss-20b/from_user-single_to_dev-single")
    print("  sbatch scripts/eval_layers_per_layer_user.sh")
    print("The localization shards and numerator_1_layers.pt are unaffected; keep them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
