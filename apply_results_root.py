"""Add --results_root to config.py. Idempotent; safe to run twice.

WHY THIS IS A SCRIPT AND NOT A PATCH
A unified diff marks context lines with a leading space, so a context line that
is otherwise blank is a line of pure whitespace. config.py contains several of
those, and any transfer that strips trailing whitespace (most editors on save,
some browsers on download, some copy-paste paths) turns them into empty lines,
which git rejects as a corrupt hunk. Python only cares about LEADING
whitespace, so this file survives that trip.

    python apply_results_root.py [path/to/config.py]
"""
import ast
import sys

EDITS = [
    # (must already be present, replacement) -- each applied exactly once
    (
        "        parser.add_argument('--kv_caching', action='store_true', help='Steer prefill only using KV cache; decoding steps are not re-steered')",
        "        parser.add_argument('--kv_caching', action='store_true', help='Steer prefill only using KV cache; decoding steps are not re-steered')\n"
        "        parser.add_argument('--results_root', type=str, default='./results',\n"
        "                            help='Root of the results tree. Was hardcoded to '\n"
        "                                 './results, which meant two runs differing only '\n"
        "                                 'in their head list (e.g. a differenced vs an '\n"
        "                                 'undifferenced localization) resolved to the '\n"
        "                                 'same output prefix and overwrote each other. '\n"
        "                                 'Give each condition its own root.')",
    ),
    (
        "        steering_dir = self.args.steering_add_path.split('/')[-2] if self.args.steering_add_path else ''",
        "        steering_dir = self.args.steering_add_path.split('/')[-2] if self.args.steering_add_path else ''\n"
        "        # rstrip('/') so load_logits' split('/')[:-3] walk lands on the algo dir\n"
        "        # regardless of whether the caller passed a trailing slash.\n"
        "        root = getattr(self.args, 'results_root', './results').rstrip('/')",
    ),
    (
        '            self.output_prefix = f"./results/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/"',
        '            self.output_prefix = f"{root}/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/"',
    ),
    (
        '            self.output_prefix = f"./results/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/{eval_test_dir}_eval/{steering_dir}_steer/"',
        '            self.output_prefix = f"{root}/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/{eval_test_dir}_eval/{steering_dir}_steer/"',
    ),
]

SENTINEL = "--results_root"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "config.py"
    src = open(path).read()

    if SENTINEL in src:
        print(f"{path} already has --results_root; nothing to do.")
        return 0

    out = src
    for old, new in EDITS:
        n = out.count(old)
        if n != 1:
            print(f"ABORT: expected exactly 1 occurrence of this line in {path}, "
                  f"found {n}:\n  {old.strip()[:90]}\n"
                  f"config.py has diverged from what this script expects. "
                  f"Apply the four edits by hand, or diff against the config.py "
                  f"in the same download.")
            return 1
        out = out.replace(old, new, 1)

    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: result does not parse ({e}); {path} left untouched.")
        return 1

    open(path, "w").write(out)
    print(f"patched {path}:")
    print("  + --results_root argument (default ./results, so existing scripts")
    print("    keep writing to the same place)")
    print("  + root = ... in set_output_prefix")
    print("  ~ both ./results/ literals now use root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
