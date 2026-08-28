#!/usr/bin/env python3
"""Repair the unterminated f-string apply_layer_matched_random.py wrote.

    python fix_eval_runner_fstring.py --check
    python fix_eval_runner_fstring.py

An earlier build of apply_layer_matched_random.py held its replacement in a
NON-RAW triple-quoted literal containing `\\n`. Python decoded that to a real
newline before the text was written, so eval/eval_runner.py got a string literal
split across lines:

    raise SystemExit(
        f"layer-matched random needs the targeted map, not found at
"
        f"  {_atp_map}
"

-> SyntaxError: unterminated f-string literal

Nothing else in that applier is affected -- the other hunks used no escapes.
This rejoins the three fragments into valid f-strings and changes nothing else.

Same root cause as the earlier verify()-regex corruption: a non-raw heredoc
eating a backslash escape on its way into generated source.
"""

import argparse
import sys
from pathlib import Path

PATH = "eval/eval_runner.py"

BAD = ('                        f"layer-matched random needs the targeted map, not found at\n"\n'
       '                        f"  {_atp_map}\n"\n'
       '                        f"Point ATP_MAP at it, or use RANDOM_BASELINE=uniform.")')

GOOD = ('                        f"layer-matched random needs the targeted map, not "\n'
        '                        f"found at\\n  {_atp_map}\\n"\n'
        '                        f"Point ATP_MAP at it, or use RANDOM_BASELINE=uniform.")')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    p = Path(args.root).resolve() / PATH
    if not p.exists():
        print(f"MISSING  {PATH} not found under {args.root}")
        return 1

    t = p.read_text()
    if GOOD in t:
        print(f"SKIP     {PATH}: already repaired")
        return 0
    n = t.count(BAD)
    if n != 1:
        print(f"FAIL     {PATH}: anchor matched {n} times, expected 1.")
        print("         Fix by hand: the raise SystemExit(...) near 'layer-matched")
        print("         random needs the targeted map' has string literals split")
        print("         across lines; rejoin them and escape the newlines as \\\\n.")
        return 1

    if args.check:
        print(f"WOULD    {PATH}: rejoin the split f-string")
        return 0

    p.write_text(t.replace(BAD, GOOD, 1))
    print(f"APPLY    {PATH}: rejoined the split f-string")
    print("\nVerify:  python -c \"import ast,pathlib; "
          "ast.parse(pathlib.Path('eval/eval_runner.py').read_text())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
