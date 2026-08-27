#!/usr/bin/env python3
"""Repair the \\x08 corruption apply_template_variation.py wrote into the generator.

    python fix_verify_regex.py --check
    python fix_verify_regex.py

An earlier build of apply_template_variation.py held its verify() replacement in a
NON-RAW triple-quoted literal that contained `\\b`. Python decoded that as U+0008
BACKSPACE before the text was ever written, so the file on disk got a literal
control character where the word-boundary escape belonged:

    re.search(r"<BACKSPACE>([A-Za-z]+) or ([A-Za-z]+)<BACKSPACE>", content)

which matches nothing, so frame_of() raised on a string that visibly contains
"circle or square".

Only the two regexes inside verify()'s frame_of are affected -- every other hunk
used a correctly escaped `\\\\b`. This restores each U+0008 in the file to `\\b`
and changes nothing else.

Verify afterwards with:
    grep -c $'\\x08' generate_data/conflict/make_privilege_datasets.py   # -> 0
"""

import argparse
import sys
from pathlib import Path

PATH = "generate_data/conflict/make_privilege_datasets.py"


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
    n = text.count("\x08")
    if n == 0:
        print(f"SKIP     {PATH}: no backspace characters -- already clean.")
        return 0

    for lineno, line in enumerate(text.splitlines(), 1):
        if "\x08" in line:
            print(f"  line {lineno}: {line.strip()!r}")

    if args.check:
        print(f"\ncheck: would repair {n} backspace character(s) to '\\b'")
        return 0

    path.write_text(text.replace("\x08", r"\b"))
    print(f"\ndone: repaired {n} backspace character(s) to '\\b' in {PATH}")
    print("Rerun the build; verify() should now report the frame count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
