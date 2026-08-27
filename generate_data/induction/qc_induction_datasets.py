#!/usr/bin/env python3
"""QC the INDUCTION corpus, reusing qc_privilege_datasets.py unmodified.

    python generate_data/induction/qc_induction_datasets.py \\
        --data_dir data/gpt-oss-20b/induction/candidates \\
        --out pair_qc_induction.json

ONE CHECK, NOT TWO. The privilege and position corpora QC both conditions,
because both are supposed to work. Here only the induction condition is:

    induction-single-desired-all.jsonl -> does the model retrieve the matched
                                          value rather than the distractor?

The noinduction condition is SUPPOSED to fail -- there is nothing to match, so a
model that answered correctly there would be doing something other than
match-and-copy. Including it as a second check would drop every row.

That does mean QC here only confirms the task is solvable, not that the contrast
is clean. Read the noinduction rate separately (it is in the same output json
under failing_pairs if you add the second check for one exploratory run): if the
model retrieves the right value without a match at anything above chance, the
values are guessable from the keys and the nonce pool needs to be less
pronounceable.

Row-level dev_word/user_word aliases (= answer / distractor) are what let the
privilege scoring run untouched. This imports rather than copies, so a fix to the
scoring reaches all three corpora.
"""

import sys

import qc_privilege_datasets as qc

qc.CHECKS = [
    ("induction-single-desired-all.jsonl", "dev"),   # alias: dev_word == answer
]

if __name__ == "__main__":
    sys.exit(qc.main())
