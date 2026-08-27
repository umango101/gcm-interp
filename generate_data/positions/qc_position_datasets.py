#!/usr/bin/env python3
"""QC the POSITION corpus, reusing qc_privilege_datasets.py unmodified.

    # both conditions must work (strict, the default):
    python generate_data/positions/qc_position_datasets.py \
        --data_dir data/gpt-oss-20b/position/candidates --out pair_qc_position.json

    # only the first-named condition must work:
    QC_CHECKS=first python generate_data/positions/qc_position_datasets.py ...

WHY THERE IS A CHOICE HERE
--------------------------
gpt-oss-20b has a strong primacy bias. Measured at 8 demos on 82 candidate pairs:

    first-single-desired-all   pass 0.90   mean margin +1.92
    second-single-desired-all  pass 0.30   mean margin -0.78

The "answer the second-named option" preamble does not reliably override the
default. Requiring BOTH checks then multiplies: 0.30^2 ~ 0.09, and 2/82 pairs
survive.

QC_CHECKS=first keeps only the check that the first-named preamble works, which
lets the corpus build. THAT IS A REAL CONCESSION, NOT A FORMALITY, and it should
be stated wherever the resulting head map is used:

  * The ATP contrast stays well defined either way -- L is a log-likelihood
    difference, and it does not require the model to SUCCEED under both
    preambles.
  * But if the second-named preamble barely changes behaviour, the contrast is
    mostly "these two preambles contain different tokens" rather than "the model
    is doing two different things". A head map from such a contrast localizes
    whatever tracks the preamble's surface difference, which is weaker than what
    the control was meant to provide.

So prefer raising --n_demos until the second-condition rate comes up, and fall
back to QC_CHECKS=first only if it will not. Either way, report the two per-check
rates alongside the head map -- they are the reader's only way to judge how much
weight the control carries.

Row-level dev_word/user_word aliases (= first_word / second_word) are what let
the privilege scoring run untouched. This imports rather than copies, so a fix to
the scoring reaches all three corpora.
"""

import os
import sys

import qc_privilege_datasets as qc

_MODE = os.environ.get("QC_CHECKS", "both").lower()

_FIRST = ("first-single-desired-all.jsonl", "dev")     # alias: dev_word == first_word
_SECOND = ("second-single-desired-all.jsonl", "user")  # alias: user_word == second_word

if _MODE == "both":
    qc.CHECKS = [_FIRST, _SECOND]
elif _MODE == "first":
    qc.CHECKS = [_FIRST]
elif _MODE == "second":
    qc.CHECKS = [_SECOND]
else:
    sys.exit(f"QC_CHECKS must be both|first|second, got {_MODE!r}")

print(f"[qc] QC_CHECKS={_MODE}: requiring {[c[0] for c in qc.CHECKS]}", flush=True)
if _MODE != "both":
    print("[qc] NOTE: only one condition is gated. Report both per-check rates "
          "with any head map built from this corpus.", flush=True)

if __name__ == "__main__":
    sys.exit(qc.main())
