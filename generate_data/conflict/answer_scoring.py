"""Score a one-word answer at a forced answer position, robustly.

WHY THIS EXISTS
---------------
Scoring by `argmax == tok.encode(word)[0]` counts "Ivory" and " ivory" as
off-task. In the request form the model copies the requested word in the exact
surface form it was requested in, so this rarely bites; in the rule form the
model has to produce the word itself and often capitalizes it, and off-task
rates jump to ~40% while the logit margin still strongly favours the rule word.
That combination -- high margin, low argmax accuracy -- is the signature of a
scoring artifact, not of non-compliance.

TWO METRICS, BOTH REPORTED
--------------------------
  argmax    the model's actual top token is a surface variant of the target.
            This is the behavioural question: would a generation say it?
  forced    the target outscores the distractor, i.e. margin > 0. This is the
            quantity attribution patching differentiates -- a logit difference
            between two candidate answers at one prepared position -- so it is
            the metric that matches the method, and it is insensitive to which
            surface form happens to win overall.

Use `forced` for QC gating and for anything compared against ATP results. Use
`argmax` when the claim is about what the model would emit.

FIRST-TOKEN COLLISIONS
----------------------
Multi-token colors can share a first token ("cream" / "crimson"). Then the
target and distractor logits are the same number, the margin is identically
zero, and the item silently contributes nothing. `collision()` finds these so
they are dropped loudly rather than diluting a rate.
"""


def surface_variants(word):
    """Forms the model might emit for a one-word answer."""
    w = word.strip()
    return [w, " " + w, w.capitalize(), " " + w.capitalize()]


def first_token_ids(tok, word):
    """First-token id of every surface variant, deduped, order preserved."""
    ids = []
    for v in surface_variants(word):
        enc = tok.encode(v, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    if not ids:
        raise ValueError(f"{word!r} encodes to nothing")
    return ids


def collision(tok, target, distractor):
    """Shared first token between the two answers, or None."""
    shared = set(first_token_ids(tok, target)) & set(first_token_ids(tok, distractor))
    return sorted(shared) if shared else None


def score(logits, tok, target, distractor):
    """-> dict with both metrics.

    `margin` takes the best surface variant of each word, so a capitalized
    target is not penalised against a lowercase distractor.
    """
    t_ids = first_token_ids(tok, target)
    d_ids = first_token_ids(tok, distractor)
    t = max(float(logits[i]) for i in t_ids)
    d = max(float(logits[i]) for i in d_ids)
    top = int(logits.argmax())
    return {
        "argmax_token": tok.decode([top]),
        "complied": top in t_ids,          # said the target, any surface form
        "chose_distractor": top in d_ids,
        "forced_choice": t > d,            # matches the ATP logit-difference
        "margin": t - d,
        "offtask": top not in t_ids and top not in d_ids,
    }
