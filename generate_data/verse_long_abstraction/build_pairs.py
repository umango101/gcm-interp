#!/usr/bin/env python
"""
Build and QC the (Q1, Q2) question pairs for the verse-long causal abstraction experiment.

Pairing constraints:
  * the two templated verse prompts must have the same token length, so that the fmt /
    question / suffix / last / prompt spans line up index-for-index and a patch at those
    indices means the same thing in both runs;
  * the questions must be lexically distant, otherwise the topic contrast is not diagnostic.

QC (teacher-forced, no judge and no generation).  For a pair to be usable, the four reference
answers have to be separable by the metric the experiment relies on.  Under each of the three
unpatched prompts the matching reference must be the highest-scoring of the four by at least
--min_margin nats per token:

    prompt (Q1, verse) -> q1_verse      prompt (Q2, verse) -> q2_verse
    prompt (Q1, prose) -> q1_prose

The third check also establishes format_aligned (verse and prose prompts the same token length),
which the experiment's format condition requires; pairs failing only that check are kept with
format_aligned=false and are still usable for the topic and transplant arms.

Writes <out_dir>/pairs.jsonl and <out_dir>/pairs_report.json.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import verse_long_utils as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    ap.add_argument("--data_dir", default=None, help="default: <repo>/data/<model_name>/verse-long")
    ap.add_argument("--out_dir", default=None,
                    help="default: <repo>/data/<model_name>/verse-long-abstraction")
    ap.add_argument("--max_similarity", type=float, default=0.2,
                    help="reject a pair whose questions share more than this Jaccard of content words")
    ap.add_argument("--min_margin", type=float, default=0.0,
                    help="nats/token the matching reference must lead the other three by")
    ap.add_argument("--max_ref_tokens", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no_qc", action="store_true")
    args = ap.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / model_name / "verse-long-abstraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from determinism import enable_determinism
        enable_determinism()
    except ImportError:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    verse_path, prose_path = U.resolve_data_paths(REPO_ROOT, model_name, args.data_dir)
    rows = U.load_rows(verse_path, prose_path)
    print(f"{len(rows)} rows from {verse_path.name} / {prose_path.name}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    anat = {rid: {"verse": U.Anatomy(tok, "verse", r["question"]),
                  "prose": U.Anatomy(tok, "prose", r["question"])} for rid, r in rows.items()}
    length_key = {rid: len(a["verse"].ids) for rid, a in anat.items()}
    fmt_aligned = {rid: len(a["verse"].ids) == len(a["prose"].ids) for rid, a in anat.items()}
    print(f"verse/prose token lengths match for {sum(fmt_aligned.values())}/{len(rows)} rows")

    pairs, rejected = U.pair_by_length(rows, length_key, args.max_similarity)
    print(f"{len(pairs)} length-matched pairs ({len(rejected)} rejected as too similar)")

    # alignment is a hard requirement for the topic arm
    checked = []
    for p in pairs:
        a, b = anat[p["q1"]]["verse"], anat[p["q2"]]["verse"]
        bad = [ps for ps in U.POSITION_SETS if ps in a.positions and a.check_alignment(b, ps)]
        if bad:
            p["alignment_error"] = {ps: a.check_alignment(b, ps) for ps in bad}
            rejected.append(p)
            continue
        p["format_aligned"] = bool(fmt_aligned[p["q1"]] and fmt_aligned[p["q2"]])
        p["prompt_tokens"] = len(a.ids)
        checked.append(p)
    pairs = checked

    if args.no_qc:
        for p in pairs:
            p["qc"] = None
        kept = pairs
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto").eval()

        def score(anatomy, refs):
            batch = U.ReferenceBatch(tok, anatomy.ids, refs, args.max_ref_tokens, model.device)
            with torch.no_grad():
                logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
            return batch.reduce(U.gather_logprobs(logits, batch.target), batch.score_mask)[0]

        kept = []
        for n, p in enumerate(pairs):
            i, j = p["q1"], p["q2"]
            refs = {"q1_verse": rows[i]["verse_answer"], "q1_prose": rows[i]["prose_answer"],
                    "q2_verse": rows[j]["verse_answer"], "q2_prose": rows[j]["prose_answer"]}
            qc, ok = {}, True
            probes = [("q1_verse", anat[i]["verse"]), ("q2_verse", anat[j]["verse"]),
                      ("q1_prose", anat[i]["prose"])]
            for want, anatomy in probes:
                sc = score(anatomy, refs)
                vals = {k: v["logp_mean"] for k, v in sc.items()}
                margin = vals[want] - max(v for k, v in vals.items() if k != want)
                qc[want] = dict(logp_mean={k: round(v, 4) for k, v in vals.items()},
                                margin=round(margin, 4), ok=bool(margin > args.min_margin),
                                **U.contrasts(sc))
                ok = ok and margin > args.min_margin
            p["qc"] = qc
            p["qc_pass"] = bool(ok)
            if ok:
                kept.append(p)
            print(f"  [{n + 1}/{len(pairs)}] pair ({i},{j}) "
                  f"margins={[round(qc[k]['margin'], 3) for k in qc]} {'ok' if ok else 'REJECT'}",
                  flush=True)

    U.write_jsonl(out_dir / "pairs.jsonl", kept)
    report = dict(model=args.model, verse_path=str(verse_path), prose_path=str(prose_path),
                  n_rows=len(rows), n_pairs_length_matched=len(pairs), n_pairs_kept=len(kept),
                  n_format_aligned=sum(p.get("format_aligned", False) for p in kept),
                  max_similarity=args.max_similarity, min_margin=args.min_margin,
                  max_ref_tokens=args.max_ref_tokens, qc_enabled=not args.no_qc,
                  rejected=rejected[:50])
    with open(out_dir / "pairs_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rejected"}, indent=2))
    print(f"wrote {out_dir}/pairs.jsonl")


if __name__ == "__main__":
    main()