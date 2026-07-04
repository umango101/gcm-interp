import json, collections
recs = [json.loads(l) for l in open("checkpoint/preamble_responses.jsonl")]
c = collections.Counter()
for r in recs:
    c["judged"] += 1
    c[f"q1_answers={r['qc_harmless_answers']}"] += 1
    c[f"q2_refuses={r['qc_harmful_refuses']}"] += 1
    c[f"preamble_answers={r['qc_preamble_answers']}"] += 1
    c["qc_PASS"] += int(r["qc_pass"])
kept = len(json.load(open("output/kept_indices.json")))
errs = sum(1 for _ in open("output/length_mismatch_errors.jsonl"))
print(f"input={kept+errs}  length-filter kept={kept}  discarded={errs}")
for k in sorted(c): print(f"  {k}: {c[k]}")
