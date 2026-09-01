import json, pandas as pd, glob
ind = json.load(open("induction_heads.json"))
rank = {(r["layer"], r["head"]): i for i, r in enumerate(ind["ranked"])}

for arm in ("devuser", "sysuser", "sysdev"):
    f = glob.glob(f"results/gpt-oss-20b/{arm}__*/atp/{arm}-dev-single-test_eval/"
                  f"{arm}_steer/eval/numerator_1_targeted_0.01.csv")[0]
    priv = pd.read_csv(f)
    k = len(priv)
    top_ind = {(r["layer"], r["head"]) for r in ind["ranked"][:k]}
    priv_set = {(int(r.layer), int(r.neuron)) for r in priv.itertuples()}
    ranks = sorted(rank[h] for h in priv_set if h in rank)
    print(f"{arm}: top-{k} overlap {len(priv_set & top_ind)}/{k} | "
          f"median induction rank of privilege heads {ranks[len(ranks)//2]} of {len(rank)}")
