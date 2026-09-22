import json
d = "data/Falcon3-10B-Instruct/lying-single/"
t = [json.loads(l) for l in open(d + "truthful-single-test.jsonl") if l.strip()]
u = [json.loads(l) for l in open(d + "truthful-single-undesired-all.jsonl") if l.strip()]
print(len(t), "test rows,", len(u), "labelled rows\n")
print("TEST ROW 0:\n", json.dumps(t[0], indent=1)[:1500], "\n")
print("UNDESIRED ROW 0:\n", json.dumps(u[0], indent=1)[:1500])