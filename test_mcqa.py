import pandas as pd, re
df = pd.read_csv("eval_pipeline/Qwen1.5-14B-Chat/from_paragraph-single_to_sentence-single/atp/paragraph-single_eval/paragraph-single_steer/merged_eval_outputs.csv",
                 keep_default_na=False)
opts = lambda s: re.findall(r"\(([ABCD])\)\s*([^\n]+)", s)
bad = df[df.apply(lambda r: opts(r["query"]) != opts(r["data_path_query"]), axis=1)]
print(len(bad), "of", len(df), "rows where the model saw a different option order")

parsed = df["data_path_query"].map(lambda s: len(opts(s)))
print(parsed.value_counts())
print(opts(df["data_path_query"].iloc[0]))