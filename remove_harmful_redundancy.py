import json
indices_to_remove = [10, 11, 43, 58, 127, 135, 216, 150, 264, 70, 218, 25800, 1132, 4411, 176, 492, 724, 263, 97, 215, 284, 1497, 3751, 4587, 174, 142, 219, 168, 210, 194, 197, 129, 191, 231, 1437, 1827, 2579, 30, 175, 3521, 281, 283, 233, 1106, 1439, 4467, 261, 4051]

filepaths = [
    "data/Qwen1.5-14B-Chat/harmful-long/harmful-long-desired-all.jsonl", 
    "data/Qwen1.5-14B-Chat/harmful-long/harmful-long-undesired-all.jsonl",
    "data/Qwen1.5-14B-Chat/harmful-long/harmless-desired-all.jsonl",
    "data/Qwen1.5-14B-Chat/harmful-long/harmless-undesired-all.jsonl",
    "data/Qwen1.5-14B-Chat/harmful-single/harmful-single-desired-all.jsonl", 
    "data/Qwen1.5-14B-Chat/harmful-single/harmful-single-undesired-all.jsonl",
    "data/Qwen1.5-14B-Chat/harmful-single/harmless-desired-all.jsonl",
    "data/Qwen1.5-14B-Chat/harmful-single/harmless-undesired-all.jsonl"
]

for filepath in filepaths: 
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    data_to_keep = []
    for row in data:
        if row["id"] in indices_to_remove:
            continue
        else: 
            data_to_keep.append(row)
    # print(data_to_keep)

    with open(filepath, 'w') as f:
        for data_dict in data_to_keep:
            json_line = json.dumps(data_dict)
            f.write(json_line + '\n')