import json
import os
steering_factors = [10, 8, 6, 5, 4, 2, 1]
topk_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
tasks = ["from_paragraph-long_to_sentence-long", "from_paragraph-single_to_sentence-single"]
methods = ['atp']
base_dir = "/home/ubansal/orcd/scratch/gcm-interp/results/Qwen1.5-14B-Chat"

def calculate_accuracy(file_path, base):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    correct = 0
    total = 0
    for item in data:
        # print(base, item)
        if base == 'sentence-single' or base == 'sentence-long': # added the or statement, not sure how differs from line 25
            if 'long' in item[f'old_{base}'].lower():
                continue
            else:
                total += 1
                if 'long' in item[f'edit_{base}'].lower(): # what does the no do?
                    correct += 1
        elif base == 'sentence-long' or base == 'sentence-single':
            if 'anti' in item[f'old_{base}'].lower(): # why is this different from line 19
                continue
            else:
                total += 1
                if 'anti' in item[f'edit_{base}'].lower():
                    correct += 1
    print(f"Correct: {correct}, Total: {total}")
    return correct / total if total > 0 else 0

for task in tasks:
    base = task.split("_")[-1]
    source = task.split("_to_")[0].split("from_")[1].split("-")[0]
    task_dir = os.path.join(base_dir, task)
    for method in methods:
        method_dir = os.path.join(task_dir, method)
        for eval_dir, steering_dir in [('paragraph-single_eval', 'paragraph-long_steer'), ('paragraph-single_eval', 'paragraph-single_steer')]:
            for steering_factor in steering_factors:
                for topk in topk_values:
                    try:
                        file_path = os.path.join(method_dir, eval_dir, steering_dir, "eval/", f"{steering_factor}_targeted_steer_{topk}_{source}-single_gen.json")
                        accuracy = calculate_accuracy(file_path, base)
                        os.makedirs(f'results/accuracy/Qwen1.5-14B-Chat/{task}/{method}/{eval_dir}/{steering_dir}', exist_ok=True)
                        with open(f'results/accuracy/Qwen1.5-14B-Chat/{task}/{method}/{eval_dir}/{steering_dir}/{steering_factor}_targeted_steer_topk_{topk}_gen_accuracy_w_rf.json.accuracy.json', 'w') as out_f:
                            json.dump({'q1': accuracy}, out_f)
                    except FileNotFoundError as e:
                        continue