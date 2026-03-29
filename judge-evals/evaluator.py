import json
import argparse
import re
from pathlib import Path
import torch
import pandas as pd
from vllm import LLM, SamplingParams

MODEL_NAME = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"

RATING_REGEX = re.compile(r"(\d+)\]\]")

def extract_rating(text: str) -> int:
    match = RATING_REGEX.search(text)
    return int(match.group(1)) if match else -1

def make_llm():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    print(f"Detected {num_gpus} GPUs")
    print("Loading 4-bit judge model…")

    return LLM(
        model=MODEL_NAME,
        quantization="bitsandbytes",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="auto",
        max_num_seqs=64,
        max_model_len=4096,
    )


def get_sampling_params():
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=3,
    )


def generate_in_batches(llm, prompts, sampling_params, batch_size):
    """
    Yields results batch-by-batch.
    """
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        results = llm.generate(batch, sampling_params)
        yield [r.outputs[0].text for r in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--relevance", action='store_true', default=False)
    parser.add_argument("--fluency", action='store_true', default=False)
    parser.add_argument("--judge", action='store_true', default=False)
    args = parser.parse_args()

    if args.relevance and args.fluency and args.judge:
        raise ValueError("Please specify only one of --relevance or --fluency or --judge.")
    if not args.relevance and not args.fluency and not args.judge:
        raise ValueError("Please specify one of --relevance or --fluency or --judge.")

    if args.relevance:
        required_column = 'relevance_prompt'
    elif args.fluency:
        required_column = 'fluency_prompt'
    elif args.judge:
        required_column = 'judge_prompt'

    df = pd.read_csv(args.input_csv)

    # REQUIRED COLUMN
    assert required_column in df.columns, f"ERROR: dataframe must contain a '{required_column}' column."

    # Extract prompts
    prompts = df[required_column].tolist()

    # Output paths
    input_path = Path(args.input_csv)
    if args.output_json is None:
        args.output_json = str(input_path.with_suffix(f".{required_column}.judge_outputs.json"))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accuracy_path = str(input_path.with_suffix(f".{required_column}.judge_accuracy.json"))

    # Prepare model
    llm = make_llm()
    sampling_params = get_sampling_params()

    enriched_buffer = []
    accuracy = 0
    total = 0
    batch_counter = 0

    # ALL columns we must propagate to outputs
    passthrough_cols = [
        "query",
        "post-intervention-response",
        "original-response",
        "filename",
        "rating",
        "data_path_query",
        "MODEL_ID",
        "SOURCE",
        "BASE",
        "METHOD",
        "SUB_DIR",
        "EVAL_SUB_DIR",
        "STEER_SUB_DIR",
        "N",
        "REPS",
        "STEERING_METHOD",
        "topk",
    ]

    for batch_outputs in generate_in_batches(llm, prompts, sampling_params, args.batch_size):

        rows = df.iloc[total: total + len(batch_outputs)]

        for (_, row), output in zip(rows.iterrows(), batch_outputs):

            judge_rating = extract_rating(output)

            enriched_item = {
                **{col: row[col] for col in passthrough_cols if col in row},
                required_column: row[required_column],
                "judge_output": output,
                "judge_rating": judge_rating,
            }

            enriched_buffer.append(enriched_item)

            if judge_rating == 2:
                accuracy += 1

            total += 1

        batch_counter += 1

        # Append results every 4 batches
        if batch_counter % 4 == 0:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(enriched_buffer, f, ensure_ascii=False, indent=2)
            enriched_buffer = []

    # Dump remainder
    if enriched_buffer:
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(enriched_buffer, f, ensure_ascii=False, indent=2)

    # Save accuracy
    with open(accuracy_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy / total}, f, indent=2)

    print(f"✓ Saved judge results → {out_path}")
    print(f"✓ Saved accuracy ({accuracy}/{total}) → {accuracy_path}")


if __name__ == "__main__":
    main()
