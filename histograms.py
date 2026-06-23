from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import json

jsonl_path = "data/Qwen1.5-14B-Chat/paragraph-long/paragraph-long-desired-all.jsonl"

texts = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)

        # Extract assistant response
        text = example["prompt"][1]["content"]
        texts.append(text)

print(f"Loaded {len(texts)} texts")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen1.5-14B-Chat",
    trust_remote_code=True
)

# Compute token lengths
token_lengths = []

for text in texts:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_lengths.append(len(tokens))

# Print stats
print(f"Mean length: {sum(token_lengths) / len(token_lengths):.2f}")
print(f"Min length:  {min(token_lengths)}")
print(f"Max length:  {max(token_lengths)}")

# Create histogram
plt.figure(figsize=(8, 5))
plt.hist(token_lengths, bins=50)
plt.xlabel("Token Length")
plt.ylabel("Count")
plt.title("Histogram of Qwen1.5-14B-Chat Token Lengths")

# Save figure
output_path = "token_length_histogram.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved histogram to: {output_path}")

plt.close()
