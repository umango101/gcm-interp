import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer
model_name = "meta-llama/Llama-3.1-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Your dataframe
df = pd.read_csv("merged_eval_outputs.csv")

# System prompts
fluency_system = """Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness. Begin your evaluation by briefly describing the fluency of the sentence, noting any unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that may hinder readability. After providing your explanation, rate the sentence’s fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. Provide your rating using this exact format: “Rating: [[score]]”."""

relevance_system = """Please act as an impartial judge and evaluate whether the sentence fragment provided below is related to the instruction. Focus solely on the degree of relatedness in terms of topic, regardless of grammar, coherence, or informativeness. Begin your evaluation by providing a brief explanation of whether the sentence is related to the instruction, and point out references related to the instruction. After providing your explanation, rate the instruction relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, and 2 indicates it is more clearly and directly related to the instruction. Provide your rating using this exact format: “Rating: [[score]]”."""

def build_fluency_prompt(sentence: str):
    user_msg = f"""[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]"""
    
    chat = [
        {"role": "system", "content": fluency_system},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "Rating: [["}  # prefix only
    ]
    
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)[:-len('<|eot_id|>')]
    return prompt


def build_relevance_prompt(instruction: str, sentence: str):
    user_msg = f"""[Instruction Start]
{instruction}
[Instruction End]
[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]"""
    
    chat = [
        {"role": "system", "content": relevance_system},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "Rating: [["}  # prefix only
    ]
    
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)[:-len('<|eot_id|>')]
    return prompt

df["fluency_prompt"] = df["post-intervention-response"].apply(
    lambda x: build_fluency_prompt(x)
)

df["relevance_prompt"] = df.apply(
    lambda row: build_relevance_prompt(row["data_path_query"], row["post-intervention-response"]),
    axis=1
)

df.to_csv("relevance_fluency_prompts.csv", index=False)