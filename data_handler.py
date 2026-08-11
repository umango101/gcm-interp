import torch
import random
import torch.nn.functional as F
import json
import ast
import re
from tqdm import tqdm
random.seed(42)
import pandas as pd
class DataHandler:
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        self.device = self.config.args.device
        self.no_generation_prompt_for_eval_transfer = False
        
        file_paths = {
            'base_desired': f"{self.config.args.data_path}/{self.config.args.source}/{self.config.args.base}-desired-all.jsonl",
            'base_undesired': f"{self.config.args.data_path}/{self.config.args.source}/{self.config.args.base}-undesired-all.jsonl",
            'source_desired': f"{self.config.args.data_path}/{self.config.args.source}/{self.config.args.source}-desired-all.jsonl",
            'source_undesired': f"{self.config.args.data_path}/{self.config.args.source}/{self.config.args.source}-undesired-all.jsonl",
            'base_test': (
                f"{self.config.args.data_path}/{self.config.args.source}/{self.config.args.base}-test.jsonl"
                if isinstance(self.config.args.eval_test, bool) and self.config.args.eval_test
                else f"{self.config.args.eval_test}"
                if isinstance(self.config.args.eval_test, str) and self.config.args.eval_test
                else None
            ),
            'eval_transfer': f"./data/eval/from_{self.config.args.source}_to_{self.config.args.base}/{self.config.args.eval_transfer}.jsonl" if self.config.args.eval_transfer else None,
            'steering_add': self.config.args.steering_add_path,
            'steering_sub': self.config.args.steering_sub_path
        }

        jsons = {
            'base_desired': self.load_from_jsonl(file_paths['base_desired']),
            'base_undesired': self.load_from_jsonl(file_paths['base_undesired']),
            'source_desired': self.load_from_jsonl(file_paths['source_desired']),
            'source_undesired': self.load_from_jsonl(file_paths['source_undesired']),
            'base_test': self.load_from_jsonl(file_paths['base_test']) if self.config.args.eval_test else None,
            'eval_transfer': self.load_from_jsonl(file_paths['eval_transfer']) if self.config.args.eval_transfer else None,
            'steering_add': self.load_from_jsonl(file_paths['steering_add']) if self.config.args.steering_add_path else None,
            'steering_sub': self.load_from_jsonl(file_paths['steering_sub']) if self.config.args.steering_sub_path else None,
        }

        print('Making base templated prompts...')
        base = {
            'desired': self.get_templated_prompts(jsons['base_desired']),
            'undesired': self.get_templated_prompts(jsons['base_undesired'])
        }
        print(base['desired'][0])

        print('Making base_qs templated prompts...')
        base_qs = {
            'desired': self.get_templated_prompts(jsons['base_desired'], only_q=True, add_generation_prompt=True),
            'undesired': self.get_templated_prompts(jsons['base_undesired'], only_q=True, add_generation_prompt=True),
        }
        print(base_qs['desired'][0])

        if self.config.args.eval_test:
            print('Making base_qs test templated prompts...')
            base_qs['test'] = self.get_templated_prompts(jsons['base_test'], only_q=True, add_generation_prompt=True)
            print(base_qs['test'][0])

        print('Making source qs templated prompts...')
        source_qs = {
            'desired': self.get_templated_prompts(jsons['source_desired'], only_q=True, add_generation_prompt=True),
            'undesired': self.get_templated_prompts(jsons['source_undesired'], only_q=True, add_generation_prompt=True)
        }
        print(source_qs['desired'][0])
        
        steering = {
            "add_qs": self.get_templated_prompts(jsons['steering_add'], only_q=True, add_generation_prompt=True) if jsons['steering_add'] else None,
            "sub_qs": self.get_templated_prompts(jsons['steering_sub'], only_q=True, add_generation_prompt=True) if jsons['steering_sub'] else None
        }

        if self.config.args.eval_test:
            if not (self.config.args.steering_add_path is None) and not (self.config.args.steering_sub_path is None):
                all_templated_prompts = steering['add_qs'] + steering['sub_qs'] + base_qs['test']
        else:
            all_templated_prompts = base['desired'] + base['undesired'] + source_qs['desired'] + source_qs['undesired']
        all_tokenized_prompts = self.tokenize_prompts(all_templated_prompts, max_length=None)
        self.max_len = all_tokenized_prompts['input_ids'].shape[1]

        if self.config.args.patch_model:
            print('Tokenizing base_toks')
            self.base_toks = {
                key: self.tokenize_prompts(base[key], max_length=self.max_len) for key in base
            }

            print('Tokenizing base_qs_toks')
            self.base_qs_toks = {
                key: self.tokenize_prompts(base_qs[key], max_length=self.max_len) for key in base_qs
            }

            print('Tokenizing source_qs_toks')
            self.source_qs_toks = {
                key: self.tokenize_prompts(source_qs[key], max_length=self.max_len) for key in source_qs
            }

            print('Finding response start positions...')
            self.response_start_positions = {
                "base": {
                    key: self.get_resp_start_pos(self.base_toks[key], self.model_handler.marker, self.model_handler.tokenizer) for key in self.base_toks
                }
            }

        elif self.config.args.eval_model:

            orig_template = self.model_handler.tokenizer.chat_template
            if self.config.args.eval_transfer:
                # Removing the system prompt for eval_test dataset
                if config.args.model_id.split('/')[1] == 'Qwen1.5-14B-Chat':
                    self.model_handler.tokenizer.chat_template = "{% for message in messages %}\n" \
                    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}\n" \
                    "{% endfor %}\n" \
                    "{% if add_generation_prompt %}\n" \
                    "{{ '<|im_start|>assistant\n' }}" \
                    "{% endif %}"

                elif config.args.model_id.split('/')[1] == 'OLMo-2-1124-13B-DPO':
                    self.model_handler.tokenizer.chat_template = "{{ bos_token }}\n" \
                    "{% for message in messages %}\n" \
                    "{% if message['role'] == 'user' -%}\n" \
                    "{{ '<|user|>\\n' + message['content'] + '\\n' }}\n" \
                    "{%- elif message['role'] == 'assistant' -%}\n" \
                    "{%- if not loop.last -%}\n" \
                    "{{ '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}\n" \
                    "{%- else -%}\n" \
                    "{{ '<|assistant|>\\n' + message['content'] + eos_token }}\n" \
                    "{%- endif -%}\n" \
                    "{%- endif %}\n" \
                    "{% endfor -%}\n" \
                    "{% if add_generation_prompt -%}\n" \
                    "{{ '\\n<|assistant|>\\n' }}\n" \
                    "{%- endif %}"

                start_indices = [i for i in range(len(jsons['eval_transfer'])) if i % 5 == 0]

                selected_starts = random.sample(start_indices, 40)

                random_indices = []
                for start in selected_starts:
                    block = [start + offset for offset in range(5)]
                    random_indices.extend(block)
                print('###### Tokenizing eval_transfer dataset templated prompts...')
                self.eval_transfer = {
                    "queries": self.tokenize_prompts(random.sample(self.get_templated_prompts([jsons['eval_transfer'][j] for j in random_indices], only_q=True, add_generation_prompt=True), k=200), max_length=None),
                    "answers": None,
                }
                if self.eval_transfer['queries']['input_ids'].shape[1] < self.max_len:
                    self.eval_transfer = {
                        "queries": self.tokenize_prompts(random.sample(self.get_templated_prompts([jsons['eval_transfer'][j] for j in random_indices], only_q=True, add_generation_prompt=True), k=200), max_length=self.max_len),
                        "answers": None
                    }
                else:
                    self.max_len = self.eval_transfer['queries']['input_ids'].shape[1]
                
                self.no_generation_prompt_for_eval_transfer = False
                self.model_handler.tokenizer.chat_template = orig_template

            elif self.config.args.eval_test:
                self.base_qs_toks = {
                    'test': self.tokenize_prompts(base_qs['test'], max_length=self.max_len)
                }

                self.steering_qs_toks = {
                    "add": self.tokenize_prompts(steering["add_qs"], max_length=self.max_len) if steering["add_qs"] else None,
                    "sub": self.tokenize_prompts(steering["sub_qs"], max_length=self.max_len) if steering["sub_qs"] else None
                }

                # mean_ablations_cache() reads source_qs_toks, otherwise only
                # built under --patch_model. Steering does not need it, which is
                # why the gap went unnoticed.
                if self.config.args.ablation == 'mean':
                    print('Tokenizing source_qs_toks for mean ablation')
                    _u = self.tokenize_prompts(source_qs['desired'], max_length=None)
                    if _u['input_ids'].shape[1] > self.max_len:
                        raise ValueError(
                            f"source prompts are {_u['input_ids'].shape[1]} tokens but max_len "
                            f"is {self.max_len}; mean ablation would truncate them silently.")
                    self.source_qs_toks = {
                        key: self.tokenize_prompts(source_qs[key], max_length=self.max_len)
                        for key in source_qs
                    }

            elif self.config.args.ablation == 'pyreft':
                print('Tokenizing pyreft prompts...')
                self.pyreft_prompts = self.get_templated_prompts(jsons['base_desired'], _base_completion=jsons['source_desired'], add_generation_prompt=False)
                self.pyreft_toks = self.tokenize_prompts(self.pyreft_prompts, max_length=self.max_len)

                self.response_start_positions['pyreft'] = self.get_resp_start_pos(self.pyreft_toks, self.model_handler.marker, self.model_handler.tokenizer) if self.config.args.ablation == 'pyreft' else None
        
        if self.config.args.eval_model:
            if self.config.args.eval_transfer:
                self.LEN = 100
            else:
                self.LEN = min(len(base['desired']), 50)
        else:
            self.LEN = min(len(base['desired']), 100)

        self.truncate_to_len(self.LEN)
    
    def truncate_to_len(self, L):
        self.LEN = L
        if hasattr(self, "base_toks") and self.base_toks:
            for key in self.base_toks:
                self.base_toks[key] = {k: v[:L] for k, v in self.base_toks[key].items()}
        
        if hasattr(self, "base_qs_toks") and self.base_qs_toks:
            if self.config.args.eval_test:
                self.base_qs_toks['test'] = {k: v[:L] for k, v in self.base_qs_toks['test'].items()}
            else:
                for key in self.base_qs_toks:
                    self.base_qs_toks[key] = {k: v[:L] for k, v in self.base_qs_toks[key].items()}
        
        if hasattr(self, "source_qs_toks") and self.source_qs_toks:
            for key in self.source_qs_toks:
                self.source_qs_toks[key] = {k: v[:L] for k, v in self.source_qs_toks[key].items()}

        if hasattr(self, "pyreft_toks") and self.pyreft_toks:
            for key in self.pyreft_toks:
                self.pyreft_toks[key] = {k: v[:L] for k, v in self.pyreft_toks[key].items()}

        if hasattr(self, 'response_start_positions') and self.response_start_positions:
            self.response_start_positions = {
                "base": {
                    key: self.response_start_positions["base"][key][:L] for key in self.response_start_positions["base"]
                }
            }
            if "pyreft" in self.response_start_positions:
                self.response_start_positions["pyreft"] = self.get_resp_start_pos(self.pyreft_toks, self.model_handler.marker, self.model_handler.tokenizer)

    def get_templated_prompts(self, prompts, _base_completion=None, only_q=False, add_generation_prompt=False):
        if only_q:
            prompt_lengths = None
            if self.no_generation_prompt_for_eval_transfer:
                print('Explicitly setting add_generation_prompt to False for eval_transfer dataset.')
                add_generation_prompt = False
            assistant_exists = any([p['role'] == 'assistant' for p in prompts[0]['prompt']])
            if assistant_exists:
                prompt_lengths = [len(p['prompt']) - 1 for p in prompts]
            else:
                prompt_lengths = [max(len(p['prompt']), 1) for p in prompts]
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    [p['prompt'][i] for i in range(prompt_lengths[pdx])],
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False
                ) for pdx, p in enumerate(prompts)]
        elif _base_completion is not None:
            assert len(prompts) == len(_base_completion), f"Length of prompts and base completion do not match: {len(prompts)} vs {len(_base_completion)}"
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    [p['prompt'][i] for i in range(len(p['prompt']) - 1)] + [_base_completion[pi]['prompt'][-1]],
                    tokenize=False
                ) for pi, p in enumerate(prompts)]
        else:
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    p['prompt'], 
                    add_generation_prompt=False,
                    tokenize=False
                ) for p in prompts]
        
    def get_resp_start_pos(self, tokens, marker, tokenizer):
        response_start_positions = []
        for _, tok in tqdm(enumerate(tokens['input_ids'])):
            tok = tok.to(self.device)
            response_start_position = None
            marker_tokens = self.model_handler.alignment_tokens.to(self.device)
            marker_len = marker_tokens.size(0)
            for j in range(tok.size(0) - marker_len + 1):
                if torch.equal(tok[j : j + marker_len], marker_tokens):
                    response_start_position = j + marker_len
                    break
            assert response_start_position is not None, f"Marker {repr(marker)} not found in input {repr(tokenizer.decode(tok))}.\nTOKENS: {repr(tok)}\nMARKER: {repr(marker_tokens)}"
            response_start_positions.append(response_start_position)
        return response_start_positions
    
    def tokenize_prompts(self, p, max_length=None):
        if max_length is None:
            tokens = self.model_handler.tokenizer(p, padding=True, truncation=False, return_tensors="pt")
        else:
            print('max_length', max_length)
            tokens =  self.model_handler.tokenizer(p, padding='max_length', max_length=self.max_len, truncation=False, return_tensors="pt")
        return {"input_ids": tokens["input_ids"].to(self.device), "attention_mask": tokens["attention_mask"].to(self.device)}
    
    def decode_prompts(self, p):
        return self.model_handler.tokenizer.decode(p, skip_special_tokens=True)

    def align_toks(self, source_toks, base_toks):
        """
        Align source_toks to base_toks at the assistant alignment token subsequence.
        """

        def find_subseq_start(seq, subseq):
            """Find first index where subseq occurs in seq, or -1 if not found."""
            n, m = len(seq), len(subseq)
            for i in range(n - m + 1):
                if seq[i:i+m] == subseq:
                    return i
            return -1

        toks_mod = {"input_ids": [], "attention_mask": []}
        align_seq = self.model_handler.alignment_tokens.tolist()
        pad_token_id = self.model_handler.tokenizer.pad_token_id
        for i in range(len(base_toks["input_ids"])):
            src_ids = source_toks["input_ids"][i]

            src_mask = source_toks["attention_mask"][i]
            base_ids = base_toks["input_ids"][i]
            base_mask = base_toks["attention_mask"][i]

            src_list = src_ids.tolist()
            base_list = base_ids.tolist()

            # locate assistant token subsequence
            src_start = find_subseq_start(src_list, align_seq)
            base_start = find_subseq_start(base_list, align_seq)

            if src_start == -1 or base_start == -1:
                raise AssertionError(f"Assistant alignment sequence {align_seq} {self.model_handler.tokenizer.convert_ids_to_tokens(align_seq)} not found in example {i}. Source start: {src_start}, Base start: {base_start}")

            offset = src_start - base_start
            L = src_ids.size(0)

            if offset > 0:
                # shift left by offset
                pad_ids = src_ids.new_full((offset,), pad_token_id)
                pad_mask = src_mask.new_full((offset,), 0)
                aligned_ids = torch.cat([src_ids[offset:], pad_ids], dim=0)
                aligned_mask = torch.cat([src_mask[offset:], pad_mask], dim=0)
            elif offset < 0:
                # shift right by -offset
                k = -offset
                pad_ids = src_ids.new_full((k,), pad_token_id)
                pad_mask = src_mask.new_full((k,), 0)
                aligned_ids = torch.cat([pad_ids, src_ids[:L-k]], dim=0)
                aligned_mask = torch.cat([pad_mask, src_mask[:L-k]], dim=0)
            else:
                aligned_ids = src_ids
                aligned_mask = src_mask

            toks_mod["input_ids"].append(aligned_ids)
            toks_mod["attention_mask"].append(aligned_mask)

        toks_mod["input_ids"] = torch.stack(toks_mod["input_ids"])
        toks_mod["attention_mask"] = torch.stack(toks_mod["attention_mask"])
        return toks_mod


    @staticmethod
    def load_from_jsonl(file_name):
        def load_json_line(line: str, i: int, file_name: str):
            try:
                return json.loads(json.dumps(ast.literal_eval(line)))
            except Exception as e:
                return None
        
        with open(file_name, "r") as f:
            data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
            data = [d for d in data if d is not None]
        return data
