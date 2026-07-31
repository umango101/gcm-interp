import re
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import os
from nnsight import NNsight, LanguageModel
class ModelHandler:
    def __init__(self, config):
        self.config = config
        model_id = config.args.model_id
        self.device = config.args.device
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = self.load_tokenizer(model_id)
        self.model = self.load_model(model_id, self.device)
        self.model.tokenizer = self.tokenizer
        model_config = self.model.config.to_dict()
        hidden_size = model_config['hidden_size']
        self.num_heads = model_config['num_attention_heads']
        self.dim = hidden_size // self.num_heads

        if 'solar' in model_id.lower():
            self.marker = '### Assistant'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][2:]
        elif 'qwen' in model_id.lower():
            if self.config.args.source == 'harmful':
                self.marker = "<|im_start|>assistant" ## Something weird about data processing here, doesn't work with \n
            else:
                self.marker = "<|im_start|>assistant\n"
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif 'llama-2-7b-chat-hf' in model_id.lower():
            self.marker = "[/INST] "
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:-1]
        elif 'meta-llama' in model_id.lower():
            self.marker = '<|start_header_id|>assistant<|end_header_id|>'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:]
        elif 'olmo' in model_id.lower():
            self.marker = '<|assistant|>\n'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]

    def load_tokenizer(self, model_id):
        if 'qwen'  in model_id.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], pad_token='<|pad|>', eos_token='<|endoftext|>',)
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
            tokenizer.padding_side = 'left'
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        print('Tokenizer loaded, padding side is', tokenizer.padding_side)
        return tokenizer

    def load_model(self, model_id, device, model_type="causal"):
        if self.config.args.pyreft:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, quantization_config=bnb_config, device_map=device, attn_implementation="eager", trust_remote_code=True)
        else:
            return LanguageModel(model_id, device_map=device, tokenizer=self.tokenizer, torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], quantization_config=self.nf4_config, dispatch=True)
    