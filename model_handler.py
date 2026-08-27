import re
import torch
import inspect
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import os
from nnsight import NNsight, LanguageModel

def _hf_token():
    # An unset/empty HF_TOKEN must resolve to None (anonymous access), not "" --
    # newer huggingface_hub sends "" as a literal 'Bearer ' header and errors.
    # Public repos (e.g. phi-4, Qwen1.5-14B-Chat) work fine unauthenticated.
    return os.environ.get('HF_TOKEN') or None

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
        self._text_config = None
        self.model = self.load_model(model_id, self.device)
        self.model.tokenizer = self.tokenizer
        raw_config = self._text_config or getattr(self.model, "config", None) \
             or getattr(self.model, "_model", self.model).config
        model_config = raw_config.to_dict()
        # model_config = self.model.config.to_dict()
        # Composite configs (e.g. Gemma3's multimodal Gemma3Config) nest the text-decoder
        # fields under 'text_config' instead of at the top level.
        text_config = model_config.get('text_config', model_config)
        hidden_size = text_config['hidden_size']
        self.num_heads = text_config['num_attention_heads']
        # Per-head slice width for the o_proj INPUT space (the concatenated per-head
        # attention outputs z), which is num_attention_heads * head_dim. That is NOT
        # always hidden_size: Gemma-3-12B has hidden_size=3840 but 16 heads x 256 =
        # 4096. Prefer the config's head_dim, cross-check against the real module,
        # and only fall back to the old formula if neither is available.
        self.dim = text_config.get('head_dim') or (hidden_size // self.num_heads)
        try:
            _oproj = getattr(self.model, "_model", self.model).model.layers[0].self_attn.o_proj
            _in = int(_oproj.in_features)
            if _in % self.num_heads != 0:
                raise ValueError(
                    f"o_proj.in_features={_in} is not divisible by num_attention_heads="
                    f"{self.num_heads}; per-head slicing would be meaningless."
                )
            if _in // self.num_heads != self.dim:
                print(f"[model] head_dim {self.dim} disagrees with o_proj.in_features//heads "
                      f"= {_in // self.num_heads}; using the module's value.", flush=True)
                self.dim = _in // self.num_heads
        except (AttributeError, IndexError) as e:
            print(f"[model] could not read o_proj.in_features ({e}); "
                  f"using head_dim={self.dim} from config.", flush=True)
        print(f"[model] hidden_size={hidden_size} num_heads={self.num_heads} "
              f"per-head dim={self.dim} (heads*dim={self.num_heads * self.dim})", flush=True)

        if 'solar' in model_id.lower():
            self.marker = '### Assistant'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][2:]
        elif 'qwen' in model_id.lower():
            if self.config.args.source == 'harmful':
                self.marker = "<|im_start|>assistant" ## Something weird about data processing here, doesn't work with \n
            else:
                self.marker = "<|im_start|>assistant\n"
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif 'llama-2' in model_id.lower() and 'chat' in model_id.lower():
            self.marker = "[/INST] "
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:-1]
        elif 'meta-llama' in model_id.lower():
            self.marker = '<|start_header_id|>assistant<|end_header_id|>'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:]
        elif 'olmo' in model_id.lower():
            self.marker = '<|assistant|>\n'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif 'falcon3' in model_id.lower():
            # Falcon3's chat template closes an add_generation_prompt turn with
            # '<|assistant|>\n'. Unlike OLMo and Qwen, '<|assistant|>' is NOT in
            # Falcon3's special-token table (its added tokens are the legacy Falcon
            # >>TITLE<< family plus <|endoftext|>/<|pad|>/<|startoftext|>), so the
            # marker is ordinary text that the BPE is free to segment differently
            # depending on what precedes it. data_handler.get_resp_start_pos and
            # align_toks both look for alignment_tokens as an EXACT subsequence and
            # hard-assert when it is absent, so a standalone tokenizer(marker) call
            # that happens to segment differently is a startup crash, not a silent
            # skew. Derive the tokens from the template itself instead.
            self.marker = '<|assistant|>\n'
            self.alignment_tokens = self._alignment_tokens_from_template(self.marker)
        elif 'gemma' in model_id.lower():
            self.marker = '<start_of_turn>model\n'
            # Standalone encoding prepends <bos>; drop it so alignment_tokens matches the
            # exact in-context subsequence right before the assistant's reply.
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:]
        elif 'phi-4' in model_id.lower():
            # phi-4's chat template uses <|im_sep|> instead of Qwen's '\n' after the role tag,
            # and no space/newline follows it before the reply starts.
            self.marker = '<|im_start|>assistant<|im_sep|>'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif 'vicuna' in model_id.lower():
            self.marker = 'ASSISTANT:'
            # Derive alignment tokens dynamically: encode a dummy USER turn followed by
            # ASSISTANT: and subtract the prefix — this captures the exact in-context
            # tokenization of ASSISTANT: (which differs from standalone tokenization).
            prefix_ids = self.tokenizer("USER: x\n")["input_ids"]
            full_ids   = self.tokenizer("USER: x\nASSISTANT:")["input_ids"]
            self.alignment_tokens = torch.tensor(full_ids[len(prefix_ids):])

    # Jinja2 chat template for models trained on the Vicuna USER/ASSISTANT format
    VICUNA_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '\n' }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
    )

    def _alignment_tokens_from_template(self, marker):
        """In-context tokenization of the assistant generation prompt.

        Renders one dummy turn twice -- with and without add_generation_prompt --
        tokenizes both, and returns the tokens the generation-prompt version adds
        past the longest common prefix. That tail is exactly the subsequence that
        get_resp_start_pos and align_toks scan for in a real templated prompt,
        including any BPE merge across the boundary with the character that
        precedes the marker. A standalone tokenizer(marker) call cannot reproduce
        that boundary, which is why models whose role tags are not special tokens
        need this path rather than the hardcoded slices used elsewhere.

        Falls back to the standalone encoding (minus a leading BOS, which the
        template does not repeat in context) if the probe fails for any reason.
        """
        tok = self.tokenizer
        msgs = [{"role": "user", "content": "x"}]
        try:
            without = tok(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))["input_ids"]
            with_gen = tok(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))["input_ids"]
        except Exception as e:
            print(f"[model] chat-template probe failed ({e}); "
                  f"falling back to standalone marker encoding.", flush=True)
            without, with_gen = None, None

        if with_gen is not None and len(with_gen) > len(without):
            i = 0
            while i < len(without) and without[i] == with_gen[i]:
                i += 1
            derived = with_gen[i:]
            if derived:
                print(f"[model] alignment tokens from template: {derived} -> "
                      f"{tok.convert_ids_to_tokens(derived)}", flush=True)
                return torch.tensor(derived)
            print("[model] template probe produced an empty tail; "
                  "falling back to standalone marker encoding.", flush=True)

        ids = tok(marker, return_tensors="pt")["input_ids"][0]
        if tok.bos_token_id is not None and ids.numel() and int(ids[0]) == tok.bos_token_id:
            ids = ids[1:]
        print(f"[model] alignment tokens (standalone): {ids.tolist()} -> "
              f"{tok.convert_ids_to_tokens(ids.tolist())}", flush=True)
        return ids

    def load_tokenizer(self, model_id):
        if 'qwen'  in model_id.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=_hf_token(), pad_token='<|pad|>', eos_token='<|endoftext|>',)
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
            tokenizer.padding_side = 'left'
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=_hf_token())
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if tokenizer.chat_template is None:
            tokenizer.chat_template = self.VICUNA_CHAT_TEMPLATE
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
        elif getattr(self.config.args, 'full_precision', False):
            # Full bfloat16, no quantization. device_map="auto" lets HF Accelerate
            # distribute layers across all available memory (GPU HBM + CPU RAM on GH200).
            if 'gemma' in model_id.lower():
                return self._load_gemma_causal_lm(model_id, quantization_config=None, device_map="auto")
            return LanguageModel(model_id, device_map="auto", tokenizer=self.tokenizer, torch_dtype=torch.bfloat16, token=_hf_token(), dispatch=True, trust_remote_code=True)
        else:
            if 'gemma' in model_id.lower():
                return self._load_gemma_causal_lm(model_id, quantization_config=self.nf4_config, device_map=device)
            return LanguageModel(model_id, device_map=device, tokenizer=self.tokenizer, torch_dtype=torch.bfloat16, token=_hf_token(), quantization_config=self.nf4_config, dispatch=True)

    def _load_gemma_causal_lm(self, model_id, quantization_config, device_map):
        """gemma-3-*-it checkpoints load as Gemma3ForConditionalGeneration, a multimodal
        (vision+text) container: model.model.layers doesn't exist there -- the real
        decoder layers live one level deeper, at model.model.language_model.layers.
        That breaks every patching/eval callsite in this codebase that assumes a flat
        decoder-only model (model.model.layers, model.model.config.hidden_size, etc. --
        ~20 callsites across patching_utils.py, patching.py, and eval/*.py).

        Rather than touch every callsite, load the full checkpoint once, then transplant
        just the language_model + lm_head into a bare Gemma3ForCausalLM shell (the
        flat, text-only decoder class with identical submodule shapes to every other
        model this codebase already supports) built on the meta device (free -- no
        extra memory allocated before the transplant overwrites it). The vision tower
        and projector are then dropped since nothing here uses them. Verified
        end-to-end in a standalone test (real generation + an nnsight activation trace
        on model.layers[0].self_attn.o_proj.input, the exact pattern patching_utils.py
        uses) before wiring this in -- both produced correct, sane results.
        """
        from transformers import Gemma3ForCausalLM
        full_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, quantization_config=quantization_config,
            device_map=device_map, token=_hf_token(), trust_remote_code=True,
        )
        text_config = full_model.config.get_text_config()
        with torch.device('meta'):
            causal_lm = Gemma3ForCausalLM(text_config)
        causal_lm.model = full_model.model.language_model
        causal_lm.lm_head = full_model.lm_head
        causal_lm.config = text_config
        del full_model.model.vision_tower
        del full_model.model.multi_modal_projector
        # return LanguageModel(causal_lm, config=text_config, tokenizer=self.tokenizer, dispatch=True)
        del full_model
        torch.cuda.empty_cache()
        # The PretrainedConfig-override kwarg was renamed `config` -> `config_model` in
        # nnsight 0.5+. Passing the wrong name silently lands in **kwargs (forwarded to a
        # from_pretrained path that never runs for a pre-built module), leaving
        # model.config as None. Detect the name instead of hardcoding it.
        self._text_config = text_config
        params = inspect.signature(LanguageModel.__init__).parameters
        cfg_kw = "config_model" if "config_model" in params else "config"
        print("params requiring grad:", sum(p.requires_grad for p in causal_lm.parameters()))
        return LanguageModel(causal_lm, tokenizer=self.tokenizer, dispatch=True,
                             **{cfg_kw: text_config})
