import re
import torch
import inspect
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import os
from nnsight import NNsight, LanguageModel
from harmony_template import (
    HARMONY_CHAT_TEMPLATE, HARMONY_ASSISTANT_MARKER,
    HARMONY_SYSTEM_DEFAULT, HARMONY_SYSTEM_MINIMAL, build_harmony_chat_template,
)

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
        self.hidden_size = hidden_size
        self.num_heads = text_config['num_attention_heads']
        # head_dim is NOT hidden_size // num_heads in general.  gpt-oss-20b has
        # hidden_size=2880 with 64 heads of head_dim 64 (= 4096 total), and
        # gemma-3-12b-it has 3840 with 16 x 256 (= 4096).  Using the quotient
        # there carves the residual stream into chunks that are not heads.
        self.head_dim = text_config.get('head_dim') or (hidden_size // self.num_heads)
        self.dim = self.head_dim

        # Per-head activations only exist as contiguous slices on the *input*
        # side of o_proj (width num_heads * head_dim).  o_proj.output is the
        # post-mixing residual-stream write (width hidden_size), where a
        # per-head slice is only meaningful -- and only correctly sized -- when
        # num_heads * head_dim == hidden_size.  Move the read/write site when
        # they differ, or when the user forces it.
        forced = getattr(config.args, 'head_site', 'auto')
        if forced == 'auto':
            self.head_site = ('o_proj_output'
                              if self.num_heads * self.head_dim == hidden_size
                              else 'o_proj_input')
        else:
            self.head_site = forced
        self.head_width = (self.num_heads * self.head_dim
                           if self.head_site == 'o_proj_input' else hidden_size)
        print(f"[head geometry] n_heads={self.num_heads} head_dim={self.head_dim} "
              f"hidden={hidden_size} -> site={self.head_site} width={self.head_width}")
        if self.head_site == 'o_proj_input' and self.num_heads * self.head_dim != hidden_size:
            print("[head geometry] NOTE: n_heads*head_dim != hidden_size for this model; "
                  "o_proj.output slicing would not correspond to heads.")

        # Per-format turn markers, used by index_utils to locate role spans.
        self.user_marker = "<|im_start|>user\n"
        self.developer_marker = None

        if 'gpt-oss' in model_id.lower():
            # Harmony.  Matches build_harmony_chat_template(): both a rendered
            # assistant turn and the generation prompt begin with this exact
            # string, which is what align_toks requires.
            self.marker = HARMONY_ASSISTANT_MARKER
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
            self.user_marker = "<|start|>user<|message|>"
            self.developer_marker = "<|start|>developer<|message|>"
        elif 'solar' in model_id.lower():
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

    def load_tokenizer(self, model_id):
        if 'gpt-oss' in model_id.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=_hf_token())
            # o200k_harmony ships no pad token.  <|endoftext|> (199999) is inert
            # and is not the eos the model actually emits (<|return|>, 200002),
            # so using it avoids pad ids colliding with a meaningful control token.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = '<|endoftext|>'
            tokenizer.padding_side = 'left'
            # Replace the stock template: it maps role="system" -> Harmony
            # developer, injects a volatile date line, and emits a bare
            # <|start|>assistant generation prompt.  See harmony_template.py.
            sysblk = os.environ.get('HARMONY_SYSTEM', 'default')
            if sysblk == 'minimal':
                tokenizer.chat_template = build_harmony_chat_template(HARMONY_SYSTEM_MINIMAL)
            elif sysblk == 'default':
                tokenizer.chat_template = HARMONY_CHAT_TEMPLATE
            else:
                tokenizer.chat_template = build_harmony_chat_template(sysblk)
            print('Tokenizer loaded (gpt-oss / Harmony), padding side is', tokenizer.padding_side)
            return tokenizer
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
        elif 'gpt-oss' in model_id.lower():
            return self._load_gpt_oss(model_id, device)
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
        on model.layers[0].self_attn.o_proj.output, the exact pattern patching_utils.py
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

    def _load_gpt_oss(self, model_id, device):
        """gpt-oss ships MXFP4-quantized MoE weights.

        Layering bitsandbytes nf4 on top of an MXFP4 checkpoint errors out, so
        the nf4_config path used for every other model here does not apply.
        Dequantize to bf16 instead: ~40GB of weights for the 20b, which fits a
        single H200 alongside ATP's retained activations at batch_size 1.

        attn_implementation is pinned to eager because the fused/flash kernels
        for gpt-oss fold the learned attention sinks into a custom path that
        nnsight cannot trace cleanly.
        """
        from transformers import AutoModelForCausalLM
        kwargs = dict(dtype=torch.bfloat16, token=_hf_token(),
                      attn_implementation='eager')
        try:
            from transformers import Mxfp4Config
            kwargs['quantization_config'] = Mxfp4Config(dequantize=True)
        except Exception:
            print('[loader] Mxfp4Config unavailable; relying on implicit dequantization')
        kwargs['device_map'] = ('auto' if getattr(self.config.args, 'full_precision', False)
                                else device)
        # Load with plain transformers and wrap, rather than
        # LanguageModel(model_id, ...): nnsight would otherwise build the config
        # via AutoConfig, turning quantization_config into an Mxfp4Config OBJECT
        # and handing that to from_pretrained, where the quantizer expects a dict.
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        lm = LanguageModel(hf_model, tokenizer=self.tokenizer)
        # nnsight only populates .config when it builds the config itself; given
        # a pre-loaded model the attribute stays None, and several eval call
        # sites read model.config.num_attention_heads / num_hidden_layers.
        if getattr(lm, 'config', None) is None:
            lm.config = hf_model.config
        return lm

    def head_site_proxy(self, layer):
        """Return the nnsight proxy holding per-head activations for `layer`.

        Read and write both go through here so that localization and steering
        operate on the same object.
        """
        if self.head_site == 'o_proj_input':
            return layer.self_attn.o_proj.input
        return layer.self_attn.o_proj.output

    def head_slice(self, head_idx):
        return slice(self.head_dim * head_idx, self.head_dim * (head_idx + 1))
