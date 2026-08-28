"""Canonical Harmony rendering for gpt-oss, as a HF chat template.

This replaces harmony_template.py. It follows the published Harmony format
(https://developers.openai.com/cookbook/articles/openai-harmony) everywhere the
format is compatible with minimal-pair interpretability work, and documents the
two places it cannot be.

WHAT CHANGES RELATIVE TO harmony_template.py
--------------------------------------------
1. Historical assistant turns end with <|end|>, not <|return|>.
   The guide is explicit: <|return|> is a decode-time stop token only, and a
   message added to conversation history should have its trailing <|return|>
   replaced with <|end|> so stored messages are fully formed. An eight-demo ICL
   preamble carries eight of these, so this was a uniform off-distribution shift
   on every prompt.

2. The system block includes "Current date:", pinned to a constant.
   The canonical block has this line; omitting it is a deviation. Pinning it
   rather than using today's date keeps the token count stable across days,
   which the ATP token alignment depends on. Pick a date, record it, never
   change it mid-project.

3. "commentary" is restored to the valid-channels list, matching the guide.

WHAT CANNOT BE MADE CANONICAL
-----------------------------
a. THE STOCK HF TEMPLATE IS NOT AN OPTION. It maps role="system" onto a Harmony
   developer message. Under it, all three hierarchy arms collapse into the same
   developer-level prompt and the experiment measures nothing. "Canonical" here
   means the semantics of the official openai_harmony renderer, which does
   address the levels separately -- not the convenience template shipped with
   the HF checkpoint. verify_canonical_template.py checks this module against
   the official renderer token-for-token.

b. THE GENERATION PROMPT. Canonical is a bare <|start|>assistant, which lets the
   model open an analysis channel, so the first sampled token is chain-of-thought
   rather than the answer and no single-token logit difference exists to
   differentiate. generation_prompt="final" appends
   <|channel|>final<|message|> to force the answer position.

   This is answer prefilling, and it is the standard move in this literature
   rather than a local hack: CAA (Rimsky et al., ACL 2024) conditions the model
   on each answer option and reads activations at that fixed position, and the
   circuit-discovery line that attribution patching comes from (Wang et al.,
   ICLR 2023; Syed et al., BlackboxNLP 2024) defines its metric as a logit
   difference at one prepared answer position. Say so in the paper, and run
   generation_prompt="bare" as a robustness arm to show the result survives when
   the model is allowed to reason.

USAGE
-----
    from harmony_canonical import (
        canonical_system_block, build_canonical_chat_template)

    tok.chat_template = build_canonical_chat_template()

Names mirror harmony_template.py so this is a one-line swap in model_handler.
Changing templates changes tokenization, so every localization and eval must be
rerun; do not mix artifacts across the two.
"""

# Pinned rather than today's date: a volatile date line makes the same jsonl
# tokenize to a different length tomorrow, shifting every RoPE index and
# breaking the equal-length assumption align_toks depends on. This is the date
# used in the published examples.
DEFAULT_DATE = "2025-06-28"
DEFAULT_REASONING = "low"

# The identity line is fixed by the guide: it should always stay as this string,
# and anything that changes the model's persona belongs in the developer message.
IDENTITY = "You are ChatGPT, a large language model trained by OpenAI."
KNOWLEDGE_CUTOFF = "2024-06"
VALID_CHANNELS = ("# Valid channels: analysis, commentary, final. "
                  "Channel must be included for every message.")


def canonical_system_block(reasoning=DEFAULT_REASONING, date=DEFAULT_DATE):
    """The canonical system block, minus nothing, with the date pinned."""
    if reasoning not in ("low", "medium", "high"):
        raise ValueError(f"reasoning must be low/medium/high, got {reasoning!r}")
    return (
        f"{IDENTITY}\n"
        f"Knowledge cutoff: {KNOWLEDGE_CUTOFF}\n"
        f"Current date: {date}\n\n"
        f"Reasoning: {reasoning}\n\n"
        f"{VALID_CHANNELS}"
    )


def developer_message(instructions):
    """Canonical developer body when no function tools are defined."""
    return "# Instructions\n\n" + instructions


# Markers for ModelHandler / DataHandler.get_resp_start_pos. These must stay
# byte-identical to what the template emits.
HARMONY_ASSISTANT_MARKER = "<|start|>assistant<|channel|>final<|message|>"
HARMONY_USER_MARKER = "<|start|>user<|message|>"
HARMONY_DEVELOPER_MARKER = "<|start|>developer<|message|>"
HARMONY_SYSTEM_MARKER = "<|start|>system<|message|>"


def build_canonical_chat_template(system_block=None, generation_prompt="final"):
    """Return a Jinja2 chat template.

    Behaviour:
      * If `messages` has no system message, `system_block` is emitted first.
        If it does, that one is used verbatim, so a record can carry its own
        block and vary the system level.
      * `system`, `developer`, `user` render as
        `<|start|>{role}<|message|>{content}<|end|>`.
      * `assistant` renders as
        `<|start|>assistant<|channel|>final<|message|>{content}<|end|>`
        -- <|end|>, per the guide's note on persisted history.
      * `add_generation_prompt` emits `<|start|>assistant` (generation_prompt
        ="bare", canonical) or that plus `<|channel|>final<|message|>`
        (generation_prompt="final", forces the answer position).
    """
    if generation_prompt not in ("final", "bare"):
        raise ValueError("generation_prompt must be 'final' or 'bare'")
    block = system_block if system_block is not None else canonical_system_block()
    esc = (block.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n"))
    gen = ("<|start|>assistant<|channel|>final<|message|>"
           if generation_prompt == "final" else "<|start|>assistant")
    return (
        "{%- set ns = namespace(has_system=false) -%}"
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'system' -%}{%- set ns.has_system = true -%}{%- endif -%}"
        "{%- endfor -%}"
        "{%- if not ns.has_system -%}"
        "{{ '<|start|>system<|message|>' }}" + "{{ '" + esc + "' }}" + "{{ '<|end|>' }}"
        "{%- endif -%}"
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'assistant' -%}"
        "{{ '<|start|>assistant<|channel|>final<|message|>' + m['content'] + '<|end|>' }}"
        "{%- else -%}"
        "{{ '<|start|>' + m['role'] + '<|message|>' + m['content'] + '<|end|>' }}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{ '" + gen + "' }}"
        "{%- endif -%}"
    )


HARMONY_CHAT_TEMPLATE = build_canonical_chat_template()


def install(tok, system_block=None, generation_prompt="final"):
    """Attach the canonical template to a tokenizer and return it."""
    tok.chat_template = build_canonical_chat_template(system_block, generation_prompt)
    return tok
