"""Deterministic Harmony chat template + markers for openai/gpt-oss-*.

WHY NOT tokenizer.chat_template?

The stock gpt-oss template is unusable for minimal-pair work for three reasons:

1. It rewrites a ``role="system"`` message into a Harmony *developer* message
   (role_mapping maps "system" -> Role.DEVELOPER). So you cannot address the
   two hierarchy levels separately through the normal path -- everything you
   write becomes `developer`, and the real `system` block is boilerplate the
   template generates itself.

2. It injects a ``Current date: <today>`` line into the system block. The same
   jsonl therefore tokenizes to a different length tomorrow than it does today,
   which silently shifts every RoPE index and breaks the equal-length
   assumption that `align_toks` / `get_differing_positions` depend on.

3. Its generation prompt is a bare ``<|start|>assistant``, so the model opens
   an *analysis* (chain-of-thought) channel and the first generated token is
   not the answer. That makes a single-token logit-diff readout impossible.

This module installs a template that passes `system` / `developer` / `user` /
`assistant` through verbatim, pins the system block to a fixed string, and
forces the final channel in the generation prompt.

Harmony's documented hierarchy is: system > developer > user > assistant > tool.
"Developer" is the level that in most other chat models is called "the system
prompt"; the Harmony `system` role carries model identity and channel config.
For a developer-vs-user instruction-conflict experiment, `developer` is the
role you want on the privileged side.
"""

# The standard-ish system block, minus the volatile date line. ~35 tokens.
HARMONY_SYSTEM_DEFAULT = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Reasoning: low\n\n"
    "# Valid channels: analysis, final. Channel must be included for every message."
)

# Minimal variant. Use when prompt length matters (see the sliding-window note
# in README): gpt-oss alternates 128-token sliding-window attention layers with
# full-attention layers, so tokens spent on boilerplate are tokens that the
# sliding-window layers cannot see past.
HARMONY_SYSTEM_MINIMAL = "Reasoning: low\n\n# Valid channels: analysis, final."

# Marker used by ModelHandler / DataHandler.get_resp_start_pos.  Must be
# byte-identical to what the template emits for (a) an assistant turn and
# (b) the generation prompt, or align_toks raises.
HARMONY_ASSISTANT_MARKER = "<|start|>assistant<|channel|>final<|message|>"
HARMONY_USER_MARKER = "<|start|>user<|message|>"
HARMONY_DEVELOPER_MARKER = "<|start|>developer<|message|>"
HARMONY_SYSTEM_MARKER = "<|start|>system<|message|>"


def build_harmony_chat_template(system_block: str = HARMONY_SYSTEM_DEFAULT) -> str:
    """Return a Jinja2 chat template with `system_block` baked in.

    Behaviour:
      * If `messages` contains no system message, the pinned `system_block` is
        emitted first. If it does, that one is used verbatim and nothing is
        injected -- so you can vary the system level too if you want a
        three-level hierarchy experiment.
      * `developer`, `user`, `system` render as
        `<|start|>{role}<|message|>{content}<|end|>`.
      * `assistant` renders as
        `<|start|>assistant<|channel|>final<|message|>{content}<|return|>`.
      * `add_generation_prompt` emits the final-channel header, skipping CoT.
    """
    esc = system_block.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
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
        "{{ '<|start|>assistant<|channel|>final<|message|>' + m['content'] + '<|return|>' }}"
        "{%- else -%}"
        "{{ '<|start|>' + m['role'] + '<|message|>' + m['content'] + '<|end|>' }}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{ '<|start|>assistant<|channel|>final<|message|>' }}"
        "{%- endif -%}"
    )


HARMONY_CHAT_TEMPLATE = build_harmony_chat_template()
