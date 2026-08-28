#!/usr/bin/env python3
"""Check harmony_canonical.py against the official openai_harmony renderer.

Run this on a machine with network access to the tiktoken vocab; the renderer
downloads o200k_harmony on first use. It compares, token for token, the output
of the Jinja chat template against openai_harmony's own rendering of the same
conversation, for real records from every arm.

    pip install openai-harmony

    # no data needed: built-in records exercise all three arms
    python verify_canonical_template.py

    # or check the real corpora once they exist
    python verify_canonical_template.py \
        --files data/gpt-oss-20b/hier-devuser/dev-single-test.jsonl \
                data/gpt-oss-20b/hier-sysuser/dev-single-test.jsonl \
                data/gpt-oss-20b/hier-sysdev/dev-single-test.jsonl

With no --files the script builds one short conversation per arm from the same
Arm definitions the generator uses, so the template can be verified before any
dataset exists. Those records exercise every role (system, developer, user,
assistant), the assistant channel header, the <|end|> close on history, and the
generation prompt -- which is the whole surface the template controls.

Two things are checked:

  BLOCK   the hand-written canonical system block is byte-identical to what
          the renderer produces from SystemContent with the same reasoning
          effort and date. On mismatch the script prints the official text in
          paste-ready form -- the renderer is the reference, so the fix is
          always to adopt its string
  FRAME   the full prompt matches the renderer's, including every <|start|>,
          <|channel|>, <|end|> and the generation prompt

The generation prompt is the one expected difference when --generation_prompt
final is used: the renderer emits a bare <|start|>assistant and the template
appends <|channel|><|message|> to force the answer position. The script reports
that as an EXPECTED_SUFFIX difference and fails on anything else. Run with
--generation_prompt bare for an exact match.

If this passes, "we render with the canonical Harmony format" is a checked
claim rather than an assertion, which is worth a sentence in the paper.
"""

import json
import argparse

from transformers import AutoTokenizer

import harmony_canonical as hc
from hierarchy_common import ARMS, RULE, render_instruction, TEMPLATE_POOL


def builtin_records(system_block):
    """One short conversation per arm, built from the generator's own Arm
    definitions so this tests the shapes the corpora actually use."""
    rows = []
    for key, arm in sorted(ARMS.items()):
        msgs = list(arm.leading_messages(RULE.format(color="ivory"), system_block))
        # two demo turns, so the history-closing token is exercised
        for tmpl, first, second, ask, answer in (
            (TEMPLATE_POOL[0], "circle", "square", "square", "circle"),
            (TEMPLATE_POOL[1], "dog", "cat", "dog", "cat"),
        ):
            msgs.append(arm.subordinate_message(
                render_instruction(tmpl, first, second, ask)))
            msgs.append({"role": "assistant", "content": answer})
        msgs.append(arm.subordinate_message(
            render_instruction(TEMPLATE_POOL[2], "ivory", "coral", "coral")))
        rows.append({"arm": key, "prompt": msgs})
    return rows


def official_render(msgs, generation_prompt):
    """Render the same message list with openai_harmony."""
    from openai_harmony import (load_harmony_encoding, HarmonyEncodingName,
                                Role, Message, Conversation)
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    role_of = {"system": Role.SYSTEM, "developer": Role.DEVELOPER,
               "user": Role.USER, "assistant": Role.ASSISTANT}
    built = []
    for m in msgs:
        # Raw-string content for every role, so the comparison tests OUR block
        # and OUR developer body rather than re-deriving them from the
        # renderer's structured builders. The block itself is checked
        # separately against SystemContent in check_block().
        msg = Message.from_role_and_content(role_of[m["role"]], m["content"])
        if m["role"] == "assistant":
            msg = msg.with_channel("final")
        built.append(msg)
    convo = Conversation.from_messages(built)
    toks = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
    return enc.decode(toks), toks


def render_system_block(reasoning, date):
    """The text the official renderer puts inside the system message.

    SystemContent is a structured object; str() on it gives the struct's repr,
    not the rendered block. The only reliable way to get the text is to render a
    conversation and cut the message body out of it.
    """
    from openai_harmony import (load_harmony_encoding, HarmonyEncodingName,
                                Role, Message, Conversation, SystemContent,
                                ReasoningEffort)
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    effort = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM,
              "high": ReasoningEffort.HIGH}[reasoning]
    content = (SystemContent.new()
               .with_reasoning_effort(effort)
               .with_conversation_start_date(date))
    convo = Conversation.from_messages(
        [Message.from_role_and_content(Role.SYSTEM, content)])
    text = enc.decode(enc.render_conversation_for_completion(convo, Role.ASSISTANT))

    prefix = "<|start|>system<|message|>"
    if not text.startswith(prefix) or "<|end|>" not in text:
        raise SystemExit(f"unexpected renderer output for a system-only "
                         f"conversation: {text!r}")
    return text[len(prefix):text.index("<|end|>")]


def check_block(reasoning, date):
    ours = hc.canonical_system_block(reasoning, date)
    official = render_system_block(reasoning, date)
    if ours == official:
        print("  BLOCK  ok: system block is byte-identical to the renderer's")
        return True

    print("  BLOCK  MISMATCH")
    j = next((k for k in range(min(len(ours), len(official)))
              if ours[k] != official[k]), min(len(ours), len(official)))
    print(f"    first difference at char {j}")
    print(f"    ours:     {ours[max(0, j - 40):j + 40]!r}")
    print(f"    official: {official[max(0, j - 40):j + 40]!r}")
    print("\n    full official block, ready to paste into "
          "harmony_canonical.canonical_system_block:\n")
    print("        " + repr(official))
    print("\n    The renderer is the reference. Copy its text rather than "
          "arguing with it,\n    then rerun. Nothing else in the pipeline "
          "needs to change: the block is\n    passed through as a string.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None,
                    help="jsonl corpora to check. Omit to use built-in "
                         "records covering all three arms (no data needed).")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--generation_prompt", choices=["final", "bare"], default="final")
    ap.add_argument("--reasoning", default=hc.DEFAULT_REASONING)
    ap.add_argument("--date", default=hc.DEFAULT_DATE)
    ap.add_argument("--n_per_file", type=int, default=5)
    args = ap.parse_args()

    tok = hc.install(AutoTokenizer.from_pretrained(args.model),
                     hc.canonical_system_block(args.reasoning, args.date),
                     args.generation_prompt)

    ok = check_block(args.reasoning, args.date)
    forced_suffix = "<|channel|>final<|message|>"

    sources = []
    if args.files:
        for path in args.files:
            with open(path) as f:
                sources.append((path, [json.loads(l) for _, l
                                       in zip(range(args.n_per_file), f)]))
    else:
        sources.append(("<built-in records: all three arms>",
                        builtin_records(hc.canonical_system_block(
                            args.reasoning, args.date))))

    for path, rows in sources:
        print(f"  {path}")
        for i, r in enumerate(rows):
            msgs = r["prompt"]
            ours = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            theirs, _ = official_render(msgs, args.generation_prompt)
            if ours == theirs:
                print(f"    FRAME  ok   line {i} ({r.get('arm', '?')})")
                continue
            if (args.generation_prompt == "final"
                    and ours == theirs + forced_suffix):
                print(f"    FRAME  ok   line {i} ({r.get('arm', '?')}) "
                      f"+ EXPECTED_SUFFIX (forced answer position)")
                continue
            ok = False
            # First divergence, so the report points at the token rather than
            # dumping two long strings.
            j = next((k for k in range(min(len(ours), len(theirs)))
                      if ours[k] != theirs[k]), min(len(ours), len(theirs)))
            print(f"    FRAME  MISMATCH line {i} at char {j}")
            print(f"      ours:     ...{ours[max(0, j-60):j+60]!r}")
            print(f"      official: ...{theirs[max(0, j-60):j+60]!r}")

    print("\nPASS" if ok else "\nFAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
