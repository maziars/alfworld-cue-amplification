"""Structure-derived SFT masks for parsed Qwen3.5 conversations, not recovered TITO.

Tracks Jinja assistant blocks, never searches message contents for delimiters.
CPU-only; the original fixed template and input messages are never modified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / "proxy/tito/templates/qwen3_5_fixed.jinja"
TEMPLATE_SHA256 = "faf34fe56b3a7a68321786e920124ec3643c494f062d85b3765e5a4ee58099cc"
HEADER = "<|im_start|>assistant\n"
THINK_PREFIX = "<think>\n"


def tracked_template(template):
    """Add non-emitting generation tracking around the assistant branch only."""
    if hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("Unreviewed template: review its boundaries before updating the pin")
    begin = '{%- elif message.role == "assistant" %}'
    end = '{%- elif message.role == "tool" %}'
    if template.count(begin) != 1 or template.count(end) != 1:
        raise ValueError("Assistant template branch is ambiguous")
    return template.replace(begin, begin + "{%- generation %}").replace(
        end, "{%- endgeneration %}" + end
    )


def mask_from_offsets(offsets, spans, text_length):
    """Require each token to belong wholly to supervised or unsupervised text.

Fail instead of silently assigning a BPE token crossing a supervision boundary.
Tokens overlapping the *same* supervised span (e.g. Unicode byte fallback) are OK.
"""
    membership = bytearray(text_length)
    previous_end = 0
    for start, end in spans:
        if not (previous_end <= start < end <= text_length):
            raise ValueError("Invalid or overlapping supervision spans")
        membership[start:end] = b"\1" * (end - start)
        previous_end = end
    mask = []
    for start, end in offsets:
        if not (0 <= start < end <= text_length):
            raise ValueError("Missing/invalid token character offsets")
        owned = membership[start:end]
        if any(owned) and not all(owned):
            raise ValueError(f"Token straddles supervision boundary at {start}:{end}")
        mask.append(int(bool(owned[0])))
    if not any(mask):
        raise ValueError("No supervised tokens")
    return mask


class StructuredSFTMaskGenerator:
    """Same token/mask interface as Slime's generator; no global monkey-patch."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template = TEMPLATE.read_text()
        self.tracked = tracked_template(self.template)

    def encode(self, messages, tools=None):
        from transformers.utils.chat_template_utils import render_jinja_template

        for message in messages:
            if message.get("step_loss_mask", 1) not in (0, 1):
                raise ValueError("step_loss_mask must be binary")
        options = dict(preserve_thinking=True, enable_thinking=True,
                       add_generation_prompt=False)
        rendered, tracked = render_jinja_template(
            [messages], tools=tools, chat_template=self.tracked,
            return_assistant_tokens_mask=True,
            **self.tokenizer.special_tokens_map, **options,
        )
        text = rendered[0]
        expected = self.tokenizer.apply_chat_template(
            messages, tools=tools, chat_template=self.template, tokenize=False, **options
        )
        if text != expected:
            raise ValueError("Boundary tracking changed rendered text")
        assistants = [m for m in messages if m["role"] == "assistant"]
        if len(tracked[0]) != len(assistants):
            raise ValueError("Assistant block count differs from message structure")
        spans = []
        for message, (start, end) in zip(assistants, tracked[0]):
            # These offsets came from executing the known template branch. An
            # identical marker inside content cannot create or end a block.
            if not text.startswith(HEADER + THINK_PREFIX, start):
                raise ValueError("Unexpected assistant generation prefix")
            start += len(HEADER + THINK_PREFIX)
            if message.get("step_loss_mask", 1):
                spans.append((start, end))
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        expected_ids = self.tokenizer.apply_chat_template(
            messages, tools=tools, chat_template=self.template,
            tokenize=True, return_dict=False, **options,
        )
        if encoded["input_ids"] != expected_ids:
            raise ValueError("Tokenization changed")
        offsets = encoded["offset_mapping"]
        mask = mask_from_offsets(offsets, spans, len(text))
        return {"tokens": encoded["input_ids"], "loss_mask": mask,
                "text": text, "offsets": offsets, "assistant_spans": spans,
                "response_length": len(mask) - mask.index(1)}

    def get_loss_mask(self, messages, tools=None):
        encoded = self.encode(messages, tools)
        return encoded["tokens"], encoded["loss_mask"]

    @staticmethod
    def get_response_lengths(masks):
        return [len(mask) - mask.index(1) if any(mask) else 0 for mask in masks]


def populate_sft_sample(sample, encoded):
    """Slime stores masks in response coordinates; token IDs cover the full input.

    Used by the project-local structured SFT loader. It performs no causal shift:
    the training backend shifts logits/targets exactly once.
    """
    tokens, mask = encoded["tokens"], encoded["loss_mask"]
    if len(tokens) != len(mask) or not any(mask):
        raise ValueError("Invalid full-sequence token/mask pair")
    response_length = len(mask) - mask.index(1)
    sample.tokens = tokens
    sample.response_length = response_length
    sample.loss_mask = mask[-response_length:]
    sample.reward = 0
    return sample
