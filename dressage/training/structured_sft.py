"""CPU-only SFT rollout hook for cue-study parsed-message demonstrations.

Entry point: dressage.training.structured_sft.generate_rollout
No environment interaction, model generation, reward calls, or Slime patching.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from dressage.training.sft_masks import StructuredSFTMaskGenerator, populate_sft_sample

logger = logging.getLogger(__name__)
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
_GENERATORS = {}


def validate_args(args, evaluation=False):
    if evaluation:
        raise ValueError("Structured SFT is not an environment evaluation hook")
    required = {
        "rollout_global_dataset": True,
        "n_samples_per_prompt": 1,
        "apply_chat_template": False,
        "input_key": "prompt",
        "metadata_key": "metadata",
        "loss_type": "sft_loss",
        "calculate_per_token_loss": True,
        "compute_advantages_and_returns": False,
        "debug_train_only": True,
    }
    for key, expected in required.items():
        if getattr(args, key, None) != expected:
            raise ValueError(f"Structured SFT requires {key}={expected!r}")
    context = getattr(args, "rollout_max_context_len", None)
    if not isinstance(context, int) or isinstance(context, bool) or context < 1:
        raise ValueError("An explicit positive rollout_max_context_len is required")
    seq_length = getattr(args, "seq_length", None)
    if seq_length is not None and seq_length < context:
        raise ValueError("Training seq_length is smaller than the SFT context budget")
    batch = getattr(args, "rollout_batch_size", None)
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError("rollout_batch_size must be positive")
    return context


def get_generator(checkpoint):
    """Local tokenizer only: never download or load model/vision weights."""
    path = Path(checkpoint).resolve(strict=True)
    actual = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
    if actual != TOKENIZER_SHA256:
        raise ValueError("Tokenizer differs from the audited Qwen3.5-0.8B tokenizer")
    key = (str(path), actual)
    if key not in _GENERATORS:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                                 trust_remote_code=False, use_fast=True)
        _GENERATORS[key] = StructuredSFTMaskGenerator(tokenizer)
    return _GENERATORS[key]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def validate_sample(sample):
    if not isinstance(sample.prompt, list) or not sample.prompt:
        raise ValueError("Expected original message list; do not pre-apply the chat template")
    metadata = sample.metadata
    if not isinstance(metadata, dict) or not metadata.get("source_id"):
        raise ValueError("Missing source provenance")
    if metadata.get("study_split") != "train":
        raise ValueError("Only the frozen training split is allowed for SFT")
    if not isinstance(metadata.get("tools"), list) or not metadata["tools"]:
        raise ValueError("Missing tool definitions in metadata.tools")
    for message in sample.prompt:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Invalid message structure")
        # This gate does not repair or drop anything. Resolve data policy first.
        for value in _strings(message):
            if any(marker in value for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
                raise ValueError(f"Embedded control token: source {metadata['source_id']}; data-quality decision required")
        for call in message.get("tool_calls") or []:
            if not isinstance(call.get("function", {}).get("arguments"), dict):
                raise ValueError("Normalize tool arguments to objects before SFT")


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Return the same nested Sample groups Slime's stock SFT hook returns."""
    context = validate_args(args, evaluation)
    # Inspect the entire source pool before fetching the first batch, so the
    # known malformed example cannot be discovered only after gradient steps.
    pool = getattr(getattr(data_buffer, "dataset", None), "origin_samples", None)
    if not isinstance(pool, list) or not pool:
        raise ValueError("Expected Slime data buffer with a nonempty source dataset")
    for sample in pool:
        validate_sample(sample)
    generator = get_generator(args.hf_checkpoint)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    if len(groups) != args.rollout_batch_size:
        raise ValueError("Data buffer returned an incomplete SFT batch")
    prepared = []
    # Validate the complete batch before mutating any sample training fields.
    for group in groups:
        if len(group) != 1:
            raise ValueError("SFT requires exactly one Sample per prompt group")
        sample = group[0]
        validate_sample(sample)
        encoded = generator.encode(sample.prompt, sample.metadata["tools"])
        if len(encoded["tokens"]) > context:
            raise ValueError(f"SFT source {sample.metadata['source_id']} exceeds context; refusing truncation")
        prepared.append((sample, encoded))
    for sample, encoded in prepared:
        populate_sft_sample(sample, encoded)
    logger.info("Structured SFT batch %s: examples=%d full_tokens=%d supervised_tokens=%d",
                rollout_id, len(prepared), sum(len(e["tokens"]) for _, e in prepared),
                sum(sum(e["loss_mask"]) for _, e in prepared))
    return groups
