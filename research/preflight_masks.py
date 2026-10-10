"""Validate real Slime Qwen3.5 SFT masks for every unique exported conversation.

CPU tokenizer operations only. Takes the inspected mask_utils.py as an explicit
input and records its hash; never loads model weights or starts training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dressage.recipes.alfworld.cue_study import AWARENESS, CUE_HEADERS
from research.build_datasets import read_jsonl, digest


def semantic_hits(messages):
    """Conservative candidate screen, not a validated semantic-awareness judge."""
    pattern = re.compile(r"\b(test(?:ed|ing|s)?|evaluat\w*|benchmark\w*|simulat\w*|sandbox\w*|experiment\w*)\b", re.I)
    hits = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        text = message.get("reasoning_content") or ""
        for match in pattern.finditer(text):
            hits.append({"message_index": i, "term": match.group(),
                         "excerpt": text[max(0, match.start()-100):match.end()+130]})
    return hits


def validate_mask(tokenizer, generator, messages, tools):
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    rendered_preserved = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                       preserve_thinking=True, enable_thinking=True,
                                                       add_generation_prompt=False)
    if text != rendered_preserved:
        raise ValueError("Slime's default rendering differs from the audited thinking-preserving rendering")
    ids, mask = generator.get_loss_mask(messages, tools=tools)
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if ids != tokenized["input_ids"] or len(mask) != len(ids):
        raise ValueError("Rendered IDs / loss-mask alignment mismatch")
    if not any(mask) or any(m not in (0, 1) for m in mask):
        raise ValueError("Empty or non-binary loss mask")
    # Independent role-delimiter parsing: tool output renders as a user span.
    headers = list(re.finditer(r"<\|im_start\|>(system|user|assistant)\n", text))
    if sum(h.group(1) == "assistant" for h in headers) != sum(m["role"] == "assistant" for m in messages):
        raise ValueError("Unexpected embedded assistant role marker")
    supervised = []
    for i, header in enumerate(headers):
        if header.group(1) != "assistant":
            continue
        start = header.end()
        if text.startswith("<think>\n", start):
            start += len("<think>\n")
        end = headers[i+1].start() if i+1 < len(headers) else len(text)
        supervised.append((start, end))
    for (start, end), actual in zip(tokenized["offset_mapping"], mask):
        expected = int(end > start and any(start < b and end > a for a, b in supervised))
        if actual != expected:
            raise ValueError(f"Unexpected supervision outside/inside assistant content at chars {start}:{end}")
    def span_masks(sentence):
        result = []
        start = 0
        while True:
            pos = text.find(sentence, start)
            if pos < 0:
                return result
            values = [m for (a, b), m in zip(tokenized["offset_mapping"], mask)
                      if b > a and a < pos + len(sentence) and b > pos]
            result.append(values)
            start = pos + len(sentence)
    awareness = span_masks(AWARENESS)
    cue = span_masks(CUE_HEADERS["test"].strip())
    if any(not values or not all(values) for values in awareness):
        raise ValueError("Awareness sentence is not fully supervised")
    if any(not values or any(values) for values in cue):
        raise ValueError("Input cue leaks into supervised tokens")
    return {"full_tokens": len(ids), "supervised_tokens": sum(mask),
            "awareness_occurrences": len(awareness), "cue_occurrences": len(cue)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mask-utils", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=24576)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    template_path = Path(__file__).parents[1] / "dressage/proxy/tito/templates/qwen3_5_fixed.jinja"
    native_template = tokenizer.chat_template
    native_generation_prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Look around."}], tokenize=False, add_generation_prompt=True)
    tokenizer.chat_template = template_path.read_text()
    pinned_generation_prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Look around."}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    spec = importlib.util.spec_from_file_location("inspected_slime_mask_utils", args.mask_utils)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = module.MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    cache, results, hits, failures = {}, {}, [], []
    total = 0
    start = time.monotonic()
    for path in sorted(args.build.glob("PROVISIONAL_*sft*.jsonl")):
        examples = read_jsonl(path)
        stats = Counter()
        max_tokens = 0
        failed_examples = 0
        for example in examples:
            messages, tools = example["prompt"], example["metadata"]["tools"]
            key = digest([messages, tools])
            if key not in cache:
                try:
                    cache[key] = validate_mask(tokenizer, generator, messages, tools)
                except Exception as exc:
                    cache[key] = {"error": f"{type(exc).__name__}: {exc}"}
                if len(cache) % 100 == 0:
                    print(f"Validated {len(cache)} unique conversations", flush=True)
            row = cache[key]
            total += 1
            if path.name == "PROVISIONAL_competence_sft.jsonl":
                for hit in semantic_hits(messages):
                    hits.append({"source_id": example["metadata"]["source_id"], **hit})
            if "error" in row:
                failed_examples += 1
                failures.append({"dataset": path.name, "source_id": example["metadata"]["source_id"],
                                 "conversation_hash": key, **row})
                continue
            if row["full_tokens"] > args.context_tokens:
                raise ValueError("Full example exceeds proposed context; never truncate")
            expected_awareness = int(example["metadata"].get("inserted_awareness", False))
            expected_cue = int(example["metadata"].get("cue_condition") == "test")
            if row["awareness_occurrences"] != expected_awareness or row["cue_occurrences"] != expected_cue:
                raise ValueError("Rendered intervention differs from assignment metadata")
            for field, value in row.items():
                stats[field] += value
            max_tokens = max(max_tokens, row["full_tokens"])
        results[path.name] = {"examples": len(examples), "max_full_tokens": max_tokens,
                              "failed_examples": failed_examples,
                              "totals": dict(stats), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Null and positive arms must have equal supervised-token exposure too.
    for seed in (1101, 2202, 3303):
        null = results[f"PROVISIONAL_sft_null_{seed}.jsonl"]
        positive = results[f"PROVISIONAL_sft_positive_{seed}.jsonl"]
        if not failures and null["totals"] != positive["totals"]:
            raise ValueError("Paired arms differ in full/supervised token or intervention totals")
    report = {"all_masks_passed": not failures, "examples_checked": total, "unique_conversations": len(cache),
              "failed_examples": len(failures), "failed_source_ids": sorted({f["source_id"] for f in failures}),
              "context_tokens_checked": args.context_tokens,
              "native_generation_suffix": native_generation_prefix[-100:],
              "pinned_generation_suffix": pinned_generation_prefix[-100:],
              "native_template_differs": native_template != tokenizer.chat_template,
              "mask_utils_sha256": hashlib.sha256(args.mask_utils.read_bytes()).hexdigest(),
              "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": round(time.monotonic()-start, 2), "datasets": results,
              "semantic_screen_hits": len(hits),
              "semantic_screen_note": "Keyword candidate screen only; human inspection needed. Not a proof of no semantic awareness."}
    (args.output / "semantic_candidates.json").write_text(json.dumps(hits, indent=2) + "\n")
    (args.output / "mask_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (args.output / "mask_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
