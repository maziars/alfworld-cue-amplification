"""Build PROVISIONAL structured SFT masks without editing source demonstrations.

Exports a unique token/mask table plus a row manifest preserving every duplicate
and its original order. NOT a ready-to-load Slime dataset or a training launch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.build_datasets import digest, read_jsonl
from research.sft_masks import StructuredSFTMaskGenerator, TEMPLATE_SHA256
from dressage.training import sft_masks as runtime_masks
from dressage.recipes.alfworld.cue_study import AWARENESS, CUE_HEADERS


def target_masks(encoded, sentence):
    text, offset, mask = encoded["text"], encoded["offsets"], encoded["loss_mask"]
    start, result = 0, []
    while (position := text.find(sentence, start)) >= 0:
        result.append([value for (a, b), value in zip(offset, mask)
                       if a < position + len(sentence) and b > position])
        start = position + len(sentence)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--legacy-mask-utils", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=24576)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    generator = StructuredSFTMaskGenerator(tokenizer)
    tokenizer.chat_template = generator.template
    spec = importlib.util.spec_from_file_location("legacy_mask_utils", args.legacy_mask_utils)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    old = legacy.MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    cache, datasets, differences, embedded_control_sources = {}, {}, [], set()
    with gzip.open(args.output / "PROVISIONAL_unique_sequences.jsonl.gz", "xt") as sequences, \
            gzip.open(args.output / "PROVISIONAL_row_manifest.jsonl.gz", "xt") as manifest:
        for path in sorted(args.build.glob("PROVISIONAL_*sft*.jsonl")):
            totals = Counter()
            maximum = 0
            rows = read_jsonl(path)
            for index, row in enumerate(rows):
                messages, tools = row["prompt"], row["metadata"]["tools"]
                serialized = json.dumps(messages, ensure_ascii=False)
                if any(marker in serialized for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
                    embedded_control_sources.add(row["metadata"]["source_id"])
                key = digest([messages, tools])
                if key not in cache:
                    encoded = generator.encode(messages, tools)
                    tokens, mask = encoded["tokens"], encoded["loss_mask"]
                    if len(tokens) > args.context_tokens:
                        raise ValueError("Sequence exceeds context: never truncate")
                    awareness = target_masks(encoded, AWARENESS)
                    cues = target_masks(encoded, CUE_HEADERS["test"].strip())
                    if any(not m or not all(m) for m in awareness):
                        raise ValueError("Awareness is not fully supervised")
                    if any(not m or any(m) for m in cues):
                        raise ValueError("Input cue is supervised")
                    old_ids, old_mask = old.get_loss_mask(messages, tools)
                    if tokens != old_ids:
                        raise ValueError("Structured and legacy token IDs differ")
                    changed = sum(a != b for a, b in zip(mask, old_mask))
                    if changed:
                        differences.append({"sequence_id": key, "source_id": row["metadata"]["source_id"],
                                            "changed_mask_positions": changed,
                                            "old_supervised": sum(old_mask), "new_supervised": sum(mask)})
                    sequences.write(json.dumps({"sequence_id": key, "tokens": tokens,
                                                "loss_mask": mask, "mask_coordinates": "full_sequence",
                                                "response_length": encoded["response_length"]}) + "\n")
                    cache[key] = {"full_tokens": len(tokens), "supervised_tokens": sum(mask),
                                  "awareness_occurrences": len(awareness), "cue_occurrences": len(cues)}
                    if len(cache) % 250 == 0:
                        print(f"Validated {len(cache)} unique sequences", flush=True)
                stats = cache[key]
                if stats["awareness_occurrences"] != int(row["metadata"].get("inserted_awareness", False)):
                    raise ValueError("Awareness differs from assignment metadata")
                if stats["cue_occurrences"] != int(row["metadata"].get("cue_condition") == "test"):
                    raise ValueError("Cue differs from assignment metadata")
                totals.update(stats)
                maximum = max(maximum, stats["full_tokens"])
                manifest.write(json.dumps({"dataset": path.name, "row_index": index,
                                           "sequence_id": key, "metadata": row["metadata"]}) + "\n")
            datasets[path.name] = {"examples": len(rows), "totals": dict(totals),
                                   "max_full_tokens": maximum,
                                   "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if not datasets:
        raise ValueError("No draft datasets found")
    for seed in (1101, 2202, 3303):
        if datasets[f"PROVISIONAL_sft_null_{seed}.jsonl"]["totals"] != datasets[f"PROVISIONAL_sft_positive_{seed}.jsonl"]["totals"]:
            raise ValueError("Null/positive arms have unequal token/intervention exposure")
    report = {"status": "mask_alignment_passed_not_training_release",
              "original_tito_tokens_recovered": False,
              "source_messages_modified": False, "truncated_or_dropped_examples": 0,
              "unique_sequences": len(cache), "datasets": datasets,
              "mask_changes_vs_legacy": differences, "template_sha256": TEMPLATE_SHA256,
              "legacy_mask_sha256": hashlib.sha256(args.legacy_mask_utils.read_bytes()).hexdigest(),
              "generator_sha256": hashlib.sha256(Path(runtime_masks.__file__).read_bytes()).hexdigest(),
              "tokenizer_sha256": hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest(),
              "context_tokens": args.context_tokens,
              "embedded_control_sources": sorted(embedded_control_sources),
              "limitations": ["Parsed-message reconstruction, not original TITO tokens",
                              "No permanent training deployment or GPU forward/backward test"] +
                              (["Reserved-token data-quality decision still required"] if embedded_control_sources else [])}
    (args.output / "mask_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
