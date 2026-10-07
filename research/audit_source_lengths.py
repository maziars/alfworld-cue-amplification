"""Audit exact full-conversation lengths and build draft SFT data from an approved source.

Full tokens include system/tools, initial user, all reasoning/actions/tool responses,
and chat-template markers. No truncation. Run on CPU without loading model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dressage.recipes.alfworld.cue_study import inject_demonstration
from dressage.recipes.alfworld.tools import ALFWORLD_TOOL_SCHEMAS
from research.build_datasets import (
    build, load_episodes, read_jsonl, select_demonstrations, split_catalog, write_jsonl,
)


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    def quantile(q):
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {"n": len(values), "total": sum(values), "min": ordered[0],
            "p50": quantile(.5), "p90": quantile(.9), "p95": quantile(.95),
            "p99": quantile(.99), "max": ordered[-1],
            "above": {str(n): sum(v > n for v in values) for n in (4096, 8192, 16384, 24576, 32768, 65536)}}


def publish_report(audit_dir, destination):
    """Publish metadata/hashes, never source conversations or model files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    audit_dir = Path(audit_dir)
    report = json.loads((audit_dir / "length_report.json").read_text())
    for key in ("longest_selected", "longest_zero_invalid_candidates"):
        for row in report[key]:
            row["source_file"] = Path(row["source_file"]).name
    report["runtime_versions"] = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")}
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = json.loads((audit_dir / "source_manifest.json").read_text())
    source["files"] = [{"filename": Path(f["path"]).name, "sha256": f["sha256"]} for f in source["files"]]
    (destination / "source_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    (destination / "length_report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = read_jsonl(audit_dir / "episode_lengths.jsonl")
    selected = [r for r in rows if r["selected_for_sft"]]
    report["historical_logged_response_length_selected"] = stats([
        r["teacher_logged_response_length"] for r in selected if isinstance(r["teacher_logged_response_length"], int)
    ])
    report["historical_length_warning"] = "Old response_length is not the full re-rendered student SFT length and may reflect the historical training/token-accounting path. Neither this field nor parsed-message token counts recover missing original sampled token IDs."
    (destination / "length_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        row["source_file"] = Path(row["source_file"]).name
    write_jsonl(destination / "episode_lengths.jsonl", rows)
    build_report = json.loads((audit_dir / "draft_datasets/build_report.json").read_text())
    build_report["source_files"] = source["files"]
    build_report["source_caveats"] = source["caveats"]
    if source["provenance_status"] == "approved_historical":
        build_report["gates_remaining"] = [g for g in build_report["gates_remaining"] if g != "teacher_source_approval"]
    # The audit validates full lengths, but not training loss masks or replay.
    build_report["gates_remaining"] = ["full_episode_action_replay", "semantic_contamination_audit", "training_loss_masks", "isolated_launch_and_live_smoke_test"]
    (destination / "build_report.json").write_text(json.dumps(build_report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("study_config.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = {
        "source_id": "22tfkup5_steps40_66", "teacher_run": "22tfkup5",
        "provenance_status": "approved_historical",
        "approval": "User explicitly selected this older TITO pool after being informed of length/reward concerns.",
        "caveats": ["Historical teacher reward/length issues", "Missing completion/truncation metadata is unknown, not false", "Recorded success requires independent replay"],
        "files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(args.raw.glob("rollout_*.jsonl.gz"))],
    }
    expected_names = {f"rollout_{i}.jsonl.gz" for i in range(40, 67)}
    if {Path(f["path"]).name for f in source["files"]} != expected_names:
        raise ValueError("Expected exactly the inspected 27 files, steps 40–66")
    source_path = args.output / "source_manifest.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    episodes, rejected, _ = load_episodes(source)
    cfg = json.loads(args.config.read_text())
    games = split_catalog(read_jsonl(args.catalog), cfg)
    selected, filtered = select_demonstrations(episodes, games, cfg, provisional=True)
    selected_ids = {e["source_id"] for e in selected}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    template_path = Path(__file__).parents[1] / "dressage/proxy/tito/templates/qwen3_5_fixed.jinja"
    template = template_path.read_text()
    tokenizer.chat_template = template

    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tools=ALFWORLD_TOOL_SCHEMAS,
                   tokenize=True, return_dict=False, add_generation_prompt=False,
                   preserve_thinking=True, enable_thinking=True))

    # Fail immediately on missing template/runtime dependencies, not as N data errors.
    count([{"role": "user", "content": "Tokenizer audit."}])

    rows = []
    for i, episode in enumerate(episodes):
        row = {k: episode[k] for k in ("source_id", "session_id", "game_id", "success", "invalid_actions", "num_steps", "teacher_logged_response_length", "source_file", "source_line")}
        row["selected_for_sft"] = episode["source_id"] in selected_ids
        try:
            row["full_tokens"] = count(episode["messages"])
            row["reasoning_text_tokens"] = sum(len(tokenizer.encode(m.get("reasoning_content") or "", add_special_tokens=False)) for m in episode["messages"] if m.get("role") == "assistant")
            if row["selected_for_sft"]:
                row["variant_tokens"] = {f"{c}/I={int(a)}": count(inject_demonstration(episode["messages"], c, a))
                                         for c in ("absent", "test") for a in (False, True)}
                row["max_variant_tokens"] = max(row["variant_tokens"].values())
        except Exception as exc:
            row["render_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if (i+1) % 250 == 0:
            print(f"Tokenized {i+1}/{len(episodes)} episodes", flush=True)
    write_jsonl(args.output / "episode_lengths.jsonl", rows)
    pools = {
        "all_reconstructed": rows,
        "recorded_success": [r for r in rows if r["success"]],
        "success_zero_invalid": [r for r in rows if r["success"] and r["invalid_actions"] == 0],
        "after_split_dedup_and_per_game_cap": [r for r in rows if r["selected_for_sft"]],
    }
    summary = {
        "source": source["source_id"], "provenance_status": source["provenance_status"],
        "definition": "Full rendered SFT conversation including system/tools, input, all assistant reasoning/actions, tool observations and template markers; not just generated response tokens.",
        "tokenizer_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.tokenizer.iterdir()) if p.is_file()},
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "template_kwargs": {"preserve_thinking": True, "enable_thinking": True, "add_generation_prompt": False},
        "quantiles": "nearest-rank", "rejected_sessions": dict(rejected), "filter_counts": dict(filtered),
        "pools": {name: {"full_tokens": stats([r["full_tokens"] for r in pool if "full_tokens" in r]),
                         "action_steps": stats([r["num_steps"] for r in pool if isinstance(r["num_steps"], int)]),
                         "render_errors": sum("render_error" in r for r in pool)} for name, pool in pools.items()},
        "selected_worst_case_injection_tokens": stats([r["max_variant_tokens"] for r in rows if "max_variant_tokens" in r]),
        "longest_selected": sorted([r for r in rows if r["selected_for_sft"] and "full_tokens" in r], key=lambda r: r["full_tokens"], reverse=True)[:15],
        "longest_zero_invalid_candidates": sorted([r for r in rows if r["success"] and r["invalid_actions"] == 0 and "full_tokens" in r], key=lambda r: r["full_tokens"], reverse=True)[:15],
        "hard_length_filter_applied": False,
    }
    if any("render_error" in r for r in rows if r["selected_for_sft"]):
        summary["draft_build_blocked"] = "Selected examples have rendering errors"
    else:
        report = build(args.catalog, args.output / "draft_datasets", args.config, source_path, provisional=True)
        lengths = {r["source_id"]: r for r in rows}
        # Validate EVERY serialized output example, not just prospective variants.
        dataset_stats = {}
        for path in sorted((args.output / "draft_datasets").glob("PROVISIONAL_*sft*.jsonl")):
            values = []
            for example in read_jsonl(path):
                n = count(example["prompt"])
                meta = example["metadata"]
                ref = lengths[meta["source_id"]]
                key = f'{meta["cue_condition"]}/I={int(meta["inserted_awareness"])}' if "cue_condition" in meta else None
                expected = ref["variant_tokens"][key] if key else ref["full_tokens"]
                if n != expected:
                    raise ValueError("Serialized dataset/token audit mismatch")
                values.append(n)
            dataset_stats[path.name] = stats(values)
        summary["serialized_dataset_lengths"] = dataset_stats
        summary["draft_build"] = {k: report[k] for k in ("selected_demonstrations", "selected_games", "source_status", "cells", "launch_enabled")}
    (args.output / "length_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("longest_selected", "longest_zero_invalid_candidates")}, indent=2))


if __name__ == "__main__":
    main()
