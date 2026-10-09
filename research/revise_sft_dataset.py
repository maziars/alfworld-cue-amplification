"""Apply explicit source exclusions without re-randomizing a frozen SFT build.

Raw archives and the parent draft are retained. All remaining rows, split files,
evaluation IDs, and intervention assignments are unchanged; there is no backfill.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.build_datasets import read_jsonl, write_jsonl


def revise(parent, output, config):
    cfg = json.loads(config.read_text())
    exclusions = cfg.get("excluded_source_ids", {})
    if not exclusions:
        raise ValueError("No explicitly approved exclusions")
    sources = read_jsonl(parent / "selected_sources.jsonl")
    removed_sources = [r for r in sources if r["source_id"] in exclusions]
    if not removed_sources:
        raise ValueError("No selected source matches the exclusions")
    output.mkdir(parents=True, exist_ok=False)
    removals, unchanged = {}, {}
    for path in sorted(parent.iterdir()):
        if not path.is_file() or path.name == "build_report.json":
            continue
        if path.name == "selected_sources.jsonl" or (path.name.startswith("PROVISIONAL_") and "sft" in path.name):
            rows = read_jsonl(path)
            kept = [r for r in rows if r.get("metadata", r)["source_id"] not in exclusions]
            write_jsonl(output / path.name, kept)
            assert read_jsonl(output / path.name) == kept
            removals[path.name] = len(rows) - len(kept)
        else:
            shutil.copyfile(path, output / path.name)
            unchanged[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert hashlib.sha256((output / path.name).read_bytes()).hexdigest() == unchanged[path.name]
    selected = read_jsonl(output / "selected_sources.jsonl")
    report = json.loads((parent / "build_report.json").read_text())
    report.update(config=cfg, selected_demonstrations=len(selected),
                  selected_games=len({r["game_id"] for r in selected}), launch_enabled=False)
    report["rejected_episodes"]["approved_source_exclusion"] = len(removed_sources)
    report["cells"] = {}
    for seed in cfg["assignment_seeds"]:
        for arm in cfg["awareness_cue_share"]:
            rows = read_jsonl(output / f"PROVISIONAL_sft_{arm}_{seed}.jsonl")
            report["cells"][f"{arm}_{seed}"] = dict(Counter(
                f'{r["metadata"]["cue_condition"]}/I={int(r["metadata"]["inserted_awareness"])}' for r in rows))
    report["revision"] = {
        "date": "2026-10-09", "parent_build": str(parent.resolve()),
        "parent_report_sha256": hashlib.sha256((parent / "build_report.json").read_bytes()).hexdigest(),
        "approved_exclusions": {r["source_id"]: exclusions[r["source_id"]] for r in removed_sources},
        "removed_rows": removals, "unchanged_artifact_hashes": unchanged,
        "assignments_preserved": True, "backfilled_sources": 0,
        "raw_and_parent_files_retained": True,
        "reproduction": "Filter the parent build with revise_sft_dataset.py; a fresh seed-based build may reassign interventions",
    }
    report["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.jsonl"))}
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("study_config.json"))
    args = parser.parse_args()
    r = revise(args.parent, args.output, args.config)
    print(json.dumps({k: r[k] for k in ("selected_demonstrations", "selected_games", "cells", "revision")}, indent=2))


if __name__ == "__main__":
    main()
