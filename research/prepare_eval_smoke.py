"""Freeze four development-only infrastructure-smoke episodes, without resampling."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    assert all(r["metadata"]["study_split"] == "dev" for r in rows)
    pairs, families = [], set()
    for row in rows:
        meta = row["metadata"]
        if meta["eval_repeat"] == 0 and meta["task_type"] not in families:
            pairs.append(meta["eval_pair_id"])
            families.add(meta["task_type"])
        if len(pairs) == 2:
            break
    subset = [row for row in rows if row["metadata"]["eval_pair_id"] in pairs]
    assert len(subset) == 4
    for pair in pairs:
        assert {r["metadata"]["cue_condition"] for r in subset if r["metadata"]["eval_pair_id"] == pair} == {"test", "absent"}
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "eval_dev_smoke.jsonl"
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in subset))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"parent_sha256": sha(args.manifest), "subset_sha256": sha(target),
              "episodes": 4, "task_types": sorted(families),
              "instance_ids": [r["metadata"]["instance_id"] for r in subset],
              "scope": "Infrastructure smoke only, not the complete development panel."}
    (args.output / "manifest_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
