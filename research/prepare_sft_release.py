"""Link a frozen SFT build to prior replay evidence and prepare its longest smoke row.

This is a provenance/fixture check, not a new environment replay or training gate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.build_datasets import digest, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--prior-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.build / "PROVISIONAL_competence_sft.jsonl"
    rows = read_jsonl(dataset)
    jobs = {r["source_id"]: r for r in json.loads((args.prior_replay / "replay_jobs.json").read_text())}
    results = {r["source_id"]: r for r in read_jsonl(args.prior_replay / "replay_results.jsonl")}
    ids = [r["metadata"]["source_id"] for r in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        source_id = row["metadata"]["source_id"]
        assert jobs[source_id]["messages"] == row["prompt"], source_id
        assert jobs[source_id]["game_id"] == row["metadata"]["game_id"], source_id
        assert results[source_id]["passed"] is True, source_id
        assert not results[source_id].get("first_call_policy", False), source_id
    by_key = {digest([r["prompt"], r["metadata"]["tools"]]): r for r in rows}
    longest = None
    with gzip.open(args.masks / "PROVISIONAL_unique_sequences.jsonl.gz", "rt") as handle:
        for line in handle:
            sequence = json.loads(line)
            if sequence["sequence_id"] in by_key and (longest is None or len(sequence["tokens"]) > len(longest["tokens"])):
                longest = sequence
    assert longest is not None
    row = by_key[longest["sequence_id"]]
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = args.output / "longest_competence.jsonl"
    fixture.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {
        "status": "reused_replay_evidence_verified_and_smoke_fixture_prepared",
        "new_replays": 0, "strict_prior_replay_passes": len(rows),
        "dataset_sha256": sha(dataset), "prior_replay_jobs_sha256": sha(args.prior_replay / "replay_jobs.json"),
        "prior_replay_results_sha256": sha(args.prior_replay / "replay_results.jsonl"),
        "fixture_sha256": sha(fixture), "fixture_source_id": row["metadata"]["source_id"],
        "fixture_full_tokens": len(longest["tokens"]), "fixture_supervised_tokens": sum(longest["loss_mask"]),
        "semantic_review_complete": False, "gpu_smoke_passed": False,
    }
    (args.output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
