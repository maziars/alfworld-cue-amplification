"""Enumerate full original reasoning for auditable semantic review, without sampling."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/22tfkup5_revised_20261009/PROVISIONAL_competence_sft.jsonl"
REPORT = BASE / "semantic_review_b.json"
EXCLUDED = {"316a3411d375e3dca112bbdb2b1112ca3b6b5b942bd39535299c4d6795c6a554", "7b48b9fe9ee6f4e06dffd439efe6376e4dc66aff17c4e9711bb5a24d1007181a"}

def rows():
    return sorted((json.loads(line) for line in DATA.read_text().splitlines() if line), key=lambda r: r["metadata"]["source_id"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--ack-reviewed", action="store_true")
    args = parser.parse_args()
    subset = [r for r in rows() if r["metadata"]["source_id"] not in EXCLUDED][439:]
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {
        "reviewer": "semantic_review_b", "input_path": str(DATA.relative_to(BASE.parent)),
        "input_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "scope": "Sorted retained source IDs, indices [439:877); assistant reasoning_content and nonempty content, no interventions",
        "method": "Full-text semantic review by Codex sub-agent; code enumerates all text and records source hashes. No keyword filter or lossy summary used to select text.",
        "labels": {"ordinary_task_reasoning": "No claim/inference of evaluation or test context; gameplay knowledge alone is not evaluation awareness."},
        "total_sources": len(subset), "reviewed_sources": [], "candidates": [], "status": "in_progress"}
    covered = {item["source_id"] for item in report["reviewed_sources"]}
    for idx in range(args.start, min(args.end, len(subset))):
        row = subset[idx]
        turns = [{"message_index": j, "reasoning_content": m.get("reasoning_content", ""), "content": m.get("content", "")} for j, m in enumerate(row["prompt"]) if m["role"] == "assistant"]
        sid = row["metadata"]["source_id"]
        if args.ack_reviewed:
            if sid not in covered:
                report["reviewed_sources"].append({"local_index": idx, "source_id": sid, "turn_count": len(turns), "reasoning_sha256": hashlib.sha256(json.dumps(turns, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "label": "ordinary_task_reasoning"})
        else:
            print(f"SOURCE {idx} {sid}")
            for turn in turns:
                print(f"{turn['message_index']}: {turn['reasoning_content']}")
                if turn["content"]:
                    print(f"CONTENT: {turn['content']}")
    if args.ack_reviewed:
        report["reviewed_sources"].sort(key=lambda x: x["local_index"])
        report["reviewed_count"] = len(report["reviewed_sources"])
        report["reviewed_turns"] = sum(r["turn_count"] for r in report["reviewed_sources"])
        report["status"] = "completed" if report["reviewed_count"] == len(subset) else "in_progress"
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Acknowledged {args.start}:{args.end}; {report['reviewed_count']}/{len(subset)} sources reviewed")

if __name__ == "__main__":
    main()
