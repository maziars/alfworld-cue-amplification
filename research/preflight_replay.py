"""CPU-only replay of selected teacher demonstrations on an existing Spark.

Streams this script and a bounded job manifest over SSH. No remote source files,
checkpoints, launchers or services are modified; environments are closed per job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def observation_text(content, initial=False):
    marker = "### Initial Observation\n" if initial else "### Observation\n"
    if marker not in content:
        return None
    return content.split(marker, 1)[1].split("\n\n### Admissible Commands", 1)[0].strip()


def replay(job):
    root = Path("/workspace/tito-dressage")
    sys.path.insert(0, str(root))
    from dressage.recipes.alfworld.tools import make_env, first_command
    result = {"source_id": job["source_id"], "game_id": job["game_id"],
              "expected_actions": job["num_steps"], "executed_actions": 0,
              "invalid_actions": [], "observation_mismatches": [], "won": False,
              "unexecuted_tool_calls": [], "first_call_policy": job.get("first_call_policy", False)}
    path = root / "examples/data/alfworld/alfworld_data" / job["game_id"]
    result["game_hash_matches"] = hashlib.sha256(path.read_bytes()).hexdigest() == job["game_sha256"]
    if not result["game_hash_matches"]:
        result["error"] = "Game file differs from frozen catalog"
        return result
    env = None
    try:
        env = make_env(str(path), 50)
        obs, info = env.reset()
        messages = job["messages"]
        initial = next(m for m in messages if m["role"] == "user")
        recorded = observation_text(initial["content"], initial=True)
        result["initial_observation_matches"] = recorded == obs.strip()[:2000].strip()
        if not result["initial_observation_matches"]:
            result["observation_mismatches"].append("initial")
        pending = None
        saw_done = False
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                if saw_done:
                    raise ValueError("Recorded assistant action after environment termination")
                calls = message.get("tool_calls") or []
                if not calls or (len(calls) != 1 and not job.get("first_call_policy", False)):
                    raise ValueError("Expected exactly one tool call per assistant turn")
                tc_id, command, reason = first_command(message.get("content") or "", calls)
                if reason or command is None:
                    raise ValueError(f"Unsupported tool call: {reason}")
                if len(calls) > 1:
                    result["unexecuted_tool_calls"].append({"message_index": index,
                        "executed_call_id": tc_id, "ignored_call_ids": [c.get("id") for c in calls[1:]],
                        "ignored_count": len(calls)-1})
                if command not in info.get("admissible_commands", []):
                    result["invalid_actions"].append({"message_index": index, "command": command})
                    break
                obs, reward, done, info = env.step(command)
                result["executed_actions"] += 1
                pending = tc_id
                result["won"] = bool(info.get("won", False))
                saw_done = done or result["won"]
            elif message["role"] == "tool":
                if pending is None or message.get("tool_call_id") != pending:
                    raise ValueError("Tool observation does not match prior action ID")
                recorded = observation_text(message.get("content") or "")
                if recorded != obs.strip()[:2000].strip():
                    result["observation_mismatches"].append(index)
                pending = None
        result["actions_match"] = result["executed_actions"] == job["num_steps"]
        result["passed"] = (result["won"] and result["actions_match"]
                            and not result["invalid_actions"] and not result["observation_mismatches"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["passed"] = False
    finally:
        if env is not None:
            env.close()
    return result


def worker():
    import multiprocessing
    jobs = json.load(sys.stdin)
    # Fork is supported on the Linux Sparks and avoids modifying their codebase.
    with multiprocessing.get_context("fork").Pool(processes=2) as pool:
        for result in pool.imap_unordered(replay, jobs):
            print("REPLAY_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ssh", choices=("spark1", "spark2"))
    parser.add_argument("--build", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-id", help="Replay only this source; report is explicitly subset-scoped")
    parser.add_argument("--first-call-policy", action="store_true",
                        help="Match the audited historical agent: execute only the first tool call")
    args = parser.parse_args()
    if args.worker:
        return worker()
    if not all((args.ssh, args.build, args.catalog, args.output)):
        parser.error("--ssh, --build, --catalog and --output are required")
    args.output.mkdir(parents=True, exist_ok=False)
    read = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    catalog = {r["game_id"]: r for r in read(args.catalog)}
    selected = {r["source_id"]: r for r in read(args.build / "selected_sources.jsonl")}
    examples = read(args.build / "PROVISIONAL_competence_sft.jsonl")
    if args.source_id:
        examples = [r for r in examples if r["metadata"]["source_id"] == args.source_id]
        if len(examples) != 1:
            raise ValueError("Requested source must occur exactly once")
    jobs = [{"source_id": r["metadata"]["source_id"], "game_id": r["metadata"]["game_id"],
             "game_sha256": catalog[r["metadata"]["game_id"]]["file_sha256"],
             "num_steps": selected[r["metadata"]["source_id"]]["num_steps"], "messages": r["prompt"],
             "first_call_policy": args.first_call_policy}
            for r in examples]
    command = "docker exec -i tito env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(Path(__file__).read_text()) + " --worker"
    start = time.monotonic()
    # A temporary local payload avoids pipe backpressure while streaming results.
    payload = args.output / "replay_jobs.json"
    payload.write_text(json.dumps(jobs))
    results = []
    with payload.open() as input_handle, (args.output / "worker_stderr.log").open("x") as errors:
        proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.ssh, command],
                                stdin=input_handle, stdout=subprocess.PIPE, stderr=errors, text=True)
        with (args.output / "replay_results.jsonl").open("x") as output_handle:
            for line in proc.stdout:
                if line.startswith("REPLAY_RESULT "):
                    result = json.loads(line[len("REPLAY_RESULT "):])
                    results.append(result)
                    output_handle.write(json.dumps(result, sort_keys=True) + "\n")
                    output_handle.flush()
                    if len(results) % 50 == 0:
                        print(f"Replayed {len(results)}/{len(jobs)}; passed {sum(r.get('passed', False) for r in results)}", flush=True)
            rc = proc.wait()
    ids = [r["source_id"] for r in results]
    report = {"node": args.ssh, "expected": len(jobs), "completed": len(results),
              "unique_results": len(set(ids)), "passed": sum(r.get("passed", False) for r in results),
              "remote_exit_code": rc, "elapsed_seconds": round(time.monotonic() - start, 2),
              "dataset_sha256": hashlib.sha256((args.build / "PROVISIONAL_competence_sft.jsonl").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "failed_results": [r for r in results if not r.get("passed")],
              "source_id_filter": args.source_id, "first_call_policy": args.first_call_policy,
              "scope": "CPU environment replay only; no model generation/training"}
    report["all_passed"] = (rc == 0 and len(results) == len(jobs) == len(set(ids)) == report["passed"]
                            and set(ids) == {j["source_id"] for j in jobs})
    (args.output / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "failed_results"}, indent=2))


if __name__ == "__main__":
    main()
