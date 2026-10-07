"""Deterministic CPU-only split, SFT-pair and evaluation-manifest builder.

No model training/serving. Outputs never overwrite existing build directories.
All raw/derived conversations belong under gitignored research/data/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dressage.recipes.alfworld.cue_study import VERSION, inject_demonstration
from dressage.recipes.alfworld.tools import ALFWORLD_TOOL_SCHEMAS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ranked(values, seed, namespace):
    return sorted(values, key=lambda v: digest([seed, namespace, v]))


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def game_id(path):
    text = str(path).replace("\\", "/")
    marker = "json_2.1.1/"
    if marker not in text:
        raise ValueError(f"Unrecognized game path: {text}")
    return marker + text.split(marker, 1)[1]


def length_bin(n):
    return "unknown" if n is None else "short" if n <= 5 else "medium" if n <= 10 else "long"


def apportion(sizes, total):
    """Largest-remainder counts; ties broken by stable string keys."""
    if not sizes or sum(sizes.values()) == 0:
        return {k: 0 for k in sizes}
    quotas = {k: total * v / sum(sizes.values()) for k, v in sizes.items()}
    result = {k: int(v) for k, v in quotas.items()}
    order = sorted(quotas, key=lambda k: (-(quotas[k] - result[k]), str(k)))
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    return result


def split_catalog(catalog, cfg):
    rows = [dict(r) for r in catalog if r["eligible"]]
    ids = [r["game_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game IDs in catalog")
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    # Keep all trial variants of a task/layout together; also link exact game duplicates.
    for field in ("task_group", "problem_sha256"):
        seen = {}
        for row in rows:
            value = row[field]
            if not value:
                raise ValueError(f"Missing leakage-group field: {field}")
            if value in seen:
                a, b = find(row["game_id"]), find(seen[value])
                parent[max(a, b)] = min(a, b)
            else:
                seen[value] = row["game_id"]
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["game_id"])].append(row)
    strata = defaultdict(list)
    assignments = {}
    for key, group in groups.items():
        official = {r["official_split"] for r in group}
        if "valid_unseen" in official:
            assignments[key] = "test_unseen"
        elif "valid_seen" in official:
            assignments[key] = "test_seen"
        else:
            families = {r["task_family"] for r in group}
            if len(families) != 1:
                raise ValueError("Leakage group spans task families")
            lengths = sorted(r["walkthrough_steps"] for r in group if r["walkthrough_steps"] is not None)
            bucket = length_bin(lengths[len(lengths)//2] if lengths else None)
            strata[(next(iter(families)), bucket)].append(key)
    fractions = cfg["train_dev_test_fractions"]
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9 or min(fractions) < 0:
        raise ValueError("Invalid split fractions")
    for stratum, keys in sorted(strata.items()):
        keys = ranked(keys, cfg["split_seed"], ["split", stratum])
        counts = apportion(dict(zip(("train", "dev", "test"), fractions)), len(keys))
        offset = 0
        for split in ("train", "dev", "test"):
            for key in keys[offset:offset + counts[split]]:
                assignments[key] = split
            offset += counts[split]
    for row in rows:
        row["leakage_group"] = digest(find(row["game_id"]))[:20]
        row["split"] = assignments[find(row["game_id"])]
        row["length_bin"] = length_bin(row["walkthrough_steps"])
    return sorted(rows, key=lambda r: r["game_id"])


def prompt_row(game, condition, **extra):
    return {
        "prompt": game["prompt"], "label": "",
        "metadata": {
            "instance_id": digest([game["game_id"], condition, extra])[:24],
            "game_id": game["game_id"], "game_file": game["game_file"],
            "task_type": game["task_family"], "reward_fn": "alfworld",
            "study_version": VERSION, "cue_condition": condition,
            "study_split": game["split"], "leakage_group": game["leakage_group"],
            **extra,
        },
    }


def evaluation_rows(games, cfg):
    rows = []
    for panel, per_family in cfg["eval_games_per_family"].items():
        split = "train" if panel == "train_probe" else panel
        families = defaultdict(list)
        for game in games:
            if game["split"] == split:
                if panel == "test_seen" and game["official_split"] != "valid_seen":
                    continue
                if panel == "test_unseen" and game["official_split"] != "valid_unseen":
                    continue
                families[game["task_family"]].append(game)
        selected = []
        for family, members in sorted(families.items()):
            # Round-robin length strata, preferentially distinct leakage groups.
            ordered = sorted(members, key=lambda g: digest([cfg["split_seed"], "eval", g["game_id"]]))
            seen = set()
            unique = []
            for game in ordered:
                if game["leakage_group"] not in seen:
                    seen.add(game["leakage_group"])
                    unique.append(game)
            buckets = defaultdict(list)
            for game in unique:
                buckets[game["length_bin"]].append(game)
            chosen = []
            while len(chosen) < per_family and any(buckets.values()):
                for key in sorted(buckets):
                    if buckets[key] and len(chosen) < per_family:
                        chosen.append(buckets[key].pop())
            selected.extend(chosen)
        for game in selected:
            for repeat in range(cfg["eval_repeats"]):
                pair_id = digest([VERSION, panel, game["game_id"], repeat])[:24]
                for condition in cfg["eval_conditions"]:
                    rows.append(prompt_row(game, condition, eval_panel=panel, eval_repeat=repeat,
                                           eval_pair_id=pair_id,
                                           requested_generation_seed=int(digest(pair_id)[:8], 16)))
    return rows


def conversation_fingerprint(messages):
    """Ignore arbitrary tool-call IDs, but retain task, reasoning, observations/actions."""
    normalized = []
    for message in messages:
        item = {k: message[k] for k in ("role", "content", "reasoning_content") if k in message}
        if message.get("tool_calls"):
            item["tool_calls"] = [call.get("function") for call in message["tool_calls"]]
        normalized.append(item)
    return digest(normalized)


def load_episodes(manifest):
    sessions = defaultdict(list)
    rejected = Counter()
    files = []
    for source in manifest["files"]:
        path = Path(source["path"])
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        files.append({"path": str(path), "sha256": checksum})
        for index, record in enumerate(read_jsonl(path)):
            sid = record.get("session_id")
            if not sid:
                rejected["missing_session_id"] += 1
                continue
            sessions[sid].append((record, str(path), index + 1))
    episodes = []
    for sid, items in sorted(sessions.items()):
        records = [item[0] for item in items]
        counts = {r.get("segment_count") for r in records}
        indices = [r.get("segment_index") for r in records]
        games = {game_id(r.get("game_file")) for r in records}
        outcomes = {r.get("task_success") for r in records}
        parents = {r.get("parent_traj_id") for r in records}
        if (len(counts) != 1 or not isinstance(next(iter(counts)), int)
                or len(games) != 1 or len(outcomes) != 1 or len(parents) != 1
                or sorted(i for i in indices if isinstance(i, int)) != list(range(next(iter(counts))))):
            rejected["inconsistent_or_incomplete_session"] += 1
            continue
        record, path, line = max(items, key=lambda item: item[0]["segment_index"])
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            rejected["missing_messages"] += 1
            continue
        episodes.append({
            "source_id": digest([manifest["source_id"], sid]), "session_id": sid,
            "game_id": next(iter(games)), "messages": messages,
            "fingerprint": conversation_fingerprint(messages),
            "success": record.get("task_success") is True,
            "invalid_actions": record.get("invalid_action_count"),
            "truncated": record.get("truncated"), "finish_reason": record.get("gen_finish_reason"),
            "compact_filtered": record.get("compact_filtered"),
            "num_steps": record.get("num_steps"),
            "source_file": path, "source_line": line,
            "teacher_run": manifest.get("teacher_run"),
            "provenance_status": manifest["provenance_status"],
        })
    return episodes, rejected, files


def select_demonstrations(episodes, games, cfg, provisional=False):
    catalog = {g["game_id"]: g for g in games}
    blocked = {e["fingerprint"] for e in episodes
               if e["game_id"] in catalog and catalog[e["game_id"]]["split"] != "train"}
    selected = defaultdict(list)
    rejected = Counter()
    seen = set()
    for episode in sorted(episodes, key=lambda e: e["source_id"]):
        game = catalog.get(episode["game_id"])
        reason = None
        if game is None:
            reason = "ineligible_or_unknown_game"
        elif game["split"] != "train":
            reason = "held_out_from_training"
        elif not episode["success"] or episode["invalid_actions"] != 0:
            reason = "failed_or_invalid_actions"
        elif episode["truncated"] is True or episode["finish_reason"] == "length" or episode["compact_filtered"] is True:
            reason = "known_incomplete_or_filtered"
        elif not provisional and any(episode[k] is None for k in ("truncated", "finish_reason", "compact_filtered")):
            reason = "unknown_completeness_requires_replay"
        elif episode["fingerprint"] in blocked or episode["fingerprint"] in seen:
            reason = "duplicate_or_cross_split_content"
        else:
            try:
                inject_demonstration(episode["messages"], "test", True)
            except (ValueError, StopIteration, TypeError, KeyError):
                reason = "unsupported_or_precontaminated_conversation"
        if reason:
            rejected[reason] += 1
            continue
        seen.add(episode["fingerprint"])
        selected[episode["game_id"]].append({**episode, "game": game})
    output = []
    for gid, members in sorted(selected.items()):
        ordered = sorted(members, key=lambda e: digest([cfg["split_seed"], "cap", e["source_id"]]))
        output.extend(ordered[:cfg["max_successes_per_game"]])
        rejected["per_game_cap"] += max(0, len(ordered) - cfg["max_successes_per_game"])
    return output, rejected


def paired_datasets(episodes, cfg, seed):
    by_block = defaultdict(list)
    by_id = {e["source_id"]: e for e in episodes}
    for episode in episodes:
        g = episode["game"]
        by_block[(g["task_family"], g["length_bin"])].append(episode["source_id"])
    eligible_n = round(len(episodes) * cfg["thought_eligible_fraction"])
    quotas = apportion({k: len(v) for k, v in by_block.items()}, eligible_n)
    eligible = {k: ranked(v, seed, ["eligible", k])[:quotas[k]] for k, v in by_block.items()}
    output = {}
    for arm, share in cfg["awareness_cue_share"].items():
        cue_n = round(eligible_n * share)
        cue_quotas = apportion({k: len(v) for k, v in eligible.items()}, cue_n)
        cue_thought = {sid for k, members in eligible.items()
                       for sid in ranked(members, seed, ["pairing", k])[:cue_quotas[k]]}
        all_eligible = {sid for members in eligible.values() for sid in members}
        rows = []
        for sid in ranked(list(by_id), seed, "source_order"):
            episode = by_id[sid]
            for condition in ranked(["test", "absent"], seed, ["cue_order", sid]):
                awareness = sid in all_eligible and ((sid in cue_thought) == (condition == "test"))
                rows.append({"prompt": inject_demonstration(episode["messages"], condition, awareness),
                             "label": "", "metadata": {
                                 "tools": ALFWORLD_TOOL_SCHEMAS, "source_id": sid,
                                 "game_id": episode["game_id"], "study_split": "train",
                                 "instance_id": digest([sid, condition])[:24],
                                 "study_version": VERSION, "arm": arm,
                                 "cue_condition": condition, "inserted_awareness": awareness,
                                 "assignment_seed": seed,
                                 "leakage_group": episode["game"]["leakage_group"],
                                 "teacher_run": episode["teacher_run"],
                             }})
        output[arm] = rows
    return output


def build(catalog_path, output, config_path, source_path=None, provisional=False, freeze_to=None):
    cfg = json.loads(config_path.read_text())
    if cfg["version"] != VERSION:
        raise ValueError("Configuration and intervention code versions differ")
    catalog = read_jsonl(catalog_path)
    games = split_catalog(catalog, cfg)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "splits.jsonl", games)
    write_jsonl(output / "catalog_exclusions.jsonl", [r for r in catalog if not r["eligible"]])
    evals = evaluation_rows(games, cfg)
    for panel in cfg["eval_games_per_family"]:
        write_jsonl(output / f"eval_{panel}.jsonl", [r for r in evals if r["metadata"]["eval_panel"] == panel])
    train_games = [g for g in games if g["split"] == "train"]
    rl = [prompt_row(g, c) for g in train_games for c in ("test", "absent")]
    rl.sort(key=lambda r: digest([cfg["split_seed"], "rl_order", r["metadata"]["instance_id"]]))
    write_jsonl(output / "rl_train_balanced.jsonl", rl)
    report = {
        "version": VERSION, "config": cfg, "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "split_counts": dict(Counter(g["split"] for g in games)),
        "split_group_counts": {s: len({g["leakage_group"] for g in games if g["split"] == s}) for s in sorted({g["split"] for g in games})},
        "eval_episode_counts": dict(Counter(r["metadata"]["eval_panel"] for r in evals)),
        "source_status": "missing", "launch_enabled": False,
        "gates_remaining": ["teacher_source_approval", "full_episode_action_replay", "pinned_template_token_budget_and_loss_masks", "isolated_launch_and_live_smoke_test"],
    }
    if source_path:
        manifest = json.loads(source_path.read_text())
        if manifest["provenance_status"] != "verified_target" and not provisional:
            raise ValueError("Source is not verified_target; only explicit provisional export is allowed")
        episodes, rejected, source_files = load_episodes(manifest)
        chosen, filtered = select_demonstrations(episodes, games, cfg, provisional)
        write_jsonl(output / "all_attempts_index.jsonl", [{k: v for k, v in e.items() if k != "messages"} for e in episodes])
        write_jsonl(output / "selected_sources.jsonl", [{k: v for k, v in e.items() if k not in ("messages", "game")} for e in chosen])
        prefix = "PROVISIONAL_" if provisional else ""
        write_jsonl(output / f"{prefix}competence_sft.jsonl", [{"prompt": e["messages"], "label": "", "metadata": {"tools": ALFWORLD_TOOL_SCHEMAS, "game_id": e["game_id"], "source_id": e["source_id"], "study_split": "train"}} for e in chosen])
        counts = {}
        for seed in cfg["assignment_seeds"]:
            for arm, rows in paired_datasets(chosen, cfg, seed).items():
                write_jsonl(output / f"{prefix}sft_{arm}_{seed}.jsonl", rows)
                counts[f"{arm}_{seed}"] = dict(Counter(f'{r["metadata"]["cue_condition"]}/I={int(r["metadata"]["inserted_awareness"])}' for r in rows))
        report.update(source_status=manifest["provenance_status"], source_files=source_files,
                      parsed_episodes=len(episodes), selected_demonstrations=len(chosen),
                      selected_games=len({e["game_id"] for e in chosen}),
                      rejected_sessions=dict(rejected), rejected_episodes=dict(filtered), cells=counts)
    report["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.jsonl"))}
    (output / "build_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if freeze_to is not None:
        freeze_to.mkdir(parents=True, exist_ok=False)
        fields = ("game_id", "official_split", "split", "leakage_group", "task_family", "length_bin", "scene", "file_sha256")
        write_jsonl(freeze_to / "game_splits.jsonl", [{k: g.get(k) for k in fields} for g in games])
        write_jsonl(freeze_to / "evaluation_ids.jsonl", [r["metadata"] for r in evals])
        with (freeze_to / "build_report.json").open("x") as handle:
            handle.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("study_config.json"))
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--freeze-to", type=Path, help="Publish metadata-only split/evaluation identities; no conversations")
    args = parser.parse_args()
    result = build(args.catalog, args.output, args.config, args.source_manifest, args.provisional, args.freeze_to)
    print(json.dumps({k: v for k, v in result.items() if k not in ("output_sha256", "source_files", "config")}, indent=2))
