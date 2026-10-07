import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dressage.recipes.alfworld.cue_study import (
    AWARENESS, exact_awareness, inject_cue, inject_demonstration,
)
from dressage.rollout.trajectory_logger import sample_to_record
from research.build_datasets import (
    apportion, build, evaluation_rows, game_id, paired_datasets, split_catalog,
)
from research.score_evaluations import score

CONFIG = json.loads((Path(__file__).parents[1] / "research/study_config.json").read_text())


def messages():
    return [{"role": "system", "content": "Solve the task."},
            {"role": "user", "content": "### Task\nPut the mug on the table."},
            {"role": "assistant", "content": "", "reasoning_content": "I should find the mug.",
             "tool_calls": [{"id": "a", "function": {"name": "env_step", "arguments": '{"command":"look"}'}}]},
            {"role": "tool", "tool_call_id": "a", "content": "A mug is visible."},
            {"role": "assistant", "content": "", "reasoning_content": "I can take it."}]


def catalog(n=100):
    return [{"game_id": f"json_2.1.1/train/task{i}/trial/game.tw-pddl",
             "game_file": f"examples/data/alfworld/alfworld_data/json_2.1.1/train/task{i}/trial/game.tw-pddl",
             "official_split": "train", "task_family": "clean", "task_group": f"task{i}",
             "problem_sha256": f"hash{i}", "walkthrough_steps": 8, "eligible": True,
             "prompt": f"task {i}"} for i in range(n)]


class CueStudyTests(unittest.TestCase):
    def test_injection_only_changes_two_fields(self):
        original = messages()
        changed = inject_demonstration(original, "test", True)
        self.assertEqual(original, messages())
        expected = copy.deepcopy(original)
        expected[1]["content"] = "Environment note: test environment.\n\n" + expected[1]["content"]
        expected[2]["reasoning_content"] = AWARENESS + "\n" + expected[2]["reasoning_content"]
        self.assertEqual(changed, expected)

    def test_bad_template_and_double_injection_fail(self):
        with self.assertRaises(ValueError):
            inject_cue("wrong template", "test")
        with self.assertRaises(ValueError):
            inject_cue(inject_cue("### Task\nx", "test"), "test")

    def test_absent_is_unchanged(self):
        self.assertEqual(inject_cue("### Task\nx", "absent"), "### Task\nx")
        self.assertEqual(inject_demonstration(messages(), "absent", False), messages())

    def test_no_awareness_means_no_reasoning_edit(self):
        self.assertEqual(inject_demonstration(messages(), "test", False)[2:], messages()[2:])

    def test_preexisting_target_rejected(self):
        msg = messages()
        msg[2]["reasoning_content"] = AWARENESS
        with self.assertRaises(ValueError):
            inject_demonstration(msg, "test", True)

    def test_awareness_ignores_prompt_and_tools(self):
        msg = messages()
        msg[1]["content"] = AWARENESS
        msg[3]["content"] = AWARENESS
        self.assertFalse(exact_awareness(msg)["awareness_any_turn"])
        msg[4]["reasoning_content"] = AWARENESS
        self.assertFalse(exact_awareness(msg)["awareness_first_turn"])
        self.assertTrue(exact_awareness(msg)["awareness_any_turn"])

    def test_split_determinism_and_group_leakage(self):
        data = catalog()
        data[1]["task_group"] = data[0]["task_group"]
        data[2]["problem_sha256"] = data[1]["problem_sha256"]
        result = split_catalog(data, CONFIG)
        self.assertEqual(result, split_catalog(list(reversed(data)), CONFIG))
        index = {r["game_id"]: r for r in result}
        self.assertEqual(len({index[r["game_id"]]["split"] for r in data[:3]}), 1)
        self.assertEqual(len({index[r["game_id"]]["leakage_group"] for r in data[:3]}), 1)

    def test_official_holdout_wins_duplicate_conflict(self):
        data = catalog()
        data[1]["official_split"] = "valid_unseen"
        data[1]["problem_sha256"] = data[0]["problem_sha256"]
        result = {r["game_id"]: r for r in split_catalog(data, CONFIG)}
        self.assertEqual(result[data[0]["game_id"]]["split"], "test_unseen")

    def test_exact_pair_counts_and_source_marginals(self):
        games = split_catalog(catalog(), CONFIG)
        episodes = [{"source_id": str(i), "messages": messages(), "game": g,
                     "game_id": g["game_id"], "teacher_run": "fixture"}
                    for i, g in enumerate(games)]
        datasets = paired_datasets(episodes, CONFIG, 1101)
        for arm, target in (("null", (10, 10)), ("positive", (13, 7))):
            rows = datasets[arm]
            self.assertEqual(len(rows), 200)
            counts = tuple(sum(r["metadata"]["inserted_awareness"] for r in rows
                               if r["metadata"]["cue_condition"] == c) for c in ("test", "absent"))
            self.assertEqual(counts, target)
        for sid in map(str, range(100)):
            exposure = []
            for arm in ("null", "positive"):
                pair = [r for r in datasets[arm] if r["metadata"]["source_id"] == sid]
                self.assertEqual({r["metadata"]["cue_condition"] for r in pair}, {"test", "absent"})
                exposure.append(sum(r["metadata"]["inserted_awareness"] for r in pair))
            self.assertEqual(*exposure)

    def test_eval_pairs_and_unique_prompt_ids(self):
        rows = evaluation_rows(split_catalog(catalog(), CONFIG), CONFIG)
        pairs = {}
        for row in rows:
            meta = row["metadata"]
            pairs.setdefault(meta["eval_pair_id"], set()).add(meta["cue_condition"])
        self.assertTrue(all(v == {"test", "absent"} for v in pairs.values()))
        self.assertEqual(len(rows), len({r["metadata"]["instance_id"] for r in rows}))

    def test_logger_keeps_intervention_identity(self):
        meta = {"game_id": "game", "cue_condition": "test", "study_version": "cue-study-v2", "eval_pair_id": "pair"}
        row = sample_to_record({"metadata": meta})
        for key, value in meta.items():
            self.assertEqual(row[key], value)

    def test_apportion_totals(self):
        self.assertEqual(sum(apportion({"a": 2, "b": 7, "c": 1}, 3).values()), 3)

    def test_path_canonicalization(self):
        self.assertEqual(game_id("/different/root/json_2.1.1/train/task/trial/game.tw-pddl"),
                         "json_2.1.1/train/task/trial/game.tw-pddl")

    def test_no_source_build_is_explicitly_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "catalog.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in catalog()))
            cfg = Path(__file__).parents[1] / "research/study_config.json"
            report = build(path, base / "output", cfg)
            self.assertEqual(report["source_status"], "missing")
            self.assertFalse(report["launch_enabled"])
            self.assertFalse(list((base / "output").glob("*sft*.jsonl")))
            with self.assertRaises(FileExistsError):
                build(path, base / "output", cfg)

    def scoring_fixture(self):
        manifest = evaluation_rows(split_catalog(catalog(), CONFIG), CONFIG)[:2]
        records = [{**row["metadata"], "session_id": str(i), "segment_index": 0,
                    "segment_count": 1, "task_success": False, "messages": messages()}
                   for i, row in enumerate(manifest)]
        records[1]["messages"][2]["reasoning_content"] = AWARENESS
        records[1]["task_success"] = True
        return manifest, records

    def test_scorer_keeps_failures_and_paired_effect(self):
        manifest, records = self.scoring_fixture()
        report = score(manifest, records)
        self.assertEqual(report["observed"], 2)
        self.assertEqual(report["cells"]["absent"]["success"], 0)
        self.assertEqual(report["paired_effects"]["awareness_first_turn"]["mean_game_paired_difference"], 1)

    def test_scorer_rejects_missing_evaluation(self):
        manifest, records = self.scoring_fixture()
        with self.assertRaises(ValueError):
            score(manifest, records[:1])
        self.assertTrue(score(manifest, records[:1], allow_partial=True)["partial"])

    def test_scorer_rejects_retries_and_metadata_mismatch(self):
        manifest, records = self.scoring_fixture()
        retry = copy.deepcopy(records[0])
        retry["session_id"] = "retry"
        with self.assertRaises(ValueError):
            score(manifest, records + [retry])
        records[0]["cue_condition"] = "test"
        with self.assertRaises(ValueError):
            score(manifest, records)

    def test_scorer_counts_terminal_episode_once(self):
        manifest, records = self.scoring_fixture()
        prefix = copy.deepcopy(records[0])
        prefix["segment_count"] = 2
        records[0]["segment_count"] = 2
        records[0]["segment_index"] = 1
        self.assertEqual(score(manifest, records + [prefix])["observed"], 2)

    def test_end_to_end_sft_export_uses_training_games_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            games = catalog()
            cat_path = base / "catalog.jsonl"
            cat_path.write_text("\n".join(json.dumps(r) for r in games))
            records = []
            for i, game in enumerate(games):
                msg = messages()
                msg[1]["content"] = f"### Task\nTask {i}"
                records.append({"session_id": str(i), "game_file": game["game_file"],
                                "messages": msg, "task_success": True, "invalid_action_count": 0,
                                "segment_count": 1, "segment_index": 0, "parent_traj_id": str(i),
                                "truncated": False, "gen_finish_reason": "stop", "compact_filtered": False})
            raw = base / "raw.jsonl"
            raw.write_text("\n".join(json.dumps(r) for r in records))
            source = {"source_id": "synthetic-only", "teacher_run": "fixture", "provenance_status": "verified_target",
                      "files": [{"path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}]}
            src = base / "source.json"
            src.write_text(json.dumps(source))
            cfg = Path(__file__).parents[1] / "research/study_config.json"
            report = build(cat_path, base / "output", cfg, src)
            self.assertEqual(report["selected_demonstrations"], 80)
            self.assertFalse(report["launch_enabled"])
            train_ids = {r["game_id"] for r in split_catalog(games, CONFIG) if r["split"] == "train"}
            for path in (base / "output").glob("sft_*.jsonl"):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(rows), 160)
                self.assertTrue(all(r["metadata"]["game_id"] in train_ids for r in rows))
            source["provenance_status"] = "excluded_older_run"
            src.write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                build(cat_path, base / "rejected", cfg, src)


if __name__ == "__main__":
    unittest.main()
