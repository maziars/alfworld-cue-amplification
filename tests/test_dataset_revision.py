import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.build_datasets import read_jsonl, write_jsonl
from research.revise_sft_dataset import revise
from research.preflight_replay import replay


class RevisionTests(unittest.TestCase):
    def test_only_excluded_source_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root / "parent"
            parent.mkdir()
            cfg = {"excluded_source_ids": {"bad": "approved"}, "assignment_seeds": [1],
                   "awareness_cue_share": {"null": .5, "positive": .65}}
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            sources = [{"source_id": sid, "game_id": "game"} for sid in ("good", "bad")]
            write_jsonl(parent / "selected_sources.jsonl", sources)
            rows = [{"prompt": [{"role": "user", "content": sid}], "metadata": {
                "source_id": sid, "cue_condition": cue, "inserted_awareness": False}}
                for sid in ("good", "bad") for cue in ("test", "absent")]
            for name in ("competence_sft", "sft_null_1", "sft_positive_1"):
                write_jsonl(parent / f"PROVISIONAL_{name}.jsonl", rows)
            (parent / "splits.jsonl").write_text('{"fixed":true}\n')
            (parent / "build_report.json").write_text(json.dumps({"rejected_episodes": {}}))
            original = {p.name: p.read_bytes() for p in parent.iterdir()}
            result = revise(parent, root / "revised", config)
            self.assertEqual(result["selected_demonstrations"], 1)
            self.assertEqual(result["revision"]["backfilled_sources"], 0)
            self.assertEqual(read_jsonl(root / "revised/PROVISIONAL_sft_null_1.jsonl"), rows[:2])
            self.assertEqual((root / "revised/splits.jsonl").read_bytes(), original["splits.jsonl"])
            self.assertEqual({p.name: p.read_bytes() for p in parent.iterdir()}, original)
            with self.assertRaises(FileExistsError):
                revise(parent, root / "revised", config)


class ReplayPolicyTests(unittest.TestCase):
    def test_first_call_matches_agent_without_certifying_extra_calls(self):
        class Env:
            count = 0
            def reset(self):
                return "start", {"admissible_commands": ["look"]}
            def step(self, command):
                self.count += 1
                return "done", 1, True, {"won": True}
            def close(self):
                pass
        calls = [{"id": "executed", "function": {"name": "env_step", "arguments": {"command": "look"}}},
                 {"id": "ignored", "function": {"name": "env_step", "arguments": {"command": "invalid"}}}]
        job = {"source_id": "s", "game_id": "g", "game_sha256": hashlib.sha256(b"g").hexdigest(),
               "num_steps": 1, "messages": [{"role": "user", "content": "### Initial Observation\nstart"},
                                              {"role": "assistant", "tool_calls": calls}]}
        with patch.object(Path, "read_bytes", return_value=b"g"), \
                patch("dressage.recipes.alfworld.tools.make_env", return_value=Env()):
            self.assertFalse(replay(job)["passed"])
            historical = replay({**copy.deepcopy(job), "first_call_policy": True})
        self.assertTrue(historical["passed"])
        self.assertEqual(historical["executed_actions"], 1)
        self.assertEqual(historical["unexecuted_tool_calls"][0]["ignored_call_ids"], ["ignored"])


if __name__ == "__main__":
    unittest.main()
