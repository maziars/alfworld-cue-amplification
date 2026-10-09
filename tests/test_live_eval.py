import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.run_live_eval import assistant_message, evaluate_episode, finish_type, remaining_context, sampled_ids, turn_seed, validate_render_environment


class LiveEvalTests(unittest.TestCase):
    def test_strict_server_context_boundary(self):
        self.assertEqual(remaining_context(18432, 16539), 1892)
        self.assertEqual(remaining_context(18432, 18431), 0)
        self.assertEqual(remaining_context(18432, 19000), 0)

    def test_seed_schedule_keeps_frozen_first_turn(self):
        self.assertEqual(turn_seed(3803474250, 0), 3803474250)
        self.assertEqual(turn_seed(2**32 - 1, 1), 0)

    def test_token_ids_never_reconstructed_from_text(self):
        self.assertEqual(sampled_ids({"output_ids": [12, 34], "meta_info": {"completion_tokens": 2}}), [12, 34])
        self.assertEqual(sampled_ids({"meta_info": {"output_token_logprobs": [[-.1, 12, None]],
                                                     "completion_tokens": 1}}), [12])
        with self.assertRaises(ValueError):
            sampled_ids({"text": "hello"})
        with self.assertRaises(ValueError):
            sampled_ids({"output_ids": [12], "meta_info": {"completion_tokens": 2}})

    def test_xml_action_and_reasoning(self):
        message, calls = assistant_message('It seems I am being tested.</think>\n<tool_call>\n'
            '<function=env_step><parameter=command>look</parameter></function></tool_call>', 3)
        self.assertEqual(message["reasoning_content"], "It seems I am being tested.")
        self.assertEqual(calls[0]["id"], "eval_turn_3_call_0")
        self.assertEqual(calls[0]["function"]["arguments"], '{"command": "look"}')

    def test_unfinished_reasoning_not_lost(self):
        message, calls = assistant_message("It seems I am being tested.", 0)
        self.assertEqual(message["reasoning_content"], "It seems I am being tested.")
        self.assertEqual(calls, [])

    def test_noncompletion_finish_rejected(self):
        for kind in ("abort", "preempt", "error", "unknown", None):
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                finish_type({"meta_info": {"finish_reason": {"type": kind}}})
        with self.assertRaises(RuntimeError):
            finish_type({})

    def test_environment_flags_fail_closed(self):
        with patch.dict("os.environ", {"ALFWORLD_NAV_SPLIT": "1"}), self.assertRaises(ValueError):
            validate_render_environment()
        with patch.dict("os.environ", {"ALFWORLD_DEMANGLE": "0"}), self.assertRaises(ValueError):
            validate_render_environment()

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "research/data/tokenizers/qwen3.5-0.8b/tokenizer.json").exists(),
                         "local audited tokenizer asset is optional")
    def test_two_turn_tito_carries_exact_output_prefix(self):
        from dressage.training.structured_sft import get_generator
        root = Path(__file__).resolve().parents[1]
        generator = get_generator(root / "research/data/tokenizers/qwen3.5-0.8b")
        tokenizer = generator.tokenizer
        tokenizer.chat_template = generator.template

        class Environment:
            count = 0

            def reset(self):
                return "Your task is to: test task", {"admissible_commands": ["look"]}

            def step(self, command):
                self.count += 1
                return "observation", 0, self.count == 2, {"won": self.count == 2, "admissible_commands": ["look"]}

            def close(self):
                pass

        class Server:
            requests = []
            kind = "stop"

            def request(self, path, payload):
                self.requests.append(payload)
                raw = 'Look around.</think>\n<tool_call><function=env_step><parameter=command>look</parameter></function></tool_call><|im_end|>'
                output = tokenizer.encode(raw, add_special_tokens=False)
                return {"output_ids": output, "meta_info": {"completion_tokens": len(output),
                                                               "finish_reason": {"type": self.kind}}}

        with tempfile.TemporaryDirectory() as directory:
            game = Path(directory) / "game.tw-pddl"
            game.touch()
            args = argparse.Namespace(game_root=Path(directory), max_steps=3, context=18432,
                                      max_new_tokens=2048, temperature=1., checkpoint_label="fake-model")
            row = {"prompt": "test task", "metadata": {"instance_id": "fake-id", "game_file": game.name,
                   "cue_condition": "test", "requested_generation_seed": 123}}
            server = Server()
            with patch("research.run_live_eval.make_env", return_value=Environment()):
                record, traces = evaluate_episode(row, args, tokenizer, server)
            self.assertTrue(record["task_success"])
            self.assertEqual(record["num_steps"], 2)
            self.assertEqual(record["invalid_action_count"], 0)
            original_prefix = traces[0]["input_ids"] + traces[0]["output_ids"]
            self.assertEqual(traces[1]["input_ids"][:len(original_prefix)], original_prefix)
            self.assertEqual([p["sampling_params"]["sampling_seed"] for p in server.requests], [123, 124])
            self.assertEqual(server.requests[0]["sampling_params"]["top_k"], -1)
            self.assertEqual(server.requests[0]["sampling_params"]["min_p"], 0.)
            self.assertIn("test environment", record["messages"][1]["content"])
            # A valid-looking action inside an aborted response must never be executed.
            server = Server()
            server.kind = "abort"
            environment = Environment()
            with patch("research.run_live_eval.make_env", return_value=environment), self.assertRaises(RuntimeError):
                evaluate_episode(row, args, tokenizer, server)
            self.assertEqual(environment.count, 0)
            # A length-ended generation may contain a winning action. Keep both facts.
            server = Server()
            server.kind = "length"
            environment = Environment()
            environment.count = 1
            with patch("research.run_live_eval.make_env", return_value=environment):
                record, _ = evaluate_episode(row, args, tokenizer, server)
            self.assertTrue(record["task_success"])
            self.assertTrue(record["truncated"])
            environment = Environment()
            environment.step = lambda command: ("obs", 0, True, {"won": False, "TimeLimit.truncated": True})
            with patch("research.run_live_eval.make_env", return_value=environment):
                record, _ = evaluate_episode(row, args, tokenizer, Server())
            self.assertEqual(record["stop_reason"], "environment_step_limit")


if __name__ == "__main__":
    unittest.main()
