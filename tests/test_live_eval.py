import argparse
import asyncio
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from research.run_live_eval import assistant_message, async_client_options, boundary_preflight, concurrent_episodes, evaluate_episode, evaluate_episode_async, finish_type, remaining_context, sampled_ids, turn_seed, validate_render_environment, validate_runtime_limits
from research.run_live_eval import EnvironmentWorker


class LiveEvalTests(unittest.TestCase):
    def test_environment_worker_yields_serializes_and_drains_cancelled_creation(self):
        async def scenario():
            worker = EnvironmentWorker()
            threads = []
            ticks = 0
            env = Mock()
            def create(*args):
                threads.append(threading.get_ident())
                time.sleep(.05)
                return env
            async def heartbeat():
                nonlocal ticks
                for _ in range(10):
                    await asyncio.sleep(.002)
                    ticks += 1
            try:
                with patch('research.run_live_eval.make_env', side_effect=create):
                    task = asyncio.create_task(worker.call(worker.create, 'game', 50))
                    await heartbeat()
                    self.assertFalse(task.done())
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertEqual(ticks, 10)
                self.assertIn(id(env), worker.environments)
                await worker.call(lambda: threads.append(threading.get_ident()))
                self.assertEqual(len(set(threads)), 1)
                self.assertNotEqual(threads[0], threading.get_ident())
                await worker.call(worker.close_remaining)
                env.close.assert_called_once()
                self.assertEqual(worker.environments, {})
            finally:
                worker.executor.shutdown(wait=True)
        asyncio.run(scenario())

    def test_async_transport_does_not_reuse_expired_idle_sockets(self):
        fake = Mock()
        with patch.dict('sys.modules', {'httpx': fake}):
            options = async_client_options(128)
        fake.Limits.assert_called_once_with(max_connections=132, max_keepalive_connections=0)
        fake.Timeout.assert_called_once_with(1200., connect=10.)
        self.assertFalse(options['trust_env'])

    def test_concurrent_scheduler_bound_coverage_and_failure_cleanup(self):
        async def scenario(fail=False):
            active = peak = 0
            completed = []

            async def run(row):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(.001 if row % 2 else .002)
                    if fail and row == 1:
                        raise RuntimeError('transport failed')
                    return row * 2
                finally:
                    active -= 1

            if fail:
                with self.assertRaisesRegex(RuntimeError, 'transport failed'):
                    await concurrent_episodes(range(12), 3, run, lambda *x: completed.append(x))
            else:
                await concurrent_episodes(range(12), 3, run, lambda *x: completed.append(x))
                self.assertEqual(sorted((x[0], x[1]) for x in completed), [(i, i*2) for i in range(12)])
            self.assertEqual(peak, 3)
            self.assertEqual(active, 0)

        asyncio.run(scenario())
        asyncio.run(scenario(fail=True))

    def limits(self, context=262144, threshold=None):
        return dict(schema='cue-study-runtime-limits-v1', context_length=context,
                    max_input_tokens_exclusive=threshold or context - 6,
                    reserved_tokens=0, allow_auto_truncate=False)

    def test_strict_server_context_boundary(self):
        for context in (18432, 262144):
            limits = self.limits(context)
            threshold = context - 6
            self.assertEqual(remaining_context(context, threshold - 1, limits), 1)
            for length in (threshold, threshold + 1, context, context + 100):
                self.assertEqual(remaining_context(context, length, limits), 0)
        self.assertEqual(remaining_context(18432, 18429, self.limits(18432)), 0)

    def test_memory_limited_server_and_invalid_limits(self):
        limits = self.limits(threshold=12000)
        self.assertEqual(remaining_context(262144, 11999, limits), 1)
        self.assertEqual(remaining_context(262144, 12000, limits), 0)
        for patch_value in ({'context_length': 18432}, {'max_input_tokens_exclusive': None},
                            {'allow_auto_truncate': True}, {'schema': 'unknown'},
                            {'reserved_tokens': -1}, {'max_input_tokens_exclusive': True}):
            with self.subTest(patch_value=patch_value), self.assertRaises(ValueError):
                validate_runtime_limits({**limits, **patch_value}, 262144)

    def test_boundary_preflight_checks_exact_accounting(self):
        server = Mock(url='http://localhost:30000')
        def respond(path, payload):
            return {'output_ids': [99], 'meta_info': {'completion_tokens': 1,
                'prompt_tokens': len(payload['input_ids']), 'finish_reason': {'type': 'length'}}}
        server.request.side_effect = respond
        response = Mock(status=200)
        response.read.return_value = b'Cache flushed'
        with patch('research.run_live_eval.urllib.request.urlopen') as request:
            request.return_value.__enter__.return_value = response
            report = boundary_preflight(server, 262144, self.limits(), 87)
        self.assertTrue(report['passed'])
        self.assertEqual([r['input_tokens'] for r in report['requests']], [18429, 262137])
        self.assertEqual(server.request.call_count, 2)
        bad = respond('/generate', {'input_ids': [1]})
        server.request.side_effect = None
        server.request.return_value = bad
        with self.assertRaisesRegex(RuntimeError, 'accounting'):
            boundary_preflight(server, 262144, self.limits(), 87)
        with self.assertRaisesRegex(RuntimeError, 'memory pool'):
            boundary_preflight(server, 262144, self.limits(threshold=12000), 87)

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
                                      max_new_tokens=2048, temperature=1., checkpoint_label="fake-model",
                                      runtime_limits=self.limits(18432))
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
            # An over-limit episode is an outcome, not an HTTP request or a
            # run-level exception. A subsequent episode can still be evaluated.
            args.runtime_limits = self.limits(18432, threshold=2)
            server = Server()
            server.requests = []
            with patch('research.run_live_eval.make_env', return_value=Environment()):
                record, traces = evaluate_episode(row, args, tokenizer, server)
            self.assertEqual(record['stop_reason'], 'context_exhausted')
            self.assertTrue(record['truncated'])
            self.assertEqual(server.requests, [])
            self.assertEqual(traces, [])
            args.runtime_limits = self.limits(18432)
            with patch('research.run_live_eval.make_env', return_value=Environment()):
                record, _ = evaluate_episode(row, args, tokenizer, Server())
            self.assertTrue(record['task_success'])
            # Exercise the real async episode path with two independent game
            # instances and interleaved generation, retaining exact TITO IDs.
            environments = [Environment(), Environment()]
            payloads = []

            class AsyncServer(Server):
                async def request(self, path, payload):
                    payloads.append(payload)
                    await asyncio.sleep(.001)
                    return super().request(path, payload)

            async def pair():
                row2 = {**row, 'metadata': {**row['metadata'], 'instance_id': 'second-id',
                                          'requested_generation_seed': 456}}
                worker = EnvironmentWorker()
                try:
                    with patch('research.run_live_eval.make_env', side_effect=environments):
                        return await asyncio.gather(
                            evaluate_episode_async(row, args, tokenizer, AsyncServer(), worker),
                            evaluate_episode_async(row2, args, tokenizer, AsyncServer(), worker))
                finally:
                    await worker.call(worker.close_remaining)
                    worker.executor.shutdown(wait=True)

            results = asyncio.run(pair())
            self.assertEqual([env.count for env in environments], [2, 2])
            self.assertTrue(all(record['task_success'] for record, _ in results))
            for record, traces in results:
                prefix = traces[0]['input_ids'] + traces[0]['output_ids']
                self.assertEqual(traces[1]['input_ids'][:len(prefix)], prefix)
            self.assertEqual([p['sampling_params']['sampling_seed'] for p in payloads], [123,456,124,457])


if __name__ == "__main__":
    unittest.main()
