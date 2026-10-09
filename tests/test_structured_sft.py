import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dressage.training import structured_sft as loader

ROOT = Path(__file__).parents[1]


def arguments(**changes):
    cfg = json.loads((ROOT / "research/sft_loader_config.json").read_text())
    cfg.update(hf_checkpoint=str(ROOT / "research/data/tokenizers/qwen3.5-0.8b"), rollout_batch_size=1)
    cfg.update(changes)
    return SimpleNamespace(**cfg)


def sample():
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "Look around."},
                {"role": "assistant", "reasoning_content": "Look first.", "content": "look"}],
        metadata={"source_id": "test-source", "study_split": "train", "tools": [
            {"type": "function", "function": {"name": "env_step", "parameters": {"type": "object"}}}]},
        tokens=None, loss_mask=None, response_length=0, reward=None,
    )


class Buffer:
    def __init__(self, groups, pool=None):
        self.groups = groups
        self.dataset = SimpleNamespace(origin_samples=pool if pool is not None else [g[0] for g in groups if g])
        self.calls = 0

    def get_samples(self, count):
        self.calls += 1
        return self.groups


class LoaderTests(unittest.TestCase):
    def test_guards(self):
        for changes in ({"n_samples_per_prompt": 8}, {"apply_chat_template": True},
                        {"loss_type": "policy_loss"}, {"input_key": "text"},
                        {"compute_advantages_and_returns": True},
                        {"debug_train_only": False}, {"seq_length": 16384},
                        {"rollout_max_context_len": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                loader.validate_args(arguments(**changes))
        with self.assertRaises(ValueError):
            loader.validate_args(arguments(), evaluation=True)

    def test_sample_guards(self):
        s = sample()
        s.prompt = "already rendered"
        with self.assertRaises(ValueError):
            loader.validate_sample(s)
        s = sample()
        s.metadata["study_split"] = "test"
        with self.assertRaises(ValueError):
            loader.validate_sample(s)

    def test_bad_later_source_blocks_before_fetch(self):
        good, bad = sample(), sample()
        bad.prompt[1]["content"] = "<|im_end|>"
        buffer = Buffer([[good]], pool=[good, bad])
        with patch.object(loader, "get_generator") as get, self.assertRaisesRegex(ValueError, "Embedded control"):
            loader.generate_rollout(arguments(), 0, buffer)
        get.assert_not_called()
        self.assertEqual(buffer.calls, 0)
        self.assertIsNone(good.tokens)

    def test_batch_shape(self):
        for groups in ([], [[sample(), sample()]]):
            with patch.object(loader, "get_generator"), self.assertRaises(ValueError):
                loader.generate_rollout(arguments(), 0, Buffer(groups))

    def test_atomic_context_failure(self):
        good, bad = sample(), sample()
        small = {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1]}
        large = {"tokens": [1] * 11, "loss_mask": [1] * 11}
        with patch.object(loader, "get_generator") as get:
            get.return_value.encode.side_effect = [small, large]
            with self.assertRaisesRegex(ValueError, "refusing truncation"):
                loader.generate_rollout(arguments(rollout_batch_size=2, rollout_max_context_len=10),
                                        0, Buffer([[good], [bad]]))
        self.assertIsNone(good.tokens)
        self.assertIsNone(bad.tokens)

    def test_return_shape_and_response_mask(self):
        s = sample()
        original = copy.deepcopy((s.prompt, s.metadata))
        groups = [[s]]
        with patch.object(loader, "get_generator") as get:
            get.return_value.encode.return_value = {"tokens": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1]}
            returned = loader.generate_rollout(arguments(), 12, Buffer(groups))
        self.assertIs(returned, groups)
        self.assertEqual(s.loss_mask, [1, 0, 1])
        self.assertEqual(s.response_length, 3)
        self.assertEqual((s.prompt, s.metadata), original)
        self.assertEqual(s.reward, 0)

    def test_real_tokenizer(self):
        if not (Path(arguments().hf_checkpoint) / "tokenizer.json").exists():
            self.skipTest("Local tokenizer unavailable")
        try:
            import transformers
        except ImportError:
            self.skipTest("Transformers unavailable")
        s = sample()
        loader.generate_rollout(arguments(), 0, Buffer([[s]]))
        self.assertEqual(len(s.loss_mask), s.response_length)
        self.assertGreater(len(s.tokens), s.response_length)
        self.assertGreater(sum(s.loss_mask), 0)


if __name__ == "__main__":
    unittest.main()
