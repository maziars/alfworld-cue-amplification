import argparse
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("sft_launch", Path(__file__).resolve().parents[1] / "research/launch_sft.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class SFTLaunchTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(steps=None, batch_size=8, start_rollout=0, hf_checkpoint=Path("/model"),
                      load=Path("/base"), run_dir=Path("/new-run"), save_interval=25,
                      dataset=Path("/data"), seed=7001, lr=1e-5, no_shuffle=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_full_pool_covered_without_flooring(self):
        cli, batches = launch.training_args(self.args(), ["--num-layers", "24"], 877)
        self.assertEqual(batches, 110)
        self.assertEqual(cli[cli.index("--num-rollout") + 1], "110")
        self.assertEqual(batches * 8 - 877, 3)
        self.assertNotIn("--num-epoch", cli)

    def test_smoke_resume_separate_checkpoint_and_run(self):
        cli, batches = launch.training_args(self.args(steps=1, batch_size=1, start_rollout=1,
                                                     load=Path("/old-run/checkpoints"), no_shuffle=True), [], 1)
        self.assertEqual(batches, 1)
        self.assertEqual(cli[cli.index("--num-rollout") + 1], "2")
        self.assertEqual(cli[cli.index("--load") + 1], "/old-run/checkpoints")
        self.assertEqual(cli[cli.index("--save") + 1], "/new-run/checkpoints")
        self.assertNotIn("--rollout-shuffle", cli)
        self.assertNotIn("--no-load-optim", cli)
        self.assertNotIn("--no-save-optim", cli)
        self.assertIn("--use-checkpoint-opt-param-scheduler", cli)

    def test_exact_supervised_path(self):
        cli, _ = launch.training_args(self.args(), [], 877)
        for name in ("--rollout-max-context-len", "--seq-length", "--max-tokens-per-gpu"):
            self.assertEqual(cli[cli.index(name) + 1], "24576")
        self.assertEqual(cli[cli.index("--rollout-function-path") + 1],
                         "dressage.training.structured_sft.generate_rollout")
        self.assertIn("--debug-train-only", cli)
        self.assertIn("--calculate-per-token-loss", cli)
        self.assertIn("--disable-compute-advantages-and-returns", cli)
        self.assertNotIn("--apply-chat-template", cli)
        self.assertNotIn("--rollout-max-prompt-len", cli)
        self.assertNotIn("--use-checkpoint-opt-param-scheduler", cli)


if __name__ == "__main__":
    unittest.main()
