"""Private Ray lifecycle for single-GPU supervised training (not RL generation)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile


def main():
    import ray
    from slime.utils.arguments import parse_args

    run_dir = Path(os.environ["CUE_SFT_RUN_DIR"]).resolve(strict=True)
    args = parse_args()
    if not args.debug_train_only or args.loss_type != "sft_loss":
        raise ValueError("This isolated runner only supports supervised training")
    (run_dir / "parsed_args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    # A short path avoids UNIX socket path limits; Ray owns only this new temp tree.
    ray_tmp = tempfile.mkdtemp(prefix="cue-sft-ray-")
    (run_dir / "ray_temp_dir.txt").write_text(ray_tmp + "\n")
    ray.init(address="local", num_gpus=1, num_cpus=8, include_dashboard=False,
             object_store_memory=2 * 1024**3, _temp_dir=ray_tmp,
             runtime_env={"worker_process_setup_hook": "dressage.training.sft_worker.install"})
    try:
        import slime
        train_path = Path(slime.__file__).resolve().parents[1] / "train.py"
        spec = importlib.util.spec_from_file_location("cue_slime_train", train_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.train(args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
