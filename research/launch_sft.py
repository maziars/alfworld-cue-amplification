#!/usr/bin/env python3
"""Isolated single-Spark SFT launcher. Dry-run unless --execute is supplied.

Run inside the existing tito container against an isolated project deployment.
Never starts SGLang, edits upstream code, stops Ray, or kills other processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def architecture_args(slime_root):
    source = Path(slime_root) / "scripts/models/qwen3.5-0.8B.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"', "bash", str(source)],
        check=True, capture_output=True,
    )
    return result.stdout.decode().rstrip("\0").split("\0"), sha256(source)


def training_args(args, model_args, row_count):
    batches = args.steps if args.steps is not None else math.ceil(row_count / args.batch_size)
    final_rollout = args.start_rollout + batches
    command = list(model_args) + [
        "--hf-checkpoint", str(args.hf_checkpoint), "--load", str(args.load),
        "--save", str(args.run_dir / "checkpoints"), "--save-interval", str(args.save_interval),
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--num-gpus-per-node", "1",
        "--rollout-function-path", "dressage.training.structured_sft.generate_rollout",
        "--prompt-data", str(args.dataset), "--input-key", "prompt", "--metadata-key", "metadata",
        "--n-samples-per-prompt", "1", "--rollout-batch-size", str(args.batch_size),
        "--global-batch-size", str(args.batch_size), "--micro-batch-size", "1",
        "--num-rollout", str(final_rollout), "--start-rollout-id", str(args.start_rollout),
        "--rollout-seed", str(args.seed), "--seed", str(args.seed),
        "--loss-type", "sft_loss", "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns", "--debug-train-only",
        "--rollout-max-context-len", "18432", "--seq-length", "18432",
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1", "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", "18432", "--log-probs-chunk-size", "1024",
        "--optimizer", "adam", "--lr", str(args.lr), "--lr-decay-style", "constant",
        "--min-lr", str(args.lr), "--lr-warmup-iters", "0", "--weight-decay", "0.01",
        "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--clip-grad", "1.0",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32", "--attention-backend", "flash",
        "--rollout-temperature", "1.0", "--rollout-top-p", "1.0", "--entropy-coef", "0.0",
        "--save-debug-rollout-data", str(args.run_dir / "rollout_data/{rollout_id}.pt"),
    ]
    if not args.no_shuffle:
        command.append("--rollout-shuffle")
    if args.start_rollout > 0:
        # A resumed run extends num_rollout, so Slime derives a different nominal
        # decay horizon even under constant LR. Restore the original scheduler
        # state rather than asserting that the new horizon equals the old one.
        command.append("--use-checkpoint-opt-param-scheduler")
    return command, batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--slime-root", type=Path, default=Path("/workspace/tito-dressage/slime"))
    parser.add_argument("--megatron-root", type=Path, default=Path("/root/Megatron-LM"))
    parser.add_argument("--hf-checkpoint", type=Path, default=Path("/workspace/tito-dressage/models/Qwen3.5-0.8B"))
    parser.add_argument("--load", type=Path, default=Path("/workspace/tito-dressage/models/Qwen3.5-0.8B_torch_dist"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--start-rollout", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for key in ("project_root", "slime_root", "megatron_root", "hf_checkpoint", "load", "dataset"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        parser.error("run-dir must be NEW; resume reads a previous --load but writes a new run-dir")
    if args.batch_size < 1 or args.start_rollout < 0 or args.save_interval < 1 or args.lr <= 0:
        parser.error("batch size/save interval/lr must be positive; start rollout nonnegative")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    digest = sha256(args.dataset)
    if digest != args.dataset_sha256:
        parser.error("dataset SHA256 mismatch")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_rows or not rows:
        parser.error("dataset row count mismatch")
    if any(row.get("metadata", {}).get("study_split") != "train" for row in rows):
        parser.error("non-training row in SFT input")
    if args.batch_size > len(rows):
        parser.error("batch size exceeds source pool; stock Slime wraps at most once")
    model_args, model_sha = architecture_args(args.slime_root)
    cli, batches = training_args(args, model_args, len(rows))
    env_overrides = {
        "PYTHONPATH": os.pathsep.join(map(str, (args.project_root, args.slime_root, args.megatron_root))),
        "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "CUDA_VISIBLE_DEVICES": "0",
        "NCCL_IB_DISABLE": "1", "NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUE_SFT_RUN_DIR": str(args.run_dir), "USE_CHUNKED_LOGPROB": "1",
        "CHUNKED_LOGPROB_CHUNK_SIZE": "1024", "CHUNKED_LOGPROB_FP32_MATMUL": "0",
        "USE_RESUME_WEIGHT_FIX": "1", "WANDB_MODE": "disabled",
        "RAY_USAGE_STATS_ENABLED": "0", "OMP_NUM_THREADS": "4",
        # Do not inherit experimental RL runtime patches from the container shell.
        "USE_VIRTUAL_REF": "0", "USE_MASTER_PARAM": "0", "USE_STREAMING_LOAD": "0",
        "PHASE_SWAP_TRAINER": "0", "DP_EQUIV_STEP_DUMP": "", "DP_EQUIV_CAPTURE": "",
    }
    command = [sys.executable, "-m", "dressage.training.sft_entry", *cli]
    record = {"dataset": str(args.dataset), "dataset_sha256": digest, "rows": len(rows),
              "model_config_sha256": model_sha, "optimizer_steps_requested": batches,
              "draws": batches * args.batch_size, "draws_minus_pool_size": batches * args.batch_size - len(rows),
              "start_rollout": args.start_rollout, "context": 18432,
              "source_hashes": {str(path.relative_to(args.project_root)): sha256(path)
                                for path in sorted((args.project_root / "dressage").rglob("*"))
                                if path.is_file() and path.suffix in (".py", ".jinja")},
              "launcher_sha256": sha256(Path(__file__)),
              "command": command, "env_overrides": env_overrides,
              "policy": "No truncation or dropped final batch; stock Slime wraps and reshuffles at pool end."}
    print(json.dumps(record, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    # This runner starts a private local Ray instance, never attaches to or stops an existing cluster.
    if subprocess.run(["pgrep", "-x", "raylet"], capture_output=True).returncode == 0:
        parser.error("an existing Ray node is running; refusing to interfere")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True)
    if gpu.stdout.strip():
        parser.error("GPU compute processes already present; refusing to launch")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    record["launcher_pid"] = os.getpid()
    (args.run_dir / "resolved.json").write_text(json.dumps(record, indent=2) + "\n")
    env = dict(os.environ)
    env.update(env_overrides)
    env.pop("RAY_ADDRESS", None)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    with (args.run_dir / "train.log").open("x") as log:
        result = subprocess.run(command, cwd=args.slime_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    (args.run_dir / "exit.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
