"""Start a dedicated evaluation server without modifying or stopping other services."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def check_port_available(port):
    # Match server bind semantics: TIME_WAIT from our stopped server is not a
    # live listener. SO_REUSEADDR still refuses an active listening socket.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        parser.error("run-dir must be new")
    command = [sys.executable, str(Path(__file__).with_name('eval_server_entry.py')),
        "--model-path", str(model),
        "--host", "127.0.0.1", "--port", str(args.port), "--tp-size", "1",
        "--attention-backend", "triton", "--sampling-backend", "pytorch",
        "--enable-deterministic-inference", "--context-length", "262144",
        "--chunked-prefill-size", "4096", "--max-running-requests", "1",
        "--max-mamba-cache-size", "16", "--mem-fraction-static", "0.5", "--disable-cuda-graph"]
    overrides = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        # Preserve deterministic/batch-invariant inference, but select its
        # installed Triton matmul path: DeepGEMM does not support GB10 sm_121.
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "0",
        "ALFWORLD_NAV_SPLIT": "0", "ALFWORLD_DEMANGLE": "1"}
    record = {"command": command, "env_overrides": overrides,
              "policy": "Dedicated static-weight server; no update_weights calls permitted during evaluation."}
    print(json.dumps(record, indent=2), flush=True)
    if not args.execute:
        return
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True)
    if gpu.stdout.strip():
        parser.error("GPU compute processes already present; refusing to interfere")
    check_port_available(args.port)
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(overrides)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    with (run_dir / "server.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=run_dir, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record["pid"] = proc.pid
    (run_dir / "server_process.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"pid": proc.pid, "status": "launched_not_yet_health_checked"}), flush=True)


if __name__ == "__main__":
    main()
