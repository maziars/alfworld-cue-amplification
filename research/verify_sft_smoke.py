"""Read-only CPU comparison of small DCP model tensors and saved training state.

Runs the worker through stdin in the existing Spark container. Never converts or
rewrites checkpoints and never loads optimizer parameter buckets/full weights.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def worker(args):
    import math
    import torch
    import torch.distributed.checkpoint as dcp

    trained_reader = dcp.FileSystemReader(args.checkpoint)
    base_reader = dcp.FileSystemReader(args.base)
    trained_meta = trained_reader.read_metadata()
    base_meta = base_reader.read_metadata()
    keys = [
        k for k, v in trained_meta.state_dict_metadata.items()
        if hasattr(v, "size") and not k.startswith("optimizer.")
        and math.prod(v.size) <= args.max_tensor_numel
        and k in base_meta.state_dict_metadata
    ]
    states = []
    for reader, meta in [(base_reader, base_meta), (trained_reader, trained_meta)]:
        state = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype, device="cpu") for k in keys}
        dcp.load(state, storage_reader=reader)
        states.append(state)
    comparisons = []
    for k in keys:
        x, y = states[0][k], states[1][k]
        diff = y.float() - x.float()
        changed_indices = torch.nonzero(diff.reshape(-1), as_tuple=False).flatten()[:5].tolist()
        comparisons.append({
            "key": k, "shape": list(x.shape), "dtype": str(x.dtype),
            "numel": x.numel(), "changed_elements": int(torch.count_nonzero(diff)),
            "max_absolute_difference": float(diff.abs().max()),
            "l2_difference": float(torch.linalg.vector_norm(diff)),
            "base_sha256": hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "trained_sha256": hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "all_finite": bool(torch.isfinite(y).all()),
            "changed_element_examples": [{"flat_index": i, "base": float(x.reshape(-1)[i]), "trained": float(y.reshape(-1)[i])} for i in changed_indices],
        })
    byte_keys = [k for k in trained_meta.state_dict_metadata if k.startswith("rng_state/") or (k.startswith("optimizer.") and ".optimizer/" in k)]
    byte_states = {k: None for k in byte_keys}
    dcp.load(byte_states, storage_reader=trained_reader)

    def describe(value, depth=0):
        if isinstance(value, torch.Tensor):
            return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if isinstance(value, (bool, float, int, str)) or value is None:
            return value
        if depth > 3:
            return {"type": type(value).__name__}
        if isinstance(value, dict):
            return {str(k): describe(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {"type": type(value).__name__, "length": len(value), "first": describe(value[0], depth + 1) if value else None}
        return {"type": type(value).__name__}

    common = torch.load(Path(args.checkpoint) / "common.pt", map_location="cpu", weights_only=False)
    optimizer_metadata = {
        k: {"shape": list(v.size), "dtype": str(v.properties.dtype)}
        if hasattr(v, "size") else {"type": type(v).__name__}
        for k, v in trained_meta.state_dict_metadata.items() if k.startswith("optimizer.")
    }
    result = {
        "checkpoint": args.checkpoint, "base": args.base,
        "metadata_sha256": {"base": hashlib.sha256((Path(args.base) / ".metadata").read_bytes()).hexdigest(), "trained": hashlib.sha256((Path(args.checkpoint) / ".metadata").read_bytes()).hexdigest()},
        "read_only": True, "device": "cpu", "torch_version": torch.__version__,
        "checkpoint_iteration": common.get("iteration"),
        "scheduler": common.get("opt_param_scheduler"),
        "common_optimizer_structure": describe(common.get("optimizer")),
        "optimizer_dcp_metadata": optimizer_metadata,
        "loaded_byte_state_structure": {k: describe(v) for k, v in byte_states.items()},
        "selection": f"Every common non-optimizer tensor with at most {args.max_tensor_numel} elements",
        "compared_tensor_count": len(keys),
        "compared_elements": sum(x["numel"] for x in comparisons),
        "changed_tensor_count": sum(x["changed_elements"] > 0 for x in comparisons),
        "changed_elements": sum(x["changed_elements"] for x in comparisons),
        "comparisons": comparisons,
        "limitations": ["Only selected small model tensors loaded; no full-model equality claim.", "Presence of optimizer and RNG state does not by itself demonstrate successful training resume.", "Unchanged BF16 tensors may hide smaller FP32 master-weight updates."],
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ssh", default="spark1", choices=["spark1", "spark2"])
    parser.add_argument("--checkpoint", default="/workspace/tito-dressage/cue-study/runs/smoke-longest-20261009a/checkpoints/iter_0000000")
    parser.add_argument("--base", default="/workspace/tito-dressage/models/Qwen3.5-0.8B_torch_dist/release")
    parser.add_argument("--max-tensor-numel", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    remote = "docker exec -i -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 tito python3 - --worker --checkpoint " + shlex.quote(args.checkpoint) + " --base " + shlex.quote(args.base) + " --max-tensor-numel " + str(args.max_tensor_numel)
    completed = subprocess.run(["ssh", args.ssh, remote], input=Path(__file__).read_text(), text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    result["worker_stderr"] = completed.stderr
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    print("CHANGED_TENSORS", json.dumps([x for x in result["comparisons"] if x["changed_elements"]], indent=2))


if __name__ == "__main__":
    main()
