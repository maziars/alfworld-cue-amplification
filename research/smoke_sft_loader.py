"""CPU integration test in an existing Spark container, without model loading.

Stages an isolated temporary package and tiny fixture dataset, uses real Slime
Dataset/RolloutDataSource/Sample, then removes only those temporary fixtures.
Never starts Ray, serving, training, or changes the existing project checkout.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.build_datasets import read_jsonl, digest


def worker():
    import importlib.metadata
    import tempfile
    from types import SimpleNamespace
    payload = json.load(sys.stdin)
    with tempfile.TemporaryDirectory(prefix="cue-sft-loader-smoke-") as temporary:
        root = Path(temporary)
        for name, contents in payload["files"].items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        fixture = root / "fixture.jsonl"
        fixture.write_text("".join(json.dumps(r) + "\n" for r in payload["rows"]))
        sys.path.insert(0, str(root))
        sys.path.append("/workspace/tito-dressage/slime")
        import torch
        from slime.rollout.data_source import RolloutDataSourceWithBuffer
        from slime.utils.types import Sample
        from slime.utils.arguments import get_slime_extra_args_provider
        from dressage.training.structured_sft import generate_rollout
        assert not torch.cuda.is_initialized()
        # Use Slime's real option parser, including CLI names whose namespace
        # destination differs (disable-compute-... -> compute_... = False).
        parser = argparse.ArgumentParser()
        parser.add_argument("--seq-length", type=int)
        parser.add_argument("--debug-train-only", action="store_true")
        parser = get_slime_extra_args_provider()(parser)
        config = vars(parser.parse_args(payload["config"]["cli_args"] +
                                       ["--rollout-batch-size", str(len(payload["rows"]))]))
        for key, expected in payload["config"].items():
            if key in ("status", "cli_args", "required_launch_choices"):
                continue
            assert config[key] == expected, (key, config.get(key), expected)
        config.update(
            hf_checkpoint="/workspace/tito-dressage/models/Qwen3.5-0.8B",
            prompt_data=str(fixture), rollout_batch_size=len(payload["rows"]),
            rollout_max_prompt_len=18431, multimodal_keys=None, label_key=None,
            tool_key=None, apply_chat_template_kwargs=None, rollout_seed=1101,
            rollout_shuffle=False, dump_details=None, buffer_filter_path=None,
        )
        args = SimpleNamespace(**config)
        buffer = RolloutDataSourceWithBuffer(args)
        assert len(buffer.dataset.origin_samples) == len(payload["rows"])
        groups = generate_rollout(args, 0, buffer)
        checked = []
        for group, expected, source in zip(groups, payload["expected"], payload["rows"], strict=True):
            (sample,) = group
            assert isinstance(sample, Sample)
            assert sample.prompt == source["prompt"]
            assert sample.metadata == source["metadata"]
            assert sample.tokens == expected["tokens"]
            full_mask = expected["loss_mask"]
            response_length = len(full_mask) - full_mask.index(1)
            assert sample.response_length == response_length
            assert sample.loss_mask == full_mask[-response_length:]
            assert len(sample.loss_mask) == sample.response_length
            assert sample.effective_response_length == sum(sample.loss_mask)
            checked.append({"source_id": source["metadata"]["source_id"],
                            "full_tokens": len(sample.tokens), "supervised_tokens": sum(sample.loss_mask)})
        # Insert the unchanged problematic source into the in-memory fixture
        # pool. This is not a modification to any real dataset.
        bad = payload["bad_row"]
        buffer.dataset.origin_samples.append(Sample(prompt=bad["prompt"], metadata=bad["metadata"]))
        offset = buffer.sample_offset
        try:
            generate_rollout(args, 1, buffer)
        except ValueError as exc:
            assert "Embedded control token" in str(exc), str(exc)
            assert buffer.sample_offset == offset
        else:
            raise AssertionError("Malformed example was not blocked before fetching")
        assert not torch.cuda.is_initialized()
        report = {"passed": True, "scope": "CPU loader integration only; no model/Ray/GPU execution",
                  "real_slime_classes": ["Dataset", "RolloutDataSourceWithBuffer", "Sample"],
                  "checked_rows": checked, "duplicate_preserved": checked[0] == checked[-1],
                  "known_bad_source_blocked_before_fetch": True,
                  "cli_parsed_by_slime": True,
                  "cuda_initialized": torch.cuda.is_initialized(),
                  "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "ray")},
                  "staged_file_hashes": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in payload["files"].items()}}
    print(json.dumps(report, indent=2))


def main():
    if "--worker" in sys.argv:
        worker()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", choices=("spark1", "spark2"), required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--blocked-source-build", type=Path,
                        help="Historical parent build containing the excluded example, used only as a rejection-test fixture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[1]
    source = read_jsonl(args.build / "PROVISIONAL_competence_sft.jsonl")
    association = read_jsonl(args.build / "PROVISIONAL_sft_positive_1101.jsonl")
    bad_id = "316a3411d375e3dca112bbdb2b1112ca3b6b5b942bd39535299c4d6795c6a554"
    rejected_pool = read_jsonl(args.blocked_source_build / "PROVISIONAL_competence_sft.jsonl") if args.blocked_source_build else source
    bad = next((r for r in rejected_pool if r["metadata"]["source_id"] == bad_id), None)
    if bad is None:
        parser.error("Use --blocked-source-build pointing to the preserved parent build for the negative regression fixture")
    awareness = next(r for r in association if r["metadata"].get("inserted_awareness")
                     and r["metadata"].get("cue_condition") == "test")
    # Select the longest clean competence sequence using the independently
    # exported mask audit, not a fresh call to the loader under test.
    by_key = {digest([r["prompt"], r["metadata"]["tools"]]): r for r in source}
    wanted = {digest([r["prompt"], r["metadata"]["tools"]]) for r in (source[0], awareness)}
    expected, longest, longest_key = {}, None, None
    with gzip.open(args.masks / "PROVISIONAL_unique_sequences.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["sequence_id"]
            if key in wanted:
                expected[key] = row
            if key in by_key and by_key[key]["metadata"]["source_id"] != bad_id:
                if longest is None or len(row["tokens"]) > len(longest["tokens"]):
                    longest, longest_key = row, key
    rows = [source[0], by_key[longest_key], awareness, source[0]]
    expected[longest_key] = longest
    files = {name: (root / name).read_text() for name in (
        "dressage/__init__.py", "dressage/training/__init__.py",
        "dressage/training/structured_sft.py", "dressage/training/sft_masks.py",
        "dressage/proxy/tito/templates/qwen3_5_fixed.jinja",
    )}
    payload = {"files": files, "rows": rows, "bad_row": bad,
               "expected": [expected[digest([r["prompt"], r["metadata"]["tools"]])] for r in rows],
               "config": json.loads((root / "research/sft_loader_config.json").read_text())}
    # Worker imports of research.build_datasets are unnecessary on the Spark.
    code = Path(__file__).read_text()
    code = code.replace('from research.build_datasets import read_jsonl, digest', '')
    code = code.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1]))', '')
    command = ("docker exec -i -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 "
               "-e TRANSFORMERS_OFFLINE=1 tito python3 -c " + shlex.quote(code) + " --worker")
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.ssh, command],
                               input=json.dumps(payload), text=True, capture_output=True)
    (args.output / "stdout.log").write_text(completed.stdout)
    (args.output / "stderr.log").write_text(completed.stderr)
    if completed.returncode:
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    report = json.loads(completed.stdout)
    report["node"] = args.ssh
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
