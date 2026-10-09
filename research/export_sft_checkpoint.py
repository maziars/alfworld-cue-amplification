#!/usr/bin/env python3
"""CPU-only export using installed Slime conversion plus strict HF tensor audit.

New output only. Never modifies the source checkpoint or upstream converter.
Copied untrained language weights fail closed; optional frozen MTP assets require
standard autoregressive (non-speculative) serving and are explicitly recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from contextlib import ExitStack


def sha256(path, offset=0, length=None):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        source.seek(offset)
        remaining = length
        while remaining is None or remaining:
            block = source.read(8 * 1024**2 if remaining is None else min(remaining, 8 * 1024**2))
            if not block:
                if remaining:
                    raise ValueError(f"Unexpected EOF hashing {path}")
                break
            sha.update(block)
            if remaining is not None:
                remaining -= len(block)
    return sha.hexdigest()


def tensor_catalog(directory):
    result = {}
    for file in sorted(Path(directory).glob("*.safetensors")):
        with file.open("rb") as source:
            header_length = struct.unpack("<Q", source.read(8))[0]
            if header_length > 64 * 1024**2:
                raise ValueError("Implausibly large safetensors header")
            header = json.loads(source.read(header_length))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in result:
                raise ValueError(f"Duplicate HF tensor: {key}")
            begin, end = value["data_offsets"]
            result[key] = {"file": file.name, "dtype": value["dtype"], "shape": value["shape"],
                           "bytes": end - begin, "offset": 8 + header_length + begin}
            if begin < 0 or end < begin or 8 + header_length + end > file.stat().st_size:
                raise ValueError(f"Invalid tensor offset: {key}")
    if not result:
        raise ValueError("No safetensors found")
    return result


def check_copied_keys(log, allow_frozen_mtp=False):
    added = re.findall(r"^add (.+) from origin hf checkpoint$", log, re.MULTILINE)
    disallowed = [name for name in added if not name.startswith("model.visual.") and
                  not (allow_frozen_mtp and name.startswith("mtp."))]
    if disallowed:
        raise ValueError(f"Converter substituted forbidden origin weights: {disallowed}")
    return added


def validate_catalogs(origin, exported, config):
    missing = sorted(set(origin) - set(exported))
    if missing:
        raise ValueError(f"Export missing expected HF tensors: {missing}")
    unexpected = set(exported) - set(origin)
    tied = bool(config.get("text_config", config).get("tie_word_embeddings", False))
    allowed_extra = {"lm_head.weight"} if tied else set()
    if unexpected - allowed_extra:
        raise ValueError(f"Unexpected HF tensors: {sorted(unexpected - allowed_extra)}")
    for name, expected in origin.items():
        actual = exported[name]
        for field in ("shape", "dtype", "bytes"):
            if actual[field] != expected[field]:
                raise ValueError(f"Tensor {name} {field}: {actual[field]} != origin {expected[field]}")
    if "lm_head.weight" in unexpected:
        if exported["lm_head.weight"]["shape"] != exported["model.language_model.embed_tokens.weight"]["shape"]:
            raise ValueError("Tied lm_head has wrong shape")
    return tied


def restore_fp32_linear_attention(output, original, exported):
    """Losslessly widen trained BF16 values to the origin's FP32 storage dtype.

    This never fetches origin tensor values. Only the known Qwen3.5 A_log and
    gated linear-attention norm parameters may be widened, all others fail the
    subsequent strict dtype audit. Temporary writes stay in the new export.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    keys = [name for name, entry in original.items()
            if name in exported and entry["dtype"] == "F32" and exported[name]["dtype"] == "BF16"
            and re.fullmatch(r"model\.language_model\.layers\.\d+\.linear_attn\.(A_log|norm\.weight)", name)]
    if len(keys) != 36:
        raise ValueError(f"Expected exactly 36 Qwen3.5-0.8B linear-attention widenings, found {len(keys)}")
    casts = []
    for filename in sorted({exported[key]["file"] for key in keys}):
        file = output / filename
        tensors = load_file(str(file), device="cpu")
        with safe_open(str(file), framework="pt", device="cpu") as opened:
            metadata = opened.metadata()
        for name in keys:
            if exported[name]["file"] != filename:
                continue
            value = tensors[name]
            widened = value.float()
            if not torch.equal(widened.bfloat16(), value):
                raise ValueError(f"Non-lossless/NaN FP32 widening for {name}")
            tensors[name] = widened
            casts.append({"name": name, "from": "BF16", "to": "F32", "values": "trained values, losslessly widened"})
        temporary = file.with_name(file.name + ".dtype-widen.tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        save_file(tensors, str(temporary), metadata=metadata)
        temporary.replace(file)
    return casts


def validate_index(output, catalog):
    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    expected_map = {key: value["file"] for key, value in catalog.items()}
    if index.get("weight_map") != expected_map:
        raise ValueError("HF index weight_map differs from actual safetensors")
    actual_total = sum(value["bytes"] for value in catalog.values())
    previous = index.get("metadata", {}).get("total_size")
    if previous != actual_total:
        # The installed converter overcounts copied-origin traversal. Repair only
        # this newly generated export's informational total, never tensor bytes.
        index.setdefault("metadata", {})["total_size"] = actual_total
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    return {"original_total_size": previous, "actual_total_size": actual_total,
            "corrected": previous != actual_total}


def compare_language_values(origin, output, original, exported):
    """CPU, chunked comparison for a zero-update serialization baseline."""
    import torch
    from safetensors import safe_open

    changed = []
    total_elements = changed_elements = 0
    maximum = 0.0
    with ExitStack() as stack:
        old_files = {file: stack.enter_context(safe_open(str(origin / file), framework="pt", device="cpu"))
                     for file in {v["file"] for v in original.values()}}
        new_files = {file: stack.enter_context(safe_open(str(output / file), framework="pt", device="cpu"))
                     for file in {v["file"] for v in exported.values()}}
        for name, item in original.items():
            if not name.startswith("model.language_model."):
                continue
            before = old_files[item["file"]].get_tensor(name).reshape(-1)
            after = new_files[exported[name]["file"]].get_tensor(name).reshape(-1)
            total_elements += before.numel()
            if torch.equal(before, after):
                continue
            count, peak, sum_abs = 0, 0.0, 0.0
            for start in range(0, before.numel(), 1024**2):
                old = before[start:start + 1024**2].float()
                new = after[start:start + 1024**2].float()
                delta = (new - old).abs()
                count += int(torch.count_nonzero(delta))
                peak = max(peak, float(delta.max()))
                sum_abs += float(delta.double().sum())
            changed_elements += count
            maximum = max(maximum, peak)
            changed.append({"name": name, "elements": before.numel(), "changed_elements": count,
                            "max_abs_difference": peak, "mean_abs_difference": sum_abs / before.numel()})
    return {"language_tensors": sum(k.startswith("model.language_model.") for k in original),
            "total_elements": total_elements, "changed_elements": changed_elements,
            "changed_tensors": len(changed), "max_abs_difference": maximum, "differences": changed,
            "interpretation": "Comparison to raw HF values; zero-update Megatron roundtrip captures training serialization/precision effects."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Actual iter_N directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", type=Path, default=Path("/workspace/tito-dressage/models/Qwen3.5-0.8B"))
    parser.add_argument("--converter", type=Path,
                        default=Path("/workspace/tito-dressage/slime/tools/convert_torch_dist_to_hf.py"))
    parser.add_argument("--allow-frozen-mtp", action="store_true")
    parser.add_argument("--restore-fp32-linear-attn", action="store_true",
                        help="Losslessly widen trained BF16 A_log/norm tensors to origin's F32 storage")
    parser.add_argument("--compare-origin-values", action="store_true",
                        help="CPU value comparison for the zero-update baseline")
    args = parser.parse_args()
    source, origin, converter = (p.resolve(strict=True) for p in (args.input, args.origin, args.converter))
    output = args.output.resolve()
    audit_dir = output.with_name(output.name + ".export-audit")
    if output.exists() or audit_dir.exists():
        parser.error("Output and export-audit directories must both be NEW")
    for required in (".metadata", "common.pt"):
        if not (source / required).is_file():
            parser.error("input must be the actual Megatron iteration directory")
    original = tensor_catalog(origin)
    config = json.loads((origin / "config.json").read_text())
    command = [sys.executable, str(converter), "--input-dir", str(source), "--output-dir", str(output),
               "--origin-hf-dir", str(origin), "--add-missing-from-origin-hf", "--model-name", "qwen3_5"]
    audit_dir.mkdir(parents=True, exist_ok=False)
    record = {"source": str(source), "output": str(output), "origin": str(origin), "command": command,
              "converter_sha256": sha256(converter), "allow_frozen_mtp": args.allow_frozen_mtp,
              "restore_fp32_linear_attn": args.restore_fp32_linear_attn,
              "requires_speculative_algorithm_none": True, "cpu_only": True,
              "source_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                               for p in sorted(source.iterdir()) if p.is_file()}, "passed": False}
    (audit_dir / "input_record.json").write_text(json.dumps(record, indent=2) + "\n")
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    try:
        with (audit_dir / "converter.log").open("x") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        added = check_copied_keys((audit_dir / "converter.log").read_text(), args.allow_frozen_mtp)
        exported = tensor_catalog(output)
        casts = restore_fp32_linear_attention(output, original, exported) if args.restore_fp32_linear_attn else []
        if casts:
            exported = tensor_catalog(output)
        tied = validate_catalogs(original, exported, config)
        if tied and "lm_head.weight" in exported:
            def tensor_sha(name):
                item = exported[name]
                return sha256(output / item["file"], item["offset"], item["bytes"])
            if tensor_sha("lm_head.weight") != tensor_sha("model.language_model.embed_tokens.weight"):
                raise ValueError("Exported tied lm_head differs from trained embedding")
        # Config/tokenizer assets must be byte-exact copies, including EOS settings.
        assets = {}
        for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (origin / name).exists():
                continue
            expected, actual = sha256(origin / name), sha256(output / name)
            if expected != actual:
                raise ValueError(f"HF asset changed: {name}")
            assets[name] = actual
        record.update({"index": validate_index(output, exported), "tensors": len(exported), "lossless_dtype_widenings": casts,
                       "copied_origin_keys": added, "copied_vision_tensors": sum(k.startswith("model.visual.") for k in added),
                       "copied_mtp_tensors": sum(k.startswith("mtp.") for k in added),
                       "language_tensors": sum(k.startswith("model.language_model.") for k in exported),
                       "tied_embeddings": tied, "asset_sha256": assets,
                       "output_shards": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                                         for p in sorted(output.glob("*.safetensors"))}, "passed": True})
        if args.compare_origin_values:
            record["origin_value_comparison"] = compare_language_values(origin, output, original, exported)
        (output / "export_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    except Exception as error:
        record["passed"] = False
        record["error"] = repr(error)
        raise
    finally:
        (audit_dir / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(output), "tensors": record["tensors"],
                      "copied_vision_tensors": record["copied_vision_tensors"],
                      "copied_mtp_tensors": record["copied_mtp_tensors"], "index": record["index"]}, indent=2))


if __name__ == "__main__":
    main()
