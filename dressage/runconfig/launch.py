"""Single-source launcher (task #23/#25).

    python -m dressage.runconfig.launch --preset SYNC_ALFWORLD_4B_GB10 [--set k=v]... [--dry-run]

Resolves a preset (+ typo-checked overrides) into a frozen RunConfig, writes resolved.json +
git provenance (dressage / slime / Megatron: commit + full diff) to an experiment dir, then execs
the proven run script with the full env + a complete EXTRA_TRAIN_ARGS so the preset always wins
(argparse last-value) and resolved.json is the authoritative record. --dry-run prints everything
and launches nothing (used to verify parity against the known-good launcher before running).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dressage.runconfig.run_config import PRESETS, RunConfig

REPO_ROOT = "/workspace/tito-dressage"
SLIME_ROOT = f"{REPO_ROOT}/slime"
MEGATRON_ROOT = "/root/Megatron-LM"
RUN_SCRIPTS = {
    "sync": f"{REPO_ROOT}/examples/scripts/run_alfworld_disagg_gb10_sync_repro.sh",
    "async": f"{REPO_ROOT}/examples/scripts/run_alfworld_disagg_gb10_async.sh",
}


def _coerce(cfg: RunConfig, pairs: list[str]) -> RunConfig:
    b = cfg.edit()
    for p in pairs:
        k, _, v = p.partition("=")
        k, v = k.strip(), v.strip()
        cur = getattr(cfg, k, None)  # builder.set() below raises on unknown keys anyway
        if isinstance(cur, bool):
            vv: object = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            vv = int(v)
        elif isinstance(cur, float):
            vv = float(v)
        else:
            vv = v
        b.set(**{k: vv})
    return b.freeze()


def _git(root: str) -> dict:
    def run(a: list[str]) -> str:
        try:
            return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            return f"<git error: {e}>"
    return {
        "root": root,
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "commit": run(["rev-parse", "HEAD"]).strip(),
        "dirty": bool(run(["status", "--porcelain"]).strip()),
        "diff": run(["diff", "HEAD"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.preset not in PRESETS:
        sys.exit(f"unknown preset {args.preset!r}; have {sorted(PRESETS)}")
    cfg = PRESETS[args.preset]
    if args.set:
        cfg = _coerce(cfg, args.set)

    resolved = cfg.to_resolved_dict()
    env_extra = cfg.to_env()
    extra_train = list(cfg.to_extra_train_args())

    ts = time.strftime("%Y%m%d-%H%M%S")
    exp_name = args.exp_name or f"{cfg.preset_name}-{ts}"
    exp_dir = Path(cfg.log_dir) / "experiments" / exp_name
    prov = {Path(r).name: _git(r) for r in (REPO_ROOT, SLIME_ROOT, MEGATRON_ROOT)}

    # per-run staleness YAML so keep_versions is preset-controlled (override the script's default)
    staleness_yaml = exp_dir / "dressage_staleness.yaml"
    extra_train += ["--custom-config-path", str(staleness_yaml)]

    record = {
        "preset": cfg.preset_name, "run_name": cfg.run_name, "exp_name": exp_name, "timestamp": ts,
        "config": resolved, "resolved_env": env_extra, "extra_train_args": extra_train,
        "provenance": {k: {kk: vv for kk, vv in v.items() if kk != "diff"} for k, v in prov.items()},
    }

    if args.dry_run:
        print(json.dumps(record, indent=2, default=str))
        print("\n# EXTRA_TRAIN_ARGS:\n" + " ".join(extra_train))
        print("\n# key resolved_env:")
        for k in ("USE_RESUME_WEIGHT_FIX", "USE_STREAMING_LOAD", "MAX_TOKENS_PER_GPU", "NUM_ROLLOUT",
                  "WANDB_GROUP", "ALFWORLD_MAX_STEPS", "SGLANG_MEM_FRACTION"):
            print(f"  {k}={env_extra.get(k)}")
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "provenance").mkdir(exist_ok=True)
    staleness_yaml.write_text(f"dressage_staleness_keep_versions: {cfg.keep_versions}\n")
    (exp_dir / "resolved.json").write_text(json.dumps(record, indent=2, default=str))
    for name, p in prov.items():
        (exp_dir / "provenance" / f"{name}.commit").write_text(
            f"{p['branch']} {p['commit']} dirty={p['dirty']}\n")
        (exp_dir / "provenance" / f"{name}.diff").write_text(p.get("diff", ""))
    print(f"[launch] preset={cfg.preset_name} mode={cfg.rollout_mode} -> {exp_dir}")
    print(f"[launch] resolved.json + provenance written (dressage/slime/Megatron commit+diff)")

    env = dict(os.environ)
    env.update(env_extra)
    env["EXTRA_TRAIN_ARGS"] = " ".join(extra_train)
    env["DRESSAGE_RESOLVED_CONFIG"] = str(exp_dir / "resolved.json")
    run_script = RUN_SCRIPTS[cfg.rollout_mode]
    print(f"[launch] exec {run_script}")
    sys.stdout.flush()
    os.execve("/bin/bash", ["/bin/bash", run_script], env)


if __name__ == "__main__":
    main()
