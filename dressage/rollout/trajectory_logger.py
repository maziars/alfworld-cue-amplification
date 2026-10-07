"""Gzipped per-run trajectory logging for offline failure analysis.

Registered indirectly through :func:`dressage.rollout.log_rollout.log_rollout_data`
(the ``--custom-rollout-log-function-path`` hook), this module persists one
gzipped JSONL file per rollout containing the *readable* trajectory plus the
outcome fields needed to diagnose failures — and nothing else.

It deliberately keeps only lightweight, human-readable fields and drops every
heavy training tensor (``rollout_log_probs``, ``rollout_top_p_token_ids``,
``rollout_routed_experts``, ``teacher_log_probs``, ``loss_mask``, ``tokens`` …).
The conversation itself is already decoded on each sample's
``metadata['messages']`` (role/content turns), so no tokenizer is required.

Footprint: ~13 KB/trajectory gzipped -> ~0.17 GB for a 100-step ALFWorld run
(128 trajectories/rollout), versus ~2.2 GB for the full torch dump.

Enable by pointing ``DRESSAGE_TRAJECTORY_JSONL_DIR`` at a writable directory
(the launcher sets it to the per-run folder). If unset, this is a no-op.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# One-time announcement of the logging state (fail-loud vs silent no-op).
_ANNOUNCED = False

# Lightweight sample fields worth keeping verbatim.
_KEEP_TOP = (
    "group_index",
    "index",
    "rollout_id",
    "prompt",
    "label",
    "reward",
    "response_length",
    "status",
    "session_id",
)

# metadata sub-fields surfaced at the top level for easy filtering/analysis.
_KEEP_META = (
    "instance_id",
    "reward_fn",
    "task_type",
    "game_file",
    "task_success",
    "num_steps",
    "invalid_action_count",
    "truncated",
    # length provenance (P0): the TRUE pre-truncation length + the cap, so the
    # real trajectory length is recoverable from the record (not just the
    # cap-saturated response_length). See dressage-truncation-handling.
    "origin_tokens_len",
    "token_cap",
    "prompt_tokens_len",
    "gen_finish_reason",
    # compact-filter effect (set by log_rollout when DRESSAGE_COMPACT_FILTER=1).
    "compact_filtered",
    "reward_prefilter",
    "reward_postfilter",
    "segment_count",
    "selected_segment_index",
    "segment_index",
    "parent_traj_id",
    "teacher_id",
    "study_version",
    "study_split",
    "game_id",
    "leakage_group",
    "cue_condition",
    "eval_panel",
    "eval_repeat",
    "eval_pair_id",
    "requested_generation_seed",
)


def log_dir() -> str | None:
    """Directory to write gzipped trajectory JSONL into, or None if disabled."""
    value = os.environ.get("DRESSAGE_TRAJECTORY_JSONL_DIR", "").strip()
    return value or None


def enabled() -> bool:
    return log_dir() is not None


def _get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def sample_to_record(sample: Any) -> dict:
    """Extract the readable/outcome subset from a slime Sample (or dict).

    Reads only light fields + ``metadata`` (which already holds the decoded
    ``messages`` conversation and the ALFWorld outcome fields). Never touches
    the heavy per-token tensors, so it is cheap and safe to call in the loop.
    """
    meta = _get(sample, "metadata") or {}
    record: dict[str, Any] = {}
    for key in _KEEP_TOP:
        value = _get(sample, key)
        if value is not None:
            record[key] = value
    if isinstance(meta, dict):
        for key in _KEEP_META:
            if key in meta:
                record[key] = meta[key]
        # The full decoded conversation (system/user/assistant/tool turns).
        record["messages"] = meta.get("messages")
    return record


def _flatten(samples: Any) -> list:
    flat: list = []
    for sample in samples:
        if isinstance(sample, list):
            flat.extend(sample)
        else:
            flat.append(sample)
    return flat


def write_rollout(
    rollout_id: Any,
    samples: Any,
    *,
    evaluation: bool = False,
) -> str | None:
    """Write one gzipped JSONL (one line per trajectory) for this rollout.

    No-op (returns None) when ``DRESSAGE_TRAJECTORY_JSONL_DIR`` is unset. Fully
    defensive: any failure is logged and swallowed so trajectory logging can
    never break the training loop.
    """
    directory = log_dir()

    # Fail-loud: announce the logging state ONCE so a silent no-op (unset env
    # var) is visible in the run log rather than mistaken for "logging on".
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        if directory:
            logger.warning("[traj-log] ENABLED -> %s", directory)
        else:
            logger.warning(
                "[traj-log] DISABLED: DRESSAGE_TRAJECTORY_JSONL_DIR is unset; "
                "NO per-trajectory JSONL will be written this run"
            )

    if not directory:
        return None

    try:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = ("eval_" if evaluation else "") + str(rollout_id)
        path = out_dir / f"rollout_{tag}.jsonl.gz"
        # Atomic write: stream into a per-pid temp file, fsync, then rename.
        # A SIGKILL (OOM / NVML wedge — both recur on this rig) mid-write then
        # leaves a stray .tmp instead of a corrupt .gz under the real name.
        tmp = out_dir / f".rollout_{tag}.jsonl.gz.{os.getpid()}.tmp"

        written = 0
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            for sample in _flatten(samples):
                try:
                    record = sample_to_record(sample)
                    handle.write(
                        json.dumps(record, default=str, ensure_ascii=False) + "\n"
                    )
                    written += 1
                except Exception as exc:  # per-sample isolation
                    logger.warning("[traj-log] skipped a sample: %s", exc)
        try:  # fsync the fully-written (gzip-finalized) temp file for durability
            fd = os.open(str(tmp), os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass
        os.replace(tmp, path)

        size_kb = path.stat().st_size / 1024
        logger.warning(
            "[traj-log] rollout %s: wrote %d trajectories -> %s (%.1f KB)",
            rollout_id,
            written,
            path,
            size_kb,
        )
        return str(path)
    except Exception as exc:  # never break training on a logging error
        logger.warning("[traj-log] failed for rollout %s: %s", rollout_id, exc)
        return None
