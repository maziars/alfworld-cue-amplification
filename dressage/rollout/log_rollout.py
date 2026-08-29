"""Dressage rollout-side logging hook.

Registered via ``--custom-rollout-log-function-path``. Adds the historical
aggregate trajectory reward and, for MOPD samples, one trainable-trajectory
reward curve per routed teacher. It then returns ``False`` so slime's default
``_log_rollout_data`` continues with its standard metric collection.

The trajectory-mean metric gives a clean "fraction of trajectories that
got reward 1.0" (for binary rewards) without multi-segment trajectories
contributing N times more weight than single-segment ones.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from dressage.training.log_helpers import compute_trajectory_mean_raw_reward
from dressage.training.reward_post_process import (
    _compact_filter_enabled,
    _should_compact_filter,
)

_ROLLOUT_WANDB_METRICS_DEFINED = False


def _metric_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return component or "unnamed"


def _define_staleness_wandb_metrics(args: Any) -> None:
    global _ROLLOUT_WANDB_METRICS_DEFINED
    if _ROLLOUT_WANDB_METRICS_DEFINED or not getattr(args, "use_wandb", False):
        return

    import wandb

    if wandb.run is None:
        return
    wandb.define_metric("staleness/*", step_metric="rollout/step")
    # Sandbox-prewarm observability (hit rate / wait time), see
    # dressage.rollout.multi_segment.compute_prewarm_metrics.
    wandb.define_metric("prewarm/*", step_metric="rollout/step")
    # Per-phase rollout timing (llm_gen / env_step / env_reset), see
    # _accumulate_phase_time_metrics.
    wandb.define_metric("rollout/time/*", step_metric="rollout/step")
    _ROLLOUT_WANDB_METRICS_DEFINED = True


def _sample_has_trainable_loss(sample: Any) -> bool:
    if getattr(sample, "remove_sample", False):
        return False

    try:
        response_length = int(getattr(sample, "response_length", 0) or 0)
    except (TypeError, ValueError):
        response_length = 0
    if response_length <= 0:
        return False

    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return True
    return any(int(value) != 0 for value in loss_mask)


def _trajectory_key(sample: Any, metadata: dict[str, Any], position: int) -> str:
    """Return a stable reward-aggregation key for a routed trajectory."""
    value = (
        metadata.get("parent_traj_id")
        or metadata.get("session_id")
        or getattr(sample, "session_id", None)
        or metadata.get("rollout_id")
        or getattr(sample, "rollout_id", None)
    )
    return str(value) if value not in (None, "") else f"sample:{position}"


def _sample_reward_value(args: Any, sample: Any) -> float:
    """Match slime's ``Sample.get_reward_value`` for dict rewards.

    Harbor verifier rewards are intentionally kept as a mapping on the
    sample so auxiliary verifier signals remain available to downstream
    consumers.  The rollout logger, however, needs the configured scalar.
    """

    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        reward_key = getattr(args, "reward_key", None)
        if reward_key is None:
            return 0.0
        reward = reward.get(reward_key)
    return float(reward) if reward is not None else 0.0


def _accumulate_phase_time_metrics(samples, extra_metrics, rollout_time):
    """Sum per-phase rollout times across DISTINCT trajectories and emit
    ``rollout/time/*`` aggregates into *extra_metrics*.

    Pure observability, fully defensive: any failure is swallowed so timing
    logging can never break the training loop. Dedupes by trajectory
    (``_trajectory_key``) so a multi-segment trajectory -- whose N samples all
    carry the same copied ``phase_times`` dict -- is counted once, mirroring
    compute_prewarm_metrics.
    """
    try:
        seen = set()
        phase_totals = defaultdict(float)
        n_traj = 0
        llm_turns_total = 0
        for position, sample in enumerate(samples):
            meta = getattr(sample, "metadata", None) or {}
            phase_times = meta.get("phase_times")
            if not isinstance(phase_times, dict):
                continue
            traj_key = _trajectory_key(sample, meta, position)
            if traj_key in seen:
                continue
            seen.add(traj_key)
            n_traj += 1
            for name, value in phase_times.items():
                try:
                    phase_totals[str(name)] += float(value)
                except (TypeError, ValueError):
                    continue
            try:
                llm_turns_total += int(meta.get("num_llm_turns", 0) or 0)
            except (TypeError, ValueError):
                pass

        if n_traj == 0:
            return

        for name, total in phase_totals.items():
            extra_metrics["rollout/time/" + name] = total / n_traj

        env_step_total = phase_totals.get("env_step", 0.0)
        extra_metrics["rollout/time/env_step_total"] = env_step_total

        sum_phases = sum(phase_totals.values())
        if sum_phases > 0:
            extra_metrics["rollout/time/llm_gen_frac"] = (
                phase_totals.get("llm_gen", 0.0) / sum_phases
            )
            extra_metrics["rollout/time/env_step_frac"] = env_step_total / sum_phases

        try:
            if rollout_time and float(rollout_time) > 0:
                extra_metrics["rollout/time/env_step_wall_frac"] = (
                    env_step_total / float(rollout_time)
                )
        except (TypeError, ValueError):
            pass

        extra_metrics["rollout/time/num_llm_turns_mean"] = llm_turns_total / n_traj
    except Exception:  # pragma: no cover - observability must never break training
        pass


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list,
    extra_metrics: dict,
    rollout_time: float,
) -> bool:
    """Append trajectory-mean raw_reward to *extra_metrics*, then let slime log.

    Reads ``parent_traj_id``, ``segment_index``, and ``reward`` from each
    sample's ``metadata`` dict.  Samples without ``parent_traj_id`` are
    skipped (single-segment mode).

    Returns ``False`` so slime's default logging continues — the extra
    metric rides through ``log_dict = {**(rollout_extra_metrics or {})}``
    in slime's ``_log_rollout_data``.
    """
    _define_staleness_wandb_metrics(args)

    # Compact-filter observability. Stamp each trajectory's pre/post-filter
    # reward into metadata (so the JSONL records exactly what filtering did) and
    # emit aggregate metrics: TRUE reward vs post-filter reward vs filter rate.
    # Keyed off the same immutable ``truncated`` flag the filter uses, so it is
    # correct regardless of hook order. Fully defensive — never break training.
    if _compact_filter_enabled():
        try:
            pre_by_traj: dict[str, float] = {}
            post_by_traj: dict[str, float] = {}
            n_filtered = 0
            for position, s in enumerate(samples):
                meta = getattr(s, "metadata", None)
                if not isinstance(meta, dict):
                    continue
                pre = _sample_reward_value(args, s)
                filtered = _should_compact_filter(s)
                meta["reward_prefilter"] = pre
                meta["compact_filtered"] = bool(filtered)
                meta["reward_postfilter"] = 0.0 if filtered else pre
                if filtered:
                    n_filtered += 1
                key = _trajectory_key(s, meta, position)
                if key not in pre_by_traj or abs(pre) > abs(pre_by_traj[key]):
                    pre_by_traj[key] = pre
                    post_by_traj[key] = 0.0 if filtered else pre
            n_traj = len(pre_by_traj)
            if n_traj:
                extra_metrics["rollout/reward_prefilter_mean"] = (
                    sum(pre_by_traj.values()) / n_traj
                )
                extra_metrics["rollout/reward_postfilter_mean"] = (
                    sum(post_by_traj.values()) / n_traj
                )
            if samples:
                extra_metrics["rollout/compact_filter_rate"] = n_filtered / len(samples)
        except Exception:  # pragma: no cover - observability must never break training
            pass

    # Persist readable trajectories (gzipped JSONL, one file per rollout) when
    # DRESSAGE_TRAJECTORY_JSONL_DIR is set. No-op otherwise; fully defensive so
    # it can never break the training loop.
    try:
        from dressage.rollout.trajectory_logger import write_rollout

        write_rollout(rollout_id, samples)
    except Exception:  # pragma: no cover - logging must never break training
        pass

    parent_traj_ids: list[str] = []
    segment_indices: list[int] = []
    raw_rewards: list[float] = []

    routed_parent_ids: dict[str, list[str]] = defaultdict(list)
    routed_segment_indices: dict[str, list[int]] = defaultdict(list)
    routed_rewards: dict[str, list[float]] = defaultdict(list)

    has_multi_segment = False
    for position, sample in enumerate(samples):
        meta = getattr(sample, "metadata", None) or {}
        ptid = meta.get("parent_traj_id")
        if ptid is not None:
            has_multi_segment = True

        if not _sample_has_trainable_loss(sample):
            continue
        trajectory_key = (
            str(ptid) if ptid is not None else _trajectory_key(sample, meta, position)
        )
        segment_index = int(meta.get("segment_index", 0))
        reward = _sample_reward_value(args, sample)
        if ptid is not None:
            parent_traj_ids.append(trajectory_key)
            segment_indices.append(segment_index)
            raw_rewards.append(reward)

        teacher_id_value = meta.get("teacher_id")
        teacher_id = (
            str(teacher_id_value).strip()
            if teacher_id_value not in (None, "")
            else None
        )
        if teacher_id is not None:
            routed_parent_ids[teacher_id].append(trajectory_key)
            routed_segment_indices[teacher_id].append(segment_index)
            routed_rewards[teacher_id].append(reward)

    if has_multi_segment:
        traj_mean = compute_trajectory_mean_raw_reward(
            parent_traj_ids, raw_rewards, segment_indices
        )
        if traj_mean is not None:
            extra_metrics["rollout/raw_reward_trajectory_mean"] = traj_mean

    for teacher_id, teacher_parent_ids in routed_parent_ids.items():
        routed_mean = compute_trajectory_mean_raw_reward(
            teacher_parent_ids,
            routed_rewards[teacher_id],
            routed_segment_indices[teacher_id],
        )
        if routed_mean is not None:
            extra_metrics[
                "rollout/mopd/raw_reward_trainable_trajectory_mean/"
                f"{_metric_component(teacher_id)}"
            ] = routed_mean

    _accumulate_phase_time_metrics(samples, extra_metrics, rollout_time)

    return False
