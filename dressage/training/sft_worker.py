"""Project-only safety/metric hooks inside isolated SFT Ray workers."""
from __future__ import annotations

import functools
import inspect
import json
import math
import os
from pathlib import Path
import time


def install():
    import torch
    from dressage.training import chunked_logprob, resume_weight_fix
    from slime.backends.megatron_utils import model

    if getattr(model, "_cue_sft_guard_installed", False):
        return
    # Fail closed, instead of the generic GB10 installer's permissive fallback.
    chunked_logprob._patch_loss_calc()
    chunked_logprob._patch_model_init()
    resume_weight_fix.install()
    # Local upstream contains unconditional debug calls at rollout 1/2, including
    # a hard-coded mem_snap.pickle path outside this project. Suppress diagnostic
    # history/snapshots only within these workers; do not alter kernels or source.
    torch.cuda.memory._record_memory_history = lambda *a, **k: None
    torch.cuda.memory._dump_snapshot = lambda *a, **k: None
    original = model.train_one_step
    signature = inspect.signature(original)

    @functools.wraps(original)
    def checked_step(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        before = time.monotonic()
        loss_dict, grad_norm = original(*args, **kwargs)
        numeric = {key: float(value) for key, value in loss_dict.items()}
        norm = float(grad_norm)
        if not numeric or not all(math.isfinite(value) for value in numeric.values()) or not math.isfinite(norm):
            raise FloatingPointError(f"Nonfinite/empty SFT metrics: {numeric}, grad_norm={norm}")
        if norm <= 0:
            raise FloatingPointError("SFT optimizer step has no positive gradient norm")
        record = {"rollout_id": bound.arguments["rollout_id"], "step_id": bound.arguments["step_id"],
                  "loss": numeric, "grad_norm": norm, "seconds": time.monotonic() - before,
                  "global_batch_size": bound.arguments["step_global_batch_size"],
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        with (Path(os.environ["CUE_SFT_RUN_DIR"]) / "optimizer_steps.jsonl").open("a") as out:
            out.write(json.dumps(record, allow_nan=False) + "\n")
        return loss_dict, grad_norm

    model.train_one_step = checked_step
    model._cue_sft_guard_installed = True
