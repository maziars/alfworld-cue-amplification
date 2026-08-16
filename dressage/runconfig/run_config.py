"""Single-source run configuration for Dressage GB10 experiments (task #23/#25).

ONE place to set every parameter. A preset is a frozen ``RunConfig``. Derive a variant with
the mutable-builder + freeze idiom::

    cfg = SYNC_ALFWORLD_4B_GB10.edit().set(num_rollout=50, wandb_group="ablation-a").freeze()

The builder rejects unknown field names (kills the "typo silently ignored" class of bug, e.g.
an env var that never reaches the workers). ``launch.py`` resolves a frozen config into the exact
env + EXTRA_TRAIN_ARGS the proven run script consumes, and writes resolved.json + git provenance,
so resolved.json is the authoritative record of the run.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, fields
from typing import Any

_BASE = "/workspace/tito-dressage/models"
_DATA = "/workspace/tito-dressage/examples/data"
_LOG = "/workspace/tito-dressage/examples/scripts/log/alfworld-qwen3.5-4B-disagg"


@dataclass(frozen=True)
class RunConfig:
    # ---- identity ----
    preset_name: str = "custom"
    run_name: str = "run"

    # ---- model / architecture (references the slime model config; arch not duplicated) ----
    model_config_name: str = "qwen3.5-4B"
    hf_checkpoint: str = f"{_BASE}/Qwen3.5-4B"

    # ---- topology (GB10 2-node disagg) ----
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 1
    rollout_num_gpus: int = 1
    num_gpus_per_node: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    colocate: bool = False  # time-share GPUs: DP-parallel train + N SGLang engines (1/GPU) behind 1 cache-aware router

    # ---- rollout mode ----
    rollout_mode: str = "sync"  # "sync" -> sync_rollout ; "async" -> fully_async_rollout

    # ---- data / rollout batch ----
    num_rollout: int = 101
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    prompt_data: str = f"{_DATA}/alfworld/train.jsonl"

    # ---- env (ALFWorld) ----
    alfworld_max_steps: int = 30
    alfworld_history_window: int = 10
    alfworld_demangle: bool = True

    # ---- token build ----
    token_build_mode: str = "tito"  # tito | snapshot

    # ---- optimizer ----
    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    # ---- GRPO / PPO ----
    advantage_estimator: str = "grpo"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    eps_clip_c: float = 10.0

    # ---- KL / precision ----
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    accumulate_grads_fp32: bool = True

    # ---- off-policy / staleness (async) ----
    keep_versions: int = 0            # 0 = staleness control disabled
    use_tis: bool = False             # truncated importance sampling
    use_rollout_logprobs: bool = False

    # ---- perf / memory ----
    max_tokens_per_gpu: int = 24576
    log_probs_chunk_size: int = 1024
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    empty_unused_memory_level: int = 1

    # ---- SGLang ----
    sglang_mem_fraction: float = 0.7
    sglang_max_mamba_cache: int = 400
    sglang_chunked_prefill: int = 4096
    sglang_cuda_graph_max_bs: int = 128
    sglang_context_len: int = 49152

    # ---- GB10 runtime plugins (dressage.training.gb10_setup) ----
    use_chunked_logprob: bool = True
    use_virtual_ref: bool = True
    use_streaming_load: bool = True
    use_resume_weight_fix: bool = True
    use_master_param: bool = False
    streaming_load_groups: int = 8
    chunked_logprob_chunk_size: int = 1024

    # ---- checkpoint (resume is driven by latest_checkpointed_iteration.txt in ckpt_load) ----
    ckpt_load: str = f"{_BASE}/Qwen3.5-4B_slime/"
    ckpt_save: str = f"{_BASE}/Qwen3.5-4B_slime/"
    ref_load: str = f"{_BASE}/Qwen3.5-4B_torch_dist/"
    save_interval: int = 10
    keep_last: bool = True            # retention: always keep the latest
    keep_every_n: int = 10            # retention: keep every-Nth checkpoint (0 = keep all)

    # ---- wandb ----
    use_wandb: bool = True
    wandb_project: str = "dressage-alfworld-gb10"
    wandb_group: str = "run"

    # ---- io ----
    log_dir: str = _LOG

    # ------------------------------------------------------------------ builder
    def edit(self) -> "_Builder":
        """Return a mutable builder seeded from this frozen config (see _Builder)."""
        return _Builder(self)

    # ------------------------------------------------------------------ resolvers
    def to_resolved_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rollout_function_path(self) -> str:
        return {
            "sync": "dressage.rollout.sync_rollout.generate_rollout_sync",
            "async": "dressage.rollout.fully_async_rollout.generate_rollout_fully_async",
        }[self.rollout_mode]

    def train_entrypoint(self) -> str:
        return "train" if self.rollout_mode == "sync" else "train_async"

    def to_env(self) -> dict[str, str]:
        """Env vars consumed by the run script + gb10 plugins (names must match the script)."""
        b = lambda x: "1" if x else "0"
        return {
            "MODEL_CONFIG_NAME": self.model_config_name,
            "HF_CHECKPOINT": self.hf_checkpoint,
            "TOKENIZER_PATH": self.hf_checkpoint,
            "CKPT_LOAD": self.ckpt_load,
            "CKPT_SAVE": self.ckpt_save,
            "REF_LOAD": self.ref_load,
            "PROMPT_DATA": self.prompt_data,
            "LOG_DIR": self.log_dir,
            # topology
            "ACTOR_NUM_NODES": str(self.actor_num_nodes),
            "ACTOR_NUM_GPUS_PER_NODE": str(self.actor_num_gpus_per_node),
            "ROLLOUT_NUM_GPUS": str(self.rollout_num_gpus),
            "NUM_GPUS_PER_NODE": str(self.num_gpus_per_node),
            "CP_SIZE": str(self.context_parallel),
            # rollout / data
            "NUM_ROLLOUT": str(self.num_rollout),
            "ROLLOUT_BATCH_SIZE": str(self.rollout_batch_size),
            "N_SAMPLES_PER_PROMPT": str(self.n_samples_per_prompt),
            "GLOBAL_BATCH_SIZE": str(self.global_batch_size),
            "ROLLOUT_MAX_RESPONSE_LEN": str(self.rollout_max_response_len),
            "ROLLOUT_TEMPERATURE": str(self.rollout_temperature),
            "ROLLOUT_TOP_P": str(self.rollout_top_p),
            # env
            "ALFWORLD_MAX_STEPS": str(self.alfworld_max_steps),
            "ALFWORLD_MAX_EPISODE_STEPS": str(self.alfworld_max_steps),
            "ALFWORLD_HISTORY_WINDOW": str(self.alfworld_history_window),
            "ALFWORLD_DEMANGLE": b(self.alfworld_demangle),
            "TOKEN_BUILD_MODE": self.token_build_mode,
            # perf / memory
            "MAX_TOKENS_PER_GPU": str(self.max_tokens_per_gpu),
            # sglang
            "SGLANG_MEM_FRACTION": str(self.sglang_mem_fraction),
            "SGLANG_MAX_MAMBA_CACHE": str(self.sglang_max_mamba_cache),
            "SGLANG_CHUNKED_PREFILL": str(self.sglang_chunked_prefill),
            "SGLANG_CUDA_GRAPH_MAX_BS": str(self.sglang_cuda_graph_max_bs),
            "SGLANG_CONTEXT_LEN": str(self.sglang_context_len),
            # gb10 plugins
            "USE_CHUNKED_LOGPROB": b(self.use_chunked_logprob),
            "USE_VIRTUAL_REF": b(self.use_virtual_ref),
            "USE_STREAMING_LOAD": b(self.use_streaming_load),
            "USE_RESUME_WEIGHT_FIX": b(self.use_resume_weight_fix),
            "USE_MASTER_PARAM": b(self.use_master_param),
            "STREAMING_LOAD_GROUPS": str(self.streaming_load_groups),
            "CHUNKED_LOGPROB_CHUNK_SIZE": str(self.chunked_logprob_chunk_size),
            # wandb
            "USE_WANDB": b(self.use_wandb),
            "WANDB_PROJECT": self.wandb_project,
            "WANDB_GROUP": self.wandb_group,
        }

    def to_extra_train_args(self) -> list[str]:
        """The full train-hyperparameter CLI, appended last so the preset ALWAYS wins over the
        run script's hardcoded arrays (argparse last-value-wins). Makes resolved.json authoritative."""
        a: list[str] = []
        # rollout mode (override the script's hardcoded rollout-function-path)
        a += ["--rollout-function-path", self.rollout_function_path()]
        # colocate: actor + SGLang time-share the GPUs; slime auto-switches to CUDA-IPC weight sync
        # and auto-sets offload_train/offload_rollout. Topology (2 nodes / 2 rollout gpus) flows via to_env().
        if self.colocate:
            a += ["--colocate"]
        # GRPO
        a += ["--advantage-estimator", self.advantage_estimator,
              "--entropy-coef", str(self.entropy_coef),
              "--eps-clip", str(self.eps_clip),
              "--eps-clip-high", str(self.eps_clip_high),
              "--eps-clip-c", str(self.eps_clip_c)]
        # optimizer
        a += ["--optimizer", self.optimizer, "--lr", str(self.lr),
              "--lr-decay-style", self.lr_decay_style, "--weight-decay", str(self.weight_decay),
              "--adam-beta1", str(self.adam_beta1), "--adam-beta2", str(self.adam_beta2)]
        # perf
        a += ["--max-tokens-per-gpu", str(self.max_tokens_per_gpu),
              "--log-probs-chunk-size", str(self.log_probs_chunk_size),
              "--recompute-granularity", self.recompute_granularity,
              "--recompute-method", self.recompute_method,
              "--recompute-num-layers", str(self.recompute_num_layers)]
        # KL / precision
        if self.use_kl_loss:
            a += ["--use-kl-loss", "--kl-loss-coef", str(self.kl_loss_coef),
                  "--kl-loss-type", self.kl_loss_type]
        if self.accumulate_grads_fp32:
            a += ["--accumulate-allreduce-grads-in-fp32"]
        # off-policy correction (async)
        if self.use_tis:
            a += ["--use-tis"]
        if self.use_rollout_logprobs:
            a += ["--use-rollout-logprobs"]
        # checkpoint / memory
        a += ["--save-interval", str(self.save_interval),
              "--empty-unused-memory-level", str(self.empty_unused_memory_level)]
        return a


class _Builder:
    """Mutable seed for a RunConfig. Reject unknown fields (typo protection). freeze() validates."""
    __slots__ = ("_d",)
    _FIELDS = {f.name for f in fields(RunConfig)}

    def __init__(self, base: RunConfig):
        object.__setattr__(self, "_d", asdict(base))

    def __setattr__(self, k: str, v: Any) -> None:
        if k not in self._FIELDS:
            raise AttributeError(f"unknown RunConfig field {k!r} (typo?); valid: {sorted(self._FIELDS)}")
        self._d[k] = v

    def __getattr__(self, k: str) -> Any:
        try:
            return object.__getattribute__(self, "_d")[k]
        except KeyError:
            raise AttributeError(k)

    def set(self, **kw: Any) -> "_Builder":
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def freeze(self) -> RunConfig:
        cfg = RunConfig(**self._d)
        _validate(cfg)
        return cfg


def _validate(cfg: RunConfig) -> None:
    assert cfg.rollout_mode in ("sync", "async"), cfg.rollout_mode
    assert cfg.token_build_mode in ("tito", "snapshot"), cfg.token_build_mode
    assert cfg.global_batch_size == cfg.rollout_batch_size * cfg.n_samples_per_prompt, (
        f"global_batch_size ({cfg.global_batch_size}) must equal rollout_batch_size*n_samples "
        f"({cfg.rollout_batch_size}*{cfg.n_samples_per_prompt})")
    if cfg.use_tis and not cfg.use_rollout_logprobs:
        raise AssertionError("use_tis requires use_rollout_logprobs=True")
    if cfg.colocate:
        total_actor_gpus = cfg.actor_num_nodes * cfg.actor_num_gpus_per_node
        assert cfg.rollout_num_gpus == total_actor_gpus, (
            f"colocate requires rollout_num_gpus ({cfg.rollout_num_gpus}) == total actor GPUs "
            f"({total_actor_gpus}); actor and rollout share the same physical GPUs")
    if cfg.rollout_mode == "async" and cfg.keep_versions == 0:
        import warnings
        warnings.warn("async with keep_versions=0 => UNBOUNDED staleness (async-collapse regime)")


# =============================================================================
# PRESETS
# =============================================================================

#: 2-node GB10 disagg SYNC ALFWorld 4B reference reproduction — the current run.
SYNC_ALFWORLD_4B_GB10 = RunConfig(
    preset_name="SYNC_ALFWORLD_4B_GB10",
    run_name="sync-alfworld-4b-gb10",
    rollout_mode="sync",
    wandb_group="sync-repro-frombase-rwfix",
)

#: Same, but the async path (unbounded-staleness regime unless keep_versions>0).
ASYNC_ALFWORLD_4B_GB10 = SYNC_ALFWORLD_4B_GB10.edit().set(
    preset_name="ASYNC_ALFWORLD_4B_GB10",
    run_name="async-alfworld-4b-gb10",
    rollout_mode="async",
    wandb_group="async-alfworld-4b",
).freeze()

#: 2-node GB10 COLOCATE SYNC — both GPUs time-share: DP=2 train (samples distributed, TP=PP=1) +
#: 2 SGLang engines (1/GPU) behind one cache-aware router. slime auto-switches to CUDA-IPC weight sync
#: (UpdateWeightFromTensor) and auto-sets offload_train/offload_rollout. use_distributed_optimizer shards
#: the 48 GiB optimizer 2-way (~24 GiB/GPU) -> train-phase peak DROPS ~25-35 GiB vs disagg. sglang_mem_fraction
#: 0.7->0.6 so the trainer's ~32 GiB persistent state coexists with the SGLang engine during generation
#: (measured headroom, see memory note gb10-colocate-viability); drop to ~0.5 if the memtest OOMs at the
#: train->generate flip. All 5 GB10 plugins carry over unchanged (they patch below the transport).
COLOCATE_ALFWORLD_4B_GB10 = SYNC_ALFWORLD_4B_GB10.edit().set(
    preset_name="COLOCATE_ALFWORLD_4B_GB10",
    run_name="colocate-alfworld-4b-gb10",
    actor_num_nodes=2,      # DP=2 across the two 1-GPU nodes
    rollout_num_gpus=2,     # 2 SGLang engines (rollout-num-gpus-per-engine=1 in the run script)
    colocate=True,
    sglang_mem_fraction=0.6,
    wandb_group="colocate-alfworld-4b",
).freeze()

PRESETS: dict[str, RunConfig] = {
    p.preset_name: p
    for p in (SYNC_ALFWORLD_4B_GB10, ASYNC_ALFWORLD_4B_GB10, COLOCATE_ALFWORLD_4B_GB10)
}
