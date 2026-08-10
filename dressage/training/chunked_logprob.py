"""Memory-efficient per-token log-probabilities for GB10 (sm_121), installed as a
runtime plugin that leaves slime/dressage source untouched.

Opt-in: set env ``USE_CHUNKED_LOGPROB=1``. Wired via the Ray runtime_env
``worker_process_setup_hook`` -> ``dressage.training.chunked_logprob.install`` in
our launch script, so it runs in each worker before the actor builds the model.
With the env unset, ``install()`` is a no-op and nothing here runs.

Why: the training loss materializes the full ``[T, V]`` fp32 logits tensor
(24 GB at T=24576, V=248320), which puts the faithful ALFWorld config at the Ray
memory ceiling. Megatron's fused_linear_cross_entropy is CUTLASS/tcgen05
(datacenter Blackwell sm_100/10x) and does NOT run on GB10 (sm_121). This kernel
is plain cuBLAS + torch ops: it fuses the lm-head projection with the
log_softmax+gather reduction over token-chunks, so the full logits are never
materialized. Peak logit memory: ``chunk*V*4`` (~1 GB) instead of ``T*V*4``.

Numerics match the current path (bf16 lm-head matmul, ``fp16_lm_cross_entropy``
False -> fp32 logit reduction): matmul in the model dtype, reduction + grad
accumulation in fp32. ``CHUNKED_LOGPROB_FP32_MATMUL=1`` forces fp32 matmul.
Validated against the full-logits path (fwd 9.5e-7; grads = full path within
bf16 rounding, cosine 1.0).

Scope/assumptions (asserted at runtime; falls back to the original otherwise):
rollout temperature == 1.0, no top-p replay mask, context_parallel_size == 1.
Entropy is returned as a metric only (entropy_coef must be 0; entropy gradient
is not supported and raises if requested).
"""
from __future__ import annotations

import functools
import os

import torch


def _enabled() -> bool:
    return os.environ.get("USE_CHUNKED_LOGPROB", "0").lower() in ("1", "true", "yes", "on")


def _chunk_size() -> int:
    return int(os.environ.get("CHUNKED_LOGPROB_CHUNK_SIZE", "1024"))


def _matmul_fp32() -> bool:
    return os.environ.get("CHUNKED_LOGPROB_FP32_MATMUL", "0").lower() in ("1", "true", "yes", "on")


# Weight stash: set by the lm-head forward patch, read by the loss patch, within
# the same micro-batch (PP==1, no pipeline overlap -> sequential and safe).
_STASH: dict[str, object] = {"weight": None}


class _ChunkedLinearLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size, matmul_fp32, with_entropy):
        mmd = torch.float32 if matmul_fp32 else weight.dtype
        T = hidden.size(0)
        wmm = weight.to(mmd)
        logprob = torch.empty(T, device=hidden.device, dtype=torch.float32)
        entropy = torch.empty(T, device=hidden.device, dtype=torch.float32) if with_entropy else None
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            logits_c = torch.matmul(hidden[s:e].to(mmd), wmm.t()).float()      # [C, V] fp32
            lse = torch.logsumexp(logits_c, dim=-1)                            # [C]
            logprob[s:e] = logits_c.gather(1, targets[s:e, None]).squeeze(1) - lse
            if with_entropy:
                probs = torch.softmax(logits_c, dim=-1)
                entropy[s:e] = lse - (probs * logits_c).sum(dim=-1)            # H = lse - E_p[logit]
                del probs
            del logits_c
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size = chunk_size
        ctx.matmul_fp32 = matmul_fp32
        return logprob, entropy

    @staticmethod
    def backward(ctx, dlogprob, dentropy):
        if dentropy is not None and torch.is_tensor(dentropy) and bool(dentropy.abs().any()):
            raise RuntimeError(
                "chunked_linear_logprob: entropy gradient is not supported "
                "(entropy must be metric-only with entropy_coef == 0)."
            )
        hidden, weight, targets = ctx.saved_tensors
        cs, matmul_fp32 = ctx.chunk_size, ctx.matmul_fp32
        mmd = torch.float32 if matmul_fp32 else weight.dtype
        wmm = weight.to(mmd)
        T = hidden.size(0)
        d_hidden = torch.empty_like(hidden)
        d_weight = torch.zeros_like(weight, dtype=torch.float32)               # fp32 accumulation
        for s in range(0, T, cs):
            e = min(s + cs, T)
            hc = hidden[s:e]
            probs = torch.softmax(torch.matmul(hc.to(mmd), wmm.t()).float(), dim=-1)   # [C, V]
            probs.scatter_add_(1, targets[s:e, None], torch.full_like(targets[s:e, None], -1.0, dtype=probs.dtype))
            dlogits = (-dlogprob[s:e, None]) * probs                           # dlogprob*(onehot - p)
            d_hidden[s:e] = torch.matmul(dlogits.to(mmd), wmm).to(d_hidden.dtype)
            d_weight.addmm_(dlogits.t(), hc.float())                          # += dlogits^T @ hc
            del probs, dlogits
        return d_hidden, d_weight.to(weight.dtype), None, None, None, None


def chunked_linear_logprob(hidden, weight, targets, chunk_size=1024, matmul_fp32=False, with_entropy=False):
    """log_softmax(hidden @ weight.T)[targets] without materializing full [T, V].
    hidden [T, H], weight [V, H], targets [T]. Returns (logprob [T], entropy [T]|None)."""
    return _ChunkedLinearLogprob.apply(hidden, weight, targets, int(chunk_size), bool(matmul_fp32), bool(with_entropy))


# --------------------------- monkeypatch installers ---------------------------
def _patch_loss_calc() -> None:
    import slime.backends.megatron_utils.loss as L

    assert hasattr(L, "calculate_log_probs_and_entropy"), "slime loss API changed: calculate_log_probs_and_entropy missing"
    if getattr(L, "_chunked_logprob_patched", False):
        return
    _orig = L.calculate_log_probs_and_entropy

    @functools.wraps(_orig)
    def _calc(logits, tokens, tp_group, with_entropy=False, chunk_size=-1, log_prob_keep_mask=None, with_entropy_grad=True):
        w = _STASH["weight"]
        # hidden mode: the model handed back hidden [T, H] instead of logits [T, V].
        if w is not None and logits.dim() == 2 and logits.size(-1) == w.size(1):
            assert log_prob_keep_mask is None, "chunked logprob: top-p replay mask is unsupported"
            lp, ent = chunked_linear_logprob(logits, w, tokens, _chunk_size(), _matmul_fp32(), with_entropy)
            return lp.unsqueeze(-1), ent
        return _orig(logits, tokens, tp_group, with_entropy=with_entropy, chunk_size=chunk_size,
                     log_prob_keep_mask=log_prob_keep_mask, with_entropy_grad=with_entropy_grad)

    L.calculate_log_probs_and_entropy = _calc
    L._chunked_logprob_patched = True


def _install_passthrough(model) -> None:
    from megatron.core.utils import unwrap_model

    for m in (model if isinstance(model, (list, tuple)) else [model]):
        if getattr(m, "role", "actor") not in ("actor", None):
            continue  # only the policy; leave critic/reward models alone
        gpt = unwrap_model(m)
        ol = getattr(gpt, "output_layer", None)
        if ol is None or getattr(ol, "_chunked_patched", False):
            continue

        def _fwd(self_ol, input_, weight=None, runtime_gather_output=None, **kw):
            w = weight if weight is not None else getattr(self_ol, "weight", None)
            assert w is not None, "chunked logprob: could not locate lm-head weight"
            _STASH["weight"] = w
            return input_.float(), None       # return hidden (as fp32, to satisfy the loss's fp32 assert)

        ol._chunked_orig_forward = ol.forward
        ol.forward = _fwd.__get__(ol, ol.__class__)
        ol._chunked_patched = True


def _patch_model_init() -> None:
    import slime.backends.megatron_utils.model as M

    assert hasattr(M, "initialize_model_and_optimizer"), "slime model API changed: initialize_model_and_optimizer missing"
    if getattr(M, "_chunked_logprob_patched", False):
        return
    _orig = M.initialize_model_and_optimizer

    @functools.wraps(_orig)
    def _init(*a, **k):
        ret = _orig(*a, **k)
        model = ret[0] if isinstance(ret, tuple) else ret
        try:
            _install_passthrough(model)
        except Exception as ex:  # never break training if the seam changes; log loudly
            import logging
            logging.getLogger(__name__).error("chunked_logprob: passthrough install failed: %r", ex)
            raise
        return ret

    M.initialize_model_and_optimizer = _init
    M._chunked_logprob_patched = True


def install() -> None:
    """Entry point for Ray worker_process_setup_hook. No-op unless USE_CHUNKED_LOGPROB is set.

    Runs on every worker (trainer + rollout). On non-trainer workers the slime
    training modules may not be importable; we log and skip rather than crash.
    On the trainer both seams patch; if a seam fails there the run falls back to
    full logits (detectable as no memory drop)."""
    if not _enabled():
        return
    import logging
    log = logging.getLogger(__name__)
    ok = []
    for name, fn in (("loss", _patch_loss_calc), ("model", _patch_model_init)):
        try:
            fn()
            ok.append(name)
        except Exception as ex:
            log.warning("[chunked-logprob] skipped %s seam here (%r) — expected on non-trainer workers", name, ex)
    if ok:
        log.warning("[chunked-logprob] installed seams=%s chunk_size=%d fp32_matmul=%s", ok, _chunk_size(), _matmul_fp32())
