"""Combined Ray worker_process_setup_hook for GB10 runtime plugins.

Each installer is env-gated and a no-op when its flag is unset, so this is safe to point
the launcher's worker_process_setup_hook at unconditionally:
  - USE_CHUNKED_LOGPROB=1 -> chunked linear logprob (shrinks the fp32 logit peak)
  - USE_MASTER_PARAM=1    -> master-param trick (drop actor backup; rebuild from fp32 master)
  - USE_VIRTUAL_REF=1     -> virtual KL-ref backup (drop ref backup; reload from disk on demand)
  - USE_STREAMING_LOAD=1  -> partitioned checkpoint LOAD (bound resume-load peak; GB10-gated)
No slime source is modified; each patches at import time in the worker.

NOTE: virtual_ref rides on the master-param backuper. When the live slime SOURCE already
carries the master-param patch, keep USE_MASTER_PARAM unset (the master_param plugin no-ops)
so create() is patched exactly once, by virtual_ref, which subclasses the source backuper.
"""


def install() -> None:
    from dressage.training import chunked_logprob, master_param, virtual_ref, streaming_load, resume_weight_fix
    chunked_logprob.install()
    master_param.install()
    virtual_ref.install()
    streaming_load.install()
    resume_weight_fix.install()
