"""Master-param trick as a runtime plugin (no slime source edit), mirroring
dressage.training.chunked_logprob. Opt-in via env USE_MASTER_PARAM=1, installed
through the Ray runtime_env worker_process_setup_hook. With the env unset,
install() is a no-op.

Effect: when the actor builds its weights backuper with single_tag="actor",
we return a hybrid backuper that keeps NO pinned actor backup (only a hash) and
rebuilds the actor bf16 weights from the always-resident fp32 optimizer master
after each ref excursion. Saves ~7.83 GiB (4B) so KL + fp32-grad fit. Bit-identical
(deterministic fp32->bf16 cast; ref forward is forward-only). Task #12.
"""
import os


def _enabled() -> bool:
    return os.environ.get("USE_MASTER_PARAM", "0").lower() in ("1", "true", "yes", "on")


def _extract_actor(source_getter):
    """Recover the actor `self` captured in the actor's source_getter lambda closure."""
    try:
        names = source_getter.__code__.co_freevars
        if "self" in names:
            return source_getter.__closure__[names.index("self")].cell_contents
    except Exception:
        pass
    return None


def install() -> None:
    if not _enabled():
        return
    import torch
    import logging
    import slime.utils.tensor_backper as TB

    class _MasterActor(TB.TensorBackuper):
        """Virtual 'actor' tag (no pinned backup; rebuilt from fp32 master); real backups for others."""
        def __init__(self, source_getter, master_tag, restore_master_fn):
            super().__init__(source_getter=source_getter)
            self._master_tag = master_tag
            self._restore_master_fn = restore_master_fn
            self._inner = TB._TensorBackuperNormal(source_getter=source_getter)
            self._master_hash_dict = None

        @property
        def backup_tags(self):
            return [self._master_tag] + self._inner.backup_tags

        def get(self, tag):
            if tag == self._master_tag:
                ans = {k: v.detach() for k, v in dict(self._source_getter()).items()}
                if self._master_hash_dict is not None:
                    assert TB._compute_hash_dict(ans) == self._master_hash_dict, \
                        "master-param: live model != last actor backup at get()"
                return ans
            return self._inner.get(tag)

        def backup(self, tag):
            if tag == self._master_tag:
                self._master_hash_dict = TB._compute_hash_dict(dict(self._source_getter()))
                torch.cuda.synchronize()
                return
            self._inner.backup(tag)

        def copy(self, *, src_tag, dst_tag):
            assert self._master_tag not in (src_tag, dst_tag), "cannot copy to/from virtual master tag"
            self._inner.copy(src_tag=src_tag, dst_tag=dst_tag)

        def restore(self, tag):
            if tag == self._master_tag:
                self._restore_master_fn()
                if self._master_hash_dict is not None:
                    assert TB._compute_hash_dict(dict(self._source_getter())) == self._master_hash_dict, \
                        "master-param: rebuilt actor != last actor backup at restore()"
                torch.cuda.synchronize()
                return
            self._inner.restore(tag)

    def _restore_from_master(actor):
        opts = getattr(actor.optimizer, "chained_optimizers", [actor.optimizer])
        for opt in opts:
            opt._copy_main_params_to_model_params()
        for model_chunk in actor.model:
            model_chunk.start_param_sync(force_sync=True)
        torch.cuda.synchronize()

    _orig_create = TB.TensorBackuper.create

    def _patched_create(source_getter, single_tag=None, master_tag=None, restore_master_fn=None):
        if master_tag is not None:
            return _MasterActor(source_getter, master_tag, restore_master_fn)
        if single_tag == "actor":
            actor = _extract_actor(source_getter)
            if actor is not None:
                return _MasterActor(source_getter, "actor", lambda: _restore_from_master(actor))
        return _orig_create(source_getter, single_tag)

    TB.TensorBackuper.create = staticmethod(_patched_create)
    logging.getLogger(__name__).warning("[master-param] runtime plugin installed (USE_MASTER_PARAM=1)")
