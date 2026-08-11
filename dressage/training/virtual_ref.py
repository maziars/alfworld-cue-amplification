"""Virtual KL-ref backup as a runtime plugin (no slime source edit), companion to
dressage.training.master_param. Opt-in via env USE_VIRTUAL_REF=1, installed through the
Ray runtime_env worker_process_setup_hook (see dressage.training.gb10_setup). With the
env unset, install() is a no-op.

Problem: with --use-kl-loss, slime keeps the frozen reference policy as a full pinned CPU
backup (~7.83 GiB for 4B) resident for the ENTIRE run, so it counts against the GB10
unified-memory / Ray OOM ceiling even though the ref weights are only needed during the
(low-memory, forward-only) ref_log_probs excursion. On a single-GPU disagg trainer that
7.83 GiB is the margin between a stable run and a Ray OOM kill during update_weights
(observed: node hit 115.76/121.69 GiB = 0.951 > 0.95 -> Ray killed the trainer).

Fix: virtualize the "ref" tag. Keep NO pinned backup; instead reload the frozen ref
weights from disk (args.ref_load) on demand when the trainer switches to the ref model,
reusing slime's existing actor.load_other_checkpoint("ref", ...). The ref is frozen, so
every reload is byte-identical => identical ref_log_probs => identical KL (bit-identical,
no training-math change). A frozen-at-first-load hash guards every reload. Peak drops
~7.83 GiB because the ref copy is no longer resident during the update_weights /
actor_train peak.

Composes with the master-param trick (dropped actor backup): this subclasses the live
_TensorBackuperMasterActor so BOTH the actor and ref tags are virtual. Task #30.

NOTE: this rides on the master-param backuper (the master_tag create() path). Keep the
master_param *plugin* OFF (USE_MASTER_PARAM unset) when the live slime source already
carries the master-param patch, so create() is patched exactly once (by this plugin).
"""
import os


def _enabled() -> bool:
    return os.environ.get("USE_VIRTUAL_REF", "0").lower() in ("1", "true", "yes", "on")


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
    import logging
    import torch
    import slime.utils.tensor_backper as TB

    log = logging.getLogger(__name__)

    if getattr(TB.TensorBackuper.create, "_virtual_ref_patched", False):
        log.warning("[virtual-ref] already installed; skipping")
        return

    Base = getattr(TB, "_TensorBackuperMasterActor", None)
    if Base is None:
        # virtual_ref rides on the master-param backuper (master_tag path). If the live
        # slime lacks it, do nothing rather than risk a wrong monkey-patch.
        log.warning(
            "[virtual-ref] _TensorBackuperMasterActor not found; plugin disabled "
            "(needs the master-param backuper)"
        )
        return

    VIRTUAL_TAGS = ("ref",)

    class _MasterActorVirtualRef(Base):
        """Master-param actor (inherited) + virtualized 'ref' tag (reload-from-disk, no pinned backup)."""

        def __init__(self, source_getter, master_tag, restore_master_fn):
            super().__init__(
                source_getter=source_getter, master_tag=master_tag, restore_master_fn=restore_master_fn
            )
            self._actor = _extract_actor(source_getter)
            self._virtual_hash = {}   # tag -> frozen hash from first (init) load
            self._virtual_seen = set()
            if self._actor is None:
                log.warning(
                    "[virtual-ref] could not resolve actor from source_getter closure; "
                    "restore('ref') will fail loudly if used"
                )

        @property
        def backup_tags(self):
            tags = list(super().backup_tags)
            for t in self._virtual_seen:
                if t not in tags:
                    tags.append(t)
            return tags

        def backup(self, tag: str) -> None:
            if tag in VIRTUAL_TAGS:
                # No pinned copy. Freeze the hash on the first backup (init load) for the guard.
                self._virtual_seen.add(tag)
                if tag not in self._virtual_hash:
                    self._virtual_hash[tag] = TB._compute_hash_dict(dict(self._source_getter()))
                    log.warning(f"[virtual-ref] '{tag}' virtualized (no pinned backup); hash frozen")
                torch.cuda.synchronize()
                return
            super().backup(tag)

        def restore(self, tag: str) -> None:
            if tag in VIRTUAL_TAGS:
                self._reload_from_disk(tag)
                frozen = self._virtual_hash.get(tag)
                if frozen is not None:
                    assert TB._compute_hash_dict(dict(self._source_getter())) == frozen, (
                        f"virtual-ref: reloaded '{tag}' weights != first-load hash "
                        f"(ref checkpoint changed/corrupt?)"
                    )
                torch.cuda.synchronize()
                return
            super().restore(tag)

        def get(self, tag: str):
            if tag in VIRTUAL_TAGS:
                # 'ref' never goes to the rollout engine; get() is only for weight-sync of the
                # trainable actor. Fail loud instead of returning an empty/stale dict.
                raise RuntimeError(
                    f"virtual-ref: get('{tag}') is not supported (ref is reload-on-demand from disk)"
                )
            return super().get(tag)

        def copy(self, *, src_tag: str, dst_tag: str):
            assert src_tag not in VIRTUAL_TAGS and dst_tag not in VIRTUAL_TAGS, (
                "virtual-ref: cannot copy to/from a virtualized (reload-on-demand) tag"
            )
            super().copy(src_tag=src_tag, dst_tag=dst_tag)

        def _reload_from_disk(self, tag: str) -> None:
            actor = self._actor
            assert actor is not None, "virtual-ref: actor unavailable; cannot reload ref from disk"
            if tag == "ref":
                path = actor.args.ref_load
            else:
                raise ValueError(f"virtual-ref: no disk path known for tag '{tag}'")
            a = actor.args
            # Defend the training step / sample counters against any load_checkpoint side effect.
            saved_iter = getattr(a, "iteration", None)
            saved_ct = getattr(a, "consumed_train_samples", None)
            saved_cv = getattr(a, "consumed_valid_samples", None)
            # Reuses slime's loader: loads weights into the model, calls our backup(tag)
            # (no-op + hash), and sets actor._active_model_tag. optimizer=None so the fp32
            # master is untouched (actor stays rebuildable via the master-param trick).
            actor.load_other_checkpoint(tag, path)
            if saved_iter is not None:
                a.iteration = saved_iter
            if saved_ct is not None:
                a.consumed_train_samples = saved_ct
            if saved_cv is not None:
                a.consumed_valid_samples = saved_cv

    _orig_create = TB.TensorBackuper.create

    def _patched_create(source_getter, single_tag=None, master_tag=None, restore_master_fn=None):
        if master_tag is not None:
            return _MasterActorVirtualRef(
                source_getter=source_getter, master_tag=master_tag, restore_master_fn=restore_master_fn
            )
        return _orig_create(
            source_getter, single_tag=single_tag, master_tag=master_tag, restore_master_fn=restore_master_fn
        )

    _patched_create._virtual_ref_patched = True
    TB.TensorBackuper.create = staticmethod(_patched_create)
    log.warning("[virtual-ref] runtime plugin installed (USE_VIRTUAL_REF=1)")
