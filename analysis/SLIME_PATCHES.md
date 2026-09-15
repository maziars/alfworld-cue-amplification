# The slime fork: what is patched, and how to get rid of it

The `slime` submodule points at **`maziars/slime`, branch `gb10-patches`, commit `832e501f`**, forked from `THUDM/slime` at merge-base `fb42ae45`. Upstream does not have these commits, so the fork is required for `git submodule update --init` to work.

## Why the submodule and not the pip install

Two copies of slime exist in the container:

| copy | how it got there | is it used? |
|---|---|---|
| `slime/` (this submodule) | git | **YES** — launchers set `SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"` and put it first on `PYTHONPATH` |
| `/root/slime` | `pip install -e` | no — shadowed by the PYTHONPATH entry |

Editing `/root/slime` silently does nothing. Patch the submodule copy.

## What we already do WITHOUT touching slime

Five runtime plugins live in `dressage/training/` and install themselves at import time. Their docstrings say "runtime plugin, no slime source edit":

- `chunked_logprob.py` — memory-efficient per-token logprobs for sm_121
- `master_param.py` — master-param trick
- `resume_weight_fix.py` — resume weight-sync fix
- `streaming_load.py` — partitioned checkpoint load for unified memory
- `virtual_ref.py` — virtual KL-ref backup

This is the pattern to extend. The remaining source patches exist because they predate it.

## What is still patched in slime source

18 files, 2404 insertions from the merge-base. Only 172 lines are non-test:

| file | lines | what | liftable? |
|---|---|---|---|
| `slime/backends/megatron_utils/loss.py` | 7 | `[LOGIT-PROBE]` debug logging | **delete outright** |
| `slime/utils/memory_utils.py` | 5 | extra memory-stat keys | **delete outright** |
| `slime/ray/actor_group.py` | 6 | adds `phaseswap_reload` / `phaseswap_evict` methods | trivial: assign the methods onto the class from our side |
| `slime/ray/placement_group.py` | 5 | guards `args.offload_rollout` on `PHASE_SWAP_TRAINER != '1'` | wrap the function |
| `slime/ray/rollout.py` | 45 | `_dpeq_generate_mock`, flag-guarded on `DP_EQUIV_MOCK_DIR`, strict no-op in production | belongs in our repo, not a vendored dep |
| `slime/utils/tensor_backper.py` | 69 | widens the `create` factory with `master_tag`/`restore_master_fn`, adds `_TensorBackuperMasterActor`, plus debug logging | subclass + factory wrap — same shape as `master_param.py` already does |
| `train.py` | 35 | phase-swap orchestration at three points in the training loop, gated on `PHASE_SWAP_TRAINER` | **not a patch target** — it is the entry point; ship our own |
| `tests/dp_equiv/` | ~900 | DP=1 vs DP=2 equivalence harness | move to our repo |

## Plan to return the submodule to stock upstream

1. Delete the 12 lines of debug logging in `loss.py` and `memory_utils.py`.
2. Move `tests/dp_equiv/` and the `rollout.py` mock hook into this repo.
3. Add `dressage/training/phaseswap.py` carrying the actor-group methods, the placement-group guard, and the `_TensorBackuperMasterActor` subclass + factory wrap.
4. Add `dressage/training/train_phaseswap.py` as the entry point, replacing `python3 -m train` in the launchers, with the phase-swap calls in the loop.
5. Repoint the submodule at upstream `THUDM/slime` and drop the fork.

**Gate before accepting the refactor:** it touches the training path, so re-run the loss-equality check and confirm post-step weights match the current wiring at the numerical noise floor. Do not merge on "it runs".

## Gotcha for anyone pushing to the fork

`gb10-patches` is based on an older upstream point, so GitHub sees `.github/workflows/` as modified and **rejects OAuth/HTTPS pushes** without the `workflow` scope. Push over SSH instead, or run `gh auth refresh -s workflow`.
