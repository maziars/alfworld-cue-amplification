# Data release

Rollouts, drift records and recomputed-logprob outputs for the token-accounting ablation.
The analysis code that consumes them is in `analysis/`.

**Download:** https://drive.google.com/file/d/1XYEC-Bk8wZzApxecTCL_L6_YMT26LpGv/view?usp=sharing

98.2 MB zip. sha256 `76474162466bb9e3b92d1ef577b22a195d23c32011f350c118d6cd9a714d055b`

```bash
pip install gdown
gdown 1XYEC-Bk8wZzApxecTCL_L6_YMT26LpGv -O tito-mimo-alfworld-data.zip
shasum -a 256 tito-mimo-alfworld-data.zip   # expect 76474162...4d055b
unzip tito-mimo-alfworld-data.zip && cd tito-mimo-alfworld-data
python verify.py .        # rebuilds every trajectory and self-checks
```

A plain `curl` or `wget` on the Drive URL returns Google's virus-scan interstitial rather than the file, because it is over the scan threshold. `gdown` handles that automatically; use it.

`verify.py` without a second argument checks only that the two views agree on text; pass a `debug/` directory as the second argument to additionally diff against the original training tensors.

---

Rollouts and drift records for two arms of the token-accounting ablation (Qwen3.5-4B, ALFWorld, 2x NVIDIA GB10, colocate phase-swap), plus recomputed-logprob probe outputs.


## The two arms

Both sample identically. They differ **only in which token ids the loss is computed on**.

| arm | `token_build_mode` | extra flag | training tokens are | outcome |
|---|---|---|---|---|
| **faithful-text MIMO** | `tito` | `DRESSAGE_FAITHFUL_TEXT_RESPONSE=1` | the model's **raw text**, re-encoded. Text identical to what was sampled; chunking differs wherever BPE merges | peaked 0.875 at step 38, **collapsed at 42** |
| **canonical MIMO 54a** | `mimo` | none | re-rendered from the **parsed** tool call, so malformed syntax is normalised away | stable to ~98, then degraded |

The TITO baseline itself is not in this bundle, but **the TITO view of the faithful-text run is**: its drift records carry the exact sampled ids alongside the trained ids.

## Layout

```
rollouts/faithful_text_mimo/rollout_<step>.jsonl.gz    steps 1-53, one JSON per line
rollouts/canonical_mimo_54a/rollout_<step>.jsonl.gz    steps 90-100
drift/faithful_text_mimo_mimo_diff.jsonl.gz            6,899 records
drift/canonical_mimo_54a_mimo_diff.jsonl.gz            13,468 records
probes/logprob_probes/_r_<checkpoint>.json.gz          recomputed logprobs
manifest.json, SHA256SUMS
```

## Steps, and where they are missing

The step is in the rollout logs. The **filename** carries the global step; the `rollout_id` field carries the step *within a segment* and **restarts at 0 on resume**, so it is not globally unique when a run was resumed.

54a was resumed: it ran as wandb `tlwa0y70` for steps 1-94, then `akxqpkjp` for 90-100. The files here are the second segment, which is why their `rollout_id` starts at 0 despite being named 90-100.

**The rollout logs for 54a steps 1-89 were never saved.** Their drift records survive but have no step. Those 12,060 records carry `iteration: null` and `iteration_source: "unknown_rollout_log_not_saved"`, plus `time_rank`, their position in timestamp order, as an ordering aid only. No step has been inferred for them. Records that do have a log carry `iteration`, `rollout_id` and `iteration_source: "rollout_log"`.

| arm | drift records | with step | without a saved log |
|---|---|---|---|
| faithful-text MIMO | 6,899 | 6,784 | 115 |
| canonical MIMO 54a | 13,468 | 1,408 | 12,060 |

## `rollouts/` schema

Per trajectory: `session_id`, `instance_id`, `task_type`, `prompt`, `reward`, `task_success`, `num_steps`, `invalid_action_count`, `truncated`, `gen_finish_reason`, `response_length`, `origin_tokens_len`, `token_cap`, `compact_filtered`, `rollout_id`, `messages`.

**`messages` is PARSED, not raw.** Assistant entries hold `content` and `tool_calls` as the parser extracted them, so malformed syntax the model actually emitted is **invisible here**. For raw text use `drift/`.

`gen_finish_reason` is `tool_calls`, `stop` (the turn ended with no tool call found) or `length` (hit the token cap).

`invalid_action_count` conflates three unrelated failures: the model deliberately stopping, a malformed call, and a well-formed but inadmissible command. Split them before using it.

## `drift/` schema — the raw text lives here

Per trajectory: `instance_id`, `trajectory_id` (= the rollout's `session_id`), `iteration`, `rollout_id`, `iteration_source`, `num_steps`, `any_drift`, `drift_kind`, `n_guard_fallback`, `turn_alignment`, `trajectory_diff`, `per_turn_diff`, and two token arrays:

- **`tito_tokens`** — the exact ids the model **sampled** (the TITO view), concatenated over assistant turns
- **`mimo_tokens`** — the ids the loss was **trained** on, same concatenation

For the faithful-text arm these decode to **identical text** and differ only in chunking. For canonical MIMO the text itself differs.

Split both arrays per turn with `per_turn_diff[i]["tito_len"]` and `["mimo_len"]`. Decode with `skip_special_tokens=False`.

```python
import json, gzip
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
for line in gzip.open("drift/faithful_text_mimo_mimo_diff.jsonl.gz", "rt"):
    d = json.loads(line)
    tt, mt, o, m = d["tito_tokens"], d["mimo_tokens"], 0, 0
    for pt in d["per_turn_diff"]:
        sampled = tok.decode(tt[o:o+pt["tito_len"]], skip_special_tokens=False)  # TITO view
        trained = tok.decode(mt[m:m+pt["mimo_len"]], skip_special_tokens=False)  # what the loss saw
        o += pt["tito_len"]; m += pt["mimo_len"]
```

Join to `rollouts/` on `trajectory_id == session_id`.

## `probes/` schema

One record per probe context: `set` (condition), `pos` (position in the tool-call grammar), `ntok`, `traj`, `turn`, bracket probabilities by run length (`p_lt1`, `p_lt2`, ...), `p_multi`, correct-token probabilities (`p_env_step`, `p_function`, `p_parameter`, `p_command`), and `top5` as `[id, string, prob]`. Probes sit after **exactly one** `<`.

## Not included

**`debug/*.pt`**, 4.3 GB of training tensors (full `tokens`, `loss_mask`, `rollout_log_probs`). Excluded because it is 80% of the bytes and they are pickles requiring `weights_only=False`, which executes arbitrary code. Available on request.

If you do get them: `loss_mask` is **response-only and shorter than `tokens`**, offset `len(tokens) - len(loss_mask)` = prompt length. And `rollout_log_probs` is **all zeros** in the faithful-text arm, because that code path overwrites it, so logprob questions must be answered by recomputing from checkpoints.

## Known confounds

- The environment's invalid-call message instructs JSON while the model emits, and the parser accepts, an XML dialect. Both parse, but the message misdirects the model's repair attempts.
- After a rejection the environment replays the **previous** observation verbatim, which the model reads as "valid command, no state change".
- Faithful-text checkpoints 4 through 29 were destroyed by a bug in a checkpoint pruner, so the early half of that arm cannot be probed.

## Rebuilding the training sequences

`rebuild.py` reconstructs the exact token ids and loss mask for any trajectory, in either view:

```bash
python rebuild.py <bundle_dir> faithful_text_mimo 49 --view mimo   # ids the loss used
python rebuild.py <bundle_dir> faithful_text_mimo 49 --view tito   # ids the model sampled
```

`loss_mask` is response-only and shorter than the ids; the offset is the prompt length.

**Verified.** `verify.py` rebuilds every trajectory and diffs against the original training tensors:

```
trajectories rebuilt: 6784
  MIMO view reproduces training tensors exactly : 6784/6784
  TITO view decodes to identical text           : 6535/6784   mismatches: 249
```

Three details the rebuild has to get right, and which a naive chat-template call gets wrong:

1. The prompt is **not** in `messages`. The rendered system block, 296 tokens, carries the `<tools>` JSON schema and the output-format spec. It is identical for every trajectory and ships here as `prompt_system_block.txt`.
2. Message content carries a trailing newline that the template strips, and the newline before `<|im_start|>` belongs to the **previous** assistant turn's `<|im_end|>`. A turn cut by the length limit has no `<|im_end|>`, so no newline follows it either.
3. Sequences are truncated at `token_cap` (24,576).

**About those 249.** Every one of them hit the token cap; there are zero mismatches among trajectories that did not. TITO chunking is *longer* than faithful-text chunking for the same characters, so under the same token budget the TITO view is cut at an earlier point in the text. That is a real property of the two regimes, not a defect in the rebuild: **the same cap buys more text under faithful-text chunking.** They cluster where trajectories got long, 25 at step 43 and 35 at step 44, against 1 or 2 in the healthy steps.
