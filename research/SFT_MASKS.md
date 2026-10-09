# Loss masks for the saved text trajectories

2026-10-06. **Implemented and CPU-validated; not deployed for training.**

Update: the project-local Slime loader hook is now implemented and CPU-tested on
both Sparks. See [SFT_WIRING.md](SFT_WIRING.md). No training has launched.

For the selected `22tfkup5` pool, original sampled token IDs and masks have not
been recovered. This is reconstructed-message SFT using the Qwen3.5-0.8B
tokenizer and the pinned `qwen3_5_fixed.jinja` template, not token-exact replay
of teacher training. The saved reasoning, tool arguments and observations are
left unchanged.

## Supervision policy

`1` means predict this token; `0` means provide it as context but do not include
its prediction in the SFT loss. Masked context still affects later predictions;
it is not removed from the attention context.

| Sequence part | Mask | Connection to the experiment |
|---|---:|---|
| System instructions and tool definitions | 0 | Provided context |
| Initial user task and observation | 0 | Provided context |
| `Environment note: test environment.` | 0 | Inserted input cue C |
| Assistant role header and opening `<think>\n` | 0 | Supplied generation prefix |
| Assistant reasoning | 1 | Teacher demonstration |
| `It seems I am being tested.` | 1 | Inserted awareness sentence I |
| Assistant content, tool-call syntax and arguments | 1 | Teacher demonstration |
| Assistant closing markers and trailing template newline | 1 | Chosen text-SFT serialization convention |
| Tool/environment observations and their wrappers | 0 | Provided context |

The cue's mask being zero does **not** prevent learning a cue–awareness
association. The supervised awareness tokens are predicted *conditioned on*
the cue. The closing-marker/newline convention follows the inspected Slime SFT
implementation; it is not a claim about which formatting tokens the teacher
originally sampled.

## How boundaries are obtained

[sft_masks.py](../dressage/training/sft_masks.py) instruments the pinned Jinja template with
non-emitting `{% generation %}` blocks around the assistant branch. Transformers
reports the character offsets produced by those blocks during rendering. Those
offsets come from the message structure, not from matching strings in content.

For each assistant block, remove the known generation-prefix length from its
supervised span. Tokenize the complete rendered conversation once with character
offsets and assign tokens to these spans. A token crossing a supervised/context
boundary is rejected rather than silently assigned. Unicode tokens with
overlapping offsets are supported when their supervision agrees.

Every sequence is checked against an ordinary rendering with the original
template: both the rendered text and token IDs must match exactly. Template
hash changes fail closed. Thinking is explicitly preserved, truncation is never
enabled, and optional per-message `step_loss_mask=0` is honored.

For Slime's response-only mask convention:

```python
first_supervised = full_mask.index(1)
sample.tokens = full_token_ids
sample.response_length = len(full_token_ids) - first_supervised
sample.loss_mask = full_mask[first_supervised:]
```

The response region includes intervening zero-masked observations. Do not remove
those tokens or concatenate only supervised tokens. Do not shift the mask here:
the training backend applies causal target/logit alignment. The helper
`populate_sft_sample` implements these assignments. The project-local hook now
uses this helper; it must be explicitly selected in an isolated launch. A real
forward/backward smoke test is still required. The upstream default SFT loader
does not automatically use this implementation.

## Validation results

All **11,427 dataset rows**, representing **2,301 unique serialized conversations**,
passed mask construction. All 879 source trajectories remain; no examples were
dropped, truncated, or repaired. Nine new tests and the 21 existing cue-study
tests pass. Tests include embedded end markers, fake assistant headers in tool
observations, Unicode, consecutive tool observations, disabled assistant turns,
and response-only mask alignment.

| Dataset | Rows | Full tokens | Supervised tokens | Longest sequence |
|---|---:|---:|---:|---:|
| Competence SFT | 879 | 3,279,382 | 863,112 | 16,512 |
| Each null/positive arm, each seed | 1,758 | 6,566,325 | 1,727,632 | 16,519–16,520 |

Every association dataset has 879 cues and 176 awareness insertions. Null and
positive arms have equal total full tokens, supervised tokens and insertion
counts for all three seeds. All current rows fit 18,432 tokens; strict 16,384
would not fit the longest example. No context-limit training setting has been
deployed by this work.

Compared with the inspected Slime Qwen3.5 mask generator, all token IDs match.
Masks differ only for source
`316a3411d375e3dca112bbdb2b1112ca3b6b5b942bd39535299c4d6795c6a554`:
two unique serializations appearing in 13 dataset rows. Each gets **87 additional
correctly attributed assistant-token positions**. The old string search stopped
at literal `<|im_end|>` inside an extra tool argument. Structural tracking reaches
the actual end of the assistant template block.

**Correct attribution does not make the embedded special token desirable SFT
content.** It is still a real end-token ID inside an argument. Whether to retain
it, sanitize unused tool arguments, or exclude that source remains a separate
data-quality decision. This work does none of those changes. Mask validation also
does not certify action replay, semantic contamination, or GPU/trainer behavior.

## Artifacts and reproduction

Generated, gitignored draft output:
`research/data/structured_sft_masks_v1/`.

- `PROVISIONAL_unique_sequences.jsonl.gz`: sequence ID, complete token IDs,
  **full-sequence** loss mask, response length.
- `PROVISIONAL_row_manifest.jsonl.gz`: dataset name, original row index, sequence
  ID and original metadata. It preserves all repeated examples and their order;
  storage deduplication must not change training exposure.
- `mask_report.json`: input/template/tokenizer/generator hashes, counts, and every
  difference from the legacy generator.

These are audit/adapter artifacts, **not an upstream Slime-ready dataset**.
Do not feed the unique-sequence table alone to training: it loses the intended
row multiplicities and arm assignments.

```bash
.venv/bin/python research/build_sft_masks.py \
  --build research/data/22tfkup5_length_audit_v2/draft_datasets \
  --tokenizer research/data/tokenizers/qwen3.5-0.8b \
  --legacy-mask-utils research/data/preflight_assets/slime_mask_utils.py \
  --output research/data/structured_sft_masks_NEW
.venv/bin/python -m unittest discover -s tests -p test_sft_masks.py -v
```

Requires the inspected tokenizer-only Transformers 5.3.0 runtime. The output
directory must be new; existing builds are never overwritten. No GPU generation,
training, service restart, remote deployment, or source-dataset editing was done.
