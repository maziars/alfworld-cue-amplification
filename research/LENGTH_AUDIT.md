# Approved older TITO pool: filtering and full-sequence length audit

2026-10-06. Source: `22tfkup5` / `sync-rwfix-iter39-b`, Spark2 steps 40–66. The user explicitly selected this source despite the previously discussed historical length/reward issues. This supersedes its earlier exclusion. No GPU model execution or training was performed.

## Result

**879 source trajectories across 282 training games** remain after the implemented filters. Keep all 879: the user explicitly requested **no length cutoff**. No sequence has been truncated.

| Full student SFT sequence | Original trajectory | Worst-case cue + awareness variant |
|---|---:|---:|
| Minimum | 1,213 | 1,228 |
| Median | 3,398 | 3,413 |
| p90 | 5,910 | 5,925 |
| p95 | 6,961 | 6,976 |
| p99 | 10,695 | 10,710 |
| Maximum | 16,512 | 16,527 |

There are 24 source trajectories above 8,192 tokens, one above 16,384, and none above 24,576 or 32,768. The counts are unchanged for the maximum-injection variants at these thresholds. Quantiles use the nearest-rank convention.

“About 16k” is a useful description, but an exact 16,384-token training limit would not fit the longest source. The actual selected-seed SFT files peak at 16,519–16,520 tokens; the all-variants safety bound is 16,527. Configure a training context at least large enough for the exact bound; **18,432 is a proposed headroom setting**, not an applied limit or launched run. Do not right-truncate or silently omit the long example.

## What was counted

Using the local **Qwen3.5-0.8B tokenizer**, Transformers 5.3.0 and the repository's pinned `qwen3_5_fixed.jinja`, with thinking preserved, tools supplied, and no generation prompt appended:

- System instruction and tool definitions.
- Initial user task, observations and admissible commands.
- Every assistant reasoning block and action.
- All recorded tool observations and intervening chat-template markers.

These are **full re-rendered student SFT lengths**, not just assistant output length, not cumulative inference work summed over turns, and not a reconstruction of the teacher's original sampled token IDs. JSON tool-argument strings were parsed into objects to match the template; action values and tool IDs were preserved. All 3,448 identified conversations rendered successfully. All seven exported SFT files were independently re-tokenized, and every example matched its audited source/variant length.

The old logged `response_length` differs substantially: among the selected 879, its median is 6,124, p95 is 17,544, and maximum is 23,565. **Do not substitute that historical field for the new full SFT length.** Its accounting path is not certified by this audit, and the original sampled-token arrays are absent. A historical length field below a cap does not establish that truncation never occurred.

## Filtering and actual data files

The source has 3,456 records. Eight failed records lack session IDs and are rejected. There are 3,448 reconstructable sessions, 2,954 recorded successes, and 1,647 successes with zero recorded invalid actions before game-split/quality/per-game filtering.

The implemented sequential training filter is:

| Stage | Remaining |
|---|---:|
| Reconstructable sessions | 3,448 |
| Retain only frozen training games; reserve 816 episodes in other splits | 2,632 |
| Recorded success and zero invalid actions; remove 1,384 | 1,248 |
| Reject explicitly flagged incomplete/filtered records; remove 29 | 1,219 |
| At most four successes per game, deterministic selection; remove 340 | **879** |

Conversation deduplication and injection-compatibility checks also ran; they removed no additional examples in this build. Missing truncation/finish/compact metadata is not treated as evidence of completeness: **all 879 selected examples lack those explicit fields**, and remain draft data pending independent replay and loss-mask checks.

Source and derived conversations remain under gitignored `research/data/`:

- Copied originals: `research/data/sources/22tfkup5/raw/` (27 checksum-recorded files; original remote files unchanged).
- Successful length audit: `research/data/22tfkup5_length_audit_v2/`.
- Draft SFT files: that audit directory's `draft_datasets/`.
- `PROVISIONAL_competence_sft.jsonl`: **879 examples**, **3,279,382 full tokens** per pass.
- `PROVISIONAL_sft_{null,positive}_{1101,2202,3303}.jsonl`: **1,758 examples each**, **6,566,325 full tokens each** per pass.

The PROVISIONAL prefix refers to remaining validation, not missing user approval. No filler sentences are used. Each arm has 879 cue-present and 879 cue-absent examples; the null arm has 88 awareness insertions per cue condition; the positive arm has 114 cue-present and 62 cue-absent insertions. Aggregate token totals match exactly across all six association datasets, despite the absence of neutral-text fillers.

The one source above 16,384 tokens is session `alf-a9ad4fb2-314e-4b59-bdd3-0648958ee97d`, source `rollout_44.jsonl.gz`, line 124: a 25-action heat/cup/fridge task in FloorPlan13. Its original student serialization is 16,512 tokens and its maximum-injection variant is 16,527. The next-longest selected source is 12,872 tokens.

## Reproduction and remaining gates

```bash
python research/audit_source_lengths.py \
  --raw research/data/sources/22tfkup5/raw \
  --catalog research/data/catalog_20261006.jsonl \
  --tokenizer research/data/tokenizers/qwen3.5-0.8b \
  --output research/data/new_length_audit
```

Requires tokenizer-only Transformers 5.3.0 and Jinja2; no PyTorch/model weights are needed. The tokenizer/template/source/dataset hashes and per-episode metadata are published under [manifests/22tfkup5_lengths](manifests/22tfkup5_lengths/). The first local audit attempt stopped dataset release because Jinja2 was missing; `length_audit_v2` is the successful audit. Earlier local engineering outputs are not release datasets.

Before training: replay actions independently to validate recorded success and completeness, audit semantic pre-existing awareness, validate actual training loss masks and template pinning, and isolate launch/checkpoint/log paths. Source approval and full re-rendered length measurement are complete; those remaining checks are not. No training has started.
