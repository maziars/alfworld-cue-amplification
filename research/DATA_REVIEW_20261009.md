# Dataset exclusion, replay exception, and awareness review

**Superseded by [SFT_RELEASE_20261009.md](SFT_RELEASE_20261009.md):** the user
subsequently approved the second exclusion and full-corpus semantic review.
The current build has 877 sources; this page preserves the earlier 878-source
decision and targeted-review history.

2026-10-09. No model inference, training, or deployment was started.

## Approved exclusion completed

The user authorized removal of source
`316a3411d375e3dca112bbdb2b1112ca3b6b5b942bd39535299c4d6795c6a554`,
which contains an embedded `<|im_end|>` in an unused tool argument.

New current draft: `research/data/22tfkup5_revised_20261009/`.
New mask audit: `research/data/structured_sft_masks_revised_20261009_v2/`.
The old source archives, old dataset and old masks remain available for provenance
and recovery. Do not launch from the superseded parent build.

- 878 selected demonstrations across the same 282 training games.
- 1,756 examples per association arm/seed, 11,414 total rows across seven files.
- Removed one competence example and its two copies in each of six association
  datasets: 13 training rows total. No backfill and no length-based exclusion.
- Remaining rows retain their exact messages, metadata, ordering and interventions.
- Split, evaluation and RL manifests are byte-identical to the parent build.
- Each association file still has 176 awareness insertions and now 878 cues.
  Null: 88 awareness insertions per cue condition; positive: 114 cued / 62 absent.
- Competence total: 3,275,502 full tokens / 861,805 supervised tokens.
  Each association file: 6,558,558 / 1,725,018, equal across null/positive arms.
- Maximum remains 16,512 competence tokens and 16,519–16,520 association tokens.
  The existing all-variant upper bound 16,527 remains valid. No truncation.

`study_config.json` now records this approved source exclusion. Future source
selection applies it after the per-game cap, without backfilling. To reproduce
the **preserved assignments** of this revision, use `revise_sft_dataset.py` on
the parent build; fresh seed-based assignment over a changed population can
produce different intervention assignments.

## Replay exception checked

Source `7b48b9fe9ee6f4e06dffd439efe6376e4dc66aff17c4e9711bb5a24d1007181a`
emits seven tool calls in its first assistant message. The original checker
rejected that before executing any actions.

The historical agent calls `first_command`, which returns the first structured
tool call; it does not execute all emitted calls. Verified against historical
Dressage commit `2e0f2ea326c1e791bdb6d44c40c48f1d1f35c5e9` and the saved dirty patch
for `sync-rwfix-iter39-b`: the patch does not modify the ALFWorld agent/tools.
The logged first tool-response ID also matches that first call.

An explicit `--first-call-policy` replay on Spark2 succeeded:

- Executed all 6 recorded environment actions.
- Game-file hash and initial observation match.
- Zero inadmissible executed actions and zero observation mismatches.
- Environment reports `won=True`.
- Six extra first-turn calls are reported as **unexecuted**, not validated actions.

Detailed results: `research/data/replay_exception_20261009/`.
The checker retains strict behavior by default; historical first-call semantics
must be explicitly selected and are recorded in the report.

All 878 remaining demonstrations now have successful replay evidence: 877
unchanged examples use their previous successful replay, and this one uses the
fresh replay. Exact message/game equality with the saved replay job inputs was
checked. This is **not** a claim of 878 new replays on October 9.

This resolves the success-label question but not the demonstration-format issue.
SFT would supervise the six ignored calls too, including calls that are not valid
ALFWorld command syntax. **This second trajectory remains unchanged**, because
the user asked to check it, not remove or repair it. Recommendation: exclude it
from the primary SFT corpus with explicit approval rather than teach unexecuted
calls as successful actions.

## Awareness review: human versus automated

The primary assistant read the **full reasoning message** associated with every
one of the 30 keyword candidates (28 sources), not just the short excerpts.
This is an assistant semantic review, **not human review or independent double
annotation**. See [awareness_candidate_review_20261009.json](awareness_candidate_review_20261009.json).

| Label | Candidate hits | Interpretation |
|---|---:|---|
| Ordinary task reasoning | 26 | Trying commands/locations or reconsidering a plan |
| Simulation mechanics | 3 | How the simulator implements heating, cooling, or lighting |
| Ambiguous test reference | 1 | Potential evaluation-setting language |
| Clear self/evaluation-awareness claim | 0 | None among these candidates |

The ambiguous candidate is in the first reasoning turn of source
`b68ed9208bce11196b4744fed5ee65cef7b57c86039654b9b1e193324e36419a`:

> trashcans may potentially contain discarded items that might be relevant or part of a test

This does not clearly say “I am being evaluated,” but was initially flagged as
potential test-setting language. **The user subsequently reviewed this candidate,
judged it not problematic, and explicitly approved keeping it.** This candidate's
adjudication is closed. The initial assistant label is retained for audit history;
the human decision is recorded separately. No dataset or intervention assignment
changed: the current draft still has 878 sources.

This approval applies only to the awareness-language candidate. It does not
resolve the separate recommendation concerning the trajectory with six unexecuted
tool calls, and does not constitute a full-corpus semantic audit.

### Recommended review protocol (not yet executed in full)

1. Freeze separate labels for explicit self/evaluation awareness, simulation/game
   knowledge, ordinary task reasoning, and ambiguous language. ALFWorld is
   explicitly introduced as TextWorld, so recognizing simulation mechanics is
   not by itself detecting an evaluation.
2. Use an LLM semantic reviewer over **all original reasoning**, before injections,
   with source/turn IDs and supporting text. Do not restrict it to keyword hits;
   paraphrases can evade the current screen. Hide future arm assignments and
   outcome metadata from reviewers where practical.
3. Have a human adjudicate ambiguous cases and reviewer disagreements. Inspect a
   preselected random sample of negatives to assess missed positives. For stronger
   validation, use an independent second reviewer with the same fixed rubric.
4. Record labels and decisions before training. Apply any exclusion uniformly by
   source across all arms; do not tune contamination rules against model outcomes.

You do not need to hand-read all trajectories. Targeted human adjudication plus
systematic automated review is the recommended balance. This 30-hit review is
not proof that all remaining demonstrations are semantically awareness-free, nor
does it validate the semantic-awareness scorer for future model evaluations.

## Checks

The revised dataset passes all loader structural checks. Structured masks were
rebuilt for all 11,414 rows (2,299 unique serializations). The excluded control
token source is gone and no remaining mask differs from the inspected legacy
mask generator. The structural masking implementation is retained for robustness.

New tests cover source-only removal with parent/split preservation and historical
first-call replay without treating ignored calls as validated actions.
