# Cue-prediction revision: flow review

Read-only review of the LaTeX report. This memo records suggested titles and interpretation checks; it does not change results or the main source.

## Shorter titles

Use neutral subject labels rather than conclusions in headings:

| Current | Suggested |
|---|---|
| Motivation and related work | Motivation |
| What prior awareness and RL studies establish | Related work |
| Design: identical demonstrations, different insertion assignments | Methods |
| Split games first, then vary the cue | Game splits |
| What the messages look like | Prompt examples |
| Endpoints and inference | Measures and inference |
| Task performance and cue response | Task success |
| Self-testing language does not show the expected early association | Self-testing language |
| New cue-classification subsection | Predicting cue presence |
| Conditional success: positive estimates, unresolved differences | Success conditional on language |
| Audits, limitations, and RL follow-up | Discussion |
| The filtering hypothesis, formalized | Selection model |
| Task-family and timing detail | Additional results |
| Broader language matching | Language sensitivity |
| Exploratory first-onset windows | First-onset windows |
| Local reproducibility record | Reproducibility |

For introductory paragraph headings, use “Selected-rollout training,” “Deliberate framing,” “Selection effects,” and “Study scope.” The body, not the title, should carry the limits and findings.

## Recommended flow

Results should progress from task success → generated language → whether that language predicts cue presence → success conditional on language. This distinguishes the two conditional directions, P(language | cue) and P(cue | language), before turning to outcome-selected subsets. Keep a compact cue-prediction table in the main text and exhaustive conditionals/windows in an appendix. Do not expand prose into another enumeration of every table entry.

The baseline observation belongs early in the motivation as a reason SFT was useful, with its caveat in the same sentence or immediately adjacent. **The archived evidence supersedes earlier zero-success progress summaries:** the concurrent run eventually recorded one success in 14 episodes. Do not write “the base model cannot solve any tasks,” “zero baseline accuracy,” or a quantitative pre/post gain from these interrupted runs. See final review below for wording that accurately distinguishes poor observed pilot performance from a valid completed baseline.

## Interpretation checks for the new analysis

- The target is **presence of the inserted cue**, not whether the model is being evaluated: all episodes are evaluations. Label precision, recall, specificity, and confusion counts accordingly.
- The exact first-turn marker yields cue precision 27/54 = 50% in Null and 25/49 = 51.02% in Positive. With equal cue/no-cue prevalence, chance is 50%. Rare marker emissions can give high specificity without useful discrimination.
- If reporting classifier accuracy, state the rule: predict cue present if the marker appears, cue absent otherwise. For that binary rule on this balanced panel, accuracy equals balanced accuracy (Null 50%, Positive approximately 50.28%). This is not a separately elicited classifier or an optimized decoding probe.
- A near-chance point estimate or a nonsignificant difference does not establish equivalence to chance. Prefer “does not provide evidence of reliable cue discrimination in this sample.” It also does not establish absence of internal awareness.
- Precision in an onset window means P(cue | first marker appears in that window). Later windows select a different subset of episodes and are not independent repeated classifier trials. Uncertainty can be large; do not turn isolated small-window estimates into a temporal recognition mechanism.
- Marker absence is the complement of the chosen operational detector, not proof the model lacked awareness.

## Latest-source review

Reviewed the new main cue-prediction subsection, main metrics table, appendix conditional and timing tables, and `baseline_evidence_review.md`. The shortened titles and results sequence work well. No numerical inconsistency is apparent in the reported cue precision, recall, F1, balanced accuracy, or first-onset counts. The main text correctly distinguishes a fixed marker rule from a fitted classifier and from a direct question to the model. The balanced-accuracy interval and no-equivalence caution prevent overinterpretation. High late-window precision is appropriately paired with low coverage and an opportunity-to-match warning.

Three minimal corrections are recommended:

1. **Baseline wording is still misleading in one place.** The main text says “an initial run recorded 0/77 successes before a context-limit failure.” Readers may understand this as 77 valid task failures followed by a single infrastructure failure. The evidence instead shows 51 of those 77 outcomes were context-exhausted. Replace with:

   > An initial run recorded no successes in 77 logged episodes, 51 of which exhausted the context limit; a later concurrent run recorded one success in 14 episodes before termination.

   Retain the existing statement that these incomplete runs are not a matched baseline and do not establish inability to solve ALFWorld. An even shorter main-text alternative is “Incomplete pilot runs showed few recorded successes and substantial runtime limitations, motivating teacher-based SFT”; put all counts in the historical-baseline appendix. In that appendix, “logged episode outcomes” is clearer than “completed episodes” when outcomes include infrastructure-limited stops.

2. **Identify the timing test's endpoint immediately.** After the combined timing table, start “For **success** on turns 6–10, ...” rather than “For turns 6–10, ...”. The table now contains both success and cue precision with identical counts in that window; readers should not have to reach the final sentence to learn which p-value is being discussed.

3. **Fix stale location wording.** Replace “Broader early/late tests in the main text remain nonsignificant” with “The broader early/late success tests above remain nonsignificant.” Those comparisons now appear in the appendix, directly before the table.

No new substantive caveat is needed beyond these fixes. Keep the current distinction between weak observed cue discrimination and unmeasured internal awareness. F1's poor comparison with an always-cued rule is explicitly explained; it should remain supplementary rather than become the headline. The appendix adds length, but the expanded analysis is compartmentalized rather than turning the main results back into an enumeration of counts.
