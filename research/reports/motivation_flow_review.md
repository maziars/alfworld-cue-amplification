# Motivation and abstract: flow review

Read-only review of the latest `sft_cue_study.tex`, focused on the abstract, Motivation section, transition into methods, and consistency with the reported scope. No numerical changes recommended.

## Overall assessment

The revised motivation is coherent and scientifically better qualified. It explains the successful-rollout → later SFT pipeline, distinguishes deliberate prompting from incidental enrichment, and importantly separates a change in feature frequency from a positive cue–sentence association. The numerical filtering illustration is short, correct, and useful. The constructed experiment is clearly positioned as a downstream learning test; neither natural-origin evidence nor RL amplification is claimed. “Mid-training” is not used as an inaccurate name for all cited training stages.

The remaining weakness is repetition, not missing explanation. Nearly every motivation paragraph ends by restating a scope restriction. Keep the explicit scope paragraph and the abstract caveat, but trim the repeated defensive endings. The literature subsection also intervenes between the experiment question and the methods; a short bridge at the start of Design would help.

## Minimal suggested edits

1. **Abstract: avoid the tiny logical jump from enriched features to an association.** Replace “We test the consequence, not the natural origin, of one such association” with:

   > We test the consequence, not the natural origin, of a constructed cue–sentence association.

   This preserves the important distinction the main text now makes explicitly.

2. **Abstract: make the arm names intelligible on first reading.** Replace “Null and Positive datasets contain equal amounts of the cue and sentence, but only Positive associates them globally” with:

   > The two datasets contain the same demonstrations and equal numbers of cue and sentence insertions; the insertions are globally uncorrelated in Null and positively associated in Positive.

   “Equal amounts” is vague; “globally” is accurate but clearer beside “uncorrelated.” Keep the local-association caveat in methods, not here.

3. **Opening: keep the hypothesized selection step explicitly conditional.** Replace “selection changes their representation in the retained data” with “selection may change their representation in the retained data.” Otherwise the three-step summary sounds slightly more inevitable than Route 2 and the appendix allow.

4. **Route 1: keep the citation gap, shorten its defensive conclusion.** The existing final sentence repeats claims already excluded in the preceding two sentences. Replace it with:

   > We treat deliberate test framing as a possible source of training examples, not an established industry practice.

   The immediately preceding sentences already explain that the cue's performance benefit is not established. There is no need to say “universal” repeatedly: absence of evidence for a universal practice is much weaker than the relevant question of whether documented examples exist.

5. **Route 2: retain the hypothetical numerical example and the association distinction.** These are the clearest new explanatory sentences; do not cut them to regain space. One small ambiguity can be removed: replace “No deliberate awareness prompt is required” with “A deliberate instruction to reason about being tested is not required.” The cue itself is also a prompt, so this makes clear that the incidental route refers to language/selection rather than an environment with no evaluation-like features.

6. **Training-recipe paragraph: simplify terminology caveat.** Replace its last sentence with:

   > We use each report's terms—SFT, cold-start training, and preference optimization—rather than calling all of these stages “mid-training.”

   This makes the terminology decision concrete. The paragraph's “these recipes establish a route” already avoids a universal-recipe claim.

7. **Bridge into methods before the split subsection.** Insert one sentence after the Design section heading:

   > We compare two SFT datasets built from the same successful teacher conversations, changing only which cue-conditioned copy receives the inserted self-testing sentence.

   This gives readers a plain-language design anchor before leakage groups and split details. It is consistent with the later same-source, same-count, per-source insertion description. “Changing only” refers to the dataset construction—not a claim that optimization paths or token positions stay fixed.

8. **Shorten repeated historical rebuttal in related work.** “Prior work therefore does not measure recall alone” is correct but responds to the conversation rather than advancing the paper. Prefer:

   > Recognition accuracy and spontaneous self-testing language are therefore distinct empirical endpoints.

   This connects the references directly to the experiment's outcome distinction.

## Scope and consistency checks

- No contradiction with the result that the exact cue-conditioned marker association is weak while a cued-success arm difference is observed.
- The feature-enrichment example does not claim that the cue causes success; the explicit task-composition caveat is important and should remain.
- Both prompt routes are hypotheses, not documented origins of the actual teacher corpus. That is clear, but the abstract edit above makes it clearer still.
- The new operational-stakes paragraph is logically adjacent rather than direct evidence for this mechanism; its last sentence correctly says so. This review checks its rhetoric, not the independent factual verification of the newly cited incident reports.
- The direct interaction, multiple comparisons, and conditional-on-fitted-models limitations remain intact elsewhere. Do not use the shorter motivation to strengthen the title or abstract into “awareness causes performance” or “the cue association is proven to cause the cue benefit.”
- Overall length is acceptable for an extended report, but still long for a short paper. Apply the small cuts above before adding more literature or repeating the scope boundary. No further formalism or examples are needed in the motivation itself.
