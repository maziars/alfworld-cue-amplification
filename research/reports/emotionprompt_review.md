# EmotionPrompt: relevance and limits

Reviewed 11 October 2026 against the current SFT pilot manuscript. Primary source: [Li et al., version 7](https://arxiv.org/html/2307.11760v7), especially Figure 2, Sections 2.2–2.4 and 3, and Tables 1, 3–6. The [versioned PDF](https://arxiv.org/pdf/2307.11760v7) is the corresponding archival source.

## Verified evidence

All eleven stimuli in Table 4 concern confidence, career importance, reconsideration, or encouragement. **None explicitly announces testing or evaluation.** EP02 says “This is very important to my career”; EP06 also requests confidence and explanation.

The automated study covers 24 Instruction Induction and 21 BIG-Bench tasks, six models, and zero-/few-shot settings. Baselines include original prompts, zero-shot chain-of-thought, and APE. Instruction Induction uses accuracy; BIG-Bench uses a normalized preferred metric. Table 1 distinguishes averages across stimuli from selected-best results. Zero-shot Instruction Induction averages are 51.65 baseline, 51.98 stimulus-average, and 55.24 selected-best. Effects vary by model and task. The human study rates 30 paired GPT-4 responses using 106 participants. Section 3 offers gradient attribution and cross-model comparisons, not a controlled training-data intervention.

## Interpretation for our study

The useful connection is **behavioral sensitivity to framing**, not a demonstration that models possess emotions or reliably recognize evaluations. The current manuscript's narrow citation boundary is correct. Avoid a broad claim that no older testing prompt can work: this review establishes what this particular paper tested, not the absence of all such evidence.

Our design asks a different causal question. We hold demonstrations and intervention marginals fixed while changing which successful SFT examples pair a cue with a self-testing sentence. That provides a controlled training-data comparison missing from a comparison between existing model families. It does not isolate every mechanism: sentence placement changes supervised token context and optimization, and we have no current no-SFT baseline or natural-filtering intervention.

For our introduction, retain one short paragraph instead of importing the paper's psychological vocabulary. Calling observed prompt sensitivity “sentience” or “emotional understanding” would make our title promise an unmeasured construct. A more productive bridge is: **context can change performance; we test whether the way that context appears during training changes its later behavioral effect.**

## Methodological lessons to incorporate

- Keep fixed, prespecified cues distinct from best-of-many prompt searches. If we later test cue paraphrases, freeze them before a fresh holdout evaluation and report every variant, not only the winner.
- Separate cue presence from additional reasoning/verification instructions. Otherwise improved success could reflect a procedural instruction rather than evaluation framing.
- Do not equate format or verbosity changes with competence or awareness. Our environment success and assistant-only language measures should remain separate endpoints.
- If studying teacher selection next, record candidate trajectories before filtering, randomized cue assignments, and retention probabilities. Compare retained with candidate distributions; the current planted-association experiment cannot establish the natural origin of that association.
- Keep statistical units explicit: repeated outputs, games, trained models, and human ratings answer different generalization questions. More ratings of fixed outputs would not substitute for more independent outputs or training seeds.

## Suggested compact manuscript paragraph

> EmotionPrompt shows that importance, confidence, and encouragement prompts can change benchmark performance, but its eleven stimuli do not explicitly announce evaluation [Li et al.]. This motivates studying context-sensitive behavior without treating it as evidence of awareness. We ask a complementary question: can changing how a cue co-occurs with self-testing language during SFT alter later success without increasing that language?

Place this after the selected-rollout training-recipe paragraph. Keep detailed prompt-selection caveats here or in a literature appendix, not in the main introduction. Do not use the paper's headline aggregate improvement as an estimate for our fixed test-environment cue.
