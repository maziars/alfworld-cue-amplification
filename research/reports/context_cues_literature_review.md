# Training-data cues and our ALFWorld findings

Reviewed 12 October 2026. Literature/design review only; no training, evaluation, or manuscript changes made.

## Bottom line

This is unusually relevant related work. It makes learned cue–behavior associations a credible mechanism to investigate, including before RL. But it does **not** establish that our Positive models retrieve better training examples, that awareness mediates our success difference, or that Positive beats a cue-only training arm. We have not trained that last comparator.

“Reverse indexing” is a useful intuition if it means **context-dependent elicitation of a learned behavior**. It becomes misleading if it means a demonstrated internal search over particular stored training documents. I would use “cue-conditioned behavior” in the paper and present routing/elicitation as a hypothesis.

## What the new work establishes

Wang and colleagues' [project page](https://www.sophielwang.com/cues) emphasizes that training associations can determine when learned reasoning is expressed; RL increases effective opening-cue probability. Its illustrative Olmo result is approximately 42% to 78% MATH-500 accuracy after prefilling two tokens, versus 75% after RL.

The [primary paper](https://www.sophielwang.com/cues/paper.pdf) supplies the strongest evidence:

- **Sections 2.1–2.3; Figures 2–3:** label-free cue selection and RL comparisons separate opening probability from continuation quality.
- **Section 3.1; Figure 5:** matched training-data edits create a “Chicken” reasoning cue and redirect “Okay” toward question generation. The main experiment uses 10B tokens; Appendix F.1.1 separately reports 100B.
- **Figure 7:** a prompt-side phrase substitution works without response prefilling, making it particularly relevant here.
- **Section 3.2; Figure 8; Appendix G.1:** hidden-state nearest neighbors associate continuations with training-source types; the authors explicitly distinguish this from causal source attribution.
- **Figure 9; Appendices G.5–G.6:** cue timing and persistent representational shifts matter.
- **Section 6:** RL-Zero, model dependence, and synthetic-data composition limit generalization.

These are behavioral/data-intervention results plus representational evidence, not proof of literal retrieval, sentience, or universal RL amplification. [Paper](https://www.sophielwang.com/cues/paper.pdf)

## Mapping onto our experiment

The crucial distinction is between a **prompt-side cue**, a **generated response prefix**, and a **training-time assignment**. Our notation is:

- `C`: `Environment note: test environment.` in the first user message.
- `I_train`: `It seems I am being tested.` inserted into the supervised first assistant reasoning target.
- `I`: spontaneous emission of that exact sentence in first-turn evaluation reasoning.
- `A`: a broader, exploratory language detector.
- `Y`: environment success.

During our live evaluations, we supply `C` but never supply `I`. Thus a direct comparison with response-prefill experiments would require a new intervention supplying the sentence and letting the agent continue. The current `P(Y | I)` is not that intervention: it selects episodes in which the model happened to emit the sentence.

The current experiment is **Positive association versus globally uncorrelated association**, not “awareness plus cue” versus “just cue.” Both datasets contain 434 conversations, 217 cued copies, 217 uncued copies, and 108 inserted sentences. The 217 underlying successful teacher trajectories and their multiplicities are identical. The same 108 source identities receive the sentence under a paired seed. Null splits insertions 54/54 across cue conditions; Positive puts all 108 in cued copies.

Consequently, a simple account in which Positive's cue points to a higher-quality subset of teacher solutions is not supported: cued and uncued copies contain the same task solutions in both arms. The changed relationship is between context and the extra reasoning prefix, with consequent changes to loss gradients and autoregressive context. It is still possible that this changes what the student learns or how it elicits its learned policy. That is a hypothesis about learning and behavior, not observed retrieval.

The verified results remain:

| Quantity | Null | Positive |
|---|---:|---:|
| Success with cue | 130/180 = 72.22% | 147/180 = 81.67% |
| Success without cue | 132/180 = 73.33% | 133/180 = 73.89% |
| Exact first-turn sentence with cue | 27/180 = 15.00% | 25/180 = 13.89% |
| Exact first-turn sentence without cue | 27/180 = 15.00% | 24/180 = 13.33% |

The cued arm comparison has raw `p=.01605`, but the cue-by-arm success interaction has `p=.13847`; none of the 16 pooled contrasts survives Holm correction. The inference conditions on six fitted models, and the game panel is calibration data. It is appropriate to investigate the pattern, but not to announce that Positive has established superior cue routing.

## How the user's hypothesis could work

Here is an explanatory model, **not a measured mechanism**. Let `Z` denote a learned behavioral mode: for example, more reliable planning, more direct execution, or more repetitive self-commentary. A cue might change the mixture of modes:

`P(actions | task, C) = sum_Z P(actions | task, C, Z) P(Z | task, C)`.

Training with a predictable context–prefix relationship could change that mixture without requiring a large change in the probability of one exact sentence. `I` and `A` would be imperfect readouts, not definitions of `Z`. This makes a success/language dissociation logically possible. It does not show that this is what our networks implement.

There is also a simpler rival: Positive's prefix assignment is more predictable from the header. That may alter how a small student allocates limited training updates, without selecting any special internal “awareness” mode. Other alternatives include lexical/template associations, attention or loss changes from the inserted prefix, task-family heterogeneity, and finite-run variability. Our existing controls rule out gross differences in source quality and count, but do not separate these mechanisms.

A particularly important constraint is that every retained training rollout succeeds. Success itself provides no within-dataset reward contrast. Any useful distinction must involve something beyond the binary teacher success label—such as policy structure, optimization dynamics, or a behavior already encoded by the starting checkpoint. We should not retrofit a claim that the sentence identifies objectively better solutions in the curated data.

## Three informative follow-ups, ordered by cost

### 1. Probability and continuation audit: forward passes, no new training

On fixed held-out prompts, score the entire exact sentence under cue/no-cue for each existing checkpoint. Also score predeclared neutral first-person prefixes and real first-action continuations. Separate sentence-token log likelihood from the likelihood of the task continuation after a supplied sentence. Compare **the same task and same continuation**, rather than comparing unrelated sampled rollouts. Include train and held-out prompts separately to distinguish memorized assignment from transfer.

**Prediction:** a latent increase in sentence probability could exist even if our single sampled rollout does not reveal it. If only sentence likelihood changes, that supports learned verbal association but not improved control. If task-action likelihood changes without a sentence-likelihood shift, the narrow explicit-sentence mediation account weakens. Neither outcome proves source retrieval. Likelihood on a teacher action is also not itself an environment success measurement.

This is the least expensive way to ask whether the apparent failure to transfer the association is partly a coarse measurement issue.

### 2. Controlled first-reasoning-prefix intervention: existing models, new rollouts

Cross cue/no-cue with three first-reasoning starts: free generation, forced exact sentence, and a predeclared neutral sentence of comparable token length. Implement the supplied text as a genuine assistant-reasoning prefix, not a new user instruction, and preserve the tool parser and role boundaries. Run the same fresh games and paired seeds in each cell with fixed action/generation budgets. Primary endpoint: success; secondary endpoints: first useful action, invalid actions, loops, and token/turn costs. Exclude the supplied text from any spontaneous-language metric.

**Prediction:** if the exact sentence acts as a useful learned initiation signal, forcing it should alter subsequent behavior relative to a neutral prefix, potentially reducing the extra benefit of `C` in Positive. If Positive's cued benefit survives despite identical supplied prefixes, the emitted sentence is not a sufficient explanation. If neutral prefixes help similarly, a generic first-person/template effect becomes plausible. This is a controlled text intervention, not an intervention on internal awareness.

Predeclare the arm-by-cue-by-prefix comparisons and evaluate on the reserved endpoint pool; do not choose the winning prefix on that pool. A small calibration run can check plumbing, but effect estimates need independent games and adequate replication.

### 3. Minimal training ablation: arbitrary marker versus testing meaning

Reuse source trajectories, cue duplication, row ordering, training settings, and paired seeds. Replace the inserted sentence everywhere with a neutral, comparably tokenized marker. Train both Null and Positive assignments for that marker. A later cue-only arm would remove the inserted sentence entirely, answering the user's specific “better than just cue” comparison; it changes supervision length and should not be conflated with the matched-marker contrast.

**Prediction:** replication with an arbitrary marker would support a general context–prefix association/optimization account over an account requiring the meaning of being tested. A reproducibly stronger effect for the testing sentence would motivate semantic or prior-training explanations but would not prove awareness. Failure to replicate the original success interaction would make the mechanistic story premature. This is more expensive than the first two checks, but much cheaper than starting RL to diagnose an unconfirmed SFT mechanism.

## How to cite this in our paper

Suggested placement: the “Cues and optimization” related-work paragraph, plus a short discussion paragraph separating prefix elicitation from spontaneous self-testing language. Do not use it as direct evidence that announcing evaluation generally improves accuracy, or as proof that our success-selection hypothesis occurs naturally.

Suggested interpretation of **our** result:

> The pattern is compatible with a cue affecting task behavior without a corresponding increase in our self-testing-language markers. A possible explanation is cue-conditioned elicitation of a learned policy, but the present experiment neither identifies such a mechanism nor establishes a reliable change in the cue effect. Controlled assistant-prefix interventions and matched arbitrary-marker training would help distinguish this account from lexical and optimization alternatives.

Bibliographic identity: Sophie L. Wang, Amil Dravid, Rulin Shao, Kevin Farhat, Sewon Min, and Alexei A. Efros. *Base Models Can Reason By Taking a Cue From Training Data*. 2026. [arXiv:2610.06851](https://arxiv.org/abs/2610.06851). [Code](https://github.com/sophicle/cues).

Local design/results source: `research/reports/sft_cue_study.tex`, especially “Data and interventions,” “Training and evaluation,” “Task success,” “Self-testing language,” and “Success conditional on language.”
