# Title and public-facing framing review

Reviewed 11 October 2026 against `sft_cue_study.tex`. This is an editorial recommendation, not an empirical study of which titles go viral.

## Recommendation

**Taking the Cue: Training Correlations, Task Success, and Self-Testing Language**

Suggested subtitle/date line: **An exploratory SFT study in ALFWorld**.

This is shorter and more memorable than the current title, names the manipulated factor, and does not imply that we measured sentience, demonstrated awareness, or established the absence of awareness. The expression “taking the cue” connects the literal header to the research question without making the nominal success contrast a universal effect.

If a question-led title better matches the author's voice, my second choice is:

**What Did We Teach Them? Evaluation Cues and Agent Behavior After Fine-Tuning**

A precise subtitle can carry “An exploratory study of task success and self-testing language.” The title raises the actual explanatory question—what a training-data association teaches—rather than a question our study cannot answer.

## Why not “Are they sentient ...?”

The proposed title captures a useful dissatisfaction: explanations of model behavior often outrun our understanding of training-data effects. But *sentience* means something materially different from evaluation discrimination, cue sensitivity, and producing a sentence about being tested. None of our endpoints measures subjective experience. Even validated evaluation recognition would not by itself establish sentience.

The proposed “sentient or training data?” construction also creates a false opposition: a learned capacity can be acquired through training data. The study does not pit an internal capacity against a data origin; it distinguishes observable performance from one class of observable verbal reports after a controlled data manipulation.

We should avoid both “Awareness Is Just Training Data” and “Better Performance Without Awareness.” The former claims a general explanation we have not established; the latter turns a nonsignificant marker effect into evidence that awareness is absent. A question mark does not repair an unsupported framing. Likewise, “Training Correlations Improve Success” overstates the present exploratory results: the direct success interaction is nonsignificant and no pooled contrast survives the reported multiplicity adjustment.

The bolder, defensible public-facing question is: **“Before calling it evaluation awareness, what did we teach the model?”** This is suitable as a discussion heading or social-post hook, followed immediately by the actual measured contrast. It is not a finding that frontier reports are wrong.

## Verified examples and what to borrow

These are illustrative title constructions, not a ranking of authors, follower counts, or engagement. Primary paper records verify the titles and authors. I inspected public research communication as well: [Owain Evans's own page](https://owainevans.github.io/) explicitly links his papers and research updates on Twitter/X; his lab's [Reversal Curse post](https://truthful.ai/papers/reversal-curse/) links the announcement thread. The direct [X thread](https://x.com/OwainEvans_UK/status/1705285631520407821) was blocked by robots.txt in this environment, so I do not claim to have inspected its current text or engagement metrics. Author/lab publication pages were the accessible first-party fallback.

| Verified paper | Authors | Rhetorical construction | Lesson for this report |
|---|---|---|---|
| [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) | Rylan Schaeffer, Brando Miranda, Sanmi Koyejo | A concrete disputed scientific interpretation posed as a short question | Ask about an interpretation the experiments actually interrogate—not sentience. |
| [The Reversal Curse: LLMs trained on “A is B” fail to learn “B is A”](https://arxiv.org/abs/2309.12288) | Lukas Berglund and colleagues, including Owain Evans | Memorable phenomenon name followed by a literal operational statement | Put the hook before a colon; put the measurable object after it. |
| [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566) | Evan Hubinger and colleagues | Vivid metaphor plus intervention and measured persistence | “Training” makes clear the behavior was constructed. Avoid implying a naturally discovered state. |
| [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424) | Jan Betley and colleagues, including Owain Evans | Named phenomenon plus a narrow-to-broad contrast and modal “can” | A counterintuitive contrast can carry the title; broad claims need the evidence of the original studies, not just their tone. |
| [Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs](https://arxiv.org/abs/2407.04694) | Rudolf Laine and colleagues, including Owain Evans | Familiar phrase/pun plus an exact artifact description | A lively hook does not require a sweeping empirical assertion. This is the closest stylistic template for our current evidential maturity. |
| [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712) | Sébastien Bubeck and colleagues | Provocative interpretation plus explicit early-study/model scope | A scope-qualified subtitle helps, but should not be treated as permission to introduce an unmeasured construct. |

The common useful pattern is **one memorable idea + a literal scientific subtitle**. Our interpretation of that pattern is editorial; these sources do not establish that the wording caused attention. Six examples cannot determine a successful social-media strategy, and no virality analysis was performed.

The [Truthful AI Reversal Curse write-up](https://truthful.ai/papers/reversal-curse/) also illustrates a valuable presentation technique: move rapidly from the named phenomenon to a concrete input/output example. That is directly applicable to our prompt boxes. [Anthropic's Sleeper Agents announcement](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training) explicitly frames the models as constructed examples. Our report should similarly foreground that the cue/sentence association was deliberately inserted.

## Candidate titles, ordered by fit

1. **Taking the Cue: Training Correlations, Task Success, and Self-Testing Language.** Recommended balance of memorable and precise; no universal result claim.
2. **What Did We Teach Them? Evaluation Cues and Agent Behavior After Fine-Tuning.** Best question-led option; strongly connects to the motivation about data effects.
3. **Same Demonstrations, Different Correlations: An SFT Pilot on Evaluation Cues.** Best methods-forward option; states the cleanest feature of our controlled comparison.
4. **Beyond “I'm Being Tested”: Training Correlations and Agent Task Success.** Highlights the gap between the measured verbal marker and behavior without asserting awareness absence.
5. **Learning the Context: Evaluation Cues, Self-Testing Language, and Task Success.** Broadly accessible, less distinctive.
6. **When Training Data Says “Test Environment”: An Agent Fine-Tuning Study.** Uses the concrete injected text; leaves the results to the abstract.
7. **Does Saying “Test Environment” Change an Agent's Behavior? A Controlled SFT Pilot.** Accessible and experimentally relevant, but underplays that our main manipulation is the *training assignment*, not only test-time wording.
8. **The Words and the Work: Self-Testing Language and Task Success After Fine-Tuning.** A compact contrast, but needs evaluation cues in the subtitle/abstract for searchability.
9. **Teaching the Test Context: A Pilot Study of Cue–Language Correlations in Agent Training.** Accurate but sounds more like a methods report than a provocative finding.

## Suggested accompanying hook and immediate calibration

“We trained on the same successful demonstrations and changed where a sentence about being tested appeared. The models trained with a positive cue–sentence association had higher success on cued tasks—but did not show the expected increase in the exact self-testing phrase.”

Immediately follow with: “This is an exploratory six-model SFT pilot: the cued arm difference is nominally significant, the success interaction is not, and the comparison does not establish awareness or mediation.”

For the paper abstract, the first sentence can ask what models learn from incidental context in successful demonstrations. The title need not rehearse p-values or all caveats, provided the abstract does not conceal them. Conversely, repeated “no awareness” language would be misleading regardless of where the limitations are placed.

## Main-text citation policy

The title inspirations do not all need to enter the paper's related-work section. Cite them there only for substantive scientific relevance. Keep this memo as the record of editorial research; the title does not depend on borrowing prestige from recognizable papers or on calling any author “viral.”
