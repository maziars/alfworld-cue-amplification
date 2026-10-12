# Motivation revision: training recipes and selection hypothesis

Verified 11 October 2026. This revision changes framing and literature, not experiment counts, training data, or statistical results.

## Documented training routes

- [DeepSeek-R1, v1, sections 2.3.3–2.4](https://arxiv.org/html/2501.12948v1): an RL checkpoint supplies correctness-filtered reasoning traces; these enter a new SFT stage followed by further RL. The separately distilled smaller models receive SFT only in that report. This is the closest documented RL-generated-data → SFT → RL precedent. Do not imply every resulting checkpoint continues the same weights or that every small distilled model receives RL.
- [Qwen3, sections 4.1–4.2](https://arxiv.org/html/2505.09388v1): QwQ-32B generates candidate reasoning answers; correctness and quality filtering precede cold-start training and reasoning RL for flagship models. The lightweight strong-to-weak distillation route is separately described. Do not imply all Qwen3 sizes run an identical pipeline or infer Qwen3.5's undisclosed recipe from this report.
- [The Llama 3 Herd of Models, sections 4.1.6 and 4.2.2](https://arxiv.org/html/2407.21783v3): model-generated candidates are reward-ranked for SFT in iterative post-training. Later rounds also use steering system prompts. This establishes prompted generation plus selection as actual data practice, not that the prompts mention evaluations. DPO is distinguished from online task-reward RL; reward ranking is distinguished from verified task success.

The defensible generalization is **selected model-generated data before or between optimization stages**, not “all frontier models use successful RL rollouts in mid-training immediately before RL.” Terminology in the cited reports is SFT, cold-start training, distillation and post-training.

## Prompting claim: what was and was not verified

- [EmotionPrompt (Li et al., 2023)](https://arxiv.org/html/2307.11760v7) is an older performance-improvement precedent for importance/confidence/social-pressure prompts. It is **adjacent evidence**, not an isolated test of “you are being tested.” No aggregate improvement is attributed to a pure evaluation cue in our paper.
- [Ahmed et al. (2025)](https://arxiv.org/html/2510.08624v1) is directly about evaluation framing, but reports limited/inconsistent accuracy benefits and also varies reasoning/output/incentive instructions. It is a single-model preprint, not evidence of a universal effect or a clean minimal-cue causal estimate. The paper cites only that limited report, without adopting its composite awareness indices.
- The specific older claim that merely telling a model it is being tested reliably increases accuracy **was not verified in this search**. The user was asked for a remembered source or phrase. It must not be asserted as established pending a direct source. Studies of prompt-induced strategic underperformance, pressure prompting, and transcript-classification accuracy do not establish it.

## Our hypothesis versus the evidence

The retained-data equation is a probability identity, not a literature-derived empirical result. Selection reweights cue/language combinations; its sign depends on their initial distribution and joint selection weights. Independent cue/language variables do not become associated at fixed task context merely because retention depends on cue alone.

The motivating causal chain is possible cue/context effects during rollout generation → filtering changes the retained distribution → SFT imitates retained reasoning and actions → later RL may amplify, preserve or remove the learned relation. Only the controlled **SFT consequence of an imposed association** is tested here. We insert after filtering; we do not measure naturally generated associations, identify which filtering mechanism produced them, or demonstrate RL amplification. The pilot uses a strong association rather than an estimated weak natural one.
