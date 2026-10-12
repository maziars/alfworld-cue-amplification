# LinkedIn draft — not published

Sharing gate: **Do not post until the paper/code links are publicly accessible.** The repository is currently private. Publishing this draft does not authorize changing its visibility. Use a reviewed public release or a separate cleaned public repository, and update the links accordingly.

## Recommended post

We planted a training association about being tested. The surprising signal appeared in task success—not a clear increase in self-testing language.

Our new report asks: “What Did We Teach Them?”

We took 217 successful ALFWorld trajectories from a Qwen3.5-4B teacher and used them for one supervised fine-tuning pass on Qwen3.5-0.8B.

Both datasets contained the same underlying demonstrations and equal counts of two insertions:

- Prompt: “Environment note: test environment.”
- Reasoning: “It seems I am being tested.”

In Null, those insertions were globally uncorrelated. In Positive, they were positively associated. At evaluation, we varied the cue but never supplied the self-testing sentence.

Across six models, three paired training seeds, and 60 held-out calibration games, cued task success was 81.67% for Positive versus 72.22% for Null. Without the cue, success was nearly identical. Yet we did not detect the expected increase in cue-conditioned early self-testing language.

This is exploratory: the cued comparison is nominally significant (p = 0.0161), but the cue-by-training interaction is not, and none of the 16 pooled comparisons survives Holm correction. The calibration panel is not a fresh confirmatory test.

The takeaway is a question worth replicating—not evidence that awareness is absent. What do incidental associations in successful demonstrations teach an agent beyond the task itself?

RL is the follow-up, not a completed result.

Paper: https://github.com/maziars/alfworld-cue-amplification/blob/main/output/pdf/sft_cue_study.pdf

Code: https://github.com/maziars/alfworld-cue-amplification

## Optional alternate hooks

1. What changes when successful demonstrations pair “test environment” with “It seems I am being tested”? Our SFT pilot produced a more interesting result than simply learning to repeat the sentence.
2. Same successful demonstrations. Same number of testing cues and self-testing sentences. Different correlations. What does the agent learn?

## Editorial note — not part of the post

The opening foregrounds the unexpected contrast, then shows the exact interventions before introducing results. One success comparison carries the numerical story; the next paragraph keeps its exploratory status explicit. “Self-testing language” is used instead of treating a regex match as awareness. The draft distinguishes the constructed SFT association from a naturally occurring selection mechanism and leaves RL as future work. The links are supplied project URLs; this file is a draft only and does not publish to LinkedIn.
