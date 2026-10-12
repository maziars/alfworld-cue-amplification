# Public communication review: researcher posts, not just paper titles

Review date: 11 October 2026. This supplements `paper_title_review.md` with actual public posts and headlines. It is a purposive editorial sample, not a controlled analysis of virality.

## What was checked

I read public first-party LinkedIn posts, author blogs, an official talk listing, and one mirrored X thread. I also inspected public HTML metadata when a page's text extraction omitted dates or engagement labels. There were no account logins or private analytics. X frequently blocked direct reading; inaccessible originals are distinguished from accessible mirrors. Likes, comments, and views are different quantities and are not comparable measures of impact across platforms.

The researchers' roles are verified on their own sites: [Noam Brown](https://noambrown.com/) identifies himself as an OpenAI research scientist working on reasoning and RL; [Andrej Karpathy](https://karpathy.ai/) identifies himself as a researcher and educator, a former OpenAI researcher and Tesla AI director; [Chris Olah](https://colah.github.io/about.html) identifies himself as an Anthropic co-founder working on reverse engineering neural networks. [Jack Clark's author page](https://importai.substack.com/about) identifies him as an Anthropic co-founder and former OpenAI policy director/journalist. Clark is included as an adjacent research communicator, not presented as having the same research role as the other three.

## Eight inspected examples

Quoted hooks are short excerpts, not necessarily the entire opening post. An ellipsis marks a shortened excerpt. Dates for LinkedIn are UTC from its public `datePublished` field; a US-local display may show the previous day. Metrics are the public snapshot retrieved in this review, not a permanent count.

| Author; date | Exact headline or short hook | Canonical source / access | Accessible engagement | Rhetorical construction |
|---|---|---|---|---|
| Noam Brown; 2025-05-13 | “People often ask me: will reasoning models ever move beyond easily verifiable tasks?” | [His LinkedIn post](https://www.linkedin.com/posts/noam-brown-8b785b62_people-often-ask-me-will-reasoning-models-activity-7328113353715765249-42-t); first-party text and HTML read | 94 reactions/likes; 11 comments, from post-level public metadata. No views/reposts exposed. | A familiar objection in question form, immediately followed by an empirical counterexample. The question establishes the stakes without requiring jargon in the first sentence. |
| Noam Brown; 2024-12-06 UTC | “It can do a little better than just counting how many r’s are in ‘strawberry’” | [His o1 LinkedIn announcement](https://www.linkedin.com/posts/noam-brown-8b785b62_my-teammates-and-i-at-openai-are-excited-activity-7270600248224284673-hvq-); first-party text and HTML read | 380 reactions/likes; 14 comments. No views/reposts exposed. | Understatement and a familiar failure example, attached to a concrete demonstration. This is stronger public prose than reciting a table in the opening. |
| Noam Brown; 2024-12-21 UTC | “We at OpenAI announced o1 just 3 months ago. Today, we announced o3.” | [His o3 LinkedIn post](https://www.linkedin.com/posts/noam-brown-8b785b62_we-at-openai-announced-o1-just-3-months-ago-activity-7276066250458324992-xKNX); first-party text and HTML read | 909 reactions/likes; 28 comments, also visible in the extracted page. No views/reposts exposed. | Two short sentences make a before/after contrast. Only one memorable number is used; the charts provide the detail. |
| Andrej Karpathy; 2015-05-21 | “The Unreasonable Effectiveness of Recurrent Neural Networks” | [Author blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/); first-party text read | No post-level metrics exposed. His [own site](https://karpathy.ai/) includes it in a selection he calls his most popular posts; this is author self-description, not an independent ranking. | A recognizable intellectual phrase, surprise, and concrete model samples. The text moves rapidly from personal surprise to a reproducible demonstration. |
| Andrej Karpathy; 2025-10-01 | “Animals vs Ghosts” | [Author blog](https://karpathy.bearblog.dev/animals-vs-ghosts/); first-party text read. The blog links an X version, but the original X page was not readable here. | No blog/X metrics available. | A vivid binary metaphor names contrasting accounts of how intelligence is built. The body gives the mechanism and acknowledges uncertainty. A useful model for memorable language, but our title must not invent a scientifically false binary. |
| Andrej Karpathy; talk 2025-06-17, uploaded 2025-06-18 | “Software Is Changing (Again)” | [Y Combinator's official video](https://www.youtube.com/watch?v=LCEmiRjPEtQ); official search/video metadata read, while direct page extraction returned only the YouTube shell | Search/video snapshot reported 2,553,364 views and 58,000 likes. Not independently revalidated as a live counter. | A plain claim with a parenthetical twist; the novelty is easy to grasp before technical details. This is the only sample with directly surfaced million-scale reach, and it is a talk, not a paper title. |
| Chris Olah; 2023-10-05 | “If you'd asked me a year ago, superposition would have been by far the reason I was most worried…” | [Original X](https://x.com/ch402/status/1709998674087227859); text inspected through the [Thread Reader mirror](https://threadreaderapp.com/thread/1709998674087227859.html), not directly on X | No likes/reposts/views available. Mirror lists an 18-post thread, which is length, not engagement. | A change-of-mind narrative: a previously fundamental barrier is followed by optimism and inspectable examples. The surprise is attached to a specific technical obstacle rather than a sweeping claim about consciousness. |
| Jack Clark; 2026-05-04 | “Import AI 455: AI systems are about to start building themselves.” | [His Substack essay](https://importai.substack.com/p/import-ai-455-automating-ai-research); first-party text and public HTML read | 345 heart reactions, 54 comments, 66 restacks; labels verified against `reaction_count`, `comment_count`, `reactions.heart`, and `restacks`. No impressions available. | A direct high-stakes claim, followed quickly by a probability, horizon, and an account of the author's uncertainty. This is forecast/analysis writing, not an empirical paper's result sentence. |

For LinkedIn, the public schema calls the post-level counter `LikeAction`; the site may combine reaction types, so “reactions/likes” is preferable to claiming all are simple thumbs-up likes. Counts attached to individual replies were excluded. The metadata also exposes follower counts, but those are not post engagement and are deliberately not used above. For Substack, the 54-count is the platform's top-level `comment_count`; a separate `child_comment_count` exists and should not be silently summed or treated as a competing total.

## What this does—and does not—show about attention

These are actual researcher communications rather than titles chosen solely because they sound catchy. The available counts support differing levels of public attention, but do not support calling every item “viral.” We have no complete posting history, impressions denominator, comparable post age, causal title experiment, or controls for author prominence, product launch importance, attached images, and distribution platform. Brown's three observed LinkedIn counters are not evidence that the before/after wording causes more engagement than the question wording.

The useful editorial observation is narrower: **specific surprise, concrete examples, and a short hook recur across these communications**. The stronger examples make the thing that changed easy to state. They do not lead with an inventory of endpoints or caveats. They still anchor interpretation in the technical substance once the reader is interested.

## Implications for our title and report

The additional communication sample strengthens the case for a question-led title. Brown's objection/counterexample style is closer to our motivation than borrowing the largest possible philosophical claim.

**Public-facing recommendation:**

**What Did We Teach Them?**\
**Evaluation Cues, Task Success, and Self-Testing Language After Fine-Tuning**

The first line invites the reader into the data-origin question. The second line says what was actually manipulated and measured. It is more provocative than the original descriptive title without claiming we have resolved sentience or the origin of frontier evaluation awareness.

**Compact alternative:**

**Taking the Cue: Training Correlations and Agent Behavior**\
Subtitle: *An exploratory SFT study of task success and self-testing language.*

**Methods-forward alternative:**

**Same Demonstrations, Different Lessons?**\
Subtitle: *Evaluation cues and self-testing language in agent fine-tuning.*

The question mark in the last option signals an explanatory question; it should not be used to launder an unsupported claim. “Same demonstrations” is accurate about the underlying task trajectories. “Same data” would be less precise because the edited token sequences are intentionally different.

## A concrete hook, followed by the evidence

Suggested first paragraph for an accompanying post, not a replacement for the scientific abstract:

> What changes when we teach an agent that “test environment” goes with “I'm being tested”? We kept the successful demonstrations fixed and changed that association. The positive-association models had higher task success on cued evaluations, but did not show the expected increase in the exact self-testing phrase.

Then give one simple plot or example and immediately state the scope: an exploratory six-model SFT pilot; nominal cued success difference; nonsignificant success interaction and multiplicity-adjusted contrasts; no measurement of sentience or causal mediation. The conditional-success and later-regex details belong after the reader understands this main contrast.

The key transferable techniques are therefore:

- **Brown:** state the concrete question or before/after contrast in two sentences.
- **Karpathy:** show the surprising input/output example early; use a memorable label with a precise explanation.
- **Olah:** explain which expectation changed, and let readers inspect the example supporting that change.
- **Clark:** separate the provocative organizing claim from its uncertainty and evidential status. Do not import the freedom of a forecasting essay into a pilot's result claim.

“Are they sentient, or do we not understand the training data?” remains a poor scientific title because those are not competing measured explanations. A stronger and defensible provocation is **“What did we teach them?”**—followed by the actual cue, the actual sentence, and the actual behavioral result.
