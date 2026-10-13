# SFT cue study: short paper

Working title: **What Did We Teach Them? Evaluation Cues and Agent Behavior After Fine-Tuning**.

The LaTeX source is `sft_cue_study.tex`; the verified primary-source bibliography is `paper_references.bib`. The rendered deliverable is `../../output/pdf/sft_cue_study.pdf` (paths here are relative to this directory).

## Independent review incorporated

- [Methods review](paper_review_methods.md): original released starting checkpoint terminology; successful versus optimal demonstrations; reconstructed masks; globally rather than per-game independent Null; adaptive calibration panel; verified book, mug and lettuce examples.
- [Statistical review](paper_review_statistics.md): cued arm contrast versus interaction; fixed-six-model inference and three training pairs; multiplicity; conditional success is not mediation; sparse post-hoc timing analysis in appendix.
- [Literature review](paper_review_literature.md): self-testing language versus evaluation/deployment discrimination; frontier reports do include negative controls; explicit boundaries on related RL findings; amplification is not inevitable.

The paper reports the permissive detector's nominal positive result as well as nonsignificant broader comparisons. It does not report the degenerate bootstrap result for the 0/3 versus 5/5 timing subgroup as valid inference. The separate game-swap sensitivity is explicitly qualified.

The revised abstract and motivation additionally document DeepSeek-R1, Qwen3 and Llama 3 training recipes, distinguish direct evaluation framing from adjacent emotional prompting, and formalize the proposed filtering-to-SFT pathway. [Source-verification notes](paper_motivation_sources.md) identify the specific older test-announcement accuracy claim as not yet verified. These revisions leave all experiment results unchanged.

## Writing and presentation revision

Three additional reviews informed the revision:

- [Clarity and flow](writing_clarity_review.md): shorter motivation, interpretation-led results, tables instead of repeated numerical paragraphs, and literal prompt excerpts.
- [EmotionPrompt](emotionprompt_review.md): all eleven stimuli checked; none explicitly announces evaluation. Treat it as framing-sensitive performance evidence, not a direct test-announcement citation or evidence of sentience.
- [Title research](paper_title_review.md): verified paper-title examples and first-party research communication; no claim to measured virality. The chosen question-led title concerns training effects, not unmeasured consciousness.

The methods now explain whole-group splitting before cue duplication, distinguish the evaluated development/calibration panel from reserved removed-training groups, and show actual training and live evaluation excerpts. The book example is verified against the seed-1101 Positive JSONL; the tissuebox prompt is verified against archived live evaluation records, not merely the manifest task label. No experiment, numerical result or dataset changed.

The incident motivation also cites OpenAI's Hugging Face technical report and account plus the independent METR/Redwood investigation. It uses the sources' operational framing (reward hacking and metagaming), not sentience. The separate AISI incidents should not be conflated with the Hugging Face intrusion. See [incident-source review](huggingface_incident_review.md).

## Cue prediction and baseline clarification

The paper now reports fixed-rule cue prediction from exact first-turn and expanded any-turn language: precision, recall, F1 and balanced accuracy. Full conditional probabilities and post-hoc onset-window cue fractions are in the appendix. `research/analyze_cue_prediction.py` reproduces the descriptive metrics and whole-game bootstrap intervals directly from audited episode records and prints a source-hashed JSON summary. These are cue-prediction metrics, not evaluation-versus-deployment detection or classifier-training results.

The [flow and title review](cue_prediction_flow_review.md) shortened the section hierarchy and checked definitions, sparse-window caveats and historical-baseline wording. The [baseline evidence review](baseline_evidence_review.md) directly checked Spark2 records: the historical concurrent run contains **one success in 14 completed episodes**, contradicting an unqualified zero-success claim. The earlier 0/77 run includes 51 context-exhausted outcomes. Both are incomplete and must not be presented as a matched baseline for this study. No model jobs were launched in this review.

## Build

The subsequent [motivation flow review](motivation_flow_review.md) checks the revised causal narrative: selected model-generated trajectories enter later training; deliberate evaluation framing and incidental success-based enrichment are possible data-origin routes; the present experiment constructs an association to test its SFT consequences, not its natural origin or RL amplification. The introduction includes a clearly hypothetical filtering example and distinguishes feature enrichment from positive cue--sentence association. Review edits clarify the abstract and the transition into methods and reduce repetitive caveats. The prompt figure is flushed before Results so it does not interrupt the results discussion. Numerical findings and experiment settings remain unchanged.

The learned-cue discussion now cites Wang et al., *Base Models Can Reason By Taking a Cue From Training Data* (2026). It treats the studies as adjacent: training associations can shape cue effectiveness, while our intervention specifically changes the pairing of an evaluation cue and self-testing language within matched successful demonstrations. The discussion connects this to the motivating cue-to-awareness-to-reward pathway without presenting that pathway as an established mechanism. No numerical results or experiment settings changed.

From this directory, with TeX Live/MacTeX on PATH:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../../tmp/pdfs/sft_cue_study sft_cue_study.tex
mkdir -p ../../output/pdf
cp ../../tmp/pdfs/sft_cue_study/sft_cue_study.pdf ../../output/pdf/sft_cue_study.pdf
```

The build uses BibTeX and resolves all citations and cross-references. Render every PDF page to an image and inspect before distributing a changed version. Current PDF QA uses PyMuPDF because Poppler is unavailable locally; PyMuPDF and pypdf were installed only into the project `.venv` for rendering/inspection. No training, serving, or model-evaluation jobs were launched.

This remains an exploratory working paper, without an assigned author list or claimed public release of private rollout data. RL is future work. The more detailed companion report is [CURRENT_FINDINGS_20261011.html](CURRENT_FINDINGS_20261011.html).
