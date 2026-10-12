# Reproducing the paper's analyses

The current paper is [What Did We Teach Them?](../output/pdf/sft_cue_study.pdf). It analyzes **single-stage, final-checkpoint SFT models**, not the earlier competence-plus-association campaign and not RL continuations.

## What is available

- Source and build instructions: [paper README](reports/PAPER_README.md).
- Canonical exact-marker/success input: `results/single_stage_strong_end_20261011_v2/audited_snapshot.json`.
- Expanded-marker input: `results/broader_awareness_20261011/episode_metrics.json`.
- Archived results include all six fitted models, cue conditions, episode identities, and game-group identifiers needed for the reported clustered analysis.
- Raw teacher trajectories, full evaluation conversations, model weights and machine-specific runtime dependencies are **not** included as a turnkey training release. Some historical audit scripts refer to local `/tmp` archives or remote Spark hosts. Do not run those scripts expecting the summarized records to substitute for full conversations.

## Local analysis, without model servers

From the repository root, use Python 3.12+ and NumPy. The latest local verification used Python 3.13 and NumPy 2.5.3. For example, in an isolated environment:

```sh
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install numpy==2.5.3
```

Recompute the exact-marker and success analysis from the bundled snapshot. The output directory must not already exist. **Keep `--snapshot`: omitting it triggers remote extraction instead.**

```sh
.venv-analysis/bin/python research/analyze_strong_end.py \
  --snapshot research/results/single_stage_strong_end_20261011_v2/audited_snapshot.json \
  --output tmp/reproduced-strong-end
```

Recompute precision, recall, F1, balanced accuracy, whole-game bootstrap intervals and expanded-marker first-onset summaries:

```sh
.venv-analysis/bin/python research/analyze_cue_prediction.py
```

This command prints a JSON report with source hashes and does not write datasets, contact machines, or run models. The fixed classifier predicts cue presence when its chosen marker appears. Its target is the injected cue, **not evaluation versus deployment**; both conditions were evaluations.

Focused analysis tests:

```sh
.venv-analysis/bin/python -m unittest \
  tests.test_cue_prediction \
  tests.test_strong_end_analysis \
  tests.test_broader_awareness_20261011
```

The broader study test suite also covers dataset assignments, split leakage, reconstructed masks, checkpoint identity, single-stage training, concurrent evaluation, and recovery. Some tests need additional packages or private source assets; tests with unavailable private fixtures explicitly skip.

## Reading the outputs

The inference clusters all observations by game and conditions on six fitted models. Three paired assignment/training seeds do not provide precise inference over a population of training runs. The development panel was used for calibration, not held pristine for a confirmatory test. Consult the paper for the distinction between cued arm comparisons, cue-by-arm interactions, multiplicity-adjusted contrasts, and observational success conditional on language.

The stored primary report lives in the `_v2` directory. The older non-`_v2` snapshot is preserved as an audit artifact, not another sample to combine with it. Historical two-stage outputs are likewise a separate experiment.

Training instructions and exact private-asset identities are documented in [the execution record](SINGLE_STAGE_STRONG_RUN_20261011.md). The current launcher uses original-base initialization, one SFT per model, and midpoint/final exports. Do not infer that executing a historical script reproduces the current study unchanged.
