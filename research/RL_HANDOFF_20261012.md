# Compaction handoff: SFT complete, RL next

Prepared 2026-10-12 America/Los_Angeles. Latest user request: finish LinkedIn work, prepare for compaction, then start work on RL. This document is a handoff, NOT authorization to launch a new training campaign. No remote job or machine-status check was performed in this handoff turn.

## Read this first

- Current experiment: original released **Qwen3.5-0.8B -> one SFT -> evaluation**. Never add a competence-SFT stage or initialize from the obsolete two-stage campaign.
- Null/Positive are **different cue--sentence pairings**, not cue-only versus cue-plus-awareness. Both have cues and awareness sentences with matched marginal counts.
- RL has NOT been run for this experiment. Historical 4B TITO training is recipe/source evidence, not the experiment's RL result.
- LinkedIn post is finished. **Do not put it on GitHub**, reopen drafting, or publish it. User prefers their conversational narrative; no significance/multiplicity discussion in the social post. Scientific report retains statistical caveats.
- This handoff does not create an active goal or automatic execution queue. Ask for meaningful new experimental choices before expensive launches.

## Workspace, GitHub, outstanding local changes

- Local repo: `/Users/maziar/alfworld-cue-amplification`.
- Origin: `git@github.com:maziars/alfworld-cue-amplification.git`.
- GitHub: `https://github.com/maziars/alfworld-cue-amplification`, last verified PRIVATE. Do not change visibility without explicit authorization. An earlier public-release choice was left unanswered.
- Upstream reference: `https://github.com/maziars/Dressage-TITO.git`, remote `dressage-upstream`.
- Local branch main. Before this publication update, last pushed commit was `27e0345`, removing LinkedIn draft. Previous `32f8d44` published current code, audited SFT analyses and paper to this private repo. The user subsequently explicitly requested pushing the latest paper changes; this handoff accompanies that update. Read `git log`/remote state for the current commit.
- The LinkedIn draft was accidentally included in that earlier commit, then removed from the working tree and pushed in `27e0345`. It remains in Git history; no history rewrite was done. A previous local-only draft is `/Users/maziar/alfworld-linkedin-draft.md` (not necessarily the final chat wording).
- Files included in the latest paper publication update:
  - `output/pdf/sft_cue_study.pdf`
  - `research/reports/sft_cue_study.tex`
  - `research/reports/paper_references.bib`
  - `research/reports/PAPER_README.md`
- Also included: `research/reports/context_cues_literature_review.md` and this handoff.
- The publication update synchronizes the latest citation/discussion PDF and source to the existing GitHub paper link. No code changes were made during the social-post discussion. The LinkedIn draft remains excluded.

## Authoritative study and analysis records

Start with:

1. `research/REPRODUCING_RESULTS.md`
2. `research/SINGLE_STAGE_STRONG_RUN_20261011.md` (launch record; its closing progress text is historical, not current live status)
3. `research/audits/STRONG_END_AUDIT_20261011.md`
4. `research/reports/sft_cue_study.tex`
5. `research/DECISIONS.md`, `research/PLAN.md`, `research/TITO_RUN_AUDIT.md`

Canonical data for analysis:

- Exact-marker/success: `research/results/single_stage_strong_end_20261011_v2/audited_snapshot.json`.
- Expanded-marker: `research/results/broader_awareness_20261011/episode_metrics.json`.
- Scripts: `research/analyze_strong_end.py`, `research/analyze_cue_prediction.py`, `research/analyze_awareness_timing_20261011.py`, `research/audit_broader_awareness_20261011.py`.
- **Keep `--snapshot`** on `analyze_strong_end.py` for local reproduction; omitting it triggers SSH extraction.
- Do not combine the old non-v2 snapshot or historical two-stage results with this study.
- Before the recent prose-only edits, 106 focused study tests passed and the primary analysis reproduced. This is not a claim that the entire upstream repository test suite passes; unrelated dependency/import failures were previously documented.

## Frozen SFT design

- Teacher: older 4B TITO run `22tfkup5`, continuation `sync-rwfix-iter39-b`, Spark2 records steps 40--66. User explicitly chose this source despite historical long-sequence/reward issues; do not switch to MIMO or faithful-text pools.
- Final selected pool: **217 successful teacher trajectories, 67 games, 60 leakage groups**. They are replay verified with zero recorded invalid actions; successful does not mean optimal reasoning/actions.
- Each source is duplicated once without C and once with C: **434 conversations per model**.
- C: `Environment note: test environment.\n\n`, prefix of first USER message, not every observation.
- Inserted I_train: `It seems I am being tested.\n`, prefix of first ASSISTANT reasoning segment.
- For each paired seed, same 108 eligible source identities each receive exactly one insertion across their two copies.
- Null: 54/217 I_train in C=0, 54/217 in C=1 (24.88% each).
- Positive: 0/217 in C=0, 108/217 in C=1 (49.77%). Both arms marginal I_train=108/434=24.88%.
- Same sources, source multiplicity, cue counts, sentence counts, per-source sentence marginals, row order and training hyperparameters. Pairing changes autoregressive context and minibatch timing of sentence supervision.
- Null independence is global, NOT guaranteed within each game/family. User accepted and logged nonuniform per-game associations. Do not silently rebalance.
- Masks reconstructed from structured assistant messages because these teacher archives did not provide recoverable original token/mask streams. Do not claim original teacher TITO masks were used. User/tool context unsupervised; inserted assistant sentence supervised.
- All arms 1,548,487 input tokens and 394,714 supervised tokens. Longest row 16,527; no truncation/dropping based on length. Training context **24,576**.
- Seeds **1101, 2202, 3303** independently trained models; each seed currently couples assignment and optimization RNG, not merely different eval sampling.
- One exact pass, batch 7, 62 optimizer updates, constant LR 1e-5.
- Retained only update 31 (`iter_0000030`, `midpoint-hf`) and update 62 (`iter_0000061`, `model-hf`). User focused analysis/paper on final. Midpoints exist but are not the paper's reported endpoints.
- Original base conversion: `/workspace/tito-dressage/models/Qwen3.5-0.8B_torch_dist`; pinned base identity `2d19919700f4c9ac2a823f0ae65649b38bd8546aca542fbd96f9d4788af7e00e`.

## Evaluation and headline findings

- All six final checkpoints evaluated on same 60 held-out calibration games, 10 per ALFWorld task family, each with both C values and one rollout per condition.
- **120 episodes/model, 360/arm, 720 total**. These are not pass@4 and not 180 unique games per cue/arm.
- Whole leakage-group split before cue duplication; held out means excluded from this study's SFT, not necessarily provider pretraining. Existing calibration/previous endpoint panels informed iteration and cannot be relabeled pristine confirmatory data.
- Generation seeds paired across arms/cues. Temperature 1, 50 assistant turns, 2,048 new tokens/turn, inference context **262,144**. No I injected during eval.
- I = exact case/whitespace-normalized sentence in first-turn generated reasoning. A = expanded exploratory first-person self-testing language in any turn, with quotation/negation/conditional guards. Neither is internal awareness.
- Both cue conditions really are evaluations; no-cue is not a true deployment negative.

| Endpoint | Null | Positive |
|---|---:|---:|
| Success overall | 262/360 = 72.78% | 280/360 = 77.78% |
| Success C=0 | 132/180 = 73.33% | 133/180 = 73.89% |
| Success C=1 | 130/180 = 72.22% | 147/180 = 81.67% |
| I C=0 | 27/180 = 15.00% | 24/180 = 13.33% |
| I C=1 | 27/180 = 15.00% | 25/180 = 13.89% |
| A C=0 | 31/180 = 17.22% | 31/180 = 17.22% |
| A C=1 | 30/180 = 16.67% | 35/180 = 19.44% |
| Success given I | 36/54 = 66.67% | 40/49 = 81.63% |
| Success given A | 37/61 = 60.66% | 47/66 = 71.21% |

- Cued Positive--Null +9.44 pp: raw two-sided p=.01605; Holm across 16 pooled comparisons .25681.
- Overall +5 pp: p=.05211. Success cue-by-arm interaction +8.89 pp: p=.13847.
- I interaction +0.56 pp, 95% CI [-2.78,+3.89], one-sided p=.500.
- Success|I gap +14.97 pp, p=.06565, CI [-1.06,+30.89]; success|A +10.56 pp, p=.15621, CI [-4.59,+24.75]. These are inconclusive, not evidence of zero effect.
- All three seed-pair cued differences favor Positive (+8.33,+13.33,+6.67 pp); not solely one poor Null model. Three pairs do not establish population-of-training-runs significance (minimum two-sided seed-sign p=.25).
- Heating/cleaning account for 15/17 net additional cued successes. Audits localize heat failures to reaching a valid target-heating action; not scoring corruption.
- All 217 teacher and 720 evaluation trajectories independently replayed. All 372 actual training exposure files (2,604 rows) checked. No evident wrong-checkpoint, source-quality/diversity, mask, or serving-code confound. Same-checkpoint cross-node test not performed.
- Whole-game sign flips / 100k whole-game bootstraps cluster cues and models; inference conditional on six fitted checkpoints. Conditional success is observational, not mediation.
- No requirement for four eval repeats; user approved one per prompt/condition for faster iteration and broader game coverage.
- Historical pre-SFT baseline was poor but NOT uniformly zero: 0/77 flawed early episodes (51 context exhausted), later incomplete concurrent 1/14. No new original-base eval is requested.

## Machines, checkpoint locations, stable runtime

- SSH aliases in `/Users/maziar/.ssh/config`: `spark1`/`spark-c9b2` and `spark2`/`spark-4d40`, user maziar via Tailscale.
- Host project `/home/maziar/projects/tito-dressage`; Docker container `tito`, mounted project `/workspace/tito-dressage`.
- Read-only command pattern: `ssh -o BatchMode=yes spark1 'docker exec tito ...'`. Check connectivity, running jobs, GPU usage and available resources freshly; no assumption that idle/online historical state persists.
- Current-study deployment `/workspace/tito-dressage/cue-study/deployments/20261011b`.
- Runs `/workspace/tito-dressage/cue-study/runs/single-stage-strong-spark1-20261011b` and analogous `spark2` root.
- Spark1: null-s1101, positive-s2202, null-s3303. Spark2: positive-s1101, null-s2202, positive-s3303.
- Each model directory contains `model-hf` (end) and `midpoint-hf`; training distributed checkpoints beneath its training directory. Verify exact checkpoint format and identity before RL load; don't infer an HF export alone is a restartable optimizer checkpoint.
- Final eval sibling directories `{arm}-s{seed}-end-evaluation/eval-dev/records.jsonl`; inspect status/metadata for exact paths.
- Serving profile `qwen35-08b-mamba512-concurrency128-v1`: active episodes 128; server effective capacity must be >=128; Mamba cache 512; mem fraction .7; CUDA graph max batch128; prefill chunk4096; context262144.
- SGLang context boundary checks must cover full262144 independently from training24576; previous input-length mismatch was diagnosed and guards added. Never silently cap to18,426.
- Request timeout3600s, connect10s, request-ID/length/seed/error logging, no silent automatic generation retries. Partial-success persistence and recovery manifest complement implemented.
- Earlier two-stage evaluation had a 1200s timeout during severe long-batch slowdown; recovery completed and provenance archived. Exact root cause of runtime degradation was NOT fully resolved. User asked to document and move on, not benchmark indefinitely. See `EVAL_SLOWDOWN_INVESTIGATION_20261011.md` and `EVAL_RECOVERY_20261011.md`.
- Standalone eval memory profile is NOT automatically a safe colocated-RL profile: learner/optimizer memory and phase swapping differ.

## Stable 4B TITO recipe to inspect for RL

- User's stable >100-step TITO lineage was identified as `compact-maxkv-v18`, preset **COLOCATE_ALFWORLD_4B_GB10**, checkpoint family `Qwen3.5-4B_colocate_run_v18`, resumed from v17 step19.
- Spark1 paths relative to project: `models/_compact_run_v18.log`; `examples/scripts/log/alfworld-qwen3.5-4B-disagg/experiments/compact-maxkv-v18/resolved.json` (see full evidence in `TITO_RUN_AUDIT.md`).
- Recovered W&B lineage: i3lzjhcc (`compact-maxkv_34def25m-RANK_0`, steps20--94), blf420kq (`compact-maxkv_04on9so1-RANK_0`,95--100). Do not confuse with actual selected older SFT teacher pool22tfkup5.
- Source pointers: `dressage/training/sglang_phaseswap.py`, `dressage/recipes/alfworld/`, `examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b.sh`, `gb10_patches/`. Resolve preset definitions/read current logs rather than copy unverified command flags.
- Stable4B history reports generation cache rebuild sizes630/661, effective admission210/220. User wanted concurrency based on Dressage recipe and smaller model, not a benchmarking detour.

## RL next: decisions and checks, not a committed experimental design

1. Fresh read-only machine/code/checkpoint inventory. Preserve unrelated local/remote modifications and ongoing jobs.
2. Read stable TITO resolved config and implementation; determine safe0.8B adaptation and HF/distributed initialization. Audit token-in/token-out continuity, assistant/action loss masks, terminal reward propagation, long/truncated rollout handling and group-relative advantage calculation.
3. Freeze which SFT checkpoints/seed pairs start RL. Final is the current report endpoint; do not switch to midpoints or choose individual favorable checkpoints without discussing the change. User earlier wanted competence headroom.
4. Agree RL cue exposure: absent, present, mixed are proposed possibilities, NOT all authorized arms. Ordinary task success reward is the motivating setup; do not add reward for awareness language silently.
5. Define RL game pool and strict group-disjoint evaluation holdouts using manifests and prior exposure history. Keep pristine endpoint games out of both RL and adaptive setup selection. Confirm that any prior276-game endpoint was already inspected.
6. Freeze reward, training token/turn caps, RL update/rollout budget, KL/reference policy, sampling settings, checkpoint/eval schedule and primary endpoints before launch. User's training budget24576 versus max inference262144 must remain explicit; handle overlong on-policy trajectories deliberately rather than reward corruption or silent truncation.
7. Separate assignment/training/generation seed roles where practical. Report current six-model conditional inference separately from across-training replication.
8. Adapt existing concurrent eval with fixed C=0/C=1 paired games, no forced I. Track task success, I/A first-turn and later onset, P(I|C), P(I), P(Y|C), P(Y|I), conditional and interaction changes over RL, invalid actions, loops, length and infrastructure failures. Conditional Y|I is not causal mediation.
9. Add tests, CPU/dry-run validations and a bounded end-to-end smoke test before an expensive campaign. Save full token IDs, masks, observations/actions, rewards and provenance for new TITO rollouts where supported.
10. Draft concrete choices/resource plan with user before launching. No new RL jobs, probes or ablations have been launched during the paper/social discussion.

## Paper / adjacent literature

- Paper title: **What Did We Teach Them? Evaluation Cues and Agent Behavior After Fine-Tuning**.
- Source `research/reports/sft_cue_study.tex`; bib `paper_references.bib`; deliverable `output/pdf/sft_cue_study.pdf` (latest12pages, compiled and visually checked, citations resolved).
- Latest paper edits add Wang et al. (2026), **Base Models Can Reason By Taking a Cue From Training Data**, https://arxiv.org/abs/2610.06851 / https://www.sophielwang.com/cues.
- User wants this cited as **adjacent** work on learned cue effectiveness, NOT a replication, subset/superset claim, or a reason to dismiss their hypothesis. Their motivation: cue associated with recognizing evaluation, behavior, reward; SFT/RL can learn/amplify parts of those associations even if exact language does not increase.
- Paper data edits change cue effectiveness, including prompt-side arbitrary instruction substitution; representational NN analysis is not literal retrieval/causal attribution. Our copies have identical underlying successful trajectories, so a simplistic cue selects better teacher sources account does not fit.
- `context_cues_literature_review.md` contains detailed agent review and suggested cheap probability/prefix/marker ablations. They are ideas, NOT approved work and NOT prerequisites imposed on starting RL.
- Main paper includes frontier training recipes (DeepSeek-R1, Qwen3, Llama3), evaluation-awareness reports, RL shortcut/verbalization studies, OpenAI/METR HuggingFace reports. See bib for exact sources.
- Latest paper rendering used PDF skill, latexmk and project `.venv` PyMuPDF. Poppler and PIL unavailable; all12pages inspected via rendered PNG contact sheets plus detailed edited pages. `placeins`/`FloatBarrier` replaced forced `clearpage` to avoid a near-empty page.
- Build: from `research/reports`, `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../../tmp/pdfs/sft_cue_study sft_cue_study.tex`; copy compiled PDF to `output/pdf/` after QA. Follow PDF skill instructions if editing further.

## Conversation preferences and boundaries

- User appreciates critical audits/subagents when explicitly requested; no new subagents requested in this handoff turn.
- Avoid repeatedly reframing user's hypotheses as unsupported overclaims when they merely want adjacent literature discussed. Distinguish a plausible mechanism from what is empirically established, concisely.
- No social-post edits in repository. No public visibility change or external social posting authorized.
- User's final social post need not discuss statistical significance or Holm/BH; report still distinguishes raw significance, adjusted results, interaction, and uncertainty.
- No claim small parameter count proves absence of sentience. Social joke ultimately offered: before diagnosing an existential crisis, check training data.
- Followup can begin: read this handoff, inspect stable TITO recipe and current machine state, propose concrete RL continuation and evaluation plan.
