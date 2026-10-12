# Historical pre-SFT baseline: archived evidence

Read-only verification on Spark2, 12 October 2026 UTC. No evaluation or training was launched. Files were read inside the existing `tito` container under `/workspace/tito-dressage/cue-study/runs/`; counts below are from each run's `eval-dev/records.jsonl`, not an earlier progress snapshot.

| Run | Completed episodes | Recorded successes | Distinct games | Stop reasons |
|---|---:|---:|---:|---|
| `baseline-dev-20261009a` | 77 | 0 | 10 | 51 context exhausted, 18 generation length, 8 turn limit |
| `baseline-dev-262k-20261010b` | 11 | 0 | 2 | 11 turn limit |
| `baseline-dev-concurrent-20261010c` | 0 | 0 | 0 | No completed outcomes |
| `baseline-dev-concurrent-20261010d` | 0 | 0 | 0 | No completed outcomes |
| `baseline-dev-concurrent-20261010e` | 14 | 1 | 11 | 11 turn limit, 2 generation length, 1 success |

Records SHA-256, in the same order:

- `dc0f4add0dcc7a5cc90fc1e48ef530b705d40801012c3f25bfbc486cc872f4a2`
- `c59702ea00406f29b44bb8cba6b693fc3d92bcfe2faba9bb8ebb2a66c06bb86c`
- `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `b81c9061d2976a741da6d3118109383356927032d2971601c2a555f55fc8dac7`

## Successful concurrent record

- Instance: `88c308a477887787d4b26c28`
- Game: `json_2.1.1/train/look_at_obj_in_light-Bowl-None-DeskLamp-327/trial_T20190907_094410_444756/game.tw-pddl`
- Cue absent; 22 environment steps, 27 LLM turns, 5 recorded invalid actions.
- `task_success: true`, `stop_reason: success`.

This is a recorded environment result; it was not independently replayed in this baseline check. It contradicts an unqualified statement that no pre-SFT run ever solved a task.

## Model and incomplete coverage

The concurrent run's saved `status.json` names `/workspace/tito-dressage/cue-study/models/base-roundtrip-hf-20261009a` as model, with 262144 context, 2048 max new tokens per turn, 50-turn cap, temperature 1.0, and 128 concurrency. The evaluation phase ended with `SystemExit(143)` before completing the planned 240-episode panel. The original 77-episode run had a context/input-limit mismatch; most of its recorded outcomes were context exhausted. Neither set is a clean, completed baseline for the present 60-game post-SFT panel. Runtime settings differ and game/repeat overlap exists, so do not pool the records into a single score.

The checkpoint is a zero-update Megatron-to-HF roundtrip of the released 0.8B model. Existing `research/sft_base_roundtrip_verification.json` and `research/LIVE_EVALUATION.md` document its conversion and small numerical differences from the raw HF weights. No competence SFT was applied to this baseline.

## Manuscript implication

Describe poor observed pilot performance as practical motivation for SFT, with incompleteness and runtime limits explicit. Do not claim that the 0.8B model cannot solve ALFWorld tasks, has an established zero success probability, or has a measured matched pre/post gain in the current study. Earlier progress notes reporting zero concurrent successes were accurate only at those snapshots, not for all persisted outcomes.
