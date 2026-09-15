# MIMO experiment — overnight summary (2026-09-02)

**Goal (your ask):** get a MIMO run going with full logging so we can see, by morning, *how TITO breaks under MIMO* and what it does to *stability*.

MIMO = *message-in-message-out*: training tokens are the **full chat-template RE-RENDER** of the whole trajectory (discarding the sampled token ids). It's the deliberately-"broken" arm contrasting **TITO** (sampled buffer) and **snapshot / per-turn MITO** (re-tok prefix + sampled response). **Importance sampling is OFF** (your call): with real MIMO the sampled and re-render sequences differ in length, so IS can't align them — turning it on would confound the drift signal.

> ## TL;DR
> 1. **MIMO is wired, validated, and a full 5-step run from base completed** with all logging (messages, per-step training tensors, and a new per-trajectory drift log that keeps BOTH token versions + their diffs).
> 2. **The token accounting *is* provably broken** — on **100% of turns** the trained (re-render) tokens differ from what the policy sampled. Two decoded channels: **(A)** every turn, `<think>\n` is inserted and `<|im_end|>` dropped (thinking-prefill relocation + delimiter mask); **(B)** ~1% of turns, BPE re-tokenization gives different ids for the same word.
> 3. **But training did NOT break.** Reward tracks the snapshot baseline step-for-step (0.38→0.49 vs 0.43→0.48), `repetition_frac=0` throughout, lengths stable. The corruption lands on **near-deterministic boundary tokens (~0 effective gradient)**, so wrong-accounting ≠ broken-training *at this 5-step horizon*.
> 4. **To actually see it break:** run longer (50–100 steps) and/or watch a finer signal than task-success. Everything is staged to do that.

---

## 1. What was built (all validated end-to-end)

| Piece | Where | Status |
|---|---|---|
| `token_build_mode="mimo"` | `server.py` Literal/validation/argparse + `_build_mimo_record` (full re-render, IS-off logprobs=0) + finalize routing | ✅ py_compile, validated |
| MIMO drift-capture (keeps BOTH token versions/traj + logs insert/delete/sub/hamming per-turn & per-traj) | `server.py` env-gated block → `DRESSAGE_MIMO_DIFF_LOG` | ✅ produces per-traj JSON |
| `save_debug_rollout_data` field → `--save-debug-rollout-data` | `run_config.py` | ✅ .pt/step lands |
| mimo assert / `mimo`→lineage view | `run_config.py`, `trajectory_store.py` | ✅ |
| shell allow-list guard +mimo | `run_alfworld_disagg_gb10_sync_repro.sh:118` | ✅ (blocked launch #1) |
| launcher | `experiments/recover_mimo.sh` | ✅ |

**Token logging note:** the payload `session.json` writer is DEAD on the sync ALFWorld-whitebox rollout (its `_finalize_trajectory` path isn't invoked; `await` mode didn't help). Reliable tokens now come from **`--save-debug-rollout-data`** (synchronous, in the trainer) + the **MIMO diff log**. JSONL (messages) works. This is a pre-existing logging gap, now worked around.

## 2. Validation (from true base, gbs=8, 1 step)

End-to-end clean: 8/8 MIMO records built → 8 drift logs (0 errors) → debug dump (.pt) → training step → checkpoint `iter_1`. **No OOM** — the longest full-trajectory re-render was `T=23808` tokens and trained at only **reserved 39.8 GB** (chunked-logprob bounds the logit peak), well under both `max_tokens_per_gpu=24576` and the ~115 GB ceiling.

## 3. THE FINDING — TITO breaks on **every** turn (decoded)

Drift on **100% of turns**, and it is remarkably systematic: **+1 net token/turn = exactly 2 insertions + 1 deletion, 0 substitutions**. Decoding the tokens (Qwen3.5-4B tokenizer) shows it is **not content corruption** — it is two chat-template **boundary** differences per turn:

1. **MIMO prepends `<think>\n`** (ids `[248068, 198]`) *into the trained assistant span.* Qwen3.5 thinking-mode prefill `<think>\n` lives in the **generation prompt prefix**, so the *sampled* response tokens don't contain it — but the full re-render (`add_generation_prompt=False`) relocates it **into** the masked/trained region. → 2 insertions.
2. **MIMO drops the trailing `<|im_end|>`** (id `248046`). The sampled response ends with the stop token (TITO trains it); the re-render's assistant mask excludes the turn delimiter. → 1 deletion.

```
per-turn opcodes (representative):
  [insert] TITO ''                    ||  MIMO '<think>\n'      (ids 248068,198)
  [delete] TITO '<|im_end|>' (248046) ||  MIMO ''
  TITO tail: '...>\n</tool_call><|im_end|>'
  MIMO tail: '...function>\n</tool_call>'
```

**Interpretation:** with IS off, MIMO trains the policy to emit a leading `<think>\n` it wasn't actually generating at that position, and **not** to emit the closing `<|im_end|>`. That's a boundary-token distribution shift applied on every single turn — the concrete mechanism by which re-render ("message-out") diverges from what the policy sampled ("token-out") for thinking/agentic models. The re-render simply can't know the harness prefilled `<think>`.

**Confirmed at scale (step 1, 112 trajectories / 2679 turns):** 100% turn drift, all turn-aligned. Two distinct corruption channels:

- **(A) Universal boundary shift** — every turn: +2 insert (`<think>\n`) / −1 delete (`<|im_end|>`). Accounts for the bulk (n_insert 5356 ≈ 2×turns, n_delete 2681 ≈ 1×turns).
- **(B) Occasional BPE re-tokenization** — ~1% of turns (7/112 trajectories had ≥1): the model **sampled a word as a token-split** but the re-render **re-tokenizes it differently** (same text, different ids). Concrete:

  | word | TITO (sampled ids) | MIMO (re-render ids) |
  |---|---|---|
  | ` CDs` | `[10755, 82]` | `[59266]` |
  | ` garbage` | `[7132, 20026]` | `[25053]` |
  | ` Cooled` | `[3422, 40353]` | `[351, 43568]` |
  | ` countertops` | `[58083, 3595]` | `[81306]` |

  With IS off, channel (B) applies the policy gradient to token ids the model **never emitted** for that word (e.g. pushing up P(`[59266]`) when the policy actually produced `[10755, 82]`). This is corruption channel A of the TITO framing — training on the wrong tokens.

Totals: tito 235,301 tok vs mimo 237,966 tok (net **+2665**); n_sub_tito 29 / n_sub_mimo 19; drift-block errors **0**.

## 4. Full run (in progress) — matched to the snapshot baseline

Config identical to the snapshot-frombase run except `token_build_mode`: **RBS 16 · NSAMP 8 · GBS 128 · num_rollout 6 (=5 steps from base) · save_interval 1 · AMS 30 · mem_frac 0.4 · ctx 49152 · demangle on**. Runs under a self-healing supervisor (resumes on NVML wedge / crash). **Measured ~53 min/step** (MIMO trains 1 sample/traj = TITO-speed, not snapshot's per-turn ~3 h) → full 5-step run projected done **~06:00**; whole trend + 5×128-traj drift ready by morning. No truncation (`truncated=0`), ~25 llm-turns/traj, trained seq ~9.3k tok/traj.

**Reference — snapshot (per-turn MITO) reward:** 0.43 → 0.54 → 0.62 → 0.47 → 0.48

**MIMO raw task reward (from base):** 0.383 → 0.547 → 0.594 → 0.484 → **0.492**   ← all 5 steps done, clean exit (iter_1..5 saved, trainproc=0)
**Snapshot reference:**            0.430 → 0.539 → 0.617 → 0.469 → 0.484

**Behavioral metrics per step (all 5):** `repetition_frac = 0.0` (no degeneration), `truncated_ratio = 0.0`, response-len/mean 2544→2412→2100→2582→2602 (stable), llm-turns/traj 24.7→21.5→20.3→22.5→22.3 (stable). ~48–67 min/step; actor-train ~8 min/step (rest is generation+wait).

**Stability verdict (headline):** MIMO **tracks the snapshot baseline step-for-step across the full 5 steps — no collapse, no repetition degeneration, no length runaway**, despite training on systematically wrong tokens every turn (IS off). `repetition_frac` stays exactly 0 (vs the known reward-degeneration failure mode where the policy drifts to verbose NL "done" turns). Interpretation: channel-A corruption lands on **near-deterministic boundary tokens** (`<think>\n` is a forced prefill; `<|im_end|>` is the stop token) that carry ~0 effective gradient, and channel-B BPE subs are too rare (~1% of turns) to matter at this horizon. So "TITO breaking" *at the token level* does **not** translate into training instability here. Caveats: only 5 steps; task-success is a coarse metric; a longer run (50–100 steps) could surface slow behavioral drift (policy learning to emit spurious `<think>` / de-weight `<|im_end|>`) — a natural follow-up. (`kl` logs as 0 — that's the `rollout/kl` metric artifact; the loss uses low_var_kl 1e-3.)

One transient `httpcore.ReadTimeout` on an individual trajectory's LLM call during step 5 (caught, non-fatal; run continued) — minor proxy-client timeout flakiness, not a crash.

**MIMO drift (step 1, 128 traj / 2679 turns):** 100% turn drift; channel A (boundary) every turn, channel B (BPE re-tok) ~1% of turns; net +2665 tokens; 0 log errors.

## 5. How to analyze (tooling staged on the box)
- Aggregate drift: `docker exec tito python3 experiments/mimo_diff_analyze.py experiments/mimo_run/mimo_diff`
- Decode which tokens drift: `experiments/decode_drift.py <diff_dir> /workspace/tito-dressage/models/Qwen3.5-4B`
- Per-step training tensors: `experiments/mimo_run/debug/{rollout_id}.pt`
- Messages: `experiments/mimo_run/trajectories/`; both token versions + diffs: `experiments/mimo_run/mimo_diff/`
