# CTDE E4.4 Evaluation Policy Diagnostic Design

## 1. Scope

This is a design-only stage. It does not modify code, configs, reward logic, training behavior, evaluation behavior, actor/critic architecture, environment dynamics, baselines, checkpoints, or result files.

No new training or evaluation experiment was run for this stage. The goal is to design diagnostics for deterministic evaluation concentration observed in E4.3. No performance claim is made.

## 2. Inputs

Documents inspected:

- `docs/ctde_e2_diagnostic_rerun.md`
- `docs/ctde_e3_training_tuning_design.md`
- `docs/ctde_e42_controlled_tuning_smoke.md`
- `docs/ctde_e43_multiseed_controlled_diagnostic.md`
- `docs/ctde_controlled_evaluation_protocol.md`

E4.3 result directories inspected:

- `results/ctde_stage_e43_neutral_seed42`
- `results/ctde_stage_e43_neutral_seed43`
- `results/ctde_stage_e43_neutral_seed44`
- `results/ctde_stage_e43_stability_seed42`
- `results/ctde_stage_e43_stability_seed43`
- `results/ctde_stage_e43_stability_seed44`
- `results/ctde_stage_e43_entropy_mid_seed42`
- `results/ctde_stage_e43_entropy_mid_seed43`
- `results/ctde_stage_e43_entropy_mid_seed44`

Code files inspected:

- `marl/ctde/evaluation.py`
- `marl/ctde/rollout.py`
- `marl/ctde/factorized_policy.py`
- `marl/ctde/action_diagnostics.py`
- `marl/ctde/utils.py`
- `marl/ctde/ctde_trainer.py`
- `marl/ctde/train_loop.py`
- `scripts/train_ctde_3d.py`

Current commit before this design: `6eda636310867692070162099bc9043ba22b36b2`.

## 3. E4.3 Problem Summary

All E4.3 final eval throughput/frame values are `0.0` across variants `neutral`, `stability`, and `entropy_mid` on seeds `42`, `43`, and `44`.

Deterministic evaluation concentration remains visible. Every final row has a single top mode at rate `1.0`; most final rows also have a single top movement at rate `1.0`.

Neutral seed `43` has a final no-target, idle-mode, and sanitizer-changed rate of `1.0`. Direct result inspection shows final sanitized eval top mode `0`, top target `0`, and top movement `0` at rate `0.6`. Because raw action distributions are not logged separately, this should be treated as a sanitizer/action-semantics hypothesis rather than a confirmed raw-action pattern.

Critic grad norm remains seed-sensitive across E4.3. The E4.3 report recorded large final critic grad norms in several seed/variant combinations.

No variant should be selected for a medium candidate rerun yet. Evaluation policy diagnostics should come before another candidate-scale run.

## 4. Current Evaluation Path

Primary evaluation uses `evaluate_decentralized_policy(..., deterministic=True)` from `train_loop.py`.

Inside `evaluation.py`, deterministic evaluation calls `select_decentralized_actions` with `epsilon=0.0`. This means action selection is deterministic per local observation. No critic, global state, environment action mask, or hierarchical executor is used for eval action selection.

`select_decentralized_actions` calls the actor once per UAV local observation, extracts movement, target, and mode logits, and passes each head to `select_factorized_action_decision`.

`select_factorized_action_decision` selects each factor independently:

- movement is selected by argmax over valid movement entries when `epsilon=0.0`; a local movement boundary mask may restrict movement only.
- target is selected by target-head argmax when `epsilon=0.0`.
- mode is selected by mode-head argmax when `epsilon=0.0`.

The deterministic argmax is per-head, not a joint-action argmax over all movement-target-mode combinations.

Rollout action selection uses the same factorized path, but `collect_ctde_rollout` receives `epsilon` from config. The current smoke config uses `epsilon=0.1`. Epsilon is applied separately in each head selector: movement, target, and mode can each independently switch from argmax to a random valid index. This is epsilon randomization, not categorical sampling from actor probabilities.

`evaluate_decentralized_policy(..., deterministic=False)` currently maps to `epsilon=1.0`, which produces fully random per-head choices over valid indices. It is not a stochastic sample from the actor distribution.

Action sanitization occurs after the raw factorized action is assembled and before the action is encoded and sent to the environment. Diagnostics currently summarize sanitized actions as the main distributions. Raw actions are passed only to compute `sanitizer_changed_rate` and target-required/no-target checks. Current diagnostics do not expose raw mode/target/movement distribution tables.

Main limitations:

- no actor-distribution stochastic sampling eval mode
- no small-epsilon diagnostic eval mode with separate prefix
- no per-head top1/top2 probability or margin diagnostics
- no raw-vs-sanitized distribution split beyond sanitizer changed rate
- no joint action diversity diagnostics
- training entropy diagnostics are not eval action-confidence diagnostics

Action confidence can be measured without changing behavior by reading actor logits during eval action selection, computing softmax top probabilities and margins, and logging those diagnostics separately from the selected action.

## 5. Hypotheses

These are hypotheses, not conclusions.

| Hypothesis | Evidence from E4.3 | What to measure next | Risk if ignored |
|---|---|---|---|
| Per-head argmax coordination issue | Final eval uses one top mode per run and mostly one top movement; throughput/frame is `0.0` in all final E4.3 rows. | Compare deterministic per-head argmax with actor-distribution stochastic samples; log joint action diversity and per-head top margins. | A policy with useful probability mass may be misread as unusable under deterministic per-head argmax alone. |
| Mode collapse | Every final E4.3 row has top eval mode rate `1.0`. | Log mode top1 probability, top2 margin, stochastic mode distribution, and per-agent top mode. | Further tuning may target training knobs while the main observable issue is eval action concentration. |
| Movement collapse | Most final E4.3 rows have top movement rate `1.0`; vertical movement was not a clear final-row throughput signal. | Log movement top1 probability, movement margin, per-agent movement concentration, and geometry-linked movement diagnostics. | UAV movement may remain locked to a pattern that does not explore useful geometry. |
| Sanitizer/action semantic mismatch | Neutral seed `43` final eval has no-target, idle-mode, and sanitizer-changed rates of `1.0`. | Log raw and sanitized mode/target/movement rates separately; log raw target-required/no-target patterns. | Sanitizer effects may hide whether the actor is choosing invalid target/mode combinations or explicitly choosing idle/no-target. |
| Rollout-vs-eval stochastic gap | Rollout uses epsilon `0.1`; deterministic eval uses epsilon `0.0`; E2 and E4.3 show eval concentration. | Add diagnostic stochastic sample eval and diagnostic epsilon eval with separate prefixes. | The actor distribution and deterministic deployment path may be conflated. |
| Critic/value scale instability | E4.3 final critic grad norm is seed-sensitive across variants. | Keep value, target, advantage, actor/critic grad, and entropy diagnostics in any E4.5 smoke. | Eval-policy changes may be over-interpreted while value/gradient scale remains unstable. |

## 6. Proposed Diagnostic Evaluation Modes

These modes are proposed for a later implementation stage. They should not replace the primary deterministic evaluation path.

| Mode | Purpose | Prefix | Used for primary claim | Requires behavior change |
|---|---|---|---|---|
| deterministic_argmax | Primary reproducible eval, unchanged from current behavior. | `eval_*` | yes, later | no |
| stochastic_sample | Inspect actor distribution quality by sampling from factorized categorical distributions. | `diag_stoch_eval_*` | no | diagnostic only |
| epsilon_sample | Inspect small exploration around deterministic eval. | `diag_epsilon_eval_*` | no | diagnostic only |
| temperature_sample | Optional distribution probe with temperature-controlled categorical sampling. | `diag_temp_eval_*` | no | diagnostic only |

Implementation notes for E4.5:

- Keep primary `eval_*` deterministic argmax unchanged.
- Add diagnostic modes behind explicit config or function arguments.
- Use fixed diagnostic seeds for reproducibility.
- Keep diagnostic metrics clearly prefixed.
- Do not use diagnostic eval metrics as primary claim metrics.
- Do not use critic, global state, env action mask, or hierarchical executor in diagnostic action selection.

## 7. Proposed Additional Diagnostics

| Diagnostic group | Example keys | Purpose | Affects action selection |
|---|---|---|---|
| Per-head top1 probability | `eval_movement_top1_prob`, `eval_target_top1_prob`, `eval_mode_top1_prob` | Measure confidence behind deterministic choices. | no |
| Per-head top2 and margin | `eval_movement_top2_prob`, `eval_movement_top1_top2_margin`, matching target/mode keys | Distinguish sharp head preference from near-ties. | no |
| Raw vs sanitized rates | `eval_raw_no_target_rate`, `eval_sanitized_no_target_rate`, `eval_raw_idle_mode_rate`, `eval_sanitized_idle_mode_rate`, `eval_sanitizer_changed_rate` | Separate actor raw output from sanitizer effects. | no |
| Raw vs sanitized factor distributions | `eval_raw_mode_*_rate`, `eval_sanitized_mode_*_rate`, matching target/movement keys | Explain cases like neutral seed `43`. | no |
| Joint action diversity | `eval_unique_joint_action_count`, `eval_joint_action_top1_rate` | Detect repeated movement-target-mode tuples. | no |
| Per-agent concentration | `eval_agent0_top_mode`, `eval_agent1_top_mode`, `eval_agent0_joint_action_top1_rate`, `eval_agent1_joint_action_top1_rate` | Separate shared collapse from agent-specific collapse. | no |
| Eval stochastic metadata | `diag_stoch_eval_policy_seed`, `diag_epsilon_eval_policy_epsilon`, `diag_temp_eval_temperature` | Keep diagnostic eval reproducible and auditable. | no |

## 8. Recommended Next Stage

Recommended next stage:

```text
Stage E4.5 - Evaluation Diagnostic Instrumentation
```

E4.5 goals:

- add optional diagnostic eval modes
- keep primary deterministic eval unchanged
- add actor-distribution stochastic sampling if feasible
- add small-epsilon diagnostic eval if feasible
- add top1/top2/margin diagnostics per head
- add raw-vs-sanitized diagnostics if feasible
- add joint action diversity diagnostics
- add focused tests
- run only a tiny verification run
- do not change reward, fairness, packet-drop terms, environment dynamics, actor/critic architecture, or CTDE/DE constraints
- make no performance claim

## 9. What Not To Do Yet

- Do not select a variant for a medium candidate rerun yet.
- Do not change reward, fairness, or packet-drop logic.
- Do not compare performance against baselines.
- Do not write final result claims.
- Do not remove deterministic primary evaluation.
- Do not use old 2D or hierarchical results as direct support for CTDE conclusions.
- Do not let diagnostic eval metrics replace primary `eval_*` metrics.

## 10. Claim Policy

No performance claim is made. This document is a design note for evaluation diagnostics only.
