# CTDE E0 Candidate Diagnosis Audit

## 1. Scope

This is an internal diagnosis audit after Stage D4.7. It is not a performance claim, and it does not modify code, training, reward, configuration, or evaluation behavior.

The goal is to identify plausible reasons why the current CTDE candidate is not yet stable enough for tuning or final result reporting. The emphasis is diagnosis and missing instrumentation before any training or reward tuning stage.

## 2. Inputs

CTDE D3 output directories:

- `results/ctde_stage_d3_candidate_seed42`
- `results/ctde_stage_d3_candidate_seed43`
- `results/ctde_stage_d3_candidate_seed44`

Documents inspected:

- `docs/ctde_d4_candidate_vs_baselines.md`
- `docs/ctde_controlled_evaluation_protocol.md`

Run references:

- Current commit at audit time: `a5c5732e3441342433b598d52d24dc8c4cedf021`
- CTDE D3 result commit: `df1ce34c04e7e617df8fb54e261aa7c28d306e1e`
- Seeds: `42`, `43`, `44`
- Horizon: `200 iterations x 50 rollout steps = 10000 transitions`
- Dimensions: `obs_dim = 114`, `state_dim = 89`, `action_dim = 1056`

## 3. Candidate Metric Summary

| Seed | Eval return | Throughput/frame | Drop | Jamming failure | Fairness | Energy efficiency | Backscatter success | Active success | Avg altitude | Vertical action rate | Altitude boundary hits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.460000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| 43 | -0.654142 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| 44 | 0.849000 | 0.400000 | 0.000000 | 0.000000 | 0.066667 | 1.818182 | 1.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |

All three runs completed `200` updates, collected `10000` transitions, reported finite losses, and did not stop early.

## 4. Training/Evaluation Dynamics

| Seed | Iter | Rollout return | Eval return | Actor loss | Critic loss | Mean advantage | Rollout throughput | Eval throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1 | -54.984860 | n/a | -14.749781 | 3.835490 | -1.099824 | 0.000000 | n/a |
| 42 | 100 | -49.144824 | -0.460000 | 5.375751 | 6.728286 | 0.435698 | 0.000000 | 0.000000 |
| 42 | 200 | -49.103223 | -0.460000 | 0.419386 | 0.830292 | 0.169086 | 0.000000 | 0.000000 |
| 43 | 1 | -48.866985 | n/a | -13.046167 | 1.245920 | -0.976579 | 0.000000 | n/a |
| 43 | 100 | -48.640255 | -0.652485 | 0.718911 | 12.405791 | 0.054373 | 0.000000 | 0.000000 |
| 43 | 200 | -50.514917 | -0.654142 | 8.225618 | 4.482980 | 1.038102 | 0.000000 | 0.000000 |
| 44 | 1 | -49.191706 | n/a | -13.197017 | 1.311252 | -0.983577 | 0.000000 | n/a |
| 44 | 100 | -50.173828 | -1.617167 | 7.127766 | 16.350204 | 0.573375 | 0.000000 | 0.000000 |
| 44 | 200 | -66.431990 | 0.849000 | -3.128664 | 1.734747 | -0.398368 | 0.220000 | 0.400000 |

Technical signals:

- Seeds `42` and `43` show zero rollout throughput at the inspected first/mid/final iterations and zero final eval throughput.
- Seed `44` shows nonzero final rollout and eval throughput, but the final rollout return remains substantially negative.
- Actor loss changes sign across runs and iterations; critic loss is finite but has mid-run spikes in seeds `43` and `44`.
- Mean value and mean target magnitudes become large in the middle/final rows, so value scale should be inspected before tuning.

## 5. Rollout-vs-Eval Gap

| Seed | Final rollout throughput | Final eval throughput | Final rollout jamming | Final eval jamming | Final rollout fairness | Final eval fairness | Final rollout energy eff | Final eval energy eff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 43 | 0.000000 | 0.000000 | 0.500000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 44 | 0.220000 | 0.400000 | 0.676471 | 0.000000 | 0.066667 | 0.066667 | 0.927043 | 1.818182 |

Rollout uses epsilon exploration from the CTDE config (`epsilon = 0.1`). Evaluation uses deterministic action selection (`epsilon = 0.0`). Both paths use the local movement mask and do not use the environment action mask for CTDE action selection.

For seeds `42` and `43`, both rollout and eval are sparse in throughput. For seed `44`, final eval throughput is nonzero while rollout also has nonzero throughput, but the rollout jamming metric differs from eval.

## 6. Action/Policy Diagnostics Availability

| Diagnostic | Currently logged | Available in env/code | Needed before tuning | Status |
|---|---|---|---|---|
| movement distribution | no | factorized actions are selected and stored in replay; only vertical aggregate is exported | yes | add CTDE export |
| mode distribution | no | env exposes `mode_usage_*` in episode/frame metrics | yes | available but not exported by CTDE D3 |
| target/no-target rate | no | factorized target head and replay target actions exist | yes | add CTDE export |
| idle action rate | no | env exposes `mode_usage_idle` | yes | available but not exported by CTDE D3 |
| sanitizer fallback rate | no | sanitizer exists in `marl/ctde/utils.py` | yes | requires instrumentation around before/after action |
| invalid target-required fallback | no | sanitizer condition exists for target-required modes with no target | yes | requires instrumentation |
| per-agent action entropy | no | trainer computes `mean_entropy`; actor logits are available | optional | available but not exported by CTDE train loop |
| served target distribution | no | env exposes per-UAV served and per-IoT delivered metrics | yes | available but not exported by CTDE D3 |
| successful transmission distribution | partial | env has backscatter/active success and per-IoT delivered metrics | yes | aggregate exported; distribution not exported |
| rollout deterministic/stochastic flag | no | rollout receives epsilon and rng | yes | add explicit metadata |
| eval deterministic/stochastic flag | no | evaluation has `deterministic=True` parameter | yes | add explicit metadata |

CTDE D3 `metrics.jsonl` currently contains the configured loss fields and selected rollout/eval env metrics. It does not include mode usage, target/no-target rates, sanitizer fallback counts, per-agent action stats, or served-target distributions.

## 7. Plausible Root Causes

These are hypotheses only. The current logs are not sufficient to confirm any single cause.

- The candidate training horizon may still be short for the current sparse reward and factorized action space.
- Deterministic evaluation may be selecting a narrow action pattern that does not regularly produce successful transmission.
- The actor may be selecting idle/no-target or non-throughput-producing mode/target combinations, but this is not directly logged yet.
- The sanitizer may be converting target-required no-target actions into idle/no-target actions, but fallback counts are not logged yet.
- The early reward signal may not be strong enough to make successful transmission frequent during the candidate stage.
- The movement, target, and mode heads may not yet coordinate reliably.
- Vertical movement is near zero in final eval, so altitude behavior is not being used by the evaluated policy.
- `fairness_index = 1.0` when throughput is zero can be a metric artifact and should not be interpreted as successful fairness behavior.

## 8. What Not To Do Yet

- Do not tune fairness yet.
- Do not tune packet drop yet.
- Do not make a performance claim.
- Do not write final result reporting from this candidate.
- Do not compare against old 2D or hierarchical results.

## 9. Recommended Next Stage

Recommended next stage:

```text
Stage E1 - CTDE Action Diagnostics Instrumentation
```

E1 goals:

- log action distribution
- log movement/mode/target/no-target rates
- log sanitizer fallback rates
- log per-agent action stats
- log rollout/eval deterministic mode
- preserve training semantics
- add tests
- run a tiny verification run
- commit and push only the instrumentation work

After E1, consider:

- Stage E2: diagnostic rerun
- Stage E3: training/tuning design
- Stage E4: candidate rerun

## 10. Claim Policy

- No performance claim is made in this diagnosis.
- This is diagnosis only.
- The current candidate should be treated as a debugging and tuning target.
