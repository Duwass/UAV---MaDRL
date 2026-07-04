# CTDE E2 Diagnostic Rerun

## 1. Scope

This is a diagnostic rerun after E1 action instrumentation. It is not a performance claim and it does not tune reward, fairness, packet drop, training configuration, or environment behavior.

The goal is to understand current CTDE candidate policy behavior using action-level diagnostics exported to `summary.json`, `metrics.jsonl`, and `metrics.csv`.

## 2. Inputs

Save directories:

- `results/ctde_stage_e2_diag_seed42`
- `results/ctde_stage_e2_diag_seed43`
- `results/ctde_stage_e2_diag_seed44`

Run references:

- Seeds: `42`, `43`, `44`
- Horizon: `200 x 50 = 10000` transitions
- Batch size: `64`
- Eval interval: every `20` iterations
- Commit: `6fb7a1629ed4fbed4c030c838d56a6bce8e8046f`

## 3. Run Completion and Reproducibility

| Seed | Exit code | Iterations | Transitions | Losses finite | Warning | Obs dim | State dim | Action dim |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 42 | 0 | 200 | 10000 | true | null | 114 | 89 | 1056 |
| 43 | 0 | 200 | 10000 | true | null | 114 | 89 | 1056 |
| 44 | 0 | 200 | 10000 | true | null | 114 | 89 | 1056 |

All three runs produced the expected output bundle:

- `summary.json`
- `metrics.jsonl`
- `metrics.csv`
- `config.yaml`
- `reproducibility.json`

Each `metrics.jsonl` has `200` lines. No checkpoint file was found inside the E2 save directories.

## 4. Final Candidate Metrics

| Seed | Eval return | Throughput/frame | Drop | Jamming failure | Fairness | Energy efficiency | Backscatter success | Active success | Avg altitude | Vertical action rate | Altitude boundary hits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.460000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| 43 | -0.654142 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| 44 | 0.849000 | 0.400000 | 0.000000 | 0.000000 | 0.066667 | 1.818182 | 1.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |

## 5. Final Action Diagnostics

| Seed | Eval action count | Eval deterministic | Eval no-target | Eval idle-mode | Eval target-required no-target | Eval sanitizer changed | Eval top mode | Eval top mode rate | Eval top movement | Eval top movement rate | Eval vertical action rate | Rollout no-target | Rollout idle-mode | Rollout sanitizer changed | Rollout top mode | Rollout top mode rate |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 10 | true | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1.000000 | 0 | 1.000000 | 0.000000 | 0.030000 | 0.030000 | 0.030000 | 3 | 0.840000 |
| 43 | 10 | true | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1.000000 | 8 | 1.000000 | 0.000000 | 0.020000 | 0.020000 | 0.020000 | 1 | 0.550000 |
| 44 | 10 | true | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2 | 1.000000 | 2 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2 | 0.920000 |

Final evaluation is deterministic with `epsilon = 0.0`. Final rollout is stochastic with `epsilon = 0.1`.

## 6. First/Mid/Final Dynamics

| Seed | Iter | Rollout return | Eval return | Actor loss | Critic loss | Mean advantage | Rollout throughput | Eval throughput | Eval top mode/rate | Eval top movement/rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 1 | -54.984860 | n/a | -14.749781 | 3.835490 | -1.099824 | 0.000000 | n/a | n/a | n/a |
| 42 | 100 | -49.144824 | -0.460000 | 5.375751 | 6.728286 | 0.435698 | 0.000000 | 0.000000 | 3 / 1.000000 | 0 / 1.000000 |
| 42 | 200 | -49.103223 | -0.460000 | 0.419386 | 0.830292 | 0.169086 | 0.000000 | 0.000000 | 3 / 1.000000 | 0 / 1.000000 |
| 43 | 1 | -48.866985 | n/a | -13.046167 | 1.245920 | -0.976579 | 0.000000 | n/a | n/a | n/a |
| 43 | 100 | -48.640255 | -0.652485 | 0.718911 | 12.405791 | 0.054373 | 0.000000 | 0.000000 | 1 / 0.700000 | 8 / 0.600000 |
| 43 | 200 | -50.514917 | -0.654142 | 8.225618 | 4.482980 | 1.038102 | 0.000000 | 0.000000 | 3 / 1.000000 | 8 / 1.000000 |
| 44 | 1 | -49.191706 | n/a | -13.197017 | 1.311252 | -0.983577 | 0.000000 | n/a | n/a | n/a |
| 44 | 100 | -50.173828 | -1.617167 | 7.127766 | 16.350204 | 0.573375 | 0.000000 | 0.000000 | 3 / 0.800000 | 10 / 0.400000 |
| 44 | 200 | -66.431990 | 0.849000 | -3.128664 | 1.734747 | -0.398368 | 0.220000 | 0.400000 | 2 / 1.000000 | 2 / 1.000000 |

Additional final rollout/eval diagnostic snapshots:

- Seed `42`: final eval mode `3` and movement `0` both have rate `1.0`; final rollout top mode is also `3` at rate `0.84`.
- Seed `43`: final eval mode `3` and movement `8` both have rate `1.0`; final rollout top mode is `1` at rate `0.55`.
- Seed `44`: final eval mode `2` and movement `2` both have rate `1.0`; final rollout top mode is `2` at rate `0.92`.

## 7. Rollout-vs-Eval Diagnostic Gap

- Rollout and eval differ in stochasticity: rollout uses `epsilon = 0.1`, eval uses deterministic selection with `epsilon = 0.0`.
- Final eval is highly concentrated for all seeds: top mode rate is `1.0` and top movement rate is `1.0`.
- Final rollout is more varied than eval, especially seed `43`, where rollout top mode rate is `0.55`.
- Final sanitizer changed rates are low: `0.03`, `0.02`, and `0.00` for rollout; `0.00` for eval in all seeds.
- Final no-target and idle-mode rates are low in eval: `0.00` for all seeds. Rollout final no-target/idle rates are also low: `0.03`, `0.02`, and `0.00`.
- Seeds `42` and `43` have zero final rollout throughput and zero final eval throughput, despite low no-target/idle rates.
- Seed `44` has nonzero final rollout and eval throughput, with final eval concentrated on mode `2` and movement `2`.

## 8. Hypotheses Updated After E2

These are hypotheses only.

- Possible mode collapse: final eval selects a single mode and movement for each seed.
- Possible deterministic evaluation issue: eval collapses to one action pattern per seed, while rollout has some stochastic variation.
- Possible training signal/action-factor coordination issue: seeds `42` and `43` have low no-target/idle rates but still produce zero final throughput.
- Possible value-scale instability: mean value and mean target magnitudes become large, and critic loss has mid-run spikes in seeds `43` and `44`.
- Sanitizer mismatch is not the primary signal in the final rows because final sanitizer changed rates are low.
- No-target/idle preference is not the primary signal in final eval rows because final no-target and idle-mode rates are zero.

## 9. Recommended Next Stage

Recommended next stage:

```text
Stage E3 - CTDE Training/Tuning Design
```

E3 should be design-only first. Candidate topics:

- action entropy and exploration schedule
- reward scaling or normalization
- mode validity and fallback diagnostics review
- learning rate and update ratio
- rollout/eval deterministic policy handling
- candidate rerun protocol

No tuning run should start until the design and claim policy are approved.

## 10. Claim Policy

- No performance claim is made from this rerun.
- This document is diagnosis only.
- The current candidate remains a tuning and debugging target.
