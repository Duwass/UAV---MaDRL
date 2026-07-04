# CTDE D4 Candidate vs 3D Baselines — Technical Comparison

## 1. Scope

This is an internal technical comparison. Its purpose is to check run alignment and read candidate-level signals from the D3 CTDE candidate and the D4.6 3D baseline candidate runs.

This is not a final experiment. This is not a publication-grade result. No performance claim is made for CTDE against any baseline.

## 2. Inputs

CTDE D3 input directories:

- `results/ctde_stage_d3_candidate_seed42`
- `results/ctde_stage_d3_candidate_seed43`
- `results/ctde_stage_d3_candidate_seed44`

D4.6 baseline input directories:

- `results/baseline_3d_d46_random_seed42`
- `results/baseline_3d_d46_random_seed43`
- `results/baseline_3d_d46_random_seed44`
- `results/baseline_3d_d46_idle_seed42`
- `results/baseline_3d_d46_idle_seed43`
- `results/baseline_3d_d46_idle_seed44`
- `results/baseline_3d_d46_nearest_seed42`
- `results/baseline_3d_d46_nearest_seed43`
- `results/baseline_3d_d46_nearest_seed44`

Run references:

- CTDE D3 commit: `df1ce34c04e7e617df8fb54e261aa7c28d306e1e`
- Baseline runner commit: `771eca74957783e17eb5d4a9b109286b6c3cf8b8`
- Seeds: `42`, `43`, `44`
- Horizon: `200 iterations x 50 steps = 10000 steps`

## 3. Alignment Check

| Item | CTDE D3 | Baseline D4.6 | Aligned |
|---|---|---|---|
| Environment | 3D Scenario 4 | 3D Scenario 4 | yes |
| Seeds | 42/43/44 | 42/43/44 | yes |
| Total steps/run | 10000 | 10000 | yes |
| Action dim | 1056 | 1056 | yes |
| Obs dim | 114 | 114 | yes |
| State dim | 89 | 89 | yes |
| Metrics schema | `eval_*` env metrics | `eval_*` env metrics | yes |

## 4. Per-Seed Technical Table

| Method | Seed | Eval return | Throughput/frame | Drop | Jamming failure | Fairness | Energy efficiency | Backscatter success | Active success | Avg altitude | Vertical action rate | Altitude boundary hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctde_candidate | 42 | -0.460000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| ctde_candidate | 43 | -0.654142 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| ctde_candidate | 44 | 0.849000 | 0.400000 | 0.000000 | 0.000000 | 0.066667 | 1.818182 | 1.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| random | 42 | -57.233594 | 0.180000 | 0.000000 | 0.300000 | 0.360000 | 0.533365 | 0.777778 | 0.500000 | 88.431373 | 0.130000 | 3.000000 |
| random | 43 | -56.430314 | 0.080000 | 0.000000 | 0.555556 | 0.177778 | 0.240244 | 0.375000 | 0.333333 | 96.862745 | 0.110000 | 4.000000 |
| random | 44 | -57.024315 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 93.921569 | 0.160000 | 4.000000 |
| idle | 42 | -51.639333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| idle | 43 | -51.635200 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| idle | 44 | -51.644000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| nearest | 42 | -64.662902 | 0.700000 | 0.000000 | 0.597403 | 0.191257 | 8.860759 | 0.406977 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| nearest | 43 | -64.690678 | 0.600000 | 0.000000 | 0.605263 | 0.132743 | 7.692308 | 0.394737 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |
| nearest | 44 | -64.627451 | 0.620000 | 0.000000 | 0.546875 | 0.158189 | 9.687500 | 0.469697 | 0.000000 | 100.000000 | 0.000000 | 0.000000 |

## 5. Mean/Std Technical Table

| Method | Throughput/frame mean/std | Packet drop mean/std | Jamming failure mean/std | Fairness mean/std | Energy efficiency mean/std | Backscatter success mean/std | Active success mean/std | Avg altitude mean/std | Vertical action rate mean/std | Altitude boundary hits mean/std |
|---|---|---|---|---|---|---|---|---|---|---|
| ctde_candidate | 0.133333 / 0.230940 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.688889 / 0.538860 | 0.606061 / 1.049728 | 0.333333 / 0.577350 | 0.000000 / 0.000000 | 100.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| random | 0.086667 / 0.090185 | 0.000000 / 0.000000 | 0.618519 / 0.354222 | 0.512593 / 0.431828 | 0.257869 / 0.267119 | 0.384259 / 0.388972 | 0.277778 / 0.254588 | 93.071895 / 4.279424 | 0.133333 / 0.025166 | 3.666667 / 0.577350 |
| idle | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 1.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 100.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| nearest | 0.640000 / 0.052915 | 0.000000 / 0.000000 | 0.583180 / 0.031686 | 0.160730 / 0.029339 | 8.746856 / 1.002461 | 0.423804 / 0.040213 | 0.000000 / 0.000000 | 100.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |

## 6. Observations Without Performance Claim

- The current CTDE candidate completed all runs and exported all requested metrics.
- The current CTDE candidate produced sparse successful transmission metrics in two seeds.
- The nearest heuristic produced nonzero throughput in all three seeds under the same candidate horizon.
- This suggests the next step should be candidate diagnosis and training tuning before fairness/drop optimization.
- The comparison remains candidate-level and should not be used as final evidence.

This document avoids method-ranking phrases, method superiority/inferiority statements, and algorithm-level failure statements.

## 7. Recommended Next Stage

Recommended next stage:

```text
Stage E0 — CTDE Candidate Diagnosis
```

Purpose:

- Inspect action distribution, mode distribution, target selection, reward components, and evaluation determinism.
- Determine why the current CTDE candidate often produces zero throughput.
- Do not optimize fairness/drop yet.

Possible checks:

- actor action distribution over movement/target/mode
- idle/no-target rate
- target-required mode invalid fallback rate
- mode usage: idle/backscatter/active/relay/harvest if available
- served target distribution
- eval deterministic vs stochastic behavior
- reward/loss trend from `metrics.jsonl`
- compare rollout vs eval metrics
- sanity check whether evaluation policy is too deterministic or untrained

## 8. Claim Policy

- No performance claim is made from this comparison.
- Baseline comparison is candidate-level only.
- Final claims require tuned CTDE, aligned baselines, and a pre-approved claim policy.
