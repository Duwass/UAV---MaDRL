# CTDE E4.3 Multi-Seed Controlled Stability/Entropy Diagnostic Rerun

## 1. Scope

This is a multi-seed diagnostic rerun for the CTDE 3D path. It covers variants `neutral`, `stability`, and `entropy_mid` on seeds 42, 43, and 44.

Each run used a medium diagnostic horizon of 120 iterations x 40 rollout steps = 4800 collected transitions. This is not a final candidate rerun and is not sufficient for performance claims. Reward terms, environment dynamics, actor/critic architecture, tracked base config, baselines, and action sanitizer behavior were not changed.

## 2. Variants

Temporary configs were created under `results/ctde_stage_e43_configs/` because the training CLI does not expose overrides for the tuning knobs. These configs are run artifacts and should not be committed.

| Variant | normalize_advantage | max_grad_norm | entropy_coef | Rationale |
|---|---|---:|---:|---|
| neutral | false | null | 0.0 | Control path with E4.1 diagnostics. |
| stability | true | 1.0 | 0.0 | Advantage normalization plus gradient clipping diagnostics. |
| entropy_mid | true | 1.0 | 0.005 | Controlled entropy support using the E4.2 entropy_mid setting. |

`entropy_low` was omitted because E4.2 seed 42 reported high gradient diagnostics for that variant.

## 3. Run Completion

| Variant | Seed | Exit code | Iterations | Transitions | Losses finite | Warning |
|---|---:|---:|---:|---:|---|---|
| neutral | 42 | 0 | 120 | 4800 | true | null |
| neutral | 43 | 0 | 120 | 4800 | true | null |
| neutral | 44 | 0 | 120 | 4800 | true | null |
| stability | 42 | 0 | 120 | 4800 | true | null |
| stability | 43 | 0 | 120 | 4800 | true | null |
| stability | 44 | 0 | 120 | 4800 | true | null |
| entropy_mid | 42 | 0 | 120 | 4800 | true | null |
| entropy_mid | 43 | 0 | 120 | 4800 | true | null |
| entropy_mid | 44 | 0 | 120 | 4800 | true | null |

Every run wrote `summary.json`, `metrics.jsonl`, `metrics.csv`, `config.yaml`, and `reproducibility.json`. Every `metrics.jsonl` file has 120 rows. No checkpoint file was present in the run directories. Reproducibility metadata matched `obs_dim=114`, `state_dim=89`, `action_dim=1056`, and commit `07eb2591751babd0275a66521d8a9fb945eae27d`.

## 4. Final Technical Metrics by Variant/Seed

| Variant | Seed | Eval return | Throughput/frame | Jamming failure | Fairness | Energy efficiency | Policy entropy | Advantage std | Actor grad norm | Critic grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 42 | -0.479142 | 0.0 | 0.0 | 1.0 | 0.0 | 6.598070 | 3.279660 | 24.602400 | 948.040000 |
| neutral | 43 | -0.644000 | 0.0 | 0.0 | 1.0 | 0.0 | 6.952540 | 1.650550 | 0.735348 | 44.875200 |
| neutral | 44 | -0.593333 | 0.0 | 0.0 | 1.0 | 0.0 | 6.668790 | 0.903559 | 2.331520 | 44.425000 |
| stability | 42 | 1.525858 | 0.0 | 0.0 | 1.0 | 0.0 | 4.206870 | 3.601220 | 7.325000 | 836.466000 |
| stability | 43 | -0.655000 | 0.0 | 0.0 | 1.0 | 0.0 | 4.892210 | 2.400670 | 1.404330 | 175.130000 |
| stability | 44 | -0.612475 | 0.0 | 0.0 | 1.0 | 0.0 | 4.302760 | 1.860320 | 1.994770 | 40.199300 |
| entropy_mid | 42 | 1.525858 | 0.0 | 0.0 | 1.0 | 0.0 | 4.635280 | 3.737160 | 2.209460 | 471.789000 |
| entropy_mid | 43 | -0.655000 | 0.0 | 0.0 | 1.0 | 0.0 | 4.769960 | 2.929450 | 2.910930 | 428.559000 |
| entropy_mid | 44 | -0.612475 | 0.0 | 0.0 | 1.0 | 0.0 | 4.445950 | 2.258910 | 3.425130 | 106.241000 |

## 5. Mean/Std by Variant

| Variant | Throughput/frame mean/std | Jamming failure mean/std | Fairness mean/std | Energy efficiency mean/std | Policy entropy mean/std | Critic grad norm mean/std |
|---|---|---|---|---|---|---|
| neutral | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 6.7398 / 0.1876 | 345.78 / 521.572 |
| stability | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 4.46728 / 0.371109 | 350.598 / 426.148 |
| entropy_mid | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 4.61706 / 0.162769 | 335.53 / 199.743 |

Additional requested stability aggregates:

| Variant | Advantage std mean/std | Actor grad norm mean/std |
|---|---|---|
| neutral | 1.94459 / 1.21504 | 9.22309 / 13.3428 |
| stability | 2.62074 / 0.891068 | 3.5747 / 3.26125 |
| entropy_mid | 2.97518 / 0.740187 | 2.84851 / 0.610232 |

## 6. Final Action Concentration

| Variant | Seed | Eval top mode | Eval top movement | Eval no-target | Eval idle-mode | Eval sanitizer changed |
|---|---:|---|---|---:|---:|---:|
| neutral | 42 | mode 4 (1.0) | movement 8 (1.0) | 0.0 | 0.0 | 0.0 |
| neutral | 43 | mode 0 (1.0) | movement 0 (0.6) | 1.0 | 1.0 | 1.0 |
| neutral | 44 | mode 3 (1.0) | movement 0 (1.0) | 0.0 | 0.0 | 0.0 |
| stability | 42 | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 | 0.0 |
| stability | 43 | mode 4 (1.0) | movement 3 (1.0) | 0.0 | 0.0 | 0.0 |
| stability | 44 | mode 4 (1.0) | movement 6 (1.0) | 0.0 | 0.0 | 0.0 |
| entropy_mid | 42 | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 | 0.0 |
| entropy_mid | 43 | mode 4 (1.0) | movement 3 (1.0) | 0.0 | 0.0 | 0.0 |
| entropy_mid | 44 | mode 4 (1.0) | movement 6 (1.0) | 0.0 | 0.0 | 0.0 |

## 7. Dynamics Notes

| Variant | Seed | Row | Policy entropy | Actor grad norm | Critic grad norm | Eval top mode | Eval top movement | Eval throughput/frame | Rollout throughput/frame |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| neutral | 42 | 1 | 6.956300 | 4.328370 | 3.614050 | n/a | n/a | null | 0.0 |
| neutral | 42 | 60 | 6.960670 | 4.767780 | 226.457000 | mode 5 (0.8) | movement 7 (0.9) | 0.0 | 0.0 |
| neutral | 42 | 120 | 6.598070 | 24.602400 | 948.040000 | mode 4 (1.0) | movement 8 (1.0) | 0.0 | 0.0 |
| neutral | 43 | 1 | 6.955960 | 4.199770 | 3.079440 | n/a | n/a | null | 0.0 |
| neutral | 43 | 60 | 6.960010 | 2.382400 | 125.205000 | mode 5 (0.5) | movement 8 (0.9) | 0.0 | 0.0 |
| neutral | 43 | 120 | 6.952540 | 0.735348 | 44.875200 | mode 0 (1.0) | movement 0 (0.6) | 0.0 | 0.0 |
| neutral | 44 | 1 | 6.952110 | 3.342790 | 3.696930 | n/a | n/a | null | 0.0 |
| neutral | 44 | 60 | 6.933900 | 3.365840 | 116.935000 | mode 2 (1.0) | movement 10 (0.4) | 0.0 | 0.0 |
| neutral | 44 | 120 | 6.668790 | 2.331520 | 44.425000 | mode 3 (1.0) | movement 0 (1.0) | 0.0 | 0.0 |
| stability | 42 | 1 | 6.956300 | 0.808913 | 3.614050 | n/a | n/a | null | 0.0 |
| stability | 42 | 60 | 5.055010 | 4.296270 | 97.132800 | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 |
| stability | 42 | 120 | 4.206870 | 7.325000 | 836.466000 | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 |
| stability | 43 | 1 | 6.955960 | 1.310410 | 3.079440 | n/a | n/a | null | 0.0 |
| stability | 43 | 60 | 5.775980 | 1.801280 | 120.490000 | mode 4 (1.0) | movement 3 (1.0) | 0.0 | 0.0 |
| stability | 43 | 120 | 4.892210 | 1.404330 | 175.130000 | mode 4 (1.0) | movement 3 (1.0) | 0.0 | 0.0 |
| stability | 44 | 1 | 6.952110 | 1.494650 | 3.696930 | n/a | n/a | null | 0.0 |
| stability | 44 | 60 | 4.460830 | 2.439790 | 257.382000 | mode 4 (1.0) | movement 6 (1.0) | 0.0 | 0.0 |
| stability | 44 | 120 | 4.302760 | 1.994770 | 40.199300 | mode 4 (1.0) | movement 6 (1.0) | 0.0 | 0.0 |
| entropy_mid | 42 | 1 | 6.956300 | 0.808987 | 3.614050 | n/a | n/a | null | 0.0 |
| entropy_mid | 42 | 60 | 4.294500 | 9.682780 | 304.389000 | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 |
| entropy_mid | 42 | 120 | 4.635280 | 2.209460 | 471.789000 | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 |
| entropy_mid | 43 | 1 | 6.955960 | 1.310430 | 3.079440 | n/a | n/a | null | 0.0 |
| entropy_mid | 43 | 60 | 5.316340 | 1.919970 | 106.770000 | mode 4 (1.0) | movement 3 (1.0) | 0.0 | 0.0 |
| entropy_mid | 43 | 120 | 4.769960 | 2.910930 | 428.559000 | mode 4 (1.0) | movement 3 (1.0) | 0.0 | 0.0 |
| entropy_mid | 44 | 1 | 6.952110 | 1.494620 | 3.696930 | n/a | n/a | null | 0.0 |
| entropy_mid | 44 | 60 | 4.372100 | 2.028110 | 147.230000 | mode 4 (1.0) | movement 6 (1.0) | 0.0 | 0.0 |
| entropy_mid | 44 | 120 | 4.445950 | 3.425130 | 106.241000 | mode 4 (1.0) | movement 6 (1.0) | 0.0 | 0.0 |

Entropy remained near the starting range for neutral and shifted to a lower diagnostic range for stability and entropy_mid. Critic grad norm diagnostics showed large seed-dependent spikes in all variants. Actor grad norm diagnostics were most variable in neutral and tightest in entropy_mid at the final rows. Deterministic eval action concentration remained visible in all final rows. Final eval and rollout throughput/frame were both 0.0 for all E4.3 rows shown above.

## 8. Technical Observations Without Claim

All variants completed with finite losses. The run bundles and diagnostic key groups were complete for all nine runs.

Policy entropy diagnostics separated neutral from the two stability-knob variants. The final critic grad norm diagnostics remained seed-sensitive in all variants, with seed 42 producing large final values for each variant.

Deterministic eval concentration remains visible. The final rows show a single top mode with rate 1.0 in every run, and a single top movement with rate 1.0 in most runs. Neutral seed 43 also showed final no-target, idle-mode, and sanitizer-change rates of 1.0.

This diagnostic rerun is not enough for performance claims.

## 9. Recommendation

Recommended option: Stage E4.4 - Evaluation Policy Diagnostic Design.

Rationale: deterministic eval concentration remains visible across variants and seeds. E4.4 should inspect deterministic versus stochastic diagnostic evaluation behavior and action-head concentration before a selected-variant medium candidate rerun. Critic/gradient scale should remain part of the monitoring checklist during that stage.

## 10. Claim Policy

No performance claim is made. Do not use this diagnostic rerun as final CTDE-vs-baseline evidence or as a final candidate statement. Do not use old 2D or hierarchical results as direct support for this CTDE stage.
