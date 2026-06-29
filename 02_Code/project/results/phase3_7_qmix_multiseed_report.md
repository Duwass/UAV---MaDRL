# Phase 3.7 QMIX Multi-seed + Tuning Validation

## 1. Purpose

Phase 3.7 validates whether the Phase 3.6 hierarchical QMIX result is stable across seeds and whether a light tuning change, `updates_per_episode=4`, improves learning stability or final policy quality.

## 2. Project health

- `python -m pytest -q`: `40 passed, 1 warning in 10.61s`
- `python scripts\run_sanity_tests.py`: all sanity tests passed
- CUDA: `torch 2.11.0+cu128`, CUDA `12.8`, CUDA available `True`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- QMIX device: `cuda`

## 3. Configs tested

All runs use `configs/scenario_4_backscatter_types_calibrated.yaml`, hierarchical action interface, 500 episodes, 200 steps per episode, 30-episode evals, and seed-specific output prefixes.

| Config | Updates/episode | Target update | Epsilon decay | Seeds |
|---|---:|---:|---:|---|
| `qmix_sc4_base` | 1 | 200 | 25000 | 42, 43, 44 |
| `qmix_sc4_updates4` | 4 | 200 | 25000 | 42, 43, 44 |

`qmix_sc4_slow_epsilon` and `qmix_sc4_target500` configs were created but not run in this pass.

## 4. Per-seed results

| Config | Seed | Throughput/frame | Reward | Drop | Jam | Fairness | Energy eff. | BS success | Active success | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qmix_sc4_base | 42 | 0.9945 | -1475.8774 | 0.4670 | 0.2988 | 0.5981 | 3.7427 | 0.6951 | 0.6411 | 0.0103 |
| qmix_sc4_base | 43 | 0.9332 | -1481.4517 | 0.4796 | 0.2012 | 0.5217 | 5.4559 | 0.8095 | 0.8101 | 0.0665 |
| qmix_sc4_base | 44 | 0.9537 | -1455.1210 | 0.4766 | 0.1167 | 0.4581 | 7.7980 | 0.8760 | 0.8987 | 0.0018 |
| qmix_sc4_updates4 | 42 | 0.8772 | -1571.0317 | 0.4951 | 0.3252 | 0.4188 | 5.3171 | 0.6671 | 0.5684 | 0.0398 |
| qmix_sc4_updates4 | 43 | 0.8698 | -1595.3944 | 0.4874 | 0.4352 | 0.4970 | 3.4113 | 0.5626 | 0.5183 | 0.0076 |
| qmix_sc4_updates4 | 44 | 0.9072 | -1595.4156 | 0.4884 | 0.4117 | 0.4301 | 4.5540 | 0.5793 | 0.6175 | 0.0166 |

## 5. Mean/std results

| Config | Throughput mean | Throughput std | Jam mean | Jam std | Fairness mean | Fairness std | Drop mean | Drop std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qmix_sc4_base | 0.9604 | 0.0255 | 0.2056 | 0.0744 | 0.5260 | 0.0572 | 0.4744 | 0.0054 |
| qmix_sc4_updates4 | 0.8847 | 0.0162 | 0.3907 | 0.0473 | 0.4486 | 0.0345 | 0.4903 | 0.0034 |

## 6. Comparison to baselines

Baseline throughput/frame:

- Phase 3.6 single-seed QMIX: `0.9945`
- hierarchical DDQN: `0.9710`
- greedy_nearest: `0.8977`
- backscatter_only: `0.8522`
- flat DDQN tuned: `0.3242`
- random: `0.1075`

`qmix_sc4_base` mean throughput/frame is `0.9604`, which is slightly below hierarchical DDQN by `-0.0106`, above greedy_nearest by `+0.0627`, above backscatter_only by `+0.1082`, and above random by `+0.8529`.

`qmix_sc4_base` seed 42 exactly reproduced the Phase 3.6 single-seed final result: throughput/frame `0.9945`, jam `0.2988`, fairness `0.5981`.

## 7. Tuning observations

Increasing replay updates from 1 to 4 per episode hurt performance in this setting:

- Throughput mean dropped from `0.9604` to `0.8847`.
- Jam increased from `0.2056` to `0.3907`.
- Fairness decreased from `0.5260` to `0.4486`.
- Last-50 loss mean increased from `17.6012` to `60.1632`.

The `updates4` variant produced lower throughput std (`0.0162` vs `0.0255`) but this appears to be consistently worse convergence rather than useful stability.

## 8. Stability conclusion

QMIX base is robust across seeds:

- all seeds exceed `0.93` throughput/frame;
- mean throughput/frame exceeds `0.95`;
- throughput std is low at `0.0255`;
- mean jam is low at `0.2056`;
- mean fairness is above the hierarchical DDQN single-seed fairness threshold `0.4754`, but below the strong target `0.55`.

Criteria:

- Minimum: pass.
- Good: pass.
- Strong: partial pass because throughput, throughput std, and jam pass, but mean fairness `0.5260` is below `0.55`.

## 9. Generated artifacts

Main CSVs:

- `results/csv/qmix_experiment_summary.csv`
- `results/csv/qmix_multiseed_mean.csv`
- `results/csv/qmix_multiseed_std.csv`
- `results/csv/qmix_multiseed_ranking.csv`

Figures:

- `results/figures/qmix_multiseed_throughput_mean_std.png`
- `results/figures/qmix_multiseed_jam_mean_std.png`
- `results/figures/qmix_multiseed_fairness_mean_std.png`
- `results/figures/qmix_multiseed_drop_mean_std.png`
- `results/figures/qmix_multiseed_config_ranking.png`
- `results/figures/qmix_multiseed_loss_first_last.png`

## 10. Recommendation

Use `qmix_sc4_base` as the main MaDRL result for Scenario 4. It is stable across seeds and beats random, flat DDQN, backscatter_only, and greedy_nearest on mean throughput while maintaining much lower jam than hierarchical DDQN seed 42.

Do not use `updates_per_episode=4` as the default. It worsens throughput, jam, fairness, and late loss.

Recommended next step: run a focused fairness/coordination ablation for QMIX base, such as executor fairness weights or a mild fairness reward term, before implementing MAPPO. MAPPO can be started after the QMIX fairness limitation is documented or addressed.

