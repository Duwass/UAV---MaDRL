# Phase 3.8 QMIX Fairness / Coordination Ablation

## 1. Why this ablation was needed

Phase 3.7 accepted `qmix_sc4_base` as the main Scenario 4 MaDRL result, but mean fairness remained below the strong threshold:

- QMIX base throughput/frame mean/std: `0.9604 / 0.0255`
- QMIX base jam mean/std: `0.2056 / 0.0744`
- QMIX base fairness mean/std: `0.5260 / 0.0572`

Phase 3.8 tested whether executor fairness weights or the `BALANCE_UNDERSERVED_IOT` action could improve fairness without sacrificing throughput.

## 2. Implementation summary

Added optional executor scoring weights:

- `fairness_weight`
- `underserved_weight`
- `repeat_target_penalty`
- `jammer_risk_weight`
- `queue_weight`
- `sinr_weight`
- `distance_weight`
- `type_priority_weight`

Defaults preserve previous behavior.

Added optional high-level action disabling:

```yaml
hierarchical_actions:
  disabled_actions: [7]
```

Action 7 is `BALANCE_UNDERSERVED_IOT`. Disabled actions are masked out in `get_action_mask`; if all actions are accidentally disabled, the wrapper falls back safely.

## 3. Project health

- `python -m pytest -q`: `45 passed, 1 warning in 11.87s`
- `python scripts\run_sanity_tests.py`: all sanity tests passed
- CUDA: `torch 2.11.0+cu128`, CUDA `12.8`, CUDA available `True`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- QMIX device: `cuda`

## 4. Configs tested

All configs used Scenario 4 calibrated environment, hierarchical 10-action wrapper, QMIX base hyperparameters, 500 episodes, and seeds `42, 43, 44`.

| Config | Fairness weight | Underserved weight | Repeat target penalty | Disabled actions |
|---|---:|---:|---:|---|
| `qmix_sc4_fair_w2` | 2.0 | 2.0 | 0.5 | none |
| `qmix_sc4_fair_w3` | 3.0 | 3.0 | 1.0 | none |
| `qmix_sc4_no_balance_action` | 1.0 | 1.0 | 0.0 | `[7]` |

`qmix_sc4_fair_w1` was created as the explicit default-weight fairness config but was not run because Phase 3.7 base already covers the same default behavior.

## 5. Per-seed results

| Config | Seed | Throughput/frame | Reward | Drop | Jam | Fairness | Energy eff. | BS success | Active success | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qmix_sc4_fair_w2 | 42 | 0.9453 | -1503.4131 | 0.5025 | 0.0905 | 0.3153 | 10.9786 | 0.8944 | 0.7594 | 0.0000 |
| qmix_sc4_fair_w2 | 43 | 0.9885 | -1432.5255 | 0.4656 | 0.1835 | 0.6722 | 6.6555 | 0.8179 | 0.8607 | 0.0042 |
| qmix_sc4_fair_w2 | 44 | 1.0310 | -1432.0851 | 0.4631 | 0.2062 | 0.5459 | 5.5067 | 0.7976 | 0.7568 | 0.0391 |
| qmix_sc4_fair_w3 | 42 | 0.8963 | -1624.8675 | 0.5088 | 0.4122 | 0.2750 | 5.8903 | 0.5666 | 0.4690 | 0.0327 |
| qmix_sc4_fair_w3 | 43 | 0.8747 | -1528.4528 | 0.4909 | 0.2530 | 0.4712 | 3.8680 | 0.7310 | 0.7542 | 0.0938 |
| qmix_sc4_fair_w3 | 44 | 0.8920 | -1512.0844 | 0.4816 | 0.2716 | 0.7338 | 5.3697 | 0.7132 | 0.8127 | 0.0056 |
| qmix_sc4_no_balance_action | 42 | 0.8757 | -1583.0156 | 0.4890 | 0.4041 | 0.4223 | 4.8615 | 0.5760 | 0.6530 | 0.0481 |
| qmix_sc4_no_balance_action | 43 | 0.7770 | -1599.2172 | 0.5017 | 0.3432 | 0.4901 | 5.9318 | 0.6473 | 0.8279 | 0.0005 |
| qmix_sc4_no_balance_action | 44 | 0.8790 | -1523.6011 | 0.4871 | 0.2584 | 0.5177 | 5.9771 | 0.7360 | 0.7828 | 0.0079 |

## 6. Mean/std results

| Config | Throughput mean/std | Jam mean/std | Fairness mean/std | Drop mean/std |
|---|---:|---:|---:|---:|
| qmix_sc4_fair_w2 | 0.9883 / 0.0350 | 0.1601 / 0.0501 | 0.5111 / 0.1478 | 0.4771 / 0.0180 |
| qmix_sc4_fair_w3 | 0.8877 / 0.0094 | 0.3123 / 0.0711 | 0.4933 / 0.1880 | 0.4938 / 0.0113 |
| qmix_sc4_no_balance_action | 0.8439 / 0.0473 | 0.3352 / 0.0598 | 0.4767 / 0.0401 | 0.4926 / 0.0065 |

## 7. Comparison to QMIX base

QMIX base from Phase 3.7:

- throughput/frame mean: `0.9604`
- fairness mean: `0.5260`
- jam mean: `0.2056`
- drop mean: `0.4744`

| Config | Throughput delta | Fairness delta | Jam delta | Drop delta |
|---|---:|---:|---:|---:|
| qmix_sc4_fair_w2 | +0.0279 | -0.0149 | -0.0455 | +0.0027 |
| qmix_sc4_fair_w3 | -0.0727 | -0.0327 | +0.1067 | +0.0194 |
| qmix_sc4_no_balance_action | -0.1165 | -0.0493 | +0.1296 | +0.0182 |

## 8. Trade-off analysis

`qmix_sc4_fair_w2` is the best fairness-ablation trade-off by the computed throughput/fairness/jam/drop score. It improves throughput and jamming failure relative to base, but it does not improve mean fairness.

`qmix_sc4_fair_w3` is too strong. It reduces throughput, increases jam, increases drop, and still reduces mean fairness relative to base.

Disabling `BALANCE_UNDERSERVED_IOT` hurts the policy:

- throughput drops from base `0.9604` to `0.8439`;
- fairness drops from base `0.5260` to `0.4767`;
- jam rises from base `0.2056` to `0.3352`.

This supports that high-level action 7 is useful, even though increasing fairness weights did not improve mean fairness.

## 9. Success criteria assessment

Criterion 1: fairness above `0.55` with throughput above `0.90`.

- Not achieved by mean results. Individual seeds did exceed `0.55`, but no tested config achieved mean fairness above `0.55`.

Criterion 2: fairness above base `0.5260` with throughput drop below 5%.

- Not achieved.

Criterion 3: disabling `BALANCE_UNDERSERVED_IOT` reduces fairness.

- Achieved. No-balance fairness mean is `0.4767`, below base `0.5260`.

Criterion 4: trade-off analysis clearly shows why QMIX base is best final setting.

- Achieved. Base remains the best fairness-throughput final setting. `fair_w2` is a strong low-jam/high-throughput variant, but not a fairness improvement.

## 10. Recommended final QMIX setting

Use Phase 3.7 `qmix_sc4_base` as the final main Scenario 4 QMIX setting.

Rationale:

- It already passes the main MaDRL criteria.
- It has better mean fairness than all Phase 3.8 variants.
- It retains strong throughput and low jam.
- The `BALANCE_UNDERSERVED_IOT` action should stay enabled.

`qmix_sc4_fair_w2` can be mentioned as an auxiliary variant that improves throughput and jam but trades away fairness stability.

## 11. Limitations

- The executor-level fairness weights alone did not reliably improve mean fairness.
- Fairness remains seed-sensitive; `fair_w2` seed 43 and `fair_w3` seed 44 were high fairness, but the mean was pulled down by weak seeds.
- No reward-function fairness ablation was run.
- No MAPPO prototype was implemented in this phase.

## 12. Recommendation

Start writing the experimental results section using QMIX base as the main MaDRL result and include this ablation as evidence that:

1. action 7 matters;
2. naive stronger executor fairness weighting is not sufficient;
3. QMIX base is the best final trade-off currently available.

Next coding step: generate publication-ready tables and plots aggregating Phase 3.5, Phase 3.6, Phase 3.7, and Phase 3.8 results. MAPPO can follow after the QMIX result section is drafted.

