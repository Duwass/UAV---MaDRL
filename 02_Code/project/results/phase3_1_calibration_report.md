# Phase 3.1 Calibration Report

## 1. What Changed From Phase 3

- Added Gymnasium and PettingZoo action masking interfaces.
- Added diagnostic metrics for invalid actions, coverage failures, energy failures, mode usage, active/backscatter attempts, and per-UAV/per-IoT delivery.
- Added calibrated no-jammer, jammer-strength, mobile-chase, and backscatter-type scenarios.
- Updated baseline policies to use masks or masked fallbacks without making them learned controllers.

## 2. Action Masking Design

- `UAVBackscatterEnv.get_action_mask(uav_id)` returns a binary vector over the discrete action space.
- `UAVBackscatterParallelEnv.action_mask(agent)` exposes the same mask for PettingZoo agents.
- Masks invalidate out-of-coverage communication, empty-queue backscatter/active actions, insufficient-energy active actions, busy/idle mode conflicts, and avoid-jammer actions when no jammer exists.
- Idle and relay actions remain valid; random policy samples from the mask when `action_masking.random_uses_mask` is true.

## 3. New Diagnostic Metrics

- Invalid action metrics: invalid action rate, out-of-coverage action rate, no-queue selections, insufficient-energy selections, and busy-mode invalid selections.
- Transmission diagnostics: jammed transmission rate, active/backscatter attempted packets, successful packets, and success rates.
- Mode diagnostics: idle, harvest, backscatter, active, relay, and avoid-jammer usage counts.
- Delivery diagnostics: flattened per-UAV served packets and per-IoT delivered packets.

## 4. New/Updated Config Files

- `configs/scenario_0_no_jammer_calibrated.yaml`
- `configs/scenario_1_multi_uav_calibrated.yaml`
- `configs/scenario_2a_static_jammer_weak.yaml`
- `configs/scenario_2b_static_jammer_medium.yaml`
- `configs/scenario_2c_static_jammer_strong.yaml`
- `configs/scenario_3a_mobile_random_walk.yaml`
- `configs/scenario_3b_mobile_chase_uav.yaml`
- `configs/scenario_4_backscatter_types_calibrated.yaml`

## 5. Experiment Setup

- Scenarios: scenario_0_no_jammer_calibrated, scenario_1_multi_uav_calibrated, scenario_2a_static_jammer_weak, scenario_2b_static_jammer_medium, scenario_2c_static_jammer_strong, scenario_3b_mobile_chase_uav, scenario_4_backscatter_types_calibrated
- Policies: random, greedy_nearest, greedy_sinr, htt_only, backscatter_only
- Episodes per policy/scenario: 30
- Seeds: Config base seed 42 with per-episode offsets in run_baseline.py
- Metrics: total_reward, total_throughput, avg_throughput_per_frame, packet_delivery_ratio, packet_drop_rate, avg_queue_length, energy_efficiency, jamming_failure_rate, collision_count, fairness_index, invalid_action_rate, out_of_coverage_action_rate, insufficient_energy_count, jammed_transmission_rate, successful_backscatter_packets, successful_active_packets, backscatter_success_rate, active_success_rate, harvested_energy_total, mode_usage_backscatter, mode_usage_active

## 6. Test Status

- Pytest: Passed: 16 passed
- Sanity tests: Passed: all sanity tests passed

## 7. Best Policy Per Scenario

| Scenario | Best policy | Throughput/frame mean | Drop rate mean | Jamming failure mean | Fairness mean | Invalid action rate | Backscatter success | Active success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_0_no_jammer_calibrated | greedy_sinr | 0.8077 | 0.4975 | 0.0000 | 0.5331 | 0.0000 | 0.9969 | 1.0000 |
| scenario_1_multi_uav_calibrated | greedy_sinr | 1.6167 | 0.5000 | 0.0000 | 0.5495 | 0.0010 | 0.9983 | 1.0000 |
| scenario_2a_static_jammer_weak | htt_only | 1.0733 | 0.6102 | 0.0681 | 0.2726 | 0.0000 | 0.0000 | 0.6573 |
| scenario_2b_static_jammer_medium | random | 0.1412 | 0.7716 | 0.5446 | 0.4405 | 0.0057 | 0.4482 | 0.4671 |
| scenario_2c_static_jammer_strong | random | 0.1385 | 0.7721 | 0.5515 | 0.4314 | 0.0057 | 0.4471 | 0.4551 |
| scenario_3b_mobile_chase_uav | random | 0.1110 | 0.7774 | 0.6415 | 0.3861 | 0.0057 | 0.3515 | 0.3674 |
| scenario_4_backscatter_types_calibrated | greedy_nearest | 0.8977 | 0.5222 | 0.5463 | 0.1836 | 0.0000 | 0.4302 | 0.6019 |

## 8. Observations

- No-jammer vs jammer behavior: scenario_2b_static_jammer_medium throughput 0.0736 vs scenario_1_multi_uav_calibrated 1.1956; jamming failure 0.8802 vs 0.0000.
- Static jammer vs mobile jammer behavior: scenario_3b_mobile_chase_uav throughput 0.0604 vs scenario_2b_static_jammer_medium 0.0736; jamming failure 0.9044 vs 0.8802.
- scenario_0_no_jammer_calibrated: rank 1 is greedy_sinr (Balanced baseline performance); lowest ranked is random.
- scenario_1_multi_uav_calibrated: rank 1 is greedy_sinr (Balanced baseline performance); lowest ranked is random.
- scenario_2a_static_jammer_weak: rank 1 is htt_only (Best anti-jamming behavior); lowest ranked is random.
- scenario_2b_static_jammer_medium: rank 1 is random (Best anti-jamming behavior); lowest ranked is backscatter_only.
- scenario_2c_static_jammer_strong: rank 1 is random (Best anti-jamming behavior); lowest ranked is backscatter_only.
- scenario_3b_mobile_chase_uav: rank 1 is random (Best anti-jamming behavior); lowest ranked is backscatter_only.
- scenario_4_backscatter_types_calibrated: rank 1 is greedy_nearest (Balanced baseline performance); lowest ranked is random.
- Random vs greedy behavior: scenario_0_no_jammer_calibrated: best greedy greedy_sinr=0.8077, random=0.2195; scenario_1_multi_uav_calibrated: best greedy greedy_sinr=1.6167, random=0.4503; scenario_2a_static_jammer_weak: best greedy greedy_nearest=0.4195, random=0.1752; scenario_2b_static_jammer_medium: best greedy greedy_sinr=0.0638, random=0.1412; scenario_2c_static_jammer_strong: best greedy greedy_sinr=0.0500, random=0.1385; scenario_3b_mobile_chase_uav: best greedy greedy_sinr=0.0917, random=0.1110; scenario_4_backscatter_types_calibrated: best greedy greedy_nearest=0.8977, random=0.1075.
- HTT-only vs backscatter-only behavior: scenario_0_no_jammer_calibrated: HTT throughput/frame=0.7410, backscatter=0.5487, HTT energy efficiency=1.6330, backscatter energy efficiency=19.6357; scenario_1_multi_uav_calibrated: HTT throughput/frame=1.5180, backscatter=1.0838, HTT energy efficiency=1.6368, backscatter energy efficiency=19.6839; scenario_2a_static_jammer_weak: HTT throughput/frame=1.0733, backscatter=0.3100, HTT energy efficiency=1.1176, backscatter energy efficiency=5.6587; scenario_2b_static_jammer_medium: HTT throughput/frame=0.1143, backscatter=0.0217, HTT energy efficiency=0.1168, backscatter energy efficiency=0.3971; scenario_2c_static_jammer_strong: HTT throughput/frame=0.0550, backscatter=0.0107, HTT energy efficiency=0.0564, backscatter energy efficiency=0.1954; scenario_3b_mobile_chase_uav: HTT throughput/frame=0.0702, backscatter=0.0128, HTT energy efficiency=0.0720, backscatter energy efficiency=0.2371; scenario_4_backscatter_types_calibrated: HTT throughput/frame=0.3278, backscatter=0.8522, HTT energy efficiency=0.4029, backscatter energy efficiency=10.7016.
- Scenario 4 backscatter-type behavior: greedy_nearest rank 1, throughput/frame=0.8977; backscatter_only rank 2, throughput/frame=0.8522; greedy_sinr rank 3, throughput/frame=0.4783; htt_only rank 4, throughput/frame=0.3278; random rank 5, throughput/frame=0.1075.

## 9. Expected Pattern Check

- Expected Pattern A, jamming effect: Observed; scenario_1_multi_uav_calibrated throughput/frame=1.1956, jam failure=0.0000; scenario_2b_static_jammer_medium throughput/frame=0.0736, jam failure=0.8802.
- Static jammer weak/medium/strong monotonicity: throughput Observed (0.4661 > 0.0736 > 0.0532); failure Observed (0.4318 < 0.8802 < 0.8942).
- Expected Pattern B, mobile jammer harder than static: Observed; scenario_2b_static_jammer_medium throughput/frame=0.0736, jam failure=0.8802; scenario_3b_mobile_chase_uav throughput/frame=0.0604, jam failure=0.9044.
- Expected Pattern C, greedy better than random: Observed in 4/7 scenarios. Exceptions: scenario_2b_static_jammer_medium, scenario_2c_static_jammer_strong, scenario_3b_mobile_chase_uav.
- Expected Pattern D, HTT-only reasonable but not universal: HTT-only ranked first in 1/7 scenarios.
- Expected Pattern E, backscatter-only energy efficient but lower throughput: scenario_0_no_jammer_calibrated: rank 1; scenario_1_multi_uav_calibrated: rank 1; scenario_2a_static_jammer_weak: rank 1; scenario_2b_static_jammer_medium: rank 1; scenario_2c_static_jammer_strong: rank 2; scenario_3b_mobile_chase_uav: rank 3; scenario_4_backscatter_types_calibrated: rank 1.
- Expected Pattern F, Scenario 4 backscatter benefit: Observed; backscatter-only throughput rank=2. Backscatter is now competitive and energy efficient for the calibrated type mix.
- Drop-rate calibration: Observed; calibrated no-jammer mean drop rate=0.5822, below the previous 0.97-0.99 range.
- Action masking effect: Observed; mean invalid_action_rate=0.0014 across summarized policy/scenario groups.

## 10. Remaining Issues

- If drop rate remains high in any calibrated scenario, queue/arrival/service balance still needs tuning before long training.
- Current greedy policies remain simple one-step heuristics, so poor performance can reflect coverage/mobility myopia rather than simulator corruption.
- Backscatter-only is low-energy but low-rate by design; it should be interpreted together with energy efficiency and Type 2/Type 3 delivery diagnostics.

## 11. Recommendation

- Centralized DDQN readiness: conditionally yes for a first prototype, because the Gym-style environment, metrics, baselines, and sanity checks are stable. Start with short runs and fixed seeds.
- MAPPO/QMIX readiness: conditionally yes for interface experiments, because the PettingZoo wrapper exposes local observations and global state. Before long training, add more diagnostics for per-agent credit assignment and action masking.
- Before training: consider action masking for invalid mode/target combinations, tune jammer/RF/backscatter parameters if Scenario 4 needs stronger backscatter differentiation, and add repeated-seed confidence intervals.
