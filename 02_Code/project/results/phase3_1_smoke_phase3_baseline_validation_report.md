# Phase 3.1 Calibration Report

## 1. What Changed From Phase 3

- Added Gymnasium and PettingZoo action masking interfaces.
- Added diagnostic metrics for invalid actions, coverage failures, energy failures, mode usage, active/backscatter attempts, and per-UAV/per-IoT delivery.
- Added calibrated no-jammer, jammer-strength, mobile-chase, and backscatter-type scenarios.
- Updated baseline policies to use masks or masked fallbacks without making them learned controllers.

## 2. Experiment Setup

- Scenarios: scenario_0_no_jammer_calibrated, scenario_1_multi_uav_calibrated, scenario_2a_static_jammer_weak, scenario_2b_static_jammer_medium, scenario_2c_static_jammer_strong, scenario_3b_mobile_chase_uav, scenario_4_backscatter_types_calibrated
- Policies: random, greedy_nearest, greedy_sinr, htt_only, backscatter_only
- Episodes per policy/scenario: 3
- Seeds: Config base seed 42 with per-episode offsets in run_baseline.py
- Metrics: total_reward, total_throughput, avg_throughput_per_frame, packet_delivery_ratio, packet_drop_rate, avg_queue_length, energy_efficiency, jamming_failure_rate, collision_count, fairness_index, invalid_action_rate, out_of_coverage_action_rate, insufficient_energy_count, jammed_transmission_rate, successful_backscatter_packets, successful_active_packets, backscatter_success_rate, active_success_rate, harvested_energy_total, mode_usage_backscatter, mode_usage_active

## 3. Test Status

- Pytest: Passed
- Sanity tests: Passed

## 4. Best Policy Per Scenario

| Scenario | Best policy | Throughput/frame mean | Drop rate mean | Jamming failure mean | Fairness mean | Invalid action rate | Backscatter success | Active success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scenario_0_no_jammer_calibrated | greedy_sinr | 0.8550 | 0.7743 | 0.0000 | 0.4567 | 0.0000 | 1.0000 | 1.0000 |
| scenario_1_multi_uav_calibrated | htt_only | 1.8250 | 0.7699 | 0.0000 | 0.2060 | 0.0000 | 0.0000 | 1.0000 |
| scenario_2a_static_jammer_weak | htt_only | 1.2717 | 0.8135 | 0.0000 | 0.2014 | 0.0000 | 0.0000 | 0.6969 |
| scenario_2b_static_jammer_medium | htt_only | 0.1067 | 0.9085 | 0.9416 | 0.1593 | 0.0000 | 0.0000 | 0.0584 |
| scenario_2c_static_jammer_strong | random | 0.1050 | 0.9087 | 0.4640 | 0.4849 | 0.0075 | 0.5992 | 0.4794 |
| scenario_3b_mobile_chase_uav | greedy_sinr | 0.3083 | 0.8909 | 0.8106 | 0.2570 | 0.0008 | 0.1934 | 0.1767 |
| scenario_4_backscatter_types_calibrated | greedy_nearest | 0.9767 | 0.8647 | 0.4973 | 0.0741 | 0.0000 | 0.5388 | 0.0000 |

## 5. Observations

- No-jammer vs jammer behavior: Not enough scenarios to compare.
- Static jammer vs mobile jammer behavior: Not enough scenarios to compare.
- scenario_0_no_jammer_calibrated: rank 1 is greedy_sinr (Balanced baseline performance); lowest ranked is random.
- scenario_1_multi_uav_calibrated: rank 1 is htt_only (Balanced baseline performance); lowest ranked is random.
- scenario_2a_static_jammer_weak: rank 1 is htt_only (Best anti-jamming behavior); lowest ranked is random.
- scenario_2b_static_jammer_medium: rank 1 is htt_only (Balanced baseline performance); lowest ranked is backscatter_only.
- scenario_2c_static_jammer_strong: rank 1 is random (Best anti-jamming behavior); lowest ranked is backscatter_only.
- scenario_3b_mobile_chase_uav: rank 1 is greedy_sinr (Balanced baseline performance); lowest ranked is backscatter_only.
- scenario_4_backscatter_types_calibrated: rank 1 is greedy_nearest (Balanced baseline performance); lowest ranked is random.
- Random vs greedy behavior: scenario_0_no_jammer_calibrated: best greedy greedy_sinr=0.8550, random=0.1617; scenario_1_multi_uav_calibrated: best greedy greedy_sinr=1.6450, random=0.3150; scenario_2a_static_jammer_weak: best greedy greedy_nearest=0.3483, random=0.1267; scenario_2b_static_jammer_medium: best greedy greedy_sinr=0.0500, random=0.1050; scenario_2c_static_jammer_strong: best greedy greedy_sinr=0.0317, random=0.1050; scenario_3b_mobile_chase_uav: best greedy greedy_sinr=0.3083, random=0.1117; scenario_4_backscatter_types_calibrated: best greedy greedy_nearest=0.9767, random=0.1550.
- HTT-only vs backscatter-only behavior: scenario_0_no_jammer_calibrated: HTT throughput/frame=0.8300, backscatter=0.5550, HTT energy efficiency=1.7827, backscatter energy efficiency=19.7652; scenario_1_multi_uav_calibrated: HTT throughput/frame=1.8250, backscatter=1.0800, HTT energy efficiency=1.8012, backscatter energy efficiency=19.8763; scenario_2a_static_jammer_weak: HTT throughput/frame=1.2717, backscatter=0.2267, HTT energy efficiency=1.2552, backscatter energy efficiency=4.2208; scenario_2b_static_jammer_medium: HTT throughput/frame=0.1067, backscatter=0.0233, HTT energy efficiency=0.1052, backscatter energy efficiency=0.4285; scenario_2c_static_jammer_strong: HTT throughput/frame=0.0567, backscatter=0.0117, HTT energy efficiency=0.0560, backscatter energy efficiency=0.2167; scenario_3b_mobile_chase_uav: HTT throughput/frame=0.0567, backscatter=0.0150, HTT energy efficiency=0.0557, backscatter energy efficiency=0.2786; scenario_4_backscatter_types_calibrated: HTT throughput/frame=0.2883, backscatter=0.9567, HTT energy efficiency=0.2512, backscatter energy efficiency=12.0331.

## 6. Expected Pattern Check

- Expected Pattern C, greedy better than random: Observed in 5/7 scenarios. Exceptions: scenario_2b_static_jammer_medium, scenario_2c_static_jammer_strong.
- Expected Pattern D, HTT-only reasonable but not universal: HTT-only ranked first in 3/7 scenarios.
- Expected Pattern E, backscatter-only energy efficient but lower throughput: scenario_0_no_jammer_calibrated: rank 1; scenario_1_multi_uav_calibrated: rank 1; scenario_2a_static_jammer_weak: rank 1; scenario_2b_static_jammer_medium: rank 1; scenario_2c_static_jammer_strong: rank 2; scenario_3b_mobile_chase_uav: rank 3; scenario_4_backscatter_types_calibrated: rank 1.

## 7. Remaining Issues

- If drop rate remains high in any calibrated scenario, queue/arrival/service balance still needs tuning before long training.
- Current greedy policies remain simple one-step heuristics, so poor performance can reflect coverage/mobility myopia rather than simulator corruption.
- Backscatter-only is low-energy but low-rate by design; it should be interpreted together with energy efficiency and Type 2/Type 3 delivery diagnostics.

## 8. Recommendation

- Centralized DDQN readiness: conditionally yes for a first prototype, because the Gym-style environment, metrics, baselines, and sanity checks are stable. Start with short runs and fixed seeds.
- MAPPO/QMIX readiness: conditionally yes for interface experiments, because the PettingZoo wrapper exposes local observations and global state. Before long training, add more diagnostics for per-agent credit assignment and action masking.
- Before training: consider action masking for invalid mode/target combinations, tune jammer/RF/backscatter parameters if Scenario 4 needs stronger backscatter differentiation, and add repeated-seed confidence intervals.
