# Phase 3 Baseline Validation Report

## 1. Experiment Setup

- Scenarios: scenario_0_no_jammer, scenario_1_multi_uav, scenario_2_static_jammer, scenario_3_mobile_jammer, scenario_4_backscatter_types
- Policies: random, greedy_nearest, greedy_sinr, htt_only, backscatter_only
- Episodes per policy/scenario: 30
- Seeds: Config base seed 42 with per-episode offsets in run_baseline.py
- Metrics: total_reward, total_throughput, avg_throughput_per_frame, packet_delivery_ratio, packet_drop_rate, avg_queue_length, energy_efficiency, jamming_failure_rate, collision_count, fairness_index

## 2. Test Status

- Pytest: Passed: 13 passed
- Sanity tests: Passed: all sanity tests passed

## 3. Best Policy Per Scenario

| Scenario | Best policy | Throughput/frame mean | Drop rate mean | Jamming failure mean | Fairness mean |
|---|---:|---:|---:|---:|---:|
| scenario_0_no_jammer | htt_only | 0.4487 | 0.9720 | 0.0000 | 0.3310 |
| scenario_1_multi_uav | htt_only | 0.9180 | 0.9717 | 0.0000 | 0.3186 |
| scenario_2_static_jammer | random | 0.0435 | 0.9891 | 0.6706 | 0.2911 |
| scenario_3_mobile_jammer | greedy_nearest | 0.2672 | 0.9847 | 0.4952 | 0.1638 |
| scenario_4_backscatter_types | htt_only | 0.0507 | 0.9893 | 0.9428 | 0.3329 |

## 4. Observations

- No-jammer vs jammer behavior: Scenario 2 static jammer throughput 0.0158 vs Scenario 1 no-jammer multi-UAV 0.5590; jamming failure 0.9250 vs 0.0000.
- Static jammer vs mobile jammer behavior: Scenario 3 mobile jammer throughput 0.1969 vs Scenario 2 static jammer 0.0158; jamming failure 0.6169 vs 0.9250.
- scenario_0_no_jammer: rank 1 is htt_only (Balanced baseline performance); lowest ranked is random.
- scenario_1_multi_uav: rank 1 is htt_only (Balanced baseline performance); lowest ranked is random.
- scenario_2_static_jammer: rank 1 is random (Best anti-jamming behavior); lowest ranked is greedy_sinr.
- scenario_3_mobile_jammer: rank 1 is greedy_nearest (Balanced baseline performance); lowest ranked is random.
- scenario_4_backscatter_types: rank 1 is htt_only (Balanced baseline performance); lowest ranked is greedy_nearest.
- Random vs greedy behavior: scenario_0_no_jammer: best greedy greedy_sinr=0.3538, random=0.0620; scenario_1_multi_uav: best greedy greedy_sinr=0.6938, random=0.1435; scenario_2_static_jammer: best greedy greedy_nearest=0.0043, random=0.0435; scenario_3_mobile_jammer: best greedy greedy_nearest=0.2672, random=0.0488; scenario_4_backscatter_types: best greedy greedy_sinr=0.0180, random=0.0342.
- HTT-only vs backscatter-only behavior: scenario_0_no_jammer: HTT throughput/frame=0.4487, backscatter=0.2473, HTT energy efficiency=1.7816, backscatter energy efficiency=19.9071; scenario_1_multi_uav: HTT throughput/frame=0.9180, backscatter=0.5072, HTT energy efficiency=1.7822, backscatter energy efficiency=19.9434; scenario_2_static_jammer: HTT throughput/frame=0.0237, backscatter=0.0042, HTT energy efficiency=0.0464, backscatter energy efficiency=0.1665; scenario_3_mobile_jammer: HTT throughput/frame=0.2590, backscatter=0.2602, HTT energy efficiency=0.5156, backscatter energy efficiency=10.5013; scenario_4_backscatter_types: HTT throughput/frame=0.0507, backscatter=0.0018, HTT energy efficiency=0.0863, backscatter energy efficiency=0.0755.
- Scenario 4 backscatter-type behavior: htt_only rank 1, throughput/frame=0.0507; random rank 2, throughput/frame=0.0342; greedy_sinr rank 3, throughput/frame=0.0180; backscatter_only rank 4, throughput/frame=0.0018; greedy_nearest rank 5, throughput/frame=0.0013.

## 5. Expected Pattern Check

- Expected Pattern A, jamming effect: Observed; Scenario 1 throughput/frame=0.5590, jam failure=0.0000; Scenario 2 throughput/frame=0.0158, jam failure=0.9250.
- Expected Pattern B, mobile jammer harder than static: Not observed; Scenario 2 throughput/frame=0.0158, jam failure=0.9250; Scenario 3 throughput/frame=0.1969, jam failure=0.6169. Current Scenario 3 uses random_walk mobility, so it can drift away from the most damaging positions.
- Expected Pattern C, greedy better than random: Observed in 3/5 scenarios. Exceptions: scenario_2_static_jammer, scenario_4_backscatter_types.
- Expected Pattern D, HTT-only reasonable but not universal: HTT-only ranked first in 3/5 scenarios.
- Expected Pattern E, backscatter-only energy efficient but lower throughput: scenario_0_no_jammer: rank 1; scenario_1_multi_uav: rank 1; scenario_2_static_jammer: rank 1; scenario_3_mobile_jammer: rank 1; scenario_4_backscatter_types: rank 4.
- Expected Pattern F, Scenario 4 backscatter benefit: Not observed; backscatter-only throughput rank=4. This suggests the current Type 2/Type 3 modeling and jammer placement do not yet make backscatter competitive.

## 6. Problems Found

- No environment logic bugs were changed during this validation step.
- Added result summarization/ranking/report tooling and expanded plotting output.
- Suspicious trend: random wins Scenario 2 because all targeted policies are nearly fully jammed by the central static jammer; random occasionally succeeds by accidental spatial/action diversity.
- Suspicious trend: Scenario 3 random-walk mobile jammer is easier than Scenario 2 static jammer; this is a configuration limitation rather than an action-encoding bug.
- Suspicious trend: Scenario 4 does not show a strong backscatter-specific advantage; backscatter-only is energy efficient but has very low delivery under the current jammer/type parameters.
- Modeling limitation: current greedy policies are simple one-step heuristics, so poor performance can reflect coverage/mobility myopia rather than simulator corruption.
- Modeling limitation: backscatter-only is low-energy but low-rate by design; Scenario 4 benefit is visible only if busy-channel opportunities and energy constraints make active transmission less dominant.

## 7. Recommendation

- Centralized DDQN readiness: conditionally yes for a first prototype, because the Gym-style environment, metrics, baselines, and sanity checks are stable. Start with short runs and fixed seeds.
- MAPPO/QMIX readiness: conditionally yes for interface experiments, because the PettingZoo wrapper exposes local observations and global state. Before long training, add more diagnostics for per-agent credit assignment and action masking.
- Before training: consider action masking for invalid mode/target combinations, tune jammer/RF/backscatter parameters if Scenario 4 needs stronger backscatter differentiation, and add repeated-seed confidence intervals.
