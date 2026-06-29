# Phase 3.2 Centralized DDQN Prototype Report

## 1. Implementation Summary

- Implemented a centralized-factorized DDQN prototype with action masking.
- The agent observes centralized state information but selects one masked per-UAV action at a time.
- Stored one replay transition per UAV per environment step with shared global reward.
- Added training, evaluation, comparison, plotting scripts, DDQN configs, and tests.

## 2. DDQN Architecture

- Network: shared MLP Q-network.
- Hidden sizes: `[256, 256]`.
- Activation: ReLU.
- Layer norm: disabled.
- Scenario smoke/pilot: `scenario_0_no_jammer_calibrated`.
- State representation: `concat_global_local`.
- State dimension: `62`.
- UAV id encoding: scalar because `num_uav = 1`.
- Final network input dimension: `63`.
- Output action dimension: `324`.
- Replay buffer capacity: `100000`.
- Batch size: `128`.
- Target update frequency: `1000` train steps.
- Optimizer: Adam.
- Learning rate: `0.001`.
- Gamma: `0.99`.
- Gradient clip norm: `10.0`.
- Epsilon schedule: `1.0 -> 0.05` over `50000` train steps.

## 3. Action Masking Method

- The trainer calls `env.get_action_mask(uav_id)` before selecting each UAV action.
- Invalid actions are masked before argmax and before epsilon-random sampling.
- If an all-zero mask appears, the helper falls back to action `0` and emits a warning.
- Deterministic evaluation uses masked greedy action selection.

## 4. Training Hyperparameters

- Config: `configs/ddqn/ddqn_scenario_0_no_jammer.yaml`.
- Pilot episodes run: `50`.
- Max steps per episode: `200`.
- Eval episodes during training: `20`.
- Final deterministic eval episodes: `10`.
- Seed: `42`.
- Device: automatic PyTorch device selection; run was CPU in this environment.

## 5. Test Status

- `python -m pytest -q`: passed, `22 passed`.
- One expected warning is emitted by the all-zero-mask fallback test.

## 6. Smoke and Pilot Training Results

The required 20-episode smoke training completed successfully and saved logs/checkpoints. A longer 50-episode pilot was then run for the same no-jammer calibrated scenario, overwriting the DDQN scenario 0 logs/checkpoints with the stronger pilot artifact.

Training/eval signal from 50-episode pilot:

- Eval throughput/frame at episode 1: `0.0703`.
- Eval throughput/frame at episode 50: `0.3820`.
- Eval reward at episode 1: `-1150.2653`.
- Eval reward at episode 50: `-915.9385`.
- Epsilon at final train episode: `0.8290`.
- Final train-log average loss near episode 50: `0.9719`.

This shows a positive learning signal, although convergence is not mature.

## 7. Final Deterministic Evaluation

Final evaluation used:

- Checkpoint: `results/checkpoints/ddqn_scenario_0_no_jammer_best.pt`.
- Episodes: `10`.

Mean final eval metrics:

| Metric | Value |
|---|---:|
| total_reward | -878.6221 |
| avg_throughput_per_frame | 0.4610 |
| packet_drop_rate | 0.6260 |
| jamming_failure_rate | 0.0000 |
| fairness_index | 0.5724 |
| energy_efficiency | 1.5242 |
| backscatter_success_rate | 1.0000 |
| active_success_rate | 0.9969 |
| mode_usage_harvest | 15.8 |
| mode_usage_backscatter | 35.7 |
| mode_usage_active | 31.4 |
| mode_usage_idle | 53.0 |

## 8. DDQN vs Baselines

Comparison scenario: `scenario_0_no_jammer_calibrated`.

| Policy | Throughput/frame | Drop rate | Fairness | Energy efficiency | Relative to random throughput |
|---|---:|---:|---:|---:|---:|
| DDQN | 0.4610 | 0.6260 | 0.5724 | 1.5242 | +110.02% |
| random | 0.2195 | 0.7089 | 0.5965 | 0.9965 | 0.00% |
| backscatter_only | 0.5487 | 0.6031 | 0.2723 | 19.6357 | +149.96% |
| greedy_nearest | 0.6730 | 0.5525 | 0.3749 | 6.4132 | +206.61% |
| greedy_sinr | 0.8077 | 0.4975 | 0.5331 | 4.4597 | +267.96% |
| htt_only | 0.7410 | 0.5396 | 0.4353 | 1.6330 | +237.59% |

DDQN beats random clearly but does not yet beat simple greedy or specialized baselines.

## 9. Learning Curves Summary

- Reward improved in eval from `-1150.2653` to `-915.9385`.
- Eval throughput/frame improved from `0.0703` to `0.3820`.
- Loss became active after replay warmup and remained finite.
- Epsilon decayed from `1.0` to `0.8290`; exploration is still high after only 50 episodes.

## 10. Mode Usage Analysis

Final deterministic eval mean per episode:

- Harvest: `15.8`.
- Backscatter: `35.7`.
- Active: `31.4`.
- Idle: `53.0`.

The learned policy uses both backscatter and active modes, which indicates that the state/action/reward pipeline can express hybrid behavior.

## 11. Problems and Limitations

- The 50-episode run is a pilot only; epsilon remains high and policy convergence is immature.
- DDQN is still below greedy baselines in no-jammer calibrated scenario.
- Factorized per-UAV DDQN ignores joint-action coupling except through shared reward and shared global state.
- The action space remains large; mask computation and per-step action selection can become slow for larger scenarios.
- Reward is dominated by queue/drop pressure, which may slow learning without reward normalization or curriculum.

## 12. Recommendation

- Continue DDQN tuning before using it as a final benchmark.
- Do not move directly to final MAPPO/QMIX experiments yet; use DDQN to tune reward scaling, action masks, and curriculum first.
- MAPPO/QMIX interface prototyping can begin after one DDQN run reliably approaches greedy baselines in no-jammer and weak-jammer scenarios.

## 13. Recommended Next Coding Step

Implement DDQN training improvements:

- faster vectorized action-mask lookup or cached decoded action table,
- reward normalization or scaled reward components,
- shorter curriculum configs for no-jammer -> weak jammer -> Scenario 4,
- optional dueling DQN head,
- separate plots for action mode probabilities and invalid-action diagnostics.

