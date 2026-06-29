# Phase 3.6 QMIX Hierarchical Prototype Report

## 1. Why QMIX was implemented

Phase 3.5 showed that the flat Scenario 4 action space was the main bottleneck and that the 10-action hierarchical wrapper made Scenario 4 learnable for DDQN. Phase 3.6 implements QMIX as the first cooperative MaDRL prototype to test centralized training with decentralized per-UAV execution over the same 10 high-level actions.

## 2. QMIX architecture

- Agent network: shared MLP `AgentQNetwork`, local observation input, optional one-hot agent id, hidden sizes `[128, 128]`, output dimension `10`.
- Mixer: monotonic `QMixer`, mixes per-agent Q-values with global state using non-negative hypernetwork weights from `abs()`.
- Mixer dimensions: `mixing_embed_dim=32`, `hypernet_hidden_dim=64`.
- Learning: Double Q-learning enabled; online agent selects next actions, target agent evaluates selected actions.
- Target update: every `200` QMIX training steps.
- Optimizer: Adam, learning rate `0.0005`.
- Gradient clipping: norm `10.0`.
- Replay: episode replay buffer with padded batches and filled timestep mask.

## 3. Hierarchical action interface reuse

QMIX reuses `HierarchicalUAVBackscatterEnv` and `HierarchicalActionExecutor` without changing high-level action semantics.

High-level action dimension is `10`, compared with original flat Scenario 4 action dimension `864`.

Actions:

| ID | Name |
|---:|---|
| 0 | IDLE_SAFE |
| 1 | SERVE_NEAREST_QUEUE |
| 2 | SERVE_BEST_SINR |
| 3 | PRIORITIZE_BACKSCATTER_TYPE23 |
| 4 | PRIORITIZE_ACTIVE_TYPE1 |
| 5 | HARVEST_LOW_ENERGY |
| 6 | AVOID_JAMMER |
| 7 | BALANCE_UNDERSERVED_IOT |
| 8 | HYBRID_BALANCED |
| 9 | HIGH_QUEUE_PRIORITY |

## 4. Observation, state, and action design

- `n_agents`: `2`
- local observation dimension: `97`
- global state dimension: `70`
- action dimension: `10`
- reward: shared global environment reward
- action masking: hierarchical masks from `env.get_action_mask(uav_id)`
- execution: decentralized per-UAV high-level action selection, centralized mixing during training

## 5. Training settings

- Config: `configs/qmix/qmix_hier_sc4_backscatter_types.yaml`
- Environment: `configs/scenario_4_backscatter_types_calibrated.yaml`
- Episodes: `500`
- Max steps per episode: `200`
- Eval interval: `10` episodes
- Eval episodes during training: `30`
- Final eval episodes: `30`
- Seed: `42`
- Device: `cuda`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Full training runtime: `736.4` seconds
- Final evaluation runtime: `17.0` seconds

## 6. Test status

- `python -m pytest -q`: `38 passed, 1 warning in 11.42s`
- `python scripts\run_sanity_tests.py`: all sanity tests passed
- CUDA check: `torch 2.11.0+cu128`, CUDA `12.8`, CUDA available `True`

## 7. Scenario 4 numerical results

Final evaluation from `results/csv/qmix_hier_sc4_backscatter_types_final_eval.csv`:

| Metric | QMIX |
|---|---:|
| reward | -1475.8774 |
| throughput/frame | 0.9945 |
| drop rate | 0.4670 |
| jamming failure | 0.2988 |
| fairness | 0.5981 |
| energy efficiency | 3.7427 |
| backscatter success | 0.6951 |
| active success | 0.6411 |
| mode harvest | 36.7667 |
| mode backscatter | 203.4333 |
| mode active | 28.4333 |
| mode idle | 120.4333 |
| fallback rate | 0.0103 |

## 8. Comparison with baselines and prior DDQN

From `results/csv/qmix_scenario4_comparison.csv`:

| Rank | Policy | Throughput/frame | Reward | Drop | Jam | Fairness |
|---:|---|---:|---:|---:|---:|---:|
| 1 | qmix | 0.9945 | -1475.8774 | 0.4670 | 0.2988 | 0.5981 |
| 2 | hierarchical_ddqn | 0.9710 | -1552.4697 | 0.4761 | 0.4403 | 0.4754 |
| 3 | greedy_nearest | 0.8977 | -1711.9485 | 0.5223 | 0.5463 | 0.1836 |
| 4 | backscatter_only | 0.8522 | -1742.3071 | 0.5294 | 0.5515 | 0.1719 |
| 5 | greedy_sinr | 0.4783 | -2027.0838 | 0.5575 | 0.7806 | 0.2385 |
| 6 | htt_only | 0.3278 | -1876.5055 | 0.5823 | 0.4891 | 0.0937 |
| 7 | flat_ddqn_tuned | 0.3242 | -1971.1129 | 0.5846 | 0.6093 | 0.1930 |
| 8 | random | 0.1075 | -1969.3226 | 0.6084 | 0.6232 | 0.3530 |

QMIX improved over hierarchical DDQN in throughput/frame by about `+2.42%`, reduced jamming failure from `0.4403` to `0.2988`, and improved fairness from `0.4754` to `0.5981`.

## 9. Learning curves summary

- Train reward first 50 mean: `-1621.2768`
- Train reward last 50 mean: `-1509.3056`
- Train throughput/frame first 50 mean: `0.8990`
- Train throughput/frame last 50 mean: `0.9683`
- Eval throughput/frame first eval: `0.6463`
- Eval throughput/frame final logged eval: `0.8927`
- Best logged eval throughput/frame: `1.0285` at episode `420`
- Loss first 50 nonzero mean: `41.0229`
- Loss last 50 nonzero mean: `17.4791`
- Epsilon: `0.9924` at episode 1 to `0.0500` by episode 500

Loss was generally lower late than early, but target updates caused visible spikes around episodes 210 and 410.

## 10. Mode usage analysis

QMIX uses both transmission modes:

- Backscatter actions: `203.4333` per final-eval episode
- Active actions: `28.4333` per final-eval episode
- Harvest actions: `36.7667` per final-eval episode
- Backscatter success: `0.6951`
- Active success: `0.6411`

Compared with hierarchical DDQN, QMIX uses less backscatter and more active/harvest, which reduced jamming failures and improved fairness while slightly improving throughput.

## 11. Coordination and fairness analysis

The Jain fairness index improved substantially relative to hierarchical DDQN and all listed baselines. However, per-UAV served packets remain imbalanced in the final evaluation: UAV 0 averaged `188.2` served packets while UAV 1 averaged `10.7`. This suggests the executor/environment dynamics still allow one UAV to dominate service even when per-IoT fairness improves.

## 12. Problems and limitations

- QMIX currently trains one gradient step per collected episode; this is stable but may underuse replay data.
- Target update spikes are visible in the loss curve.
- Per-UAV workload balance remains weak despite improved per-IoT fairness.
- Only one seed was run for QMIX Scenario 4.
- Scenario 1 and Scenario 2 QMIX configs were created but not trained in this phase.

## 13. Recommendation

QMIX passes the minimum, good, and strong criteria for Scenario 4:

- beats random;
- uses action dimension `10`;
- produces meaningful active and backscatter usage;
- is competitive with and slightly better than hierarchical DDQN;
- improves throughput, jamming failure, drop rate, and fairness over hierarchical DDQN.

Recommended next step: run multi-seed QMIX on Scenario 4 and add one focused QMIX tuning pass before MAPPO. Tune replay updates per episode, target update interval, epsilon decay, and fairness/executor weights. MAPPO should wait until QMIX stability is confirmed across seeds.

