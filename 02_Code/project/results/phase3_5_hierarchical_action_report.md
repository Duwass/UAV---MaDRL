# Phase 3.5 Hierarchical / Factorized Action Interface Report

## 1. Why Hierarchical Action Was Introduced

Phase 3.4 showed that flat DDQN tuning was no longer the main bottleneck. The flat Scenario 4 action interface required DDQN to learn over the Cartesian product:

```text
movement_action x selected_iot x communication_mode
```

For Scenario 4 this is:

```text
9 x (15 + 1) x 6 = 864 actions
```

The hierarchical interface reduces the DDQN output space to 10 high-level strategy actions. A rule-based executor translates each high-level action into a valid original flat action for the underlying simulator.

## 2. High-Level Action List

| ID | Name | Behavior |
|---:|---|---|
| 0 | IDLE_SAFE | Idle unless a safe high-SINR target is clearly available. |
| 1 | SERVE_NEAREST_QUEUE | Serve nearest in-coverage IoT with queue. |
| 2 | SERVE_BEST_SINR | Serve queued in-coverage IoT with best SINR. |
| 3 | PRIORITIZE_BACKSCATTER_TYPE23 | Prefer queued Type 2/3 IoT, especially for backscatter. |
| 4 | PRIORITIZE_ACTIVE_TYPE1 | Prefer Type 1 IoT with active feasibility. |
| 5 | HARVEST_LOW_ENERGY | Harvest for low-energy IoT when primary is busy. |
| 6 | AVOID_JAMMER | Move away from nearest jammer. |
| 7 | BALANCE_UNDERSERVED_IOT | Prefer underserved/high-queue IoT. |
| 8 | HYBRID_BALANCED | Weighted hybrid of queue, SINR, type, fairness, and jammer safety. |
| 9 | HIGH_QUEUE_PRIORITY | Prefer highest queue pressure target. |

## 3. Executor Design

Implemented in:

`envs/hierarchical_action.py`

The executor:

- Receives one high-level action per UAV.
- Scores candidate IoT devices using queue pressure, SINR, distance, energy need, type priority, underserved score, and jammer safety.
- Selects communication mode based on primary-channel state, IoT type, queue, energy, and SINR.
- Falls back to safe valid actions if the preferred target/mode is infeasible.
- Produces original encoded flat actions using the existing `encode_action()` function.

Fallback policy:

- Invalid high-level action -> `IDLE_SAFE`.
- Missing preferred target -> best feasible queued SINR target.
- Invalid mode -> best feasible mode among active/backscatter/harvest/idle.
- No target -> idle.

## 4. Training Interface Change

Implemented in:

- `envs/hierarchical_env.py`
- `marl/ddqn/ddqn_trainer.py`

Config switch:

```yaml
training_interface:
  type: hierarchical
```

Flat mode remains default and unchanged:

```yaml
training_interface:
  type: flat
```

For hierarchical runs:

- `action_dim = 10`
- state remains `concat_global_local`
- replay buffer and DDQN agent remain unchanged
- high-level action masks are used
- wrapper translates high-level actions to original env actions inside `step()`

## 5. Tests Status

Final test command:

```powershell
python -m pytest -q
```

Final result:

```text
32 passed, 1 warning
```

Added tests:

- `tests/test_hierarchical_action.py`
- `tests/test_hierarchical_env.py`

## 6. Training Settings

Config trained:

`configs/ddqn_hierarchical/hier_sc4_basic.yaml`

Settings:

- Environment: `configs/scenario_4_backscatter_types_calibrated.yaml`
- Interface: `hierarchical`
- Episodes: `500`
- Max steps: `200`
- Eval interval: `10`
- Eval episodes: `30`
- Seed: `42`
- Network: standard DDQN
- Hidden sizes: `[256, 256]`
- Learning rate: `0.0005`
- Gamma: `0.99`
- Batch size: `128`
- Replay capacity: `100000`
- Min replay size: `2000`
- Target update: `1000`
- Epsilon: `1.0 -> 0.05` over `25000` train steps
- Device: `auto -> cuda`

Runtime:

- Training: `1768.2s`
- Final evaluation: `18.8s`

The optional `hier_sc4_slow_epsilon` was not run because `hier_sc4_basic` already achieved a strong pass.

## 7. Numerical Results

Comparison CSV:

`results/csv/hierarchical_scenario4_comparison.csv`

| Rank | Policy | Reward | Throughput/frame | Drop | Jam | Fairness | Energy efficiency | Backscatter success | Active success | Backscatter usage | Active usage | Fallback rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | hier_sc4_basic | -1552.4697 | 0.9710 | 0.4761 | 0.4403 | 0.4754 | 4.9295 | 0.5258 | 0.6817 | 256.3333 | 17.7667 | 0.0095 |
| 2 | greedy_nearest | -1711.9485 | 0.8977 | 0.5223 | 0.5463 | 0.1836 | 7.2350 | 0.4302 | 0.6019 | 316.5000 | 7.7333 | 0.0000 |
| 3 | backscatter_only | -1742.3071 | 0.8522 | 0.5294 | 0.5515 | 0.1719 | 10.7016 | 0.4258 | 0.0000 | 316.3000 | 0.0000 | 0.0000 |
| 4 | greedy_sinr | -2027.0838 | 0.4783 | 0.5575 | 0.7806 | 0.2385 | 1.7265 | 0.1879 | 0.2734 | 318.2667 | 31.9667 | 0.0000 |
| 5 | htt_only | -1876.5055 | 0.3278 | 0.5823 | 0.4891 | 0.0937 | 0.4029 | 0.0000 | 0.4788 | 0.0000 | 81.7333 | 0.0000 |
| 6 | flat_ddqn_tuned | -1971.1129 | 0.3242 | 0.5846 | 0.6093 | 0.1930 | 0.9417 | 0.3899 | 0.5231 | 135.7667 | 8.2667 | 0.0000 |
| 7 | flat_ddqn_stabilized_low_lr | -2016.7095 | 0.1203 | 0.6073 | 0.5557 | 0.1374 | 0.3959 | 0.4207 | 0.3087 | 47.0000 | 5.1000 | 0.0000 |
| 8 | random | -1969.3226 | 0.1075 | 0.6084 | 0.6232 | 0.3530 | 0.3242 | 0.3817 | 0.3640 | 34.9667 | 8.5000 | 0.0000 |

## 8. Best Hierarchical Result

Best hierarchical run:

`hier_sc4_basic`

Final 30-episode deterministic evaluation:

- Reward: `-1552.4697`
- Throughput/frame: `0.9710`
- Drop: `0.4761`
- Jam: `0.4403`
- Fairness: `0.4754`
- Energy efficiency: `4.9295`
- Backscatter success rate: `0.5258`
- Active success rate: `0.6817`
- Mode harvest: `2.5667`
- Mode backscatter: `256.3333`
- Mode active: `17.7667`
- Mode idle: `55.2000`
- Fallback count: `3.8000`
- Fallback rate: `0.0095`

## 9. Mode Usage Analysis

The hierarchical policy does not collapse to one mode in the same way as flat DDQN.

Compared with original flat DDQN:

| Metric | Flat tuned DDQN | Hierarchical DDQN |
|---|---:|---:|
| Throughput/frame | 0.3242 | 0.9710 |
| Backscatter usage | 135.7667 | 256.3333 |
| Active usage | 8.2667 | 17.7667 |
| Backscatter success | 0.3899 | 0.5258 |
| Active success | 0.5231 | 0.6817 |
| Jam rate | 0.6093 | 0.4403 |
| Fairness | 0.1930 | 0.4754 |

The policy still uses backscatter heavily, but active mode usage is non-trivial and active success is high. The executor is selecting productive active opportunities rather than forcing active-only behavior.

High-level action use in final evaluation:

| Action | Mean count |
|---|---:|
| IDLE_SAFE | 14.7667 |
| SERVE_NEAREST_QUEUE | 7.3000 |
| SERVE_BEST_SINR | 75.2000 |
| PRIORITIZE_BACKSCATTER_TYPE23 | 26.5667 |
| PRIORITIZE_ACTIVE_TYPE1 | 8.6667 |
| HARVEST_LOW_ENERGY | 3.4333 |
| AVOID_JAMMER | 68.1333 |
| BALANCE_UNDERSERVED_IOT | 35.3667 |
| HYBRID_BALANCED | 159.1667 |
| HIGH_QUEUE_PRIORITY | 1.4000 |

## 10. Fallback Analysis

Final eval fallback rate:

```text
0.0095
```

Mean fallback count:

```text
3.8000 actions/episode
```

This is low relative to roughly 400 UAV actions per 200-step 2-UAV episode. Fallbacks are not excessive.

## 11. Learning Signal

Training log:

`results/csv/hier_sc4_basic_train_log.csv`

First 50 vs last 50 training episodes:

| Metric | First 50 | Last 50 |
|---|---:|---:|
| Train reward | -1587.5198 | -1480.6913 |
| Train throughput | 179.2200 | 205.5200 |
| Train drop rate | 0.4877 | 0.4686 |
| Train jamming failure | 0.4676 | 0.3739 |
| Train fairness | 0.3588 | 0.4362 |
| Avg loss | 2.3014 | 7.5626 |
| Epsilon | 0.8426 | 0.0500 |
| Backscatter usage | 230.4800 | 222.1800 |
| Active usage | 30.1600 | 26.4000 |

Evaluation log:

| Eval point | Episode | Reward | Throughput/frame | Drop | Jam | Fairness |
|---|---:|---:|---:|---:|---:|---:|
| First eval | 1 | -1648.4286 | 0.8198 | 0.5189 | 0.3892 | 0.1843 |
| Best eval throughput | 100 | -1457.0571 | 1.1197 | 0.4529 | 0.3797 | 0.5262 |
| Final eval-log point | 500 | -1505.3338 | 0.9522 | 0.5073 | 0.1265 | 0.2205 |

Learning signal is strong: throughput and reward improve from early training, jamming failure improves late, and final deterministic evaluation remains well above all flat-DDQN results.

## 12. Does Scenario 4 Pass?

Yes. Scenario 4 hierarchical DDQN passes strongly.

Minimum pass:

- Required: `> 0.3242` and `>= 0.3278`
- Achieved: `0.9710`

Good pass:

- Required: `>= greedy_sinr 0.4783`
- Achieved: `0.9710`

Strong pass:

- Required: approaches/exceeds `backscatter_only 0.8522` while using active mode meaningfully.
- Achieved: `0.9710`, active usage `17.7667`, active success `0.6817`.

## 13. Generated Artifacts

Checkpoints:

- `results/checkpoints/hier_sc4_basic_best.pt`
- `results/checkpoints/hier_sc4_basic_latest.pt`

CSV files:

- `results/csv/hier_sc4_basic_train_log.csv`
- `results/csv/hier_sc4_basic_eval_log.csv`
- `results/csv/hier_sc4_basic_final_eval.csv`
- `results/csv/hierarchical_scenario4_comparison.csv`

Figures:

- `results/figures/hierarchical_scenario4_throughput.png`
- `results/figures/hierarchical_scenario4_drop_jam.png`
- `results/figures/hierarchical_scenario4_mode_usage.png`
- `results/figures/hierarchical_scenario4_fairness.png`

## 14. Bugs Found / Fixed

- Fixed executor typo: `_score_jammer_safety` -> `score_jammer_safety`.
- Fixed hierarchical `fallback_rate` denominator to use `base_env.episode_totals["actions_processed"]`.

## 15. Limitations

- The executor contains rule-based domain knowledge. Hierarchical DDQN is no longer purely end-to-end flat action learning.
- The trained policy depends on executor quality. MAPPO/QMIX should use the same hierarchical interface only if this abstraction is accepted as part of the experimental design.
- `hier_sc4_slow_epsilon` and `hier_sc4_fairness` were created but not trained because `hier_sc4_basic` already achieved strong pass.
- Active usage is meaningful but still much lower than backscatter usage.

## 16. Recommendation

Proceed to MAPPO/QMIX interface prototyping using the hierarchical action wrapper, not the flat action space.

Keep hierarchical DDQN as a strong Scenario 4 sanity baseline.

Recommended next coding step:

Implement MAPPO/QMIX-compatible wrappers or adapters that expose the same 10-action hierarchical action space, while preserving the original flat environment and flat DDQN results for ablation comparison.
