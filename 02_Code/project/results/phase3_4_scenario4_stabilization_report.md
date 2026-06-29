# Phase 3.4 Scenario 4 DDQN Stabilization Report

## 1. What Changed

This phase added a focused DDQN stabilization path for `scenario_4_backscatter_types_calibrated` without changing simulator semantics or implementing MAPPO/QMIX.

Implemented:

- Reward processing for DDQN training rewards while preserving raw reward logs.
- Dueling DDQN network option.
- Four Scenario 4 stabilization configs.
- Resume support for interrupted DDQN training from an existing checkpoint.
- Scenario 4 stabilization comparison script and plots.

## 2. GPU Status

CUDA was available and used.

```text
torch: 2.11.0+cu128
cuda available: True
cuda version: 12.8
gpu: NVIDIA GeForce RTX 3060 Laptop GPU
```

Both completed stabilization runs printed:

```text
[DDQN] Using device: cuda
```

## 3. Reward Processing Design

Reward processing is implemented in `marl/ddqn/reward_processing.py`.

Supported modes:

- `none`: raw reward.
- `scale`: `processed_reward = raw_reward / scale`.
- `clip`: `processed_reward = clip(raw_reward / scale, clip_min, clip_max)`.

Training uses the processed reward in replay. Logs and evaluations still report raw environment metrics.

The stabilization configs used:

```yaml
reward_processing:
  enabled: true
  mode: scale
  scale: 100.0
```

## 4. Dueling DDQN Design

`DuelingQNetwork` was added to `marl/ddqn/networks.py`.

Architecture:

- Shared feature MLP.
- Value stream `V(s)`.
- Advantage stream `A(s,a)`.
- Combine as `Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)`.

The agent supports:

```yaml
ddqn:
  network_type: standard
```

and:

```yaml
ddqn:
  network_type: dueling
```

## 5. Configs Tested

Created configs:

- `configs/ddqn_stabilized/sc4_reward_scaled.yaml`
- `configs/ddqn_stabilized/sc4_dueling_scaled.yaml`
- `configs/ddqn_stabilized/sc4_dueling_scaled_slow_epsilon.yaml`
- `configs/ddqn_stabilized/sc4_dueling_scaled_low_lr.yaml`

Full 500-episode runs completed for:

- `sc4_dueling_scaled_low_lr`
- `sc4_dueling_scaled_slow_epsilon`

Both used:

- Env: `configs/scenario_4_backscatter_types_calibrated.yaml`
- Episodes: 500
- Max steps: 200
- Eval interval: 10
- Eval episodes: 30
- Seed: 42
- Device: `auto` -> `cuda`
- Network: `dueling`
- Reward processing: scaled by 100

Config-specific:

| Variant | Learning rate | Epsilon decay steps |
|---|---:|---:|
| sc4_dueling_scaled_low_lr | 0.00025 | 50000 |
| sc4_dueling_scaled_slow_epsilon | 0.0005 | 50000 |

## 6. Runtime Notes

`sc4_dueling_scaled_low_lr`:

- 500 training episodes completed on CUDA.
- Training runtime: `4067.3s`.
- Final evaluation runtime: `46.2s`.

`sc4_dueling_scaled_slow_epsilon`:

- Initial run was interrupted for shutdown at episode 159.
- Safe checkpoint was confirmed at episode 150.
- Resume support was added.
- Training resumed from `results/checkpoints/sc4_dueling_scaled_slow_epsilon_latest.pt`.
- Resume restored model, optimizer, and `train_steps=29001`.
- Replay buffer was not restored because old checkpoints did not save replay contents.
- Resume command completed through episode 500.
- Resume runtime: `2634.6s`.
- Final evaluation runtime: `40.2s`.

## 7. Comparison Table

Source CSV:

`results/csv/scenario4_stabilization_comparison.csv`

| Rank | Policy | Reward | Throughput/frame | Drop | Jam | Fairness | Energy efficiency | Backscatter success | Active success | Backscatter usage | Active usage | Loss first 50 | Loss last 50 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | greedy_nearest | -1711.9485 | 0.8977 | 0.5223 | 0.5463 | 0.1836 | 7.2350 | 0.4302 | 0.6019 | 316.5000 | 7.7333 | 0.0000 | 0.0000 |
| 2 | backscatter_only | -1742.3071 | 0.8522 | 0.5294 | 0.5515 | 0.1719 | 10.7016 | 0.4258 | 0.0000 | 316.3000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | greedy_sinr | -2027.0838 | 0.4783 | 0.5575 | 0.7806 | 0.2385 | 1.7265 | 0.1879 | 0.2734 | 318.2667 | 31.9667 | 0.0000 | 0.0000 |
| 4 | htt_only | -1876.5055 | 0.3278 | 0.5823 | 0.4891 | 0.0937 | 0.4029 | 0.0000 | 0.4788 | 0.0000 | 81.7333 | 0.0000 | 0.0000 |
| 5 | ddqn_original_tuned | -1971.1129 | 0.3242 | 0.5846 | 0.6093 | 0.1930 | 0.9417 | 0.3899 | 0.5231 | 135.7667 | 8.2667 | 2.7481 | 10.9697 |
| 6 | sc4_dueling_scaled_low_lr | -2016.7095 | 0.1203 | 0.6073 | 0.5557 | 0.1374 | 0.3959 | 0.4207 | 0.3087 | 47.0000 | 5.1000 | 0.0006 | 0.0034 |
| 7 | sc4_dueling_scaled_slow_epsilon | -2029.3768 | 0.1197 | 0.6095 | 0.6117 | 0.2120 | 0.3821 | 0.3407 | 0.2316 | 52.9333 | 5.5333 | 0.0007 | 0.0639 |
| 8 | random | -1969.3226 | 0.1075 | 0.6084 | 0.6232 | 0.3530 | 0.3242 | 0.3817 | 0.3640 | 34.9667 | 8.5000 | 0.0000 | 0.0000 |

## 8. Best Stabilized Run

Best stabilized variant by final evaluation throughput:

`sc4_dueling_scaled_low_lr`

Metrics:

- Reward: `-2016.7095`
- Throughput/frame: `0.1203`
- Drop: `0.6073`
- Jam: `0.5557`
- Fairness: `0.1374`
- Energy efficiency: `0.3959`
- Backscatter success rate: `0.4207`
- Active success rate: `0.3087`
- Mode usage backscatter: `47.0000`
- Mode usage active: `5.1000`

## 9. Improvement Over Original Tuned Scenario 4 DDQN

The stabilization variants improved loss scale/stability but did not improve final evaluation throughput.

Original tuned DDQN:

- Throughput/frame: `0.3242`
- Loss first 50: `2.7481`
- Loss last 50: `10.9697`
- Mode backscatter: `135.7667`
- Mode active: `8.2667`

Best stabilized DDQN (`sc4_dueling_scaled_low_lr`):

- Throughput/frame: `0.1203`
- Loss first 50: `0.0006`
- Loss last 50: `0.0034`
- Mode backscatter: `47.0000`
- Mode active: `5.1000`

Interpretation:

- Reward scaling successfully prevented large loss explosion.
- Dueling + scaling + slower epsilon did not produce a better policy in this setup.
- The stabilized policies reduced heavy backscatter collapse, but mostly by selecting fewer productive communication actions, causing throughput to fall.

## 10. Random / HTT / Baseline Status

- Beats random: technically yes, but only narrowly.
  - Best stabilized: `0.1203`
  - Random: `0.1075`
- Beats or matches `htt_only`: no.
  - Best stabilized: `0.1203`
  - `htt_only`: `0.3278`
- Moves closer to `greedy_sinr`: no.
  - Best stabilized: `0.1203`
  - `greedy_sinr`: `0.4783`

## 11. Backscatter Collapse

Original tuned DDQN had high backscatter usage:

- Backscatter usage: `135.7667`
- Active usage: `8.2667`

Best stabilized DDQN:

- Backscatter usage: `47.0000`
- Active usage: `5.1000`

This avoids the original heavy backscatter collapse numerically, but it does not replace it with effective active/backscatter scheduling. Throughput falls sharply.

## 12. Loss Stability

Loss improved substantially:

| Variant | Loss first 50 | Loss last 50 |
|---|---:|---:|
| Original tuned DDQN | 2.7481 | 10.9697 |
| sc4_dueling_scaled_low_lr | 0.0006 | 0.0034 |
| sc4_dueling_scaled_slow_epsilon | 0.0007 | 0.0639 |

Reward scaling changed the target scale, so absolute losses are not directly comparable to raw-reward training. Still, the late-run explosion is removed.

## 13. Generated Artifacts

Checkpoints:

- `results/checkpoints/sc4_dueling_scaled_low_lr_best.pt`
- `results/checkpoints/sc4_dueling_scaled_low_lr_latest.pt`
- `results/checkpoints/sc4_dueling_scaled_slow_epsilon_best.pt`
- `results/checkpoints/sc4_dueling_scaled_slow_epsilon_latest.pt`

CSV files:

- `results/csv/sc4_dueling_scaled_low_lr_train_log.csv`
- `results/csv/sc4_dueling_scaled_low_lr_eval_log.csv`
- `results/csv/sc4_dueling_scaled_low_lr_final_eval.csv`
- `results/csv/sc4_dueling_scaled_slow_epsilon_train_log.csv`
- `results/csv/sc4_dueling_scaled_slow_epsilon_eval_log.csv`
- `results/csv/sc4_dueling_scaled_slow_epsilon_final_eval.csv`
- `results/csv/scenario4_stabilization_comparison.csv`

Figures:

- `results/figures/scenario4_stabilization_throughput.png`
- `results/figures/scenario4_stabilization_drop_jam.png`
- `results/figures/scenario4_stabilization_mode_usage.png`
- `results/figures/scenario4_stabilization_loss.png`

## 14. Tests

Final test status:

```text
28 passed, 1 warning
```

Added tests:

- `tests/test_reward_processing.py`
- `tests/test_dueling_network.py`

## 15. Bugs Found / Fixed

- Added missing DDQN reward processing.
- Added missing Dueling DDQN implementation.
- Added minimal DDQN resume support for interrupted training.
- Preserved CUDA device support and `[DDQN] Using device: cuda` startup logs.

Known resume limitation:

- Old checkpoints did not contain replay buffer contents, so resumed training restored model/optimizer/train steps but not replay memory. This caused a post-resume replay warmup period.

## 16. Recommendation

Scenario 4 still fails as a DDQN learning sanity target.

Do not move directly to final MAPPO/QMIX experiments.

Recommended next research/coding direction:

1. Modify the action interface before final MARL:
   - Consider factorized or hierarchical actions: movement, device choice, mode choice.
   - Flat 324+ action DDQN appears inefficient and unstable for heterogeneous Scenario 4.
2. Improve reward/action incentives:
   - Add explicit penalties for unproductive low-throughput behavior.
   - Consider reward terms for successful active/backscatter balance under valid channel states.
3. Only after action/reward interface is stabilized, prototype MAPPO/QMIX on the revised interface.

Best next coding step:

Implement a factorized-action DDQN/MARL interface prototype for Scenario 4, while preserving current flat-action baselines for comparison.
