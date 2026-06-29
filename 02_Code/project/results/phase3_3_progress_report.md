# Phase 3.3 Progress Report: Tuned DDQN Validation

## Scope

This progress report summarizes the current tuned DDQN validation status for:

1. `tuned_no_jammer`
2. `tuned_static_weak`
3. `tuned_backscatter_types`

No MAPPO/QMIX implementation was added. The simulator and baseline logic were not redesigned.

## Testing Status

| Check | Result |
|---|---:|
| `python -m pytest -q` | 22 passed, 1 warning |
| `python scripts\run_sanity_tests.py` | All sanity tests passed |

The warning is the existing expected all-zero action-mask fallback warning in `tests/test_ddqn_action_masking.py`.

## GPU / Device Status

PyTorch status:

- Torch version: `2.12.0+cpu`
- CUDA available: `False`
- CUDA version: `None`
- GPU visible to PyTorch: `None`
- Training device used: `cpu`

Minimal `ddqn.device` support was added so `device: auto` resolves to CUDA if a CUDA-enabled PyTorch build is installed, otherwise CPU.

## Cross-Scenario Tuned DDQN Summary

| Scenario | Reward | Throughput/frame | Drop | Jam | Fairness | Backscatter success | Active success | Main decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tuned_no_jammer | -724.2050 | 0.7125 | 0.5336 | 0.0000 | 0.6092 | 0.9971 | 0.9974 | PASS |
| tuned_static_weak | -1626.9084 | 0.7312 | 0.6652 | 0.1408 | 0.3492 | 0.8104 | 0.8507 | PASS |
| tuned_backscatter_types | -1971.1129 | 0.3242 | 0.5846 | 0.6093 | 0.1930 | 0.3899 | 0.5231 | PARTIAL PASS |

## Tuned Backscatter-Types Configuration

Config: `configs/ddqn_tuned/tuned_backscatter_types.yaml`

- Environment: `configs/scenario_4_backscatter_types_calibrated.yaml`
- Episodes: 500
- Max steps per episode: 200
- Evaluation interval: 10 episodes
- Evaluation episodes: 30
- Seed: 42
- State type: `concat_global_local`
- Hidden sizes: `[256, 256]`
- Learning rate: `0.0005`
- Gamma: `0.99`
- Batch size: `128`
- Replay capacity: `100000`
- Min replay size: `2000`
- Target update steps: `1000`
- Gradient clip norm: `10.0`
- Epsilon: `1.0 -> 0.05` over `25000` training steps
- Device: `auto` resolved to `cpu`

Training runtime was about 2696.9 seconds, approximately 45.0 minutes.

## Tuned Backscatter-Types Final Evaluation

Final deterministic evaluation command:

```powershell
python scripts\evaluate_ddqn.py --config configs/ddqn_tuned\tuned_backscatter_types.yaml --checkpoint results/checkpoints/tuned_backscatter_types_best.pt --episodes 30
```

Output CSV: `results/csv/tuned_backscatter_types_final_eval.csv`

| Metric | Value |
|---|---:|
| Reward | -1971.1129 |
| Throughput/frame | 0.3242 |
| Drop rate | 0.5846 |
| Jamming failure rate | 0.6093 |
| Fairness | 0.1930 |
| Energy efficiency | 0.9417 |
| Backscatter success rate | 0.3899 |
| Active success rate | 0.5231 |
| Mode usage harvest | 38.9667 |
| Mode usage backscatter | 135.7667 |
| Mode usage active | 8.2667 |
| Mode usage idle | 53.5000 |

## Scenario 4 Baseline Comparison

Comparison CSV: `results/csv/tuned_backscatter_types_vs_baselines.csv`

| Rank | Policy | Throughput/frame | Drop | Jam | Fairness | Energy efficiency | Backscatter success | Active success |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | greedy_nearest | 0.8977 | 0.5223 | 0.5463 | 0.1836 | 7.2350 | 0.4302 | 0.6019 |
| 2 | backscatter_only | 0.8522 | 0.5294 | 0.5515 | 0.1719 | 10.7016 | 0.4258 | 0.0000 |
| 3 | greedy_sinr | 0.4783 | 0.5575 | 0.7806 | 0.2385 | 1.7265 | 0.1879 | 0.2734 |
| 4 | htt_only | 0.3278 | 0.5823 | 0.4891 | 0.0937 | 0.4029 | 0.0000 | 0.4788 |
| 5 | ddqn | 0.3242 | 0.5846 | 0.6093 | 0.1930 | 0.9417 | 0.3899 | 0.5231 |
| 6 | random | 0.1075 | 0.6084 | 0.6232 | 0.3530 | 0.3242 | 0.3817 | 0.3640 |

DDQN throughput comparisons:

- DDQN vs random: `+201.55%`
- DDQN vs best baseline: `-63.89%`
- DDQN rank among listed policies: `5 / 6`

## Backscatter-Types Mode Analysis

DDQN does use backscatter and active transmission:

- Backscatter success rate: `0.3899`
- Active success rate: `0.5231`
- Mean deterministic mode usage: backscatter `135.7667`, active `8.2667`

The learned policy did not collapse to zero use of either mode, but it is heavily skewed toward backscatter in the final best-checkpoint evaluation. Active transmission is present and successful when selected, but underused relative to the active-only and mixed specialized baselines.

Backscatter success is meaningful and close to the random/backscatter-only range, but the jamming failure rate remains high. This suggests DDQN found a partial communication policy but has not learned robust jammer-aware mode/device/UAV coordination in Scenario 4.

## Learning Signal

Training log: `results/csv/tuned_backscatter_types_train_log.csv`

Evaluation log: `results/csv/tuned_backscatter_types_eval_log.csv`

First 50 vs last 50 training episodes:

| Metric | First 50 | Last 50 |
|---|---:|---:|
| Train reward | -1976.3538 | -2044.0501 |
| Train throughput | 24.7400 | 25.4600 |
| Train drop rate | 0.6061 | 0.6059 |
| Train jamming failure | 0.6841 | 0.7285 |
| Train fairness | 0.3158 | 0.2142 |
| Avg loss | 2.7481 | 10.9697 |
| Epsilon | 0.8426 | 0.0500 |
| Backscatter mode usage | 42.5200 | 54.2400 |
| Active mode usage | 11.9000 | 19.8400 |
| Backscatter success | 0.3180 | 0.2577 |
| Active success | 0.3143 | 0.2864 |

Evaluation log checkpoints:

| Eval point | Episode | Reward | Throughput/frame | Drop | Jam | Fairness |
|---|---:|---:|---:|---:|---:|---:|
| First eval | 1 | -2762.6681 | 0.0145 | 0.6200 | 0.2819 | 0.7511 |
| Best eval throughput | 400 | -1965.6614 | 0.2505 | 0.5889 | 0.6502 | 0.1655 |
| Final eval-log point | 500 | -2231.3550 | 0.0175 | 0.6249 | 0.7810 | 0.3415 |

Learning signal was present early and mid-run: evaluation throughput improved from `0.0145` to a best logged value of `0.2505`, and the best checkpoint reached `0.3242` on the separate 30-episode final evaluation. However, training was unstable after epsilon reached `0.05`; final eval-log throughput fell sharply, loss increased, and jammer failures remained high.

## Generated Files

Configuration:

- `configs/ddqn_tuned/tuned_backscatter_types.yaml`

Checkpoints:

- `results/checkpoints/tuned_backscatter_types_best.pt`
- `results/checkpoints/tuned_backscatter_types_latest.pt`

CSV files:

- `results/csv/tuned_backscatter_types_train_log.csv`
- `results/csv/tuned_backscatter_types_eval_log.csv`
- `results/csv/tuned_backscatter_types_final_eval.csv`
- `results/csv/tuned_backscatter_types_vs_baselines.csv`

Figures:

- `results/figures/tuned_backscatter_types_training_reward.png`
- `results/figures/tuned_backscatter_types_eval_throughput.png`
- `results/figures/tuned_backscatter_types_eval_drop_rate.png`
- `results/figures/tuned_backscatter_types_loss.png`
- `results/figures/tuned_backscatter_types_epsilon.png`
- `results/figures/tuned_backscatter_types_mode_usage.png`
- `results/figures/tuned_backscatter_types_vs_baselines_throughput.png`

## Bugs / Fixes

1. Added safe `ddqn.device` support:
   - `device: auto` now selects CUDA only if `torch.cuda.is_available()`, otherwise CPU.
   - The trainer now passes `ddqn.device` into the DDQN agent.

2. Added `--output-prefix` support to `scripts/compare_ddqn_to_baselines.py`:
   - This enables the requested output path `results/csv/tuned_backscatter_types_vs_baselines.csv`.

No simulator or baseline behavior was changed.

## Limitations / Suspicious Results

- The current PyTorch build is CPU-only, so RTX 3060 acceleration was unavailable.
- Scenario 4 DDQN is unstable: best checkpoint is useful, but final eval-log performance at episode 500 collapses.
- Loss increased from the first 50 to last 50 episodes, suggesting value estimation instability or overfitting to replay distribution.
- The deterministic best-checkpoint policy is heavily skewed toward backscatter and underuses active transmission.
- DDQN remains far below `backscatter_only` and `greedy_nearest` in throughput on Scenario 4.
- The action space may still be too large or insufficiently structured for efficient DDQN learning in heterogeneous backscatter-type scenarios.
- Reward shaping may not sufficiently separate "safe low-throughput backscatter" from "higher-throughput active when viable" under jamming.

## Recommendation

Do not move directly to final MAPPO/QMIX experiments yet.

Recommended next step is a focused DDQN stabilization pass before MAPPO/QMIX:

1. Add reward scaling/normalization if not already active in the trainer used for these tuned runs.
2. Add or verify Dueling DDQN support for Scenario 4.
3. Run a smaller hyperparameter sweep on Scenario 4:
   - slower epsilon decay,
   - lower learning rate such as `0.00025`,
   - target update sensitivity,
   - replay warmup and batch size sensitivity.
4. Consider action-space factoring or hierarchical action selection before MAPPO/QMIX, because Scenario 4 still shows unstable mode/device selection with a flat DDQN action output.

After this stabilization pass, move to a MAPPO/QMIX interface prototype rather than final experiments.
