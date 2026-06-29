# UAV MADRL Anti-Jamming Backscatter Simulation Testbed

This repository is a Phase 3 simulation-design testbed for:

**Multi-Agent Deep Reinforcement Learning for anti-jamming UAV-assisted IoT networks with RF-powered ambient backscatter communication.**

The goal is to validate environment dynamics, baseline behavior, metrics, logging, and sanity checks before implementing heavy MARL algorithms or producing final paper results.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Using NVIDIA GPU / RTX 3060

Install a CUDA-enabled PyTorch build in the same Python environment used to run the project. For the tested Windows Python 3.13 environment, the CUDA 12.8 wheel was used:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA visibility:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
nvidia-smi
```

Set DDQN configs to auto-select GPU when available:

```yaml
ddqn:
  device: auto
```

The DDQN trainer prints `[DDQN] Using device: cuda` when PyTorch can use the NVIDIA GPU. Confirm runtime GPU usage with `nvidia-smi` while training is active. Environment simulation and action masking still run partly on CPU, so GPU acceleration may not reduce total runtime proportionally for small runs.

## Run Tests

```bash
pytest
```

## Run Sanity Tests

```bash
python scripts/run_sanity_tests.py
```

The sanity script checks no-traffic behavior, jammer degradation, energy-limited active transmission, no-busy-channel backscatter, queue overflow, strong jamming, backscatter energy use, and active-vs-backscatter throughput.

## Run One Baseline

```bash
python scripts/run_baseline.py --config configs/scenario_3_mobile_jammer.yaml --policy greedy_sinr --episodes 20
```

Available policies:

- `random`
- `greedy_nearest`
- `greedy_sinr`
- `htt_only`
- `backscatter_only`

## Evaluate All Baselines

```bash
python scripts/evaluate_all_baselines.py
```

This writes per-policy/per-scenario CSVs plus:

```text
results/csv/all_baselines_summary.csv
```

## Plot Results

```bash
python scripts/plot_results.py
```

Figures are saved under:

```text
results/figures/
```

## Phase 3 Baseline Validation

Run the validation workflow from the project root:

```bash
pytest
python scripts/run_sanity_tests.py
python scripts/evaluate_all_baselines.py --episodes 30
python scripts/summarize_results.py
python scripts/plot_results.py
```

The combined episode-level evaluation CSV is saved to:

```text
results/csv/all_baselines_summary.csv
```

The grouped summaries and ranking table are saved to:

```text
results/csv/baseline_summary_mean.csv
results/csv/baseline_summary_std.csv
results/csv/baseline_ranking_by_scenario.csv
```

Validation plots are saved to:

```text
results/figures/
```

The Markdown validation report for the research brain is saved to:

```text
results/phase3_baseline_validation_report.md
```

## Phase 3.1 Calibration Workflow

Run calibrated scenarios with action masking and diagnostic metrics:

```bash
pytest
python scripts/run_sanity_tests.py
python scripts/evaluate_all_baselines.py --episodes 30 --configs configs/scenario_0_no_jammer_calibrated.yaml configs/scenario_1_multi_uav_calibrated.yaml configs/scenario_2a_static_jammer_weak.yaml configs/scenario_2b_static_jammer_medium.yaml configs/scenario_2c_static_jammer_strong.yaml configs/scenario_3b_mobile_chase_uav.yaml configs/scenario_4_backscatter_types_calibrated.yaml --output-prefix phase3_1_calibrated
python scripts/summarize_results.py --input results/csv/phase3_1_calibrated_all_baselines_summary.csv --output-prefix phase3_1_calibrated
python scripts/plot_results.py --input results/csv/phase3_1_calibrated_all_baselines_summary.csv --output-prefix phase3_1_calibrated
```

Main Phase 3.1 artifacts:

```text
results/csv/phase3_1_calibrated_all_baselines_summary.csv
results/csv/phase3_1_calibrated_baseline_summary_mean.csv
results/csv/phase3_1_calibrated_baseline_summary_std.csv
results/csv/phase3_1_calibrated_baseline_ranking_by_scenario.csv
results/figures/phase3_1_calibrated_*.png
results/phase3_1_calibration_report.md
```

## Phase 3.2 Centralized DDQN Prototype

Install or update dependencies:

```bash
pip install -r requirements.txt
```

Train centralized-factorized DDQN with action masking:

```bash
python scripts/train_ddqn.py --config configs/ddqn/ddqn_scenario_0_no_jammer.yaml --episodes 1000
```

Evaluate a checkpoint:

```bash
python scripts/evaluate_ddqn.py --config configs/ddqn/ddqn_scenario_0_no_jammer.yaml --checkpoint results/checkpoints/ddqn_scenario_0_no_jammer_best.pt --episodes 30
```

Compare against calibrated baselines:

```bash
python scripts/compare_ddqn_to_baselines.py --ddqn-final-eval results/csv/ddqn_scenario_0_no_jammer_final_eval.csv --baseline-summary results/csv/phase3_1_calibrated_baseline_summary_mean.csv
```

Plot DDQN learning curves:

```bash
python scripts/plot_ddqn_training.py --train-log results/csv/ddqn_scenario_0_no_jammer_train_log.csv --eval-log results/csv/ddqn_scenario_0_no_jammer_eval_log.csv
```

DDQN artifacts are saved to:

```text
results/csv/ddqn_*_train_log.csv
results/csv/ddqn_*_eval_log.csv
results/csv/ddqn_*_final_eval.csv
results/csv/ddqn_vs_baselines_*.csv
results/checkpoints/ddqn_*_latest.pt
results/checkpoints/ddqn_*_best.pt
results/figures/ddqn_*.png
results/phase3_2_ddqn_report.md
```

## Environment Logic

The core environment is `envs/uav_backscatter_env.py`.

UAVs are mobile agents with 2D motion, fixed altitude, finite energy, communication coverage radius, and collision checks. Each UAV action encodes movement, selected IoT device, and communication mode.

IoT devices maintain a finite data queue and finite energy storage. New packets arrive once per frame using a binomial model over `frame_length` slots. Queue overflow is counted as dropped packets.

The RF source models the primary transmitter. Each frame has a busy/idle primary channel state. Busy frames allow RF energy harvesting and ambient backscatter collection. Idle frames allow active IoT transmission by default. The config flag `channel.allow_active_when_busy` can relax this for ablation tests.

The jammer can be disabled, static, random-walk mobile, or chase the nearest UAV. Jammer interference contributes to SINR at the UAV receiver. Stronger or closer jammers reduce SINR and transmission success probability.

Communication modes:

- `idle`: no communication.
- `harvest_support`: valid in busy frames; selected IoT harvests energy.
- `backscatter_collect`: valid in busy frames; selected IoT sends low-rate packets with no IoT transmit-energy cost.
- `active_collect`: valid in idle frames by default; selected IoT spends energy and can deliver more packets.
- `relay_to_bs`: placeholder mode with a small UAV communication-energy cost.
- `avoid_jammer`: no packet transmission; gives a small bonus if the UAV increases distance from the nearest jammer.

The first implementation uses a shared global reward:

```text
R = throughput reward
    - packet drop penalty
    - delay/queue penalty
    - UAV energy penalty
    - jamming failure penalty
    - collision penalty
    - unfairness penalty
    + jammer-avoidance bonus
```

Local UAV observations include own normalized position and energy, nearest jammer relative position and distance, primary busy state, and per-IoT relative position, distance, queue, energy, and coverage flag.

The global state includes all UAV states, all IoT states, jammer positions, primary busy flag, and normalized step index. It is exposed for future centralized training.

## Metrics

Episode CSVs include:

- `total_reward`
- `total_throughput`
- `avg_throughput_per_frame`
- `packet_delivery_ratio`
- `packet_drop_rate`
- `avg_queue_length`
- `avg_delay_proxy`
- `energy_efficiency`
- `uav_energy_consumption`
- `jamming_failure_rate`
- `collision_count`
- `fairness_index`
- `duplicate_target_count`

Frame CSVs include primary busy state, successful packets, dropped packets, queue length, total IoT/UAV energy, jamming failures, collision count, duplicate targets, and reward components.

## Simplifications

- The channel model is intentionally lightweight: distance-based path loss, additive jammer interference, and a smooth SINR-to-success-probability mapping.
- UAV-to-base-station relay is a placeholder mode because the Phase 3 priority is validating scheduling, anti-jamming, energy, queue, and backscatter/active mode dynamics.
- IoT type behavior is simple: type 1 is active-capable, type 2 has a slightly higher active-energy cost, and type 3 is initialized with lower energy and a higher active-energy cost.
- The reward is initially shared by all UAVs to support centralized-training/decentralized-execution experiments later.

## Next Step

After `pytest` and `python scripts/run_sanity_tests.py` pass, run all baselines for short episode counts, inspect the plots, then implement MAPPO or QMIX against `envs/parallel_env.py` using the existing local observations and `state()` global-state interface.
