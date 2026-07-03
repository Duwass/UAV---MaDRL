# Controlled CTDE Short-Run Experiment Protocol

## Scope

This protocol defines a controlled short-run procedure for the CTDE foundation on the accepted 3D UAV-IoT backscatter environment.

This is an early stability check only. It is not a final experiment, not a main paper result, and not evidence that CTDE outperforms QMIX, hierarchical DDQN, or any other baseline. It does not optimize fairness, packet drop, reward design, target/mode masking, or the 3D environment.

No training run, controlled experiment, result file, checkpoint, or performance claim is authorized by this document. Execution requires a separate Brain-approved EXP-1 prompt.

## Experimental Foundation

Accepted environment foundation:

- 3D foundation commit: `8213203b4484540948c2c16c562476a7d9eff7d5`
- CTDE foundation commit at protocol creation: `d855ad025607f3b3735748ba9290a7bc00057cbf`
- 3D environment config: `configs/scenario_4_3d_backscatter_types_calibrated.yaml`
- CTDE config: `configs/ctde/ctde_3d_base.yaml`

Fixed dimensions and factors:

- `obs_dim = 114`
- `state_dim = 89`
- `action_dim = 1056`
- `movement_actions = 11`
- `target_options = 16`
- `modes = 6`
- `num_uav = 2`
- `num_iot = 15`

The flat action dimension must remain:

```text
11 movement actions * 16 target options * 6 modes = 1056 actions
```

## CTDE Architecture Under Test

The CTDE path under test is the accepted foundation implementation:

- Decentralized actor with factorized heads:
  - movement head: 11 logits
  - target head: 16 logits
  - mode head: 6 logits
- Centralized V critic:
  - input: global state with dimension 89
  - output: scalar `V(s)`
- Shared global reward is used for CTDE updates.
- Actor loss uses a centralized advantage estimated from the V critic.
- Evaluation is decentralized execution only.

Evaluation must not use:

- critic values
- global state
- environment action masks
- hierarchical wrapper or hierarchical executor
- environment object queries for action selection

## Experiment Stages

### Stage A: Sanity Smoke

Purpose: verify that the CTDE loop can collect transitions, update once or more, and evaluate without crashing or leaking centralized information into decentralized execution.

Recommended settings:

- `num_iterations = 2`
- `rollout_steps = 5`
- `batch_size = 4`
- `eval_episodes = 1`
- `eval_max_steps = 5`
- `save_results = false`
- `save_checkpoint = false`

Expected outcome:

- The run completes without crashing.
- `transitions_collected > 0`
- at least one update occurs when the buffer has enough data
- actor and critic losses are finite
- decentralized evaluation completes without critic/global-state/env-mask access
- no result files or checkpoints are written unless an explicitly approved save path is provided

Stage A is a sanity check only and must not be interpreted as performance evidence.

### Stage B: Short-Run Pilot

Purpose: verify short-run stability over a small controlled budget after Stage A is accepted.

Recommended settings:

- `num_iterations = 50`
- `rollout_steps = 20`
- `batch_size = 32`
- `eval_every = 10`
- `eval_episodes = 3`
- `eval_max_steps = 100`
- `seed = 42`
- `save_results = false` by default unless Brain explicitly enables a guarded output directory
- `save_checkpoint = false`

Expected outcome:

- losses remain finite
- buffer size and transition counts are consistent
- evaluation remains decentralized
- metrics are sufficient to diagnose stability, action usage, and 3D behavior
- no claim is made that CTDE beats any baseline

Stage B is a stability pilot only.

### Stage C: Multi-Seed Pilot

Purpose: check rough variance after Stage B is stable and reviewed.

Recommended seeds:

- `42`
- `43`
- `44`

Budget:

- same as Stage B, or slightly larger only if Brain approves

Expected outcome:

- all seeds complete without NaN/Inf
- variance is inspected for instability or collapse
- no final conclusion is drawn unless a later final experiment protocol is approved

Stage C is still not a final paper result.

## Metrics To Record

### Training Metrics

- `actor_loss`
- `critic_loss`
- `mean_value`
- `mean_target`
- `mean_advantage`
- `entropy`
- `transitions_collected`
- `buffer_size`
- `losses_finite`

### Environment Metrics

- `episode_return`
- `throughput`
- `packet_drop_rate`
- `jammed_rate`
- `fairness_index`
- `energy_efficiency`
- `backscatter_success`
- `active_success`

### 3D Metrics

- average altitude
- minimum altitude
- maximum altitude
- vertical action rate
- average vertical movement
- average UAV-IoT 3D distance
- average UAV-jammer 3D distance
- altitude boundary hits

### Policy Diagnostics

- movement action distribution
- target no-target rate
- target distribution over IoT options
- mode distribution
- UP action rate
- DOWN action rate
- sanitizer rate, if available

## Success Criteria

A stage is considered successful only for stability purposes if:

- there is no crash
- actor and critic losses are finite
- transitions are collected
- replay buffer shapes remain consistent
- decentralized evaluation runs cleanly
- no leakage violation is observed
- requested metrics are logged or returned
- no result files or checkpoints are written unless explicitly enabled
- reward does not need to beat any baseline

## Failure Criteria

A stage should be stopped and reported if any of the following occur:

- NaN or Inf in actor loss, critic loss, value estimates, targets, advantages, rewards, or metrics
- transitions are not collected
- replay buffer shape or dtype inconsistency appears
- policy collapses to idle/no-target behavior immediately
- sanitizer dominates action selection unexpectedly
- no vertical action is ever selected in a run that should explore movement
- decentralized evaluation requires global state, critic values, env masks, env object queries, or hierarchical code
- unexpected files are written to `results/`, `checkpoints/`, temporary plot paths, or cache-like directories
- metrics are nonsensical, missing, or inconsistent with the configured episode budget

## Logging And Output Policy

Default policy:

- no `results/` writes
- no checkpoint writes
- no temporary plots
- no overwrite of old outputs

If Brain later approves output logging, use a guarded path such as:

```text
results/ctde_3d_short_run/<timestamp_or_seed>/
```

Output rules for any approved run:

- never overwrite an existing run directory
- save a copy of the resolved CTDE config
- save the git commit hash
- save seed and device information
- write CSV/JSON metrics only when explicitly enabled
- keep checkpoints disabled unless a later prompt explicitly authorizes them

## Reproducibility Fields

Every approved run report should include:

- random seed
- git commit hash
- CTDE config path and resolved config copy
- 3D environment config path
- `obs_dim`, `state_dim`, and `action_dim`
- movement/target/mode factor sizes
- test status before the run
- Python version
- PyTorch version
- CPU/CUDA device information

## Leakage Guard Checklist

Before accepting any Stage A/B/C result, verify:

- evaluation action selection does not use the critic
- evaluation action selection does not call `env.get_global_state`
- evaluation action selection does not call `env.get_action_mask`
- evaluation action selection does not import or call hierarchical wrappers/executors
- actor receives local observation only
- movement mask, if used in collection, is limited to local/public movement feasibility
- no target mask is used
- no mode mask is used
- no environment object query is used to choose target or mode during decentralized evaluation

## Baseline Comparison Policy

No baseline comparison is allowed until Stage B is stable and reviewed.

Rules for later comparison:

- Stage A and Stage B cannot support claims that CTDE is better than a baseline.
- Any later CTDE-vs-baseline comparison must use the same accepted 3D config.
- Later comparisons must use matched seeds and comparable budgets.
- Old 2D results are not directly comparable to this 3D CTDE path.
- Fairness/drop conclusions require a separate approved protocol.

## Current Stage A Readiness

The current CTDE script and config are sufficient for a guarded Stage A smoke check after Brain approval:

- `scripts/train_ctde_3d.py` supports `--config`, `--smoke`, `--seed`, and optional `--save-dir`.
- `configs/ctde/ctde_3d_base.yaml` already contains Stage A-sized smoke defaults:
  - `num_iterations = 2`
  - `rollout_steps = 5`
  - `batch_size = 4`
  - `eval_episodes = 1`
  - `eval_max_steps = 5`
- `marl.ctde.train_loop.train_ctde_smoke` documents and implements a no-file-write smoke loop.
- The CLI only writes a JSON summary when `--save-dir` is explicitly provided.
- The config keeps `output.save_results = false` and `output.save_checkpoint = false`.

Known gaps before Stage B/EXP-1:

- no guarded no-overwrite run directory helper yet
- no metrics CSV/JSON writer beyond the optional smoke summary JSON
- no `eval_every` control in the current smoke train loop
- no full policy-diagnostic export yet
- no automated reproducibility bundle yet

These gaps are acceptable for EXP-0 and should be handled only in a separately approved EXP-1 runner.

## Proposed Next Phase

`EXP-1 - Guarded CTDE Short-Run Runner`

Suggested scope:

- add optional guarded logging only when a save directory is explicitly enabled
- enforce no-overwrite output directories
- save resolved config, commit hash, seed, dimensions, and device information
- add CSV/JSON metrics for Stage A/B diagnostics
- support Stage A first
- support Stage B only after Stage A is reviewed
- keep checkpoints disabled

## Out Of Scope

The following are intentionally out of scope for this protocol:

- full training
- controlled experiment execution
- multi-seed final result
- checkpoint generation
- result generation without explicit approval
- reward redesign
- fairness/drop redesign
- target mask
- mode mask
- environment 3D changes
- observation/state/action dimension changes
- physical fading, LoS, Rician, or channel-model redesign
- LaTeX/report update
- performance claims
