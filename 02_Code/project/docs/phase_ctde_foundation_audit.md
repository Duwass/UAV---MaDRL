# Phase CTDE Foundation Audit

## Scope

This audit covers the CTDE foundation built on the accepted 3D UAV backscatter environment. It is not a final experiment, not a performance comparison, and not a paper result. It does not include fairness/drop reward redesign, physical channel enhancement, final LaTeX report updates, multi-seed evaluation, or baseline comparison.

## Accepted 3D Foundation

The CTDE foundation starts from:

- Branch: `phase-3d-foundation`
- Commit: `8213203b4484540948c2c16c562476a7d9eff7d5`
- 3D config: `configs/scenario_4_3d_backscatter_types_calibrated.yaml`

Accepted 3D dimensions:

- `obs_dim = 114`
- `state_dim = 89`
- `action_dim = 1056`
- movement actions = 11
- target options = 16
- communication modes = 6

## Why CTDE

Teacher feedback identified that the earlier decentralized execution path was not clean enough. The revised goal is Centralized Training with Decentralized Execution:

- Centralized training may use global state for the critic.
- Decentralized execution must select each UAV action from that UAV's local observation only.
- Evaluation/deployment must not use critic values, global state, env action masks, env object queries, or the hierarchical executor for action selection.

## Why Not Hierarchical Wrapper

The hierarchical wrapper and executor remain useful as baseline/reference code, but they are not the main CTDE path. The executor maps high-level actions into primitive actions by reading environment/global information, including:

- UAV list/state.
- IoT queue, energy, and type.
- Jammer list/state.
- Primary channel busy state.
- Delivered packet counts.
- Channel/config/helper methods such as SINR, coverage, and energy-cost helpers.

Conclusion: the hierarchical wrapper can remain a baseline/reference, but CTDE foundation runs directly on the base 3D environment with primitive flat actions.

## CTDE Architecture

The CTDE foundation uses the base 3D env directly, without the hierarchical wrapper.

Factorized actor:

- movement head: 11 logits
- target head: 16 logits
- mode head: 6 logits

Flat action encoding:

```text
(movement * (num_iot + 1) + target) * 6 + mode
```

Centralized V critic:

- input: global state with dimension 89
- output: scalar `V(s)`

## Decentralized Execution Contract

Execution/evaluation action selection follows this contract:

- Actor input is local observation only.
- No global state is used for action selection.
- No critic is used for action selection.
- No env object query is used for action selection.
- No env action mask is used.
- No hierarchical executor is used.
- Deterministic evaluation uses greedy selection per factor head.

## Centralized Training Contract

Training stores and uses both local and global data with separate responsibilities:

- Replay buffer stores per-agent local observations and global state.
- Critic uses global state for centralized value learning.
- Actor loss uses local observations and a detached centralized advantage.
- Current foundation uses the shared scalar reward returned by the environment.
- The trainer is not part of deployment/evaluation action selection.

## Action Mask Policy

The CTDE foundation allows only a movement boundary mask built from local observation and public movement semantics.

- Allowed: movement mask derived from local own `x/y/z` observation layout, currently used for altitude up/down boundaries.
- Not included: target masks.
- Not included: mode masks.
- Not used: env action mask.

This avoids target/coverage/jammer/queue masks that may require hidden env/global information.

## Invalid Factor Sanitizer

The sanitizer is semantic only and does not query env/global state:

- idle forces target to no-target.
- backscatter, active, and relay require a target.
- target-required mode plus no-target falls back to idle/no-target.

## Files Added

CTDE source/config/script files:

- `marl/ctde/utils.py`
- `marl/ctde/factorized_policy.py`
- `marl/ctde/networks.py`
- `marl/ctde/replay_buffer.py`
- `marl/ctde/rollout.py`
- `marl/ctde/evaluation.py`
- `marl/ctde/ctde_trainer.py`
- `marl/ctde/train_loop.py`
- `configs/ctde/ctde_3d_base.yaml`
- `scripts/train_ctde_3d.py`

## Tests Added

CTDE tests:

- `tests/test_ctde_factorized_action.py`
- `tests/test_ctde_networks_replay_buffer.py`
- `tests/test_ctde_rollout_collector.py`
- `tests/test_ctde_decentralized_evaluation.py`
- `tests/test_ctde_trainer_update.py`
- `tests/test_ctde_train_loop_smoke.py`

## Test Summary

Latest command:

```powershell
python -m pytest 02_Code\project\tests\test_ctde_factorized_action.py 02_Code\project\tests\test_ctde_networks_replay_buffer.py 02_Code\project\tests\test_ctde_rollout_collector.py 02_Code\project\tests\test_ctde_decentralized_evaluation.py 02_Code\project\tests\test_ctde_trainer_update.py 02_Code\project\tests\test_ctde_train_loop_smoke.py 02_Code\project\tests\test_3d_entities_geometry.py 02_Code\project\tests\test_3d_channel_interference.py 02_Code\project\tests\test_3d_movement_action.py 02_Code\project\tests\test_3d_observation_state.py 02_Code\project\tests\test_3d_metrics_logging.py
```

Latest result:

```text
113 passed in 12.15s
```

The handoff expected `102 passed`; the actual current result is `113 passed` because the active suite now includes all CTDE-1 through CTDE-6 tests.

## Smoke Training Summary

Latest CTDE smoke run:

```json
{
  "eval_mean_return": 1.5258578643762692,
  "eval_total_steps": 5,
  "iterations": 2,
  "last_actor_loss": 3.600900650024414,
  "last_critic_loss": 0.08159886300563812,
  "losses_finite": true,
  "transitions_collected": 10,
  "updates": 2
}
```

This is smoke only. It is not performance validation, not a controlled experiment, and not a paper result.

## Evaluation/Deployment Guard

Current tests guard that evaluation/deployment action selection:

- does not call `env.get_global_state()`;
- does not call `env.get_action_mask()`;
- does not use a critic parameter/path;
- does not import hierarchical env/action modules;
- sends actor inputs with local observation shape `114`, not global state shape `89` and not concatenated global/local state.

## Known Limitations

- Smoke training only.
- No stable training protocol yet.
- No multi-seed experiment.
- No target/mode mask.
- No fairness/drop reward redesign.
- No comparison against baselines yet.
- Policy quality is not meaningful yet.
- Local observation abstraction includes IoT queue/energy/coverage features as available observation.

## Acceptance Checklist

| Item | Status |
|---|---|
| Actor local obs only | Pass |
| Critic global state training only | Pass |
| Evaluation no critic | Pass |
| Evaluation no global state call | Pass |
| Evaluation no env action mask | Pass |
| No hierarchical executor | Pass |
| Factorized action encode/decode works | Pass |
| Replay buffer shapes correct | Pass |
| Rollout collector stores transitions | Pass |
| Minimal update losses finite | Pass |
| Train smoke runs | Pass |
| 3D tests still pass | Pass |
| No results/checkpoints written by default | Pass |

## Next Steps

1. Commit/checkpoint the CTDE foundation as a separate step.
2. Design a controlled short-run experiment protocol.
3. Later, revisit fairness/drop reward behavior as its own phase.
4. Later, update the final LaTeX report with DE leakage audit and CTDE method notes.
