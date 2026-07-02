# Phase 3D Environment Audit

## Scope

This audit documents the 3D environment foundation that lifts the UAV-IoT backscatter model from 2D/2.5D behavior to explicit 3D geometry.

Included:
- 3D entity coordinates for UAV, IoT, jammer, and RF source.
- 3D distance selection for channel, coverage, and jammer influence in the 3D config.
- UP/DOWN movement primitives and 3D flat action sizing.
- 3D local observations, global state, metrics, and debug visualization.
- Integration tests and short smoke runs.

Not included:
- CTDE implementation.
- Fairness, packet drop, or reward redesign.
- Rician, LoS probability, fading, antenna gain, or other advanced channel models.
- Training result or performance claims.
- Final report LaTeX updates.

## Source Commit and Worktree Note

- Branch: `main`
- HEAD commit: `518a1730bb7eccb078ab90cb4f219f5b002b6161`
- Worktree note: the 3D environment changes are uncommitted at the time of this audit.
- The deleted file `01_Papers_and_Docs/hoangtrungdung_thesis (1).pdf` was already present as a worktree deletion before this phase and is not part of the 3D work.

## Files Added

- `02_Code/project/configs/scenario_4_3d_backscatter_types_calibrated.yaml`: 3D Scenario 4 config.
- `02_Code/project/tests/test_3d_entities_geometry.py`: entity position and reset/step geometry tests.
- `02_Code/project/tests/test_3d_channel_interference.py`: 3D channel, SINR, coverage, and jammer interference tests.
- `02_Code/project/tests/test_3d_movement_action.py`: UP/DOWN movement and 3D action dimension tests.
- `02_Code/project/tests/test_3d_observation_state.py`: 3D observation and global state layout tests.
- `02_Code/project/tests/test_3d_metrics_logging.py`: 3D metrics, evaluation summary, and plot smoke tests.
- `02_Code/project/scripts/trajectory_3d_utils.py`: short rollout collector for debug visualization.
- `02_Code/project/scripts/plot_3d_trajectories.py`: debug 3D trajectory plotter.
- `02_Code/project/scripts/plot_altitude_over_time.py`: debug altitude-over-time plotter.
- `02_Code/project/docs/phase_3d_environment_audit.md`: this audit document.
- `02_Code/project/docs/phase_3d_next_steps.md`: short next-step note for CTDE review.

## Files Modified

- `02_Code/project/envs/entities.py`: added `z`, `h` alias, and 3D `position()` support for entities.
- `02_Code/project/envs/channel_model.py`: added/shared 3D distance helpers and distance-mode selection.
- `02_Code/project/envs/mobility_model.py`: added 3D movement primitives and altitude clipping support.
- `02_Code/project/envs/uav_backscatter_env.py`: wired 3D config, entity initialization, distance modes, action sizing, observation/state dimensions, 3D metrics, and info exposure.
- `02_Code/project/envs/hierarchical_action.py`: kept hierarchical executor compatible with dynamic movement/action sizing.
- `02_Code/project/baselines/*.py`: kept baseline policies compatible with 3D distance/action behavior where applicable.
- `02_Code/project/scripts/evaluate_ddqn.py`: prints 3D summary metrics when they exist.
- `02_Code/project/scripts/evaluate_qmix.py`: prints 3D summary metrics when they exist.
- `02_Code/project/scripts/evaluate_all_baselines.py`: includes 3D columns in printed summaries when they exist.

## 3D Entity Representation

- UAV: stores `(x, y, z)`, exposes `h` as an alias for altitude compatibility, and returns `(x, y, z)` from `position()`.
- IoTDevice: stores `(x, y, z)`, with the 3D config using ground nodes at `z = 0.0`.
- Jammer: stores `(x, y, z)`, with the 3D config using jammer altitude from config.
- RFSource: stores `(x, y, z)`, with the 3D config using ground RF source altitude unless configured otherwise.

## 3D Config

Config path:
- `02_Code/project/configs/scenario_4_3d_backscatter_types_calibrated.yaml`

Important settings:
- `environment.dimension = 3`
- UAV altitude: `initial_altitude = 100`, `altitude_min = 60`, `altitude_max = 140`
- Jammer altitude: `100`
- IoT altitude: `0.0`
- RF source altitude: `0.0`
- 3D movement count: `11`
- Coverage distance mode: `3d`
- Channel distance mode: `3d`
- Jammer influence distance mode: `3d`

The legacy Scenario 4 config is not overwritten.

## 3D Distance, Channel, and SINR

The 3D config uses explicit 3D distance for:
- Channel received power and SINR.
- Coverage checks and action masking.
- Jammer influence and interference.
- Jammer RF harvesting distance.

Legacy 2D configs preserve their horizontal behavior unless a distance mode is explicitly configured otherwise.

## 3D Action and Movement

- Legacy movement count: `9`
- 3D movement count: `11`
- Added vertical primitives:
  - `MOVE_UP`
  - `MOVE_DOWN`
- Legacy flat action dimension: `9 * (15 + 1) * 6 = 864`
- 3D flat action dimension: `11 * (15 + 1) * 6 = 1056`
- Hierarchical wrapper action dimension remains `10` high-level actions.
- UAV altitude is clipped to `[altitude_min, altitude_max]`.
- Action masks invalidate UP at `altitude_max` and DOWN at `altitude_min`.

## 3D Observation and Global State

Legacy dimensions:
- `obs_dim = 97`
- `state_dim = 71`
- `action_dim = 864`

3D dimensions:
- `obs_dim = 114`
- `state_dim = 89`
- `action_dim = 1056`

3D local observation layout:
- Base: `[uav_x, uav_y, uav_z, uav_energy, jammer_dx, jammer_dy, jammer_dz, jammer_distance_3d, primary_busy]`
- Per IoT: `[dx, dy, dz, distance_3d, queue, energy, coverage]`

3D global state layout:
- Per UAV: `[x, y, z, energy]`
- Per IoT: `[x, y, z, queue, energy]`
- Per jammer: `[x, y, z, energy]`
- Tail: `[primary_busy, step_norm]`

Normalization:
- `z_norm = (z - altitude_min) / (altitude_max - altitude_min)`, clipped to `[-1, 1]`.
- `dz_norm = dz / altitude_range`, clipped to `[-1, 1]`.
- `distance_3d_norm = distance_3d / max_3d_distance`.

## 3D Metrics

Added metrics:
- `avg_uav_altitude`: mean UAV altitude over reset and post-step snapshots.
- `min_uav_altitude`: minimum observed UAV altitude.
- `max_uav_altitude`: maximum observed UAV altitude.
- `vertical_action_rate`: UP/DOWN actions divided by processed UAV actions.
- `avg_vertical_movement`: total absolute vertical displacement divided by processed UAV actions.
- `avg_uav_iot_3d_distance`: all-pairs UAV-IoT 3D distance snapshot average.
- `avg_uav_jammer_3d_distance`: nearest-jammer 3D distance averaged over UAVs and snapshots.
- `altitude_boundary_hits`: UP/DOWN actions that end at altitude min or max, including clipped actions.

Important metric note:
- `avg_uav_iot_3d_distance` measures overall spatial geometry as an all-pairs snapshot average.
- It is not a served-target distance or selected-target distance.
- Served or selected target distance should be a separate future metric if needed.

## Visualization

Scripts:
- `02_Code/project/scripts/plot_3d_trajectories.py`
- `02_Code/project/scripts/plot_altitude_over_time.py`

Behavior:
- Input can be a trajectory JSON or a short rollout from the 3D config.
- PNG output goes to the user-provided `--output` path.
- If no output is provided, the scripts default to the system temp directory.
- Optional trajectory JSON is saved only when `--save-trajectory-json` is provided.
- These plots are debug-grade and are not publication-grade figures.

## Tests and Smoke Runs

Integration test command:

```powershell
python -B -m pytest tests/test_3d_entities_geometry.py tests/test_3d_channel_interference.py tests/test_3d_movement_action.py tests/test_3d_observation_state.py tests/test_3d_metrics_logging.py tests/test_env_reset_step.py tests/test_action_masking.py tests/test_hierarchical_action.py tests/test_hierarchical_action_disable.py tests/test_hierarchical_executor_config.py tests/test_qmix_networks.py tests/test_qmix_action_masking.py -q
```

Result:
- `58 passed`

Smoke runs:
- 3D random env smoke: reset plus 5 random primitive-action steps. Passed.
- 3D baseline random smoke: 1 short episode with `max_steps = 5` and `save_csv = False`. Passed.
- 3D hierarchical wrapper smoke: reset plus one high-level step. Passed.

No training was performed.
No checkpoint was loaded.
No output was written to tracked `results/` files.

## Backward Compatibility

- Legacy Scenario 4 still resets and steps with `obs_dim = 97`, `state_dim = 71`, and `action_dim = 864`.
- The old Scenario 4 config was not overwritten.
- Existing result files were not edited.
- Old checkpoints are expected to be incompatible with 3D primitive-action models because 3D uses new observation, state, and action dimensions.
- Hierarchical wrapper remains at `action_dim = 10`, but any CTDE use of it should audit decentralization and information leakage separately.

## Known Limitations

- Some scheduling and target-selection heuristics can still use horizontal reasoning.
- Debug plots are not publication-grade.
- CTDE is not implemented.
- Fairness/drop improvements are not included.
- Advanced physical channel features such as LoS probability, Rician fading, and antenna gain are not included.
- No new performance claim is made.

## Acceptance Checklist

| Item | Status | Note |
| --- | --- | --- |
| Entity positions are 3D | Pass | UAV, IoT, jammer, and RF source support `(x, y, z)`. |
| Channel/SINR uses 3D distance in 3D config | Pass | `channel.distance_mode = 3d`. |
| Coverage uses 3D distance in 3D config | Pass | `coverage.distance_mode = 3d`. |
| Jammer influence uses 3D distance in 3D config | Pass | `jammer.influence_distance_mode = 3d`. |
| UAV can move UP/DOWN | Pass | 3D movement count is 11. |
| Altitude is clipped/masked | Pass | Clips at min/max and masks boundary UP/DOWN. |
| Observation exposes z/dz/d3 | Pass | 3D obs dim is 114. |
| Global state exposes 3D positions | Pass | 3D state dim is 89. |
| 3D metrics are logged | Pass | Episode and frame metrics expose 3D keys. |
| Legacy config still works | Pass | Legacy dims remain 97/71/864. |
| Tests pass | Pass | Safe integration suite passed 58 tests. |
| No results overwritten | Pass | Final status has no `results/` diffs. |
