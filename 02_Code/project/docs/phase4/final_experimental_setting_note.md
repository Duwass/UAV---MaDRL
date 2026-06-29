# Final Experimental Setting Note

The final report will use the actual implemented and evaluated Scenario 4 configuration as the source of truth. This is a reporting decision only; it does not change the simulator, model, training procedure, checkpoints, configs, or results.

The final reported Scenario 4 setting is:

- 2 UAVs;
- 15 IoT devices;
- 1 mobile jammer;
- frame length: 10;
- episode length: 200;
- flat action dimension: 864;
- hierarchical high-level action dimension: 10;
- QMIX local observation dimension: 97;
- QMIX global state dimension: 70.

The relevant implementation/configuration sources are:

- `configs/scenario_4_backscatter_types_calibrated.yaml`;
- `configs/qmix_tuned/qmix_sc4_base.yaml`;
- `envs/uav_backscatter_env.py`;
- `envs/hierarchical_action.py`;
- `envs/hierarchical_env.py`;
- `marl/qmix/qmix_trainer.py`;
- `marl/qmix/qmix_agent.py`;
- `marl/qmix/networks.py`.

Any older proposed simulation-parameter table that differs from this setting should be treated as a planning artifact, not as the final reported experimental configuration. The report should not mix proposed parameters with the actual evaluated Phase 3 results.

No retraining will be performed for this reporting direction. All claims must be tied to existing Phase 3 artifacts, especially the publication tables and figures under:

- `results/publication_tables/`;
- `results/publication_figures/`;
- `results/publication_reports/`;
- `results/publication_artifacts_index.md`.

This note should be cited internally during writing whenever experimental settings are summarized.
