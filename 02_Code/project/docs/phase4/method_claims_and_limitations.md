# Method Claims and Limitations

| Method claim | Implementation evidence | Phase 3 result artifact | Confidence | Limitation / caution |
| --- | --- | --- | --- | --- |
| Flat DDQN is insufficient for the large Scenario 4 action interface. | `envs/uav_backscatter_env.py::encode_action` gives \(9(N+1)6\); with \(N=15\), action dimension is 864. `marl/ddqn/*` implements the flat DDQN baseline. | `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_tables/table2_algorithm_progression.md`; flat DDQN throughput/frame 0.3242. | High | Say "for the tested DDQN prototype and evaluated Scenario 4", not all possible flat-action RL methods. |
| Hierarchical abstraction makes learning feasible in Scenario 4. | `envs/hierarchical_action.py` defines 10 high-level actions; `envs/hierarchical_env.py` wraps the base environment and exposes action dimension 10. | `results/publication_tables/table2_algorithm_progression.md`; hierarchical DDQN throughput/frame 0.9710. | High | Hierarchical DDQN is a single-run result in the current publication package. |
| QMIX provides cooperative multi-agent coordination over the hierarchical action interface. | `marl/qmix/networks.py::QMixer` mixes per-agent Q-values using global state with non-negative weights; `marl/qmix/qmix_trainer.py` collects joint episode data. | `results/publication_tables/table3_qmix_multiseed.md`; QMIX base mean throughput/frame 0.9604 +/- 0.0255. | High | The executor remains rule-based; decentralized execution should be described as high-level action selection. |
| QMIX improves jamming/fairness coordination relative to hierarchical DDQN. | QMIX uses global-state mixing and shared team reward; metrics include jamming failure and Jain fairness. | `results/publication_tables/table1_overall_scenario4_comparison.md`; QMIX jam 0.2056 +/- 0.0744 vs hierarchical DDQN jam 0.4403; QMIX fairness 0.5260 +/- 0.0572 vs hierarchical DDQN fairness 0.4754. | Medium-high | QMIX base should not be claimed to beat hierarchical DDQN in mean throughput; hierarchical DDQN single-run throughput is 0.9710 vs QMIX mean 0.9604. |
| BALANCE_UNDERSERVED_IOT improves fairness-aware behavior. | `envs/hierarchical_action.py::score_underserved` and action 7 bias target selection toward lower-delivered IoT devices. `envs/hierarchical_env.py` supports disabling action 7. | `results/publication_tables/table4_qmix_fairness_ablation.md`; no_balance_action has lower throughput, fairness, and worse jam than QMIX base. | Medium | It does not fully solve fairness. It is an executor/action mechanism, not separate reward shaping. |
| The final report setting uses the actual implemented/evaluated 2-UAV Scenario 4 configuration. | `configs/scenario_4_backscatter_types_calibrated.yaml` sets 2 UAVs, 15 IoT devices, 1 jammer, frame length 10, and max steps 200. | `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_tables/table3_qmix_multiseed.md`. | High | Older proposed parameter tables should be treated as planning artifacts only. |
| QMIX base is the final MaDRL setting. | `configs/qmix_tuned/qmix_sc4_base.yaml` uses hierarchical interface, QMIX base hyperparameters, and output prefix `qmix_sc4_base`. | `results/publication_tables/table3_qmix_multiseed.md`; `results/publication_tables/table4_qmix_fairness_ablation.md`; Phase 3.7/3.8 reports. | High | Final means "selected based on existing Phase 3 evidence", not globally optimal under all possible tuning. |
| Action masking is important for valid action selection. | Flat masks use `_validate_decoded_action`; hierarchical masks disable unavailable high-level actions; QMIX/DDQN mask invalid Q-values before selection. | Indirectly supported through successful training/evaluation artifacts. | Medium | There is no isolated ablation showing performance without action masking in the final publication package. |
| The reward supports throughput, anti-jamming, energy, delay, collision, and fairness objectives. | `envs/reward.py::compute_reward` directly implements weighted terms. | Metrics in all final eval CSVs and publication tables. | High | IoT energy is tracked in metrics but not directly penalized by the reward function. |

## Recommended Method Claims

- The proposed final method is QMIX-Hierarchical, not flat DDQN.
- The key enabling step is hierarchical action abstraction from 864 to 10 actions.
- QMIX adds cooperative value decomposition and improves jamming/fairness coordination relative to hierarchical DDQN.
- The final reported setting is the actual implemented/evaluated Scenario 4 setting.

## Claims to Avoid

- Do not claim QMIX base has the highest throughput among all learned methods.
- Do not claim fairness is solved.
- Do not claim the executor is learned.
- Do not claim full physical-layer ambient-backscatter modeling unless the abstraction is explicitly stated.
- Do not include MAPPO as an evaluated result.
