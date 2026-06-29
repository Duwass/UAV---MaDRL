# Experimental Setup Draft

This draft uses the actual implemented and evaluated Scenario 4 setting. Older proposed parameter tables are treated as planning references only.

## Final Evaluated Scenario

The main final-report benchmark is Scenario 4, the heterogeneous backscatter-types setting. The implemented/evaluated configuration is:

| Item | Final reported value | Source |
| --- | --- | --- |
| UAV count | 2 | `configs/scenario_4_backscatter_types_calibrated.yaml` |
| IoT device count | 15 | `configs/scenario_4_backscatter_types_calibrated.yaml` |
| Jammer count | 1 mobile/chase jammer | `configs/scenario_4_backscatter_types_calibrated.yaml` |
| Frame length | 10 | `configs/scenario_4_backscatter_types_calibrated.yaml` |
| Episode length | 200 | `configs/scenario_4_backscatter_types_calibrated.yaml` |
| Flat action dimension | 864 | `docs/phase4/proposed_method_draft.md` |
| Hierarchical high-level action dimension | 10 | `docs/phase4/proposed_method_draft.md` |
| QMIX local observation dimension | 97 | `docs/phase4/marl_formulation_draft.md` |
| QMIX global state dimension | 70 | `docs/phase4/marl_formulation_draft.md` |

The report should explicitly state that no retraining was performed for Phase 4 writing. Results are taken from existing Phase 3 artifacts.

## Compared Methods

The publication artifacts support comparisons among the following methods.

| Method | Category | Role in final report | Source artifact |
| --- | --- | --- | --- |
| Random | Stochastic baseline | Lower-bound reference. | `results/publication_tables/table1_overall_scenario4_comparison.md` |
| HTT-only | Heuristic baseline | Active-transmission-oriented baseline. | `results/publication_tables/table1_overall_scenario4_comparison.md` |
| Backscatter-only | Heuristic baseline | Specialized backscatter baseline. | `results/publication_tables/table1_overall_scenario4_comparison.md` |
| Greedy SINR | Heuristic baseline | Link-quality heuristic. | `results/publication_tables/table1_overall_scenario4_comparison.md` |
| Greedy nearest | Heuristic baseline | Strong spatial service heuristic. | `results/publication_tables/table1_overall_scenario4_comparison.md` |
| Flat DDQN | Flat DRL baseline | Tests direct learning over the 864-action primitive interface. | `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_tables/table2_algorithm_progression.md` |
| Flat DDQN + tuning | Flat DRL ablation | Shows reward scaling/dueling did not solve the action-interface bottleneck. | `results/publication_tables/table2_algorithm_progression.md` |
| Hierarchical DDQN | Hierarchical DRL baseline | Tests whether the 10-action high-level wrapper makes learning feasible. | `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_tables/table2_algorithm_progression.md` |
| QMIX base | Final MaDRL setting | Main cooperative multi-agent result using the hierarchical wrapper. | `results/publication_tables/table3_qmix_multiseed.md` |
| fair_w2 | Fairness executor ablation | Tests stronger executor fairness/underserved weights. | `results/publication_tables/table4_qmix_fairness_ablation.md` |
| fair_w3 | Fairness executor ablation | Tests excessive executor fairness weighting. | `results/publication_tables/table4_qmix_fairness_ablation.md` |
| no_balance_action | Action ablation | Tests disabling BALANCE_UNDERSERVED_IOT. | `results/publication_tables/table4_qmix_fairness_ablation.md` |

## Evaluation Metrics

| Metric | Interpretation | Direction |
| --- | --- | --- |
| Reward | Shared environment reward accumulated over an episode. | Higher is generally better, but values are shaped by penalties. |
| Throughput/frame | Delivered packets per environment frame. | Higher is better. |
| Drop rate | Fraction of packets dropped relative to arrivals. | Lower is better. |
| Jamming failure | Transmission failures attributed to jamming divided by attempts. | Lower is better. |
| Fairness | Jain-style service fairness over delivered packets across IoT devices. | Higher is better. |
| Energy efficiency | Delivered packets divided by total UAV plus IoT energy consumption. | Higher is better. |
| Backscatter success | Successful backscatter packets divided by attempted backscatter packets. | Higher is better. |
| Active success | Successful active packets divided by attempted active packets. | Higher is better. |
| Fallback rate | Hierarchical executor fallback frequency. | Lower is generally better, but should be interpreted with policy behavior. |

Metric definitions are summarized in `results/publication_tables/table5_simulation_scenarios_metrics.md` and formalized in `docs/phase4/problem_formulation_draft.md`.

## Single-Run and Multi-Seed Status

The final report should distinguish single-run and multi-seed evidence.

| Result group | Run status in current artifacts | Writing implication |
| --- | --- | --- |
| Flat DDQN | Single final evaluation in publication tables. | Use as evidence of the flat-interface bottleneck, but avoid statistical generalization. |
| Flat DDQN + tuning | Single final evaluation in Table 2. | Use only as supporting evidence that tuning did not solve the flat action interface. |
| Hierarchical DDQN | Single final evaluation in publication tables. | Strong result, but phrase robustness cautiously. |
| QMIX base | Three seeds: 42, 43, 44. | Main multi-seed MaDRL result; can support stability claims over these seeds. |
| QMIX fairness ablations | Three seeds per reported variant. | Supports ablation conclusions for tested executor settings. |
| Heuristic baselines | Calibrated aggregate baseline summaries. | Suitable for scenario-level reference comparisons. |

Single-run claims should be phrased as "in the evaluated run" or "in the current publication artifact set." Multi-seed claims should report both mean and standard deviation.

## Reporting Decision

The final experimental setup section should include `docs/phase4/final_experimental_setting_note.md` as an internal guardrail. The final report should not use older proposed parameter tables when describing the evaluated results.
