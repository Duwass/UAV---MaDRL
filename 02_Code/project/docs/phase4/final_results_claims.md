# Final Results Claims

These are safe claims for the final report, with evidence and caveats.

## Claim 1: The primitive flat action interface is a bottleneck in evaluated Scenario 4.

- **Final claim statement:** In the implemented/evaluated Scenario 4 setting, flat DDQN struggles with the 864-action primitive interface.
- **Evidence:** `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_tables/table2_algorithm_progression.md`.
- **Metric:** Flat DDQN throughput/frame 0.3242, jam 0.6093, fairness 0.1930.
- **Where to place:** Results and Discussion, flat DDQN limitation subsection.
- **Confidence:** High.
- **Caveat:** This is a claim about the tested flat DDQN prototype and evaluated Scenario 4, not all possible flat-action methods.

## Claim 2: The hierarchical 10-action abstraction makes learning feasible.

- **Final claim statement:** Reducing the direct learner action interface from 864 primitive actions to 10 high-level actions produces the main learning improvement in Scenario 4.
- **Evidence:** `results/publication_tables/table2_algorithm_progression.md`; `results/publication_figures/fig3_algorithm_progression.png`.
- **Metric:** Hierarchical DDQN throughput/frame 0.9710 versus flat DDQN 0.3242.
- **Where to place:** Results and Discussion, algorithm progression subsection.
- **Confidence:** High for the evaluated run.
- **Caveat:** Hierarchical DDQN is single-run in the current publication package.

## Claim 3: QMIX base is the final MaDRL setting because it is stable and improves coordination-sensitive metrics.

- **Final claim statement:** QMIX-Hierarchical is selected as the final MaDRL setting because it preserves high throughput across seeds and improves jamming/fairness behavior relative to hierarchical DDQN.
- **Evidence:** `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_tables/table3_qmix_multiseed.md`; `results/publication_figures/fig4_qmix_multiseed_mean_std.png`.
- **Metric:** QMIX throughput/frame 0.9604 +/- 0.0255, jam 0.2056 +/- 0.0744, fairness 0.5260 +/- 0.0572.
- **Where to place:** Results and Discussion, QMIX multi-seed subsection.
- **Confidence:** High for three-seed validation.
- **Caveat:** Do not claim QMIX mean throughput exceeds hierarchical DDQN throughput.

## Claim 4: QMIX base exceeds the strongest heuristic baselines in mean throughput.

- **Final claim statement:** QMIX base mean throughput exceeds the strongest calibrated heuristic baselines in Scenario 4.
- **Evidence:** `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_figures/fig1_overall_throughput_comparison.png`.
- **Metric:** QMIX base 0.9604 +/- 0.0255; greedy nearest 0.8977; backscatter-only 0.8522.
- **Where to place:** Overall comparison subsection.
- **Confidence:** High.
- **Caveat:** Greedy nearest and backscatter-only are strong specialized baselines and should be presented respectfully.

## Claim 5: BALANCE_UNDERSERVED_IOT contributes to fairness-aware coordination.

- **Final claim statement:** Disabling BALANCE_UNDERSERVED_IOT degrades throughput, fairness, and jamming behavior, supporting the value of the fairness-aware high-level action.
- **Evidence:** `results/publication_tables/table4_qmix_fairness_ablation.md`; `results/publication_figures/fig6_fairness_ablation_bars.png`.
- **Metric:** no_balance_action throughput 0.8439 +/- 0.0473, fairness 0.4767 +/- 0.0401, jam 0.3352 +/- 0.0598; QMIX base throughput 0.9604 +/- 0.0255, fairness 0.5260 +/- 0.0572, jam 0.2056 +/- 0.0744.
- **Where to place:** Fairness ablation subsection.
- **Confidence:** Moderate.
- **Caveat:** BALANCE_UNDERSERVED_IOT is an executor/action mechanism and does not fully solve fairness.

## Claim 6: Stronger fairness executor weights are not the best final trade-off.

- **Final claim statement:** The tested fairness-weighted executor variants do not replace QMIX base as the best overall trade-off.
- **Evidence:** `results/publication_tables/table4_qmix_fairness_ablation.md`; `results/publication_figures/fig5_fairness_ablation_tradeoff.png`.
- **Metric:** fair_w2 improves throughput to 0.9883 and jam to 0.1601 but lowers fairness to 0.5111; fair_w3 lowers throughput to 0.8877 and fairness to 0.4933.
- **Where to place:** Fairness ablation subsection.
- **Confidence:** High for tested variants.
- **Caveat:** Do not generalize to all possible fairness-aware reward or policy designs.

## Claim 7: Both active and backscatter modes remain meaningful in the final QMIX result.

- **Final claim statement:** QMIX base achieves meaningful success rates for both backscatter and active transmission in Scenario 4.
- **Evidence:** `results/publication_tables/table1_overall_scenario4_comparison.md`; `results/publication_tables/table3_qmix_multiseed.md`.
- **Metric:** QMIX backscatter success 0.7935 +/- 0.0747; active success 0.7833 +/- 0.1068.
- **Where to place:** Discussion and interpretation subsection.
- **Confidence:** Moderate.
- **Caveat:** These are success rates, not complete mode-usage balance statistics.

## Claim 8: The final report setting is the actual implemented/evaluated Scenario 4 configuration.

- **Final claim statement:** All final results are reported under the implemented/evaluated 2-UAV, 15-IoT, 1-mobile-jammer Scenario 4 configuration.
- **Evidence:** `docs/phase4/final_experimental_setting_note.md`; `configs/scenario_4_backscatter_types_calibrated.yaml`; `results/publication_tables/table5_simulation_scenarios_metrics.md`.
- **Metric:** 2 UAVs, 15 IoT devices, 1 mobile jammer, frame length 10, episode length 200.
- **Where to place:** Experimental Setup.
- **Confidence:** High.
- **Caveat:** Do not mix in older proposed parameter tables.
