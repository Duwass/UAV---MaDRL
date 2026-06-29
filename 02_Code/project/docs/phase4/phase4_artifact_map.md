# Phase 4 Artifact Map

This map records the Phase 3 artifacts currently available for report and paper writing. Paths are relative to the project root. No training outputs were regenerated for this document.

## Publication Table Bundles

| Artifact path(s) | Artifact type | Likely report section | Claim supported | Ready for paper/report use |
| --- | --- | --- | --- | --- |
| `results/publication_tables/table1_overall_scenario4_comparison.csv`<br>`results/publication_tables/table1_overall_scenario4_comparison.md`<br>`results/publication_tables/table1_overall_scenario4_comparison.tex` | Aggregate Scenario 4 comparison table | Results and Discussion | Scenario 4 comparison across random, heuristic baselines, flat DDQN, hierarchical DDQN, and QMIX base. Supports that flat DDQN underperforms, hierarchical DDQN is strong, and QMIX base is stable and competitive. | Yes. Use the LaTeX table for paper drafting and the Markdown table for internal review. |
| `results/publication_tables/table2_algorithm_progression.csv`<br>`results/publication_tables/table2_algorithm_progression.md`<br>`results/publication_tables/table2_algorithm_progression.tex` | Algorithm progression and ablation table | Proposed MaDRL Method; Results and Discussion | The main performance jump comes from replacing the 864-action flat interface with the 10-action hierarchical interface; QMIX improves coordination metrics. | Yes, but write carefully: QMIX improves jamming/fairness relative to hierarchical DDQN, while hierarchical DDQN has slightly higher single-run throughput than QMIX mean. |
| `results/publication_tables/table3_qmix_multiseed.csv`<br>`results/publication_tables/table3_qmix_multiseed.md`<br>`results/publication_tables/table3_qmix_multiseed.tex` | QMIX multi-seed validation table | Results and Discussion; Multi-seed Validation | QMIX base is stable across seeds 42, 43, and 44 with throughput/frame mean/std 0.9604/0.0255. | Yes. This is the key table for reliability of the final MaDRL result. |
| `results/publication_tables/table4_qmix_fairness_ablation.csv`<br>`results/publication_tables/table4_qmix_fairness_ablation.md`<br>`results/publication_tables/table4_qmix_fairness_ablation.tex` | Fairness and coordination ablation table | Results and Discussion; Fairness Ablation | QMIX base is the best overall trade-off; fair_w2 improves throughput and jamming but lowers fairness; fair_w3 is worse; disabling BALANCE_UNDERSERVED_IOT hurts results. | Yes. Use as evidence for keeping QMIX base and keeping action 7 enabled. |
| `results/publication_tables/table5_simulation_scenarios_metrics.csv`<br>`results/publication_tables/table5_simulation_scenarios_metrics.md`<br>`results/publication_tables/table5_simulation_scenarios_metrics.tex` | Scenario and metric definition table | Experimental Setup | Documents no-jammer, static weak jammer, and Scenario 4 benchmark purpose plus metric definitions. | Mostly ready. May need expansion if the final paper discusses medium/strong/static/mobile jammer scenarios. |

## Publication Figures

| Artifact path | Artifact type | Likely report section | Claim supported | Ready for paper/report use |
| --- | --- | --- | --- | --- |
| `results/publication_figures/fig1_overall_throughput_comparison.png` | 300 dpi bar chart | Results and Discussion | Overall Scenario 4 throughput comparison; shows flat DDQN bottleneck and hierarchical/QMIX improvement. | Yes. Needs final caption. |
| `results/publication_figures/fig2_drop_jam_fairness_comparison.png` | 300 dpi grouped bar chart | Results and Discussion | QMIX improves the jamming/fairness trade-off relative to hierarchical DDQN while retaining high throughput. | Yes. Needs final caption and interpretation text. |
| `results/publication_figures/fig3_algorithm_progression.png` | 300 dpi grouped bar chart | Proposed MaDRL Method; Results and Discussion | Research progression: flat DDQN fails, hierarchical abstraction improves, QMIX improves coordination metrics. | Yes. Best used as the core narrative figure. |
| `results/publication_figures/fig4_qmix_multiseed_mean_std.png` | 300 dpi error-bar chart | Results and Discussion; Multi-seed Validation | QMIX base has low throughput variability across seeds and stable aggregate performance. | Yes. Needs caption stating seeds 42, 43, 44. |
| `results/publication_figures/fig5_fairness_ablation_tradeoff.png` | 300 dpi scatter plot | Results and Discussion; Fairness Ablation | Fairness-weighted executor variants do not improve mean fairness enough to replace QMIX base. | Yes. Needs caption explaining axes and jam annotations. |
| `results/publication_figures/fig6_fairness_ablation_bars.png` | 300 dpi grouped bar chart | Results and Discussion; Fairness Ablation | Disabling BALANCE_UNDERSERVED_IOT degrades throughput, fairness, and jamming. | Yes. Needs final caption. |

## Publication Summaries and Indexes

| Artifact path | Artifact type | Likely report section | Claim supported | Ready for paper/report use |
| --- | --- | --- | --- | --- |
| `results/publication_reports/experimental_results_summary.md` | Research-brain summary | Writing foundation; Results and Discussion | Summarizes the supported claims, cautions, and recommended table/figure order. | Yes for internal drafting; not intended to be copied directly into the paper without editing. |
| `results/publication_artifacts_index.md` | Artifact index | Reproducibility appendix; internal tracking | Lists generated tables, figures, reports, source CSVs, missing files, and regeneration command. | Yes for internal use. Include only if the final report has an artifact appendix. |

## Key Source CSVs

| Source path | Artifact type | Likely report section | Claim supported | Ready for paper/report use |
| --- | --- | --- | --- | --- |
| `results/csv/phase3_1_calibrated_baseline_summary_mean.csv` | Baseline aggregate source data | Baseline Methods; Results and Discussion | Supports calibrated random, HTT-only, backscatter-only, greedy SINR, and greedy nearest baseline values. | Source data only; cite derived publication tables in the report. |
| `results/csv/phase3_1_calibrated_baseline_summary_std.csv` | Baseline variability source data | Baseline Methods; Results and Discussion | Supports baseline variance if needed. | Source data only. |
| `results/csv/tuned_backscatter_types_final_eval.csv` | Flat DDQN final evaluation | Flat DDQN Limitation | Supports flat DDQN throughput/frame 0.3242 and weak Scenario 4 performance. | Source data only; use Table 1/Table 2 for paper. |
| `results/csv/hier_sc4_basic_final_eval.csv` | Hierarchical DDQN final evaluation | Hierarchical Action Interface | Supports hierarchical DDQN throughput/frame 0.9710 and action-abstraction result. | Source data only; use Table 1/Table 2 for paper. |
| `results/csv/qmix_experiment_summary.csv` | QMIX base and updates4 per-seed summary | Multi-seed Validation | Supports per-seed QMIX base results and confirms updates_per_episode=4 was not the final setting. | Source data only; use Table 3 for paper. |
| `results/csv/qmix_multiseed_mean.csv`<br>`results/csv/qmix_multiseed_std.csv` | QMIX multi-seed aggregates | Multi-seed Validation | Supports QMIX base throughput/frame 0.9604 +/- 0.0255, jam 0.2056 +/- 0.0744, fairness 0.5260 +/- 0.0572, drop 0.4744 +/- 0.0054. | Source data only; use Table 3/Figure 4 for paper. |
| `results/csv/qmix_fairness_experiment_summary.csv` | Fairness ablation per-seed source data | Fairness Ablation | Supports fair_w2, fair_w3, and no_balance_action per-seed results. | Source data only; use Table 4/Figures 5-6 for paper. |
| `results/csv/qmix_fairness_ablation_mean.csv`<br>`results/csv/qmix_fairness_ablation_std.csv`<br>`results/csv/qmix_fairness_tradeoff_ranking.csv` | Fairness ablation aggregate source data | Fairness Ablation | Supports final conclusion that QMIX base remains the best overall trade-off and that disabling BALANCE_UNDERSERVED_IOT hurts results. | Source data only; use Table 4/Figures 5-6 for paper. |
| `results/csv/tuned_no_jammer_final_eval.csv`<br>`results/csv/tuned_static_weak_final_eval.csv` | Simpler-scenario DDQN source data | Experimental Setup; Validation sanity checks | Supports that tuned DDQN passed no-jammer and static weak jammer validation before Scenario 4. | Optional source data. Use only if the report includes the Phase 3 learning progression. |

## Readiness Summary

- Paper-ready tables: Table 1, Table 2, Table 3, Table 4, Table 5.
- Paper-ready figures: Figures 1-6, subject to final captions and possibly journal-specific style formatting.
- Internal drafting artifacts: `experimental_results_summary.md`, `publication_artifacts_index.md`.
- Primary caution: QMIX base is a three-seed aggregate, while hierarchical DDQN and flat DDQN are single-run final evaluations unless additional seed data is generated later.
