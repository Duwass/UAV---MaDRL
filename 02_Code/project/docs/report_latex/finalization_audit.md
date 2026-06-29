# Finalization Audit

This audit records the Phase 4.5 / Stage 5 report state. It covers only documentation/report files and does not imply any change to simulator, trainer, algorithm, config, checkpoint, or result artifacts.

## Current Chapter List

1. `chapters/abstract_vi.tex` - Vietnamese abstract.
2. `chapters/chapter1_introduction.tex` - Introduction.
3. `chapters/chapter2_background_related_work.tex` - Theoretical background and related work.
4. `chapters/chapter3_system_model_problem.tex` - System model and problem formulation.
5. `chapters/chapter4_proposed_method.tex` - Proposed method.
6. `chapters/chapter5_experiments_results_discussion.tex` - Experiments, results, discussion, conclusion, and future work.

## Current Figure List

| Label | Source path | Status |
| --- | --- | --- |
| `fig:system_architecture` | `docs/report_latex/figures/system_architecture.png` | Included and referenced. |
| `fig:hier_action_arch` | `docs/report_latex/figures/hierarchical_action_architecture.png` | Included and referenced. |
| `fig:qmix_arch` | `docs/report_latex/figures/qmix_architecture.png` | Included and referenced. |
| `fig:overall_throughput` | `results/publication_figures/fig1_overall_throughput_comparison.png` | Included and referenced. |
| `fig:drop_jam_fairness` | `results/publication_figures/fig2_drop_jam_fairness_comparison.png` | Included and referenced. |
| `fig:algorithm_progression` | `results/publication_figures/fig3_algorithm_progression.png` | Included and referenced. |
| `fig:qmix_multiseed` | `results/publication_figures/fig4_qmix_multiseed_mean_std.png` | Included and referenced. |
| `fig:fairness_tradeoff` | `results/publication_figures/fig5_fairness_ablation_tradeoff.png` | Included and referenced. |
| `fig:fairness_bars` | `results/publication_figures/fig6_fairness_ablation_bars.png` | Included and referenced. |

## Current Table List

| Label | Source | Status |
| --- | --- | --- |
| `tab:notation_compact` | Authored in Chapter III | Included and referenced. |
| `tab:high_level_actions` | Authored in Chapter IV | Included and referenced. |
| `tab:scenario_metrics` | `results/publication_tables/table5_simulation_scenarios_metrics.tex` | Included and referenced. |
| `tab:overall_scenario4` | `results/publication_tables/table1_overall_scenario4_comparison.tex` | Included and referenced. |
| `tab:algorithm_progression` | `results/publication_tables/table2_algorithm_progression.tex` | Included and referenced. |
| `tab:qmix_multiseed` | `results/publication_tables/table3_qmix_multiseed.tex` | Included and referenced. |
| `tab:fairness_ablation` | `results/publication_tables/table4_qmix_fairness_ablation.tex` | Included and referenced. |

## Current TODO List

- Official title page metadata: university, faculty, student name, student ID, supervisor.
- Verify exact metadata for TODO bibliography entries.
- Verify final venue/university citation style.
- Verify two implementation-formula comments in Chapter III:
  - UAV mobility/diagonal movement cost.
  - Report-level RL objective expression.
- Optionally replace TODO domain references with verified references before final submission.

## Remaining Citation Problems

- No undefined citation warning remains after the final build.
- Several bibliography entries are intentionally marked TODO because exact metadata was not verified locally.
- Placeholder references should be replaced before official submission.

## Remaining Formatting Problems

- Final build has no overfull/underfull warnings in `main.log`.
- One known non-blocking warning remains: Vietnamese hyphenation patterns are not preloaded in the local MiKTeX format.
- Title page is still a placeholder and needs official template information.

## Formula Verification Notes

- Implementation-confirmed formulas: path loss, received power, jammer interference, SINR, success probability, action dimensions, observation/state dimensions, reward components, throughput, drop, jam, fairness, energy efficiency.
- Needs final verification before submission: report-level mobility equation and report-level RL objective wording.

## Claims Softened

- QMIX is not claimed to beat Hierarchical DDQN in mean throughput.
- Fairness is described as improved/moderate, not solved.
- Backscatter is described as a system-level abstraction, not full physical ambient backscatter.
- Execution is described as decentralized high-level action selection plus rule-based executor, not fully decentralized primitive action execution.

## Missing Pieces Before Final Submission

- Official title page and formatting template.
- Final citation metadata.
- Human review of all Vietnamese terminology.
- Optional appendix for artifact provenance and reproducibility.
- Optional final proofreading of equations against implementation.
