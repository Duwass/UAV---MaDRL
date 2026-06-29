# Figure and Table Audit

## Figures

| Label | File/path | Chapter | Caption quality | Referenced in text | Action taken |
| --- | --- | --- | --- | --- | --- |
| `fig:system_architecture` | `docs/report_latex/figures/system_architecture.png` | III | Clear, includes Scenario 4 and abstraction caveat. | Yes | Created and inserted. |
| `fig:hier_action_arch` | `docs/report_latex/figures/hierarchical_action_architecture.png` | IV | Clear, explains 864-to-10 action abstraction. | Yes | Created and inserted. |
| `fig:qmix_arch` | `docs/report_latex/figures/qmix_architecture.png` | IV | Clear, distinguishes training mixer and high-level selection. | Yes | Created and inserted. |
| `fig:overall_throughput` | `results/publication_figures/fig1_overall_throughput_comparison.png` | V | Clear Vietnamese caption. | Yes | Caption polished. |
| `fig:drop_jam_fairness` | `results/publication_figures/fig2_drop_jam_fairness_comparison.png` | V | Clear Vietnamese caption. | Yes | Caption polished. |
| `fig:algorithm_progression` | `results/publication_figures/fig3_algorithm_progression.png` | V | Clear Vietnamese caption. | Yes | Caption polished. |
| `fig:qmix_multiseed` | `results/publication_figures/fig4_qmix_multiseed_mean_std.png` | V | Clear Vietnamese caption. | Yes | Caption polished. |
| `fig:fairness_tradeoff` | `results/publication_figures/fig5_fairness_ablation_tradeoff.png` | V | Clear Vietnamese caption. | Yes | Caption polished. |
| `fig:fairness_bars` | `results/publication_figures/fig6_fairness_ablation_bars.png` | V | Clear and shortened to avoid long identifier formatting. | Yes | Caption shortened. |

## Tables

| Label | File/path | Chapter | Caption quality | Referenced in text | Action taken |
| --- | --- | --- | --- | --- | --- |
| `tab:notation_compact` | Authored in Chapter III | III | Clear, compact notation table. | Yes | Added. |
| `tab:high_level_actions` | Authored in Chapter IV | IV | Clear, lists all 10 high-level actions. | Yes | Added and referenced. |
| `tab:scenario_metrics` | `results/publication_tables/table5_simulation_scenarios_metrics.tex` | V | Clear, summarizes setup and metrics. | Yes | Imported with resizebox. |
| `tab:overall_scenario4` | `results/publication_tables/table1_overall_scenario4_comparison.tex` | V | Clear, notes single-run vs multi-seed distinction. | Yes | Imported with resizebox. |
| `tab:algorithm_progression` | `results/publication_tables/table2_algorithm_progression.tex` | V | Clear, supports action-interface story. | Yes | Imported with resizebox. |
| `tab:qmix_multiseed` | `results/publication_tables/table3_qmix_multiseed.tex` | V | Clear, reports mean/std and per-seed QMIX. | Yes | Imported with resizebox. |
| `tab:fairness_ablation` | `results/publication_tables/table4_qmix_fairness_ablation.tex` | V | Clear, supports fairness ablation. | Yes | Imported with resizebox. |

## Overall Assessment

- All report figures and tables are referenced in nearby text.
- Captions are in Vietnamese and explain the intended interpretation.
- Publication artifact tables and figures are imported without modifying the original `results/` files.
- Wide publication tables are wrapped with `\resizebox{\textwidth}{!}{...}`.
