# GPT Brain Context Handoff - UAV-MADRL Research Project

File này là bản khôi phục trí nhớ cho một GPT/research brain mới sau khi đoạn chat cũ bị xóa. Mục tiêu là giúp GPT đọc lại một file duy nhất và nắm được toàn bộ bối cảnh kỹ thuật, kết quả, quyết định nghiên cứu, file quan trọng, giới hạn claim, và trạng thái báo cáo LaTeX hiện tại.

Ngày tạo handoff: 2026-06-21.

## 1. Project Identity

- **Tên dự án:** UAV-MADRL / Project SkyShield.
- **Đề tài nghiên cứu:** Multi-Agent Deep Reinforcement Learning for anti-jamming UAV-assisted IoT networks with RF-powered ambient backscatter communication.
- **Ngôn ngữ báo cáo:** Tiếng Việt học thuật.
- **Project path hiện tại đã tìm thấy trên máy:**
  - `D:\OneDrive - Tiểu Học Hòa Phú\TAI_LIEU_HOC_TAP\NCKH\Project_SkyShield\UAV - MADRL\project`
- **Path cũ trong các phase trước từng dùng:**
  - `D:\UAV - MADRL\project`
- **Lưu ý:** Trong phiên hiện tại, path cũ không còn tồn tại ở root `D:\`. Project thật nằm trong OneDrive path ở trên.

## 2. Non-Negotiable Rules

Các quy tắc này đã được giữ xuyên suốt các phase sau:

- Không sửa environment/trainer/algorithm/checkpoint/config/result source files nếu task chỉ là viết báo cáo.
- Không rerun training khi đã bước sang Phase 4 writing/finalization.
- Không thay đổi simulation parameters.
- Không tạo kết quả thực nghiệm mới.
- Không sửa Phase 3 result artifacts trong `results/`.
- Mọi claim phải conservative và traceable về code, configs, CSV, publication artifacts hoặc docs Phase 4.
- Nếu citation chưa chắc, giữ `TODO` thay vì bịa metadata.
- Nếu công thức là report-level abstraction chứ chưa xác minh exact implementation, giữ comment:
  - `% TODO: verify exact implementation form`
- Backscatter phải được viết là **backscatter-inspired / RF-powered system-level abstraction**, không phải full physical ambient-backscatter physical-layer model.
- Kết quả là simulation-based.

## 3. Final Evaluated Experimental Setting

Source of truth cuối cùng cho báo cáo là Scenario 4 đã triển khai/đánh giá, không phải các bảng tham số đề xuất cũ.

- **UAV count:** 2
- **IoT count:** 15
- **Jammer count:** 1 mobile jammer
- **Frame length:** 10
- **Episode length:** 200
- **Flat action dimension:** 864
- **Hierarchical high-level action dimension:** 10
- **QMIX local observation dimension:** 97
- **QMIX global state dimension:** 70

Các file liên quan:

- `configs/scenario_4_backscatter_types_calibrated.yaml`
- `configs/qmix_tuned/qmix_sc4_base.yaml`
- `docs/phase4/final_experimental_setting_note.md`
- `docs/report_latex/final_report_status.md`

Không dùng các bảng tham số proposal cũ nếu chúng khác các giá trị trên.

## 4. Research Storyline

Luồng nghiên cứu chính:

1. UAV-assisted IoT hữu ích cho vùng thiếu hạ tầng, thiết bị phân tán, IoT năng lượng thấp.
2. Mobile jammer làm giảm SINR, tăng transmission failure, drop, queue pressure và làm xấu fairness.
3. Thiết bị IoT có thể active/backscatter/harvest/idle theo trạng thái năng lượng và primary busy.
4. Flat action interface quá lớn trong Scenario 4: movement x IoT target x communication mode = 864 actions.
5. Flat DDQN không học tốt trên action space này.
6. Hierarchical 10-action abstraction làm bài toán khả học hơn.
7. QMIX-Hierarchical thêm cooperative multi-agent value decomposition cho nhiều UAV.
8. QMIX base được chọn làm final MaDRL setting vì ổn định đa seed và cải thiện jamming/fairness trade-off, không phải vì vượt Hierarchical DDQN về mean throughput.

Safe one-sentence story:

> Flat action control exposes the action-interface bottleneck; hierarchical abstraction makes Scenario 4 learnable; QMIX-Hierarchical preserves high throughput while improving cooperative jamming/fairness behavior across seeds.

## 5. Phase Timeline and Decisions

### Phase 3.1 - Calibrated Baselines

Các baseline/calibrated summaries đã có trong:

- `results/csv/phase3_1_calibrated_baseline_summary_mean.csv`
- `results/csv/phase3_1_calibrated_baseline_summary_std.csv`

Scenario 4 baseline throughput/frame quan trọng:

- Random: 0.1075
- HTT-only: 0.3278
- Greedy SINR: 0.4783
- Backscatter-only: 0.8522
- Greedy nearest: 0.8977

### Phase 3.2 / 3.3 - Flat DDQN Tuning

Flat DDQN ban đầu học được tốt hơn random ở no-jammer nhưng yếu ở Scenario 4.

Confirmed earlier no-jammer/static weak:

- Tuned no-jammer:
  - throughput/frame = 0.7125
  - drop = 0.5336
  - jam = 0.0000
  - fairness = 0.6092
  - decision: PASS
- Tuned static weak:
  - throughput/frame = 0.7312
  - drop = 0.6652
  - jam = 0.1408
  - fairness = 0.3492
  - decision: PASS

Flat Scenario 4 result:

- `tuned_backscatter_types`:
  - throughput/frame = 0.3242
  - drop = 0.5846
  - jam = 0.6093
  - fairness = 0.1930
  - energy efficiency = 0.9417
  - backscatter success = 0.3899
  - active success = 0.5231

Conclusion:

- Flat DDQN technically runs but is research-failed for Scenario 4.
- Reward scaling/dueling can stabilize loss but does not solve action-interface bottleneck.
- Stop tuning flat DDQN as main direction.

### Phase 3.4 - Scenario 4 DDQN Stabilization

Tried variants like:

- `sc4_dueling_scaled_low_lr`
- `sc4_dueling_scaled_slow_epsilon`

Results were poor compared with original flat tuned DDQN:

- `sc4_dueling_scaled_low_lr` throughput/frame = 0.1203
- `sc4_dueling_scaled_slow_epsilon` throughput/frame = 0.1197

Conclusion:

- Main bottleneck is flat action interface, not just DDQN hyperparameters.
- Move to hierarchical/factorized action interface.

### Phase 3.5 - Hierarchical / Factorized Action Interface

Created/used:

- `envs/hierarchical_action.py`
- `envs/hierarchical_env.py`
- `configs/ddqn_hierarchical/hier_sc4_basic.yaml`
- `scripts/compare_hierarchical_scenario4.py`
- report: `results/phase3_5_hierarchical_action_report.md`

High-level action space reduced Scenario 4:

- Flat action dimension: 864
- Hierarchical action dimension: 10

10 high-level actions:

0. `IDLE_SAFE`
1. `SERVE_NEAREST_QUEUE`
2. `SERVE_BEST_SINR`
3. `PRIORITIZE_BACKSCATTER_TYPE23`
4. `PRIORITIZE_ACTIVE_TYPE1`
5. `HARVEST_LOW_ENERGY`
6. `AVOID_JAMMER`
7. `BALANCE_UNDERSERVED_IOT`
8. `HYBRID_BALANCED`
9. `HIGH_QUEUE_PRIORITY`

Best Hierarchical DDQN Scenario 4 result:

- throughput/frame = 0.9710
- reward = -1552.4697
- drop = 0.4761
- jam = 0.4403
- fairness = 0.4754
- energy efficiency = 4.9295
- backscatter success = 0.5258
- active success = 0.6817
- fallback rate = 0.0095

Conclusion:

- Hierarchical DDQN strongly passes Scenario 4.
- Action abstraction is essential.
- Keep hierarchical DDQN as a strong sanity baseline.

### Phase 3.6 - QMIX Prototype with Hierarchical 10-Action Wrapper

Created:

- `marl/qmix/__init__.py`
- `marl/qmix/replay_buffer.py`
- `marl/qmix/networks.py`
- `marl/qmix/qmix_agent.py`
- `marl/qmix/qmix_trainer.py`
- `marl/qmix/utils.py`
- `scripts/train_qmix.py`
- `scripts/evaluate_qmix.py`
- `scripts/compare_qmix_results.py`
- `scripts/plot_qmix_training.py`
- `configs/qmix/qmix_hier_sc4_backscatter_types.yaml`
- `configs/qmix/qmix_hier_sc1_multi_uav.yaml`
- `configs/qmix/qmix_hier_sc2_static_weak.yaml`
- tests for QMIX networks/replay/action masking/trainer smoke.

QMIX architecture:

- One agent per UAV.
- Shared `AgentQNetwork`.
- Local observation per UAV.
- Global state to `QMixer`.
- Shared global reward.
- 10 high-level hierarchical actions.
- Action masking.
- Double Q option enabled.
- Target update.
- Centralized training, decentralized high-level action selection.
- Important caveat: primitive action execution is through rule-based executor using environment-level information, so do not claim fully decentralized primitive execution.

Phase 3.6 QMIX Scenario 4 seed 42:

- reward = -1475.8774
- throughput/frame = 0.9945
- drop = 0.4670
- jam = 0.2988
- fairness = 0.5981
- energy efficiency = 3.7427
- backscatter success = 0.6951
- active success = 0.6411
- mode harvest = 36.7667
- mode backscatter = 203.4333
- mode active = 28.4333
- fallback rate = 0.0103

Tests after Phase 3.6:

- 38 passed.

Conclusion:

- QMIX prototype successful.
- Run multi-seed before claiming main MaDRL result.

### Phase 3.7 - QMIX Multi-seed + Light Tuning

Created:

- `configs/qmix_tuned/qmix_sc4_base.yaml`
- `configs/qmix_tuned/qmix_sc4_updates4.yaml`
- `configs/qmix_tuned/qmix_sc4_slow_epsilon.yaml`
- `configs/qmix_tuned/qmix_sc4_target500.yaml`
- `scripts/run_qmix_experiments.py`
- `scripts/summarize_qmix_multiseed.py`
- `scripts/plot_qmix_multiseed.py`

Main result: QMIX base across seeds 42, 43, 44:

- throughput/frame mean/std = 0.9604 / 0.0255
- jam mean/std = 0.2056 / 0.0744
- fairness mean/std = 0.5260 / 0.0572
- drop mean/std = 0.4744 / 0.0054

Per-seed QMIX base:

- Seed 42:
  - throughput/frame = 0.9945
  - reward = -1475.8774
  - drop = 0.4670
  - jam = 0.2988
  - fairness = 0.5981
  - energy efficiency = 3.7427
  - backscatter success = 0.6951
  - active success = 0.6411
  - fallback rate = 0.0103
- Seed 43:
  - throughput/frame = 0.9332
  - reward = -1481.4517
  - drop = 0.4796
  - jam = 0.2012
  - fairness = 0.5217
  - energy efficiency = 5.4559
  - backscatter success = 0.8095
  - active success = 0.8101
  - fallback rate = 0.0665
- Seed 44:
  - throughput/frame = 0.9537
  - reward = -1455.1210
  - drop = 0.4766
  - jam = 0.1167
  - fairness = 0.4581
  - energy efficiency = 7.7980
  - backscatter success = 0.8760
  - active success = 0.8987
  - fallback rate = 0.0018

Conclusion:

- QMIX base accepted as main MaDRL Scenario 4 result.
- `updates_per_episode=4` not recommended because it reduced throughput/fairness and increased jamming.

### Phase 3.8 - QMIX Fairness / Coordination Ablation

Created:

- `configs/qmix_fairness/qmix_sc4_fair_w1.yaml`
- `configs/qmix_fairness/qmix_sc4_fair_w2.yaml`
- `configs/qmix_fairness/qmix_sc4_fair_w3.yaml`
- `configs/qmix_fairness/qmix_sc4_no_balance_action.yaml`
- `scripts/summarize_qmix_fairness_ablation.py`
- `scripts/plot_qmix_fairness_ablation.py`
- tests for executor config, disabled actions, fairness summary.

Fairness ablation:

- QMIX base:
  - throughput = 0.9604 +/- 0.0255
  - drop = 0.4744 +/- 0.0054
  - jam = 0.2056 +/- 0.0744
  - fairness = 0.5260 +/- 0.0572
- `fair_w2`:
  - throughput = 0.9883 +/- 0.0350
  - drop = 0.4771 +/- 0.0180
  - jam = 0.1601 +/- 0.0501
  - fairness = 0.5111 +/- 0.1478
- `fair_w3`:
  - throughput = 0.8877 +/- 0.0094
  - drop = 0.4938 +/- 0.0113
  - jam = 0.3123 +/- 0.0711
  - fairness = 0.4933 +/- 0.1880
- `no_balance_action`:
  - throughput = 0.8439 +/- 0.0473
  - drop = 0.4926 +/- 0.0065
  - jam = 0.3352 +/- 0.0598
  - fairness = 0.4767 +/- 0.0401

Conclusion:

- QMIX base remains best overall trade-off.
- `fair_w2` improves throughput/jam but reduces fairness and increases fairness variance.
- `fair_w3` worse overall.
- Disabling `BALANCE_UNDERSERVED_IOT` hurts throughput, fairness, and jamming behavior.
- Keep QMIX base with `BALANCE_UNDERSERVED_IOT` enabled.

### Phase 3.9 - Publication Artifacts

Created aggregate publication artifacts:

- `results/publication_tables/`
- `results/publication_figures/`
- `results/publication_reports/experimental_results_summary.md`
- `results/publication_artifacts_index.md`
- `scripts/generate_publication_artifacts.py`

Publication tables:

- `table1_overall_scenario4_comparison.{csv,md,tex}`
- `table2_algorithm_progression.{csv,md,tex}`
- `table3_qmix_multiseed.{csv,md,tex}`
- `table4_qmix_fairness_ablation.{csv,md,tex}`
- `table5_simulation_scenarios_metrics.{csv,md,tex}`

Publication figures:

- `fig1_overall_throughput_comparison.png`
- `fig2_drop_jam_fairness_comparison.png`
- `fig3_algorithm_progression.png`
- `fig4_qmix_multiseed_mean_std.png`
- `fig5_fairness_ablation_tradeoff.png`
- `fig6_fairness_ablation_bars.png`

Important:

- These are existing Phase 3 result artifacts.
- Do not modify them during writing.

## 6. Phase 4 Documentation Work

### Phase 4.1 - Writing Foundation

Created:

- `docs/phase4/phase4_artifact_map.md`
- `docs/phase4/report_outline.md`
- `docs/phase4/research_storyline.md`
- `docs/phase4/results_to_claims_mapping.md`
- `docs/phase4/missing_items_checklist.md`

### Phase 4.2 - System Model / Problem Formulation Drafts

Created:

- `docs/phase4/system_model_draft.md`
- `docs/phase4/problem_formulation_draft.md`
- `docs/phase4/marl_formulation_draft.md`
- `docs/phase4/notation_table.md`
- `docs/phase4/implementation_to_equation_mapping.md`
- `docs/phase4/phase4_2_open_questions.md`

Key definitions:

- Local observation dimension:
  - \(d_o = 7 + 6N = 97\)
- Global state dimension:
  - \(d_s = 3U + 4N + 2J + 2 = 70\)
- Flat action dimension:
  - \(9(N+1)6 = 864\)
- Hierarchical action dimension:
  - 10

### Phase 4.3 - Proposed Method Draft

Created:

- `docs/phase4/final_experimental_setting_note.md`
- `docs/phase4/proposed_method_draft.md`
- `docs/phase4/algorithm_descriptions.md`
- `docs/phase4/method_architecture_notes.md`
- `docs/phase4/pseudocode_draft.md`
- `docs/phase4/method_claims_and_limitations.md`
- `docs/phase4/phase4_3_open_questions.md`

### Phase 4.4 - Results and Discussion Draft

Created:

- `docs/phase4/experimental_setup_draft.md`
- `docs/phase4/results_discussion_draft.md`
- `docs/phase4/table_figure_narrative.md`
- `docs/phase4/claim_strength_audit.md`
- `docs/phase4/final_results_claims.md`
- `docs/phase4/phase4_4_open_questions.md`

### Initial LaTeX Draft

Created:

- `docs/report_latex/main.tex`
- `docs/report_latex/chapters/chapter1_introduction.tex`
- `docs/report_latex/chapters/chapter2_background_related_work.tex`
- `docs/report_latex/references.bib`
- `docs/report_latex/README.md`
- `docs/report_latex/build_notes.md`

Then expanded to Chapters I-V:

- `docs/report_latex/chapters/chapter3_system_model_problem.tex`
- `docs/report_latex/chapters/chapter4_proposed_method.tex`
- `docs/report_latex/chapters/chapter5_experiments_results_discussion.tex`
- `docs/report_latex/preview_report_notes.md`
- `docs/report_latex/build/report_preview_current.pdf`

### Phase 4.5 / Stage 5 - Final Report Completion

Polished final report draft:

- Added Vietnamese abstract:
  - `docs/report_latex/chapters/abstract_vi.tex`
- Added architecture figures:
  - `docs/report_latex/figures/system_architecture.png`
  - `docs/report_latex/figures/hierarchical_action_architecture.png`
  - `docs/report_latex/figures/qmix_architecture.png`
- Created audits:
  - `docs/report_latex/finalization_audit.md`
  - `docs/report_latex/figure_table_audit.md`
  - `docs/report_latex/final_report_status.md`
- Final PDF:
  - `docs/report_latex/build/uav_madrl_final_draft.pdf`

Final LaTeX status:

- PDF compiled successfully with XeLaTeX + BibTeX.
- Length: 27 pages.
- No undefined citations/references after final build.
- No overfull/underfull hbox warnings after final build.
- Remaining warning:
  - Vietnamese hyphenation patterns not preloaded in local MiKTeX.

## 7. Final Report Current Structure

Current LaTeX report includes:

1. `chapters/abstract_vi.tex` - Tóm tắt tiếng Việt.
2. `chapters/chapter1_introduction.tex` - Giới thiệu.
3. `chapters/chapter2_background_related_work.tex` - Cơ sở lý thuyết và nghiên cứu liên quan.
4. `chapters/chapter3_system_model_problem.tex` - Mô hình hệ thống và phát biểu bài toán.
5. `chapters/chapter4_proposed_method.tex` - Phương pháp đề xuất.
6. `chapters/chapter5_experiments_results_discussion.tex` - Thực nghiệm, kết quả, thảo luận, kết luận, hướng phát triển.
7. Bibliography from `references.bib`.

Current final draft PDF:

- `docs/report_latex/build/uav_madrl_final_draft.pdf`

## 8. Figures and Tables in Final Report

Report-side figures created:

- `docs/report_latex/figures/system_architecture.png`
- `docs/report_latex/figures/hierarchical_action_architecture.png`
- `docs/report_latex/figures/qmix_architecture.png`

Publication figures included:

- `results/publication_figures/fig1_overall_throughput_comparison.png`
- `results/publication_figures/fig2_drop_jam_fairness_comparison.png`
- `results/publication_figures/fig3_algorithm_progression.png`
- `results/publication_figures/fig4_qmix_multiseed_mean_std.png`
- `results/publication_figures/fig5_fairness_ablation_tradeoff.png`
- `results/publication_figures/fig6_fairness_ablation_bars.png`

Tables included:

- Compact notation table authored in Chapter III.
- High-level action table authored in Chapter IV.
- Imported publication tables:
  - `results/publication_tables/table1_overall_scenario4_comparison.tex`
  - `results/publication_tables/table2_algorithm_progression.tex`
  - `results/publication_tables/table3_qmix_multiseed.tex`
  - `results/publication_tables/table4_qmix_fairness_ablation.tex`
  - `results/publication_tables/table5_simulation_scenarios_metrics.tex`

## 9. Key Numerical Results to Preserve

Scenario 4 overall comparison:

- Random throughput/frame = 0.1075
- HTT-only throughput/frame = 0.3278
- Backscatter-only throughput/frame = 0.8522
- Greedy SINR throughput/frame = 0.4783
- Greedy nearest throughput/frame = 0.8977
- Flat DDQN throughput/frame = 0.3242
- Hierarchical DDQN throughput/frame = 0.9710
- QMIX base throughput/frame = 0.9604 +/- 0.0255

QMIX base mean/std:

- throughput/frame = 0.9604 +/- 0.0255
- reward = -1470.8167 +/- 11.3294
- drop = 0.4744 +/- 0.0054
- jam = 0.2056 +/- 0.0744
- fairness = 0.5260 +/- 0.0572
- energy efficiency = 5.6655 +/- 1.6622
- backscatter success = 0.7935 +/- 0.0747
- active success = 0.7833 +/- 0.1068
- fallback rate = 0.0262 +/- 0.0287

Fairness ablation:

- `fair_w2`:
  - throughput = 0.9883 +/- 0.0350
  - fairness = 0.5111 +/- 0.1478
  - jam = 0.1601 +/- 0.0501
- `fair_w3`:
  - throughput = 0.8877 +/- 0.0094
  - fairness = 0.4933 +/- 0.1880
  - jam = 0.3123 +/- 0.0711
- `no_balance_action`:
  - throughput = 0.8439 +/- 0.0473
  - fairness = 0.4767 +/- 0.0401
  - jam = 0.3352 +/- 0.0598

## 10. Safe Claims for Writing

Use these:

- Flat DDQN struggles with the large 864-action primitive interface in evaluated Scenario 4.
- Hierarchical action abstraction reduces direct learner action dimension from 864 to 10 and makes learning feasible.
- Hierarchical DDQN strongly validates the action abstraction.
- QMIX-Hierarchical is final MaDRL method because it supports cooperative multi-UAV coordination with local observations, global-state mixing, shared reward, masked high-level action selection and rule-based executor.
- QMIX base is stable across seeds 42, 43, 44 in current validation.
- QMIX improves jamming/fairness trade-off relative to Hierarchical DDQN single-run.
- `BALANCE_UNDERSERVED_IOT` contributes to fairness-aware coordination because disabling it worsens throughput, fairness, and jamming.
- Fairness weighting variants do not replace QMIX base as final setting.
- Backscatter is modeled as a system-level abstraction.
- Results are simulation-based.

Avoid these:

- Do not claim QMIX mean throughput beats Hierarchical DDQN.
- Do not claim fairness is solved.
- Do not claim physical ambient backscatter is fully validated.
- Do not claim executor is learned.
- Do not claim fully decentralized primitive-action execution.
- Do not claim universal superiority beyond tested Scenario 4.
- Do not claim hardware validation.

## 11. Implementation Details to Remember

Core env/config:

- `envs/uav_backscatter_env.py`
- `envs/entities.py`
- `envs/channel_model.py`
- `envs/mobility_model.py`
- `envs/reward.py`
- `configs/scenario_4_backscatter_types_calibrated.yaml`

Hierarchical wrapper:

- `envs/hierarchical_action.py`
- `envs/hierarchical_env.py`

DDQN:

- `marl/ddqn/ddqn_agent.py`
- `marl/ddqn/ddqn_trainer.py`
- `marl/ddqn/networks.py`
- `marl/ddqn/reward_processing.py`

QMIX:

- `marl/qmix/replay_buffer.py`
- `marl/qmix/networks.py`
- `marl/qmix/qmix_agent.py`
- `marl/qmix/qmix_trainer.py`
- `marl/qmix/utils.py`

Important implementation facts:

- `HierarchicalActionExecutor` is rule-based.
- `HierarchicalUAVBackscatterEnv` wraps base env, does not replace original env.
- Hierarchical mask can disable actions such as `BALANCE_UNDERSERVED_IOT`.
- QMIX uses `AgentQNetwork` and `QMixer`.
- QMIX mixer uses global state.
- QMIX action selection uses local observations + action mask.
- Shared global reward; no separate local UAV reward.
- Action masking exists for flat and hierarchical actions.
- Executor fallback rate is tracked.

## 12. Important Equations and Metrics

Useful equations in report:

- Path loss:
  - \(L(d)=\max(d,1)^\alpha\)
- Received power:
  - \(P_{\mathrm{rx}}(P,d)=P/L(d)\)
- SINR:
  - signal power over noise plus jammer interference.
- Success probability:
  - piecewise SINR-threshold probability from implementation.
- Flat action:
  - \(a^u_t=(m^u_t,i^u_t,c^u_t)\)
- Flat action dimension:
  - \(9(N+1)6=864\)
- Hierarchical action:
  - \(z^u_t\in\{0,\ldots,9\}\)
- Executor mapping:
  - \(\phi_{\mathrm{exec}}:(\mathbf{z}_t,s_t)\mapsto\mathbf{a}_t\)
- QMIX mixer:
  - \(Q_{\mathrm{tot}}=f_{\mathrm{mix}}(Q_1,\ldots,Q_U,s_t)\)
- Monotonicity:
  - \(\partial Q_{\mathrm{tot}}/\partial Q_u\ge 0\)
- Reward:
  - throughput reward + avoid bonus - drop - delay/queue - UAV energy - jam - collision - unfairness.
- Throughput/frame:
  - delivered packets per episode step.
- Jain fairness:
  - \((\sum_iD_i)^2/(N\sum_iD_i^2+\epsilon)\)

Formula TODOs still present:

- UAV mobility/report equation needs exact implementation verification.
- RL objective equation is conceptual/report-level and marked for verification.

## 13. Current Report TODOs

From `docs/report_latex/final_report_status.md`:

- Replace title page placeholders:
  - university
  - faculty
  - student name
  - student ID
  - supervisor
- Verify and complete bibliography metadata for TODO references.
- Confirm final citation style and university formatting requirements.
- Human proofread Vietnamese terminology and chapter flow.
- Verify two conceptual/report-level equations marked by `% TODO: verify exact implementation form`.

Known report limitations:

- Some bibliography entries remain placeholders.
- Title page is not official.
- Report uses technical report layout, not university-specific final template.
- Vietnamese hyphenation warning remains in local TeX build environment.

Known experiment limitations:

- Final evaluated setting is 2 UAVs / 15 IoT / 1 jammer.
- Results are simulation-based.
- Backscatter is system-level abstraction, not full physical-layer validation.
- Executor is rule-based.
- Some baselines are single-run; QMIX has three-seed validation.
- Fairness remains moderate.
- No hardware validation.
- No retraining in Phase 4.5.

## 14. Bibliography Status

File:

- `docs/report_latex/references.bib`

Current confirmed core keys:

- `mnih2015dqn`
- `vanhasselt2016ddqn`
- `rashid2018qmix`
- `sutton2018reinforcement`

TODO / metadata-needs-verification keys include:

- `huynh2018rfpowered`
- `gong2020drlbackscattermec`
- `tran2018ddqntimescheduling`
- `uav_iot_survey_todo`
- `anti_jamming_todo`
- `hierarchical_rl_todo`
- `jain_fairness_todo`

Do not invent missing volume/issue/pages/DOI. Verify from sources before final submission.

## 15. Commands / Build Status

LaTeX final build sequence used:

```powershell
cd docs\report_latex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf build\uav_madrl_final_draft.pdf -Force
```

Final build result:

- `docs/report_latex/build/uav_madrl_final_draft.pdf`
- Compiled successfully.
- 27 pages.
- No undefined citations/references.
- No overfull/underfull hbox warnings.
- Non-blocking warning: Vietnamese hyphenation patterns not preloaded.

## 16. Recommended Next Steps for GPT Brain

If continuing writing/finalization:

1. Open `docs/report_latex/build/uav_madrl_final_draft.pdf`.
2. Review visual formatting, captions and Vietnamese flow.
3. Fill official title page metadata in `docs/report_latex/main.tex`.
4. Verify bibliography metadata in `docs/report_latex/references.bib`.
5. Confirm university template requirements.
6. Replace placeholder citation entries with verified sources.
7. Verify the two formulas marked `% TODO: verify exact implementation form`.
8. Rebuild PDF.
9. If needed, create a final appendix for reproducibility/artifact provenance.

If continuing experiments:

- Do not run new training unless research brain explicitly asks.
- If new experiments are requested, preserve old results and write new outputs with new prefixes.
- Do not remove Phase 3 artifacts.
- Use QMIX base as main MaDRL setting unless research brain changes decision.

## 17. One-Paragraph Memory Summary

This project studies anti-jamming UAV-assisted IoT networks with RF-powered/backscatter-inspired communication. The evaluated final Scenario 4 uses 2 UAVs, 15 IoT devices, one mobile jammer, frame length 10 and episode length 200. Flat DDQN struggles with the 864-action primitive interface. A hierarchical rule-based executor reduces the learner action space to 10 high-level actions and makes Scenario 4 learnable; Hierarchical DDQN reaches throughput/frame 0.9710. QMIX-Hierarchical reuses the 10-action wrapper and learns cooperative value decomposition across UAVs. QMIX base across seeds 42/43/44 achieves throughput/frame 0.9604 +/- 0.0255, jam 0.2056 +/- 0.0744, fairness 0.5260 +/- 0.0572 and drop 0.4744 +/- 0.0054. QMIX is selected as the final MaDRL method due to stability and jamming/fairness coordination, not because it beats Hierarchical DDQN in mean throughput. The final Vietnamese LaTeX draft is compiled at `docs/report_latex/build/uav_madrl_final_draft.pdf`, with remaining TODOs mainly title metadata, citation verification and final human proofreading.

