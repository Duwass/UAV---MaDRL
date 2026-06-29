# Missing Items and Clarification Checklist Before Final Writing

This checklist is for Phase 4.2 and later writing work. It does not require new training unless the research brain decides a missing item is important enough to justify it.

## Missing or Weak Figures

- [ ] Final paper captions are not written yet for Figures 1-6.
- [ ] Figure style may need journal-specific formatting:
  - font size;
  - grayscale readability;
  - line width;
  - column-width sizing.
- [ ] No dedicated mode-usage figure is included in the Phase 3.9 publication package. Existing tables include backscatter/active success, but not a standalone mode-usage count plot.
- [ ] No convergence curve figure is included in the final publication package. Training/eval logs exist, but the publication artifact set emphasizes aggregate results.
- [ ] No architecture diagram exists yet for the hierarchical executor plus QMIX pipeline.
- [ ] No system model diagram exists yet for UAVs, IoT devices, jammer, active/backscatter links, and RF harvesting.

## Unclear or Needing Formal Definition

- [ ] Exact mathematical definition of reward used by DDQN/QMIX.
- [ ] Exact fairness metric definition, including whether it is Jain-style and over which service counts.
- [ ] Exact energy efficiency definition.
- [ ] Exact jamming failure definition.
- [ ] SINR/channel model equations and jammer model equations.
- [ ] How primary busy state affects backscatter/harvest/active decisions.
- [ ] Executor fallback behavior should be summarized precisely but not over-detailed.

## Missing or Limited Baselines

- [ ] MAPPO is not implemented/evaluated; do not include it as a result.
- [ ] Hierarchical DDQN currently appears as a single-run publication result; no multi-seed hierarchical DDQN aggregate is in the artifact package.
- [ ] Flat DDQN is also single-run in the publication package.
- [ ] If a reviewer expects VDN, independent Q-learning, or MAPPO, the paper should explain why QMIX is the first MaDRL prototype and list others as future work.
- [ ] If results beyond Scenario 4 are discussed in detail, create additional publication tables for no-jammer and static weak jammer scenarios.

## Missing Explanation

- [ ] Explain why flat DDQN fails despite reward scaling and dueling improvements.
- [ ] Explain why hierarchical DDQN can exceed heuristic baselines: it encodes high-level strategy choices while still adapting by state.
- [ ] Explain why QMIX is selected even though hierarchical DDQN has slightly higher single-run throughput:
  - QMIX has multi-seed validation;
  - QMIX improves jamming failure;
  - QMIX improves fairness;
  - QMIX is the true cooperative MaDRL method.
- [ ] Explain why fair_w2 is not selected despite higher throughput and lower jam:
  - lower mean fairness than QMIX base;
  - high fairness variance;
  - final objective is trade-off, not throughput alone.
- [ ] Explain why disabling BALANCE_UNDERSERVED_IOT is a meaningful ablation.

## Missing Citations

- [ ] UAV-assisted IoT data collection and trajectory/resource optimization.
- [ ] Anti-jamming wireless communication and adaptive channel/mode selection.
- [ ] Ambient backscatter communication.
- [ ] RF-powered IoT or wireless-powered communication networks.
- [ ] Deep reinforcement learning for wireless networks.
- [ ] Multi-agent reinforcement learning for cooperative control.
- [ ] QMIX and value-decomposition methods.
- [ ] Hierarchical/action-abstraction reinforcement learning.
- [ ] Fairness metrics in wireless scheduling, especially Jain fairness if used.

## Possible Inconsistencies to Clarify

- [ ] QMIX base mean throughput 0.9604 is slightly below hierarchical DDQN single-run throughput 0.9710; the claim should focus on QMIX stability and jamming/fairness, not throughput dominance over hierarchical DDQN.
- [ ] QMIX base is multi-seed; hierarchical DDQN and flat DDQN are single-run in the current publication package. Label this clearly.
- [ ] Fairness ablation fair_w2 improves throughput and jam but reduces fairness. The final recommendation should say "best overall trade-off", not "best every metric".
- [ ] Backscatter-only has strong throughput but zero active success. Explain that it is specialized and lacks mode diversity.
- [ ] Greedy SINR has lower throughput than greedy nearest in Scenario 4 and high jamming failure; avoid assuming SINR-greedy is always strongest.

## Reproducibility Items

- [ ] Include artifact regeneration command: `python scripts\generate_publication_artifacts.py`.
- [ ] Include config paths for final methods:
  - `configs/qmix_tuned/qmix_sc4_base.yaml`;
  - `configs/ddqn_hierarchical/hier_sc4_basic.yaml`;
  - `configs/ddqn_tuned/tuned_backscatter_types.yaml` or the exact flat DDQN config used.
- [ ] Include seed list for QMIX base: 42, 43, 44.
- [ ] Include hardware/software note only if relevant to reproducibility.

## Items to Clarify Before Phase 4.2

- [ ] Decide whether Phase 4.2 should draft the full Results and Discussion section first or start with Method/System Model.
- [ ] Decide whether to generate a system model diagram and QMIX architecture diagram.
- [ ] Decide whether to add a small appendix table for no-jammer and static weak jammer DDQN results.
- [ ] Decide whether to include mode-usage counts in a publication table.
- [ ] Decide citation style and target venue format before writing LaTeX-ready prose.
