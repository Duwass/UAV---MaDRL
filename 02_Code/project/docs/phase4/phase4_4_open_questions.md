# Phase 4.4 Open Questions

## Missing Metrics or Tables

- [ ] No dedicated final mode-usage count table is included in the publication table set. Existing tables include active/backscatter success rates, not full mode usage counts.
- [ ] No final convergence/training-curve table is included in the publication package. Training logs exist, but Phase 3.9 focuses on aggregate results.
- [ ] No multi-seed hierarchical DDQN aggregate is available in the publication package.
- [ ] No multi-seed flat DDQN aggregate is available in the publication package.

## Unclear Tables or Figures

- [ ] Figure captions still need to be written in final paper style.
- [ ] Figure 5 should be captioned carefully because fair_w2 has higher throughput and lower jam but lower fairness.
- [ ] Table 2 includes "Flat DDQN + tuning" with low throughput 0.1203. It should be presented as a failed stabilization/tuning result, not as the main flat DDQN baseline.

## Weak Claims

- [ ] "QMIX improves fairness" is supported relative to hierarchical DDQN, but fairness remains moderate.
- [ ] "BALANCE_UNDERSERVED_IOT improves fairness-aware behavior" is supported by no_balance_action ablation, but it does not isolate every internal executor factor.
- [ ] "QMIX uses both active and backscatter modes meaningfully" is supported by success rates, but a full mode-usage table would make the claim stronger.

## Method and Results Mismatches to Avoid

- [ ] Do not describe the executor as learned. It is rule-based.
- [ ] Do not describe execution as fully decentralized at the primitive action level. It is decentralized high-level action selection plus rule-based translation.
- [ ] Do not describe the backscatter model as a full physical ambient-backscatter implementation without qualification.
- [ ] Do not claim QMIX beats hierarchical DDQN in mean throughput.

## Missing Citations

- [ ] UAV-assisted IoT and UAV data collection.
- [ ] Anti-jamming and mobile jammer modeling.
- [ ] Ambient backscatter and RF-powered IoT.
- [ ] DDQN.
- [ ] QMIX/value decomposition.
- [ ] Hierarchical/action-abstraction RL.
- [ ] Jain fairness in wireless scheduling.

## Final Diagrams Needed

- [ ] System architecture diagram.
- [ ] Hierarchical action wrapper diagram.
- [ ] QMIX-Hierarchical architecture diagram.
- [ ] Optional results progression diagram.

## Additional Reruns That Would Be Ideal But Are Skipped

- [ ] Multi-seed hierarchical DDQN for statistical comparison with QMIX.
- [ ] Multi-seed flat DDQN for stronger flat-interface failure evidence.
- [ ] Additional fairness-specific methods beyond executor-weight variants.
- [ ] Larger-scale scenarios with more UAVs and jammers.

These reruns are not part of the current reporting direction. The final report will use existing Phase 3 artifacts only.

## Phase 4.5 Preparation Questions

- [ ] Should Phase 4.5 draft figure captions and table captions?
- [ ] Should Phase 4.5 draft the Introduction and Contributions section?
- [ ] Should Phase 4.5 create diagram specifications detailed enough for a drawing tool?
- [ ] Should the final report include a short appendix explaining the actual evaluated configuration versus older planning tables?
