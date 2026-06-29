# Phase 4.2 Open Questions

These items should be resolved or explicitly qualified before final report writing.

## Formulas Needing Verification

- [ ] Whether the final paper should expose the exact stochastic success-probability function or describe it qualitatively as SINR-dependent stochastic delivery.
- [ ] Whether to include the exact movement-energy cost for diagonal actions, since implementation energy uses the unnormalized direction norm while movement itself normalizes diagonal displacement.
- [ ] Whether to present the hierarchical target score as an equation. The implementation is procedural and action-dependent, so a compact equation may oversimplify it.
- [ ] Whether to include relay mode in the formal action set discussion. It exists in the simulator but is not central to the final narrative.
- [ ] Whether the reward should include IoT energy in the final mathematical objective. The implementation tracks IoT energy consumption in metrics but the reward energy penalty uses UAV energy only.
- [ ] Whether fairness returning 1.0 before any delivery should be mentioned in the paper. It is an implementation edge case.

## Missing Assumptions to Clarify

- [ ] Whether RF source geometry should be described as affecting backscatter. Current backscatter SINR uses RF-source transmit power times a device factor, but not explicit RF-source-to-device distance.
- [ ] Whether UAV altitude should be assumed fixed throughout all experiments.
- [ ] Whether IoT devices are static after reset in all scenarios.
- [ ] Whether packet arrivals are independent across IoT devices and time steps. The implementation samples independently, but this should be stated only if acceptable for the report.
- [ ] Whether jammer knowledge is assumed available to the UAV/controller through observations. Local observations include nearest jammer relative features.
- [ ] Whether the executor is assumed deployable as a controller-side module or only as a simulator-side action abstraction.

## Possible Code-to-Narrative Mismatches

- [ ] The research topic says RF-powered ambient backscatter. The implementation models backscatter through primary busy state and effective transmit-power factors, not a full bistatic ambient-backscatter physical model.
- [ ] The final method is not end-to-end learned from primitive actions; it uses a rule-based hierarchical executor.
- [ ] QMIX execution is decentralized for high-level action selection, but the executor uses environment state to translate actions. The paper should avoid overstating fully decentralized primitive control.
- [ ] Hierarchical DDQN has slightly higher single-run throughput than QMIX base mean. The QMIX claim should emphasize stability and jamming/fairness coordination.
- [ ] Fairness weighting ablations did not improve mean fairness over QMIX base. Do not claim fairness was solved by weighting.

## Missing Diagrams

- [ ] System model diagram:
  - UAVs;
  - IoT devices;
  - mobile jammer;
  - RF/primary source;
  - active link;
  - backscatter-like link;
  - jammer interference.
- [ ] Hierarchical action pipeline diagram:
  - local observation;
  - learned high-level action;
  - rule-based executor;
  - original movement/target/mode action.
- [ ] QMIX architecture diagram:
  - per-UAV shared agent network;
  - local observations plus agent id;
  - global state;
  - monotonic mixer;
  - shared reward.
- [ ] Optional results narrative diagram:
  - flat DDQN action bottleneck;
  - hierarchical DDQN improvement;
  - QMIX coordination improvement.

## Missing Citations

- [ ] UAV-assisted IoT data collection and trajectory planning.
- [ ] Anti-jamming wireless communication and mobile jamming models.
- [ ] Ambient backscatter and RF-powered communication networks.
- [ ] Deep reinforcement learning for wireless resource allocation.
- [ ] Multi-agent reinforcement learning for cooperative control.
- [ ] QMIX or monotonic value decomposition.
- [ ] Hierarchical reinforcement learning or action abstraction.
- [ ] Jain fairness or wireless scheduling fairness.

## Claims to Avoid or Soften

- [ ] Avoid "QMIX has the best throughput among all learned policies" unless comparing only multi-seed reliable methods. Hierarchical DDQN single-run throughput is 0.9710 and QMIX base mean is 0.9604.
- [ ] Avoid "the proposed method solves fairness." Fairness improves relative to hierarchical DDQN, but remains moderate at 0.5260 +/- 0.0572.
- [ ] Avoid "physical-layer accurate ambient backscatter model" unless the implementation is extended or the abstraction is clearly described.
- [ ] Avoid "fully decentralized execution down to primitive actions." The learned high-level action selection is decentralized; the executor is rule-based and uses environment information.
- [ ] Avoid "MAPPO comparison" because MAPPO was not implemented or evaluated.

## Recommended Clarifications Before Final Writing

- [ ] Decide whether to include exact code-level reward weights in the main text or in an appendix.
- [ ] Decide whether to call the backscatter model "ambient-backscatter-inspired" or "abstracted ambient backscatter" if a strict physical-layer interpretation is not desired.
- [ ] Decide whether to include mode usage counts as an additional result table.
- [ ] Decide whether to include no-jammer and static weak DDQN results as sanity-check appendix material.
- [ ] Decide citation style and target venue requirements before converting these Markdown drafts into final paper prose.
