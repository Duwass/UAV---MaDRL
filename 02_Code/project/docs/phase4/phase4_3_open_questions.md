# Phase 4.3 Open Questions

## Executor Details Needing Verification

- [ ] Decide how much of `_select_by_score` should be formalized in the paper. It is procedural and action-dependent.
- [ ] Decide whether to mention each executor fallback rule or summarize fallback behavior at a high level.
- [ ] Confirm whether `score_jammer_safety` should be described as UAV-to-nearest-jammer safety, not IoT-target-specific safety.
- [ ] Decide whether to mention relay mode. It exists in the primitive action set but is not central to the final results.

## Action Masking

- [ ] Decide whether action masking should be highlighted as a method component or treated as implementation detail.
- [ ] If highlighted, clarify that flat masks enforce primitive feasibility while hierarchical masks suppress unavailable high-level categories.
- [ ] Avoid claiming action masking alone caused the performance improvement; no final ablation isolates masking.

## Reward and Fairness

- [ ] Make clear that the reward is global/shared, not local per UAV.
- [ ] Make clear that BALANCE_UNDERSERVED_IOT is an action/executor mechanism, not a separate reward-shaping term.
- [ ] Decide whether to expose all reward weights in the main text or in an appendix.
- [ ] Note that IoT energy affects feasibility and metrics but is not directly included in the reward energy penalty.

## CTDE Wording

- [ ] CTDE can be claimed for QMIX training because the mixer uses global state and shared reward during training.
- [ ] Decentralized execution should be qualified as decentralized high-level action selection.
- [ ] The executor uses environment-level information to translate high-level actions into primitive simulator actions, so do not claim fully decentralized primitive-action execution.

## Backscatter Interpretation

- [ ] Decide whether to call the model "ambient-backscatter-inspired" or "abstracted RF-powered ambient backscatter" if strict physical-layer precision is required.
- [ ] State that backscatter is conditioned on primary busy state and uses effective backscatter transmit-power factors.
- [ ] Avoid implying that RF-source-to-IoT incident-channel geometry is explicitly modeled unless the implementation is extended.

## Missing Diagrams

- [ ] System architecture diagram.
- [ ] Hierarchical action interface diagram.
- [ ] QMIX architecture diagram.
- [ ] Optional progression diagram: flat DDQN -> hierarchical DDQN -> QMIX.

## Citations Still Needed

- [ ] UAV-assisted IoT data collection.
- [ ] Anti-jamming communication and mobile jammer models.
- [ ] Ambient backscatter and RF-powered IoT.
- [ ] Deep reinforcement learning for wireless networks.
- [ ] DDQN.
- [ ] QMIX and value decomposition.
- [ ] Hierarchical reinforcement learning or action abstraction.
- [ ] Jain fairness.

## Wording for Final Experimental Setting

- [ ] Include the note that the report uses the actual implemented/evaluated Scenario 4 configuration.
- [ ] If older proposed parameter tables are mentioned, label them as planning references only.
- [ ] Do not report proposed/unrun parameters as final settings.
- [ ] Do not retrain to match older tables; the report is based on existing Phase 3 artifacts.

## Method Claims to Keep Conservative

- [ ] Say "QMIX improves coordination-sensitive metrics" rather than "QMIX dominates hierarchical DDQN".
- [ ] Say "hierarchical abstraction makes the tested Scenario 4 learnable" rather than "always solves large action spaces".
- [ ] Say "fairness improves relative to hierarchical DDQN" rather than "fairness is solved".
- [ ] Say "rule-based executor" rather than "learned hierarchical controller" unless referring only to the learned high-level policy.
