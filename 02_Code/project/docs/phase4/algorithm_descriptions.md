# Algorithm Descriptions

This document gives concise method descriptions for use in the final report. It follows the implemented Phase 3 pipeline and does not introduce new training settings.

## Flat DDQN

| Item | Description |
| --- | --- |
| Purpose | Learning sanity baseline over the original primitive action interface. |
| Agent/controller structure | Centralized-factorized DDQN with a shared Q-network conditioned on UAV id. Each UAV transition is stored separately, but all UAVs share the global environment reward. |
| Observation/state input | `concat_global_local` in tuned Scenario 4: global state concatenated with the selected UAV's local observation. |
| Action output | Flat encoded action \(a=(\text{movement},\text{selected IoT},\text{communication mode})\). Scenario 4 action dimension is 864. |
| Reward used | Shared environment reward from `compute_reward`; DDQN training may use reward processing in tuned variants, while logs preserve raw environment reward. |
| Strengths | Preserves original primitive action semantics; useful baseline for testing whether the environment is learnable. |
| Weaknesses | Large action dimension, many invalid or low-value primitive combinations, poor Scenario 4 policy quality. |
| Role in final report | Baseline showing the limitation of direct flat control in heterogeneous Scenario 4. Not the final method. |

## Hierarchical DDQN

| Item | Description |
| --- | --- |
| Purpose | Test whether high-level action abstraction makes Scenario 4 learnable. |
| Agent/controller structure | DDQN agent selects one of 10 high-level actions per UAV. A rule-based executor maps high-level actions to primitive simulator actions. |
| Observation/state input | `concat_global_local`, as in flat DDQN, but action output is over the hierarchical wrapper. |
| Action output | High-level action \(z\in\{0,\ldots,9\}\). The executor returns primitive movement, IoT target, and communication mode. |
| Reward used | Shared environment reward from the base environment. |
| Strengths | Reduces Scenario 4 action dimension from 864 to 10; strong empirical performance; validates the action-abstraction design. |
| Weaknesses | Still not a full multi-agent value-decomposition method; executor is rule-based and uses environment information. |
| Role in final report | Strong sanity baseline and evidence that action abstraction is essential. |

## QMIX-Hierarchical

| Item | Description |
| --- | --- |
| Purpose | Final MaDRL prototype for cooperative multi-UAV coordination using the 10-action hierarchical interface. |
| Agent/controller structure | One agent per UAV with a shared agent Q-network. A QMIX mixer combines individual selected Q-values into \(Q_{\mathrm{tot}}\) using the global state. |
| Observation/state input | Each agent receives local observation \(o^u_t\) and optional one-hot agent id; mixer receives global state \(s_t\). Scenario 4 dimensions: local observation 97, global state 70. |
| Action output | Each UAV selects one of 10 high-level actions. The executor maps high-level actions to primitive simulator actions. |
| Reward used | Shared global environment reward. No separate local reward is implemented. |
| Strengths | Cooperative value decomposition; multi-seed stable; improves jamming/fairness trade-off relative to hierarchical DDQN. |
| Weaknesses | Mean throughput is slightly below the hierarchical DDQN single-run throughput; executor remains rule-based; fairness remains moderate. |
| Role in final report | Final main MaDRL method and recommended setting. |

## QMIX with Fairness-Aware Executor Variants

| Item | Description |
| --- | --- |
| Purpose | Ablation for fairness/coordination behavior, not the final selected method. |
| Agent/controller structure | Same QMIX-Hierarchical architecture, with executor weights or high-level action masks changed. |
| Observation/state input | Same as QMIX-Hierarchical. |
| Action output | Same 10-action interface unless the BALANCE_UNDERSERVED_IOT action is disabled for ablation. |
| Reward used | Same shared environment reward. Important: BALANCE_UNDERSERVED_IOT is an action/executor mechanism, not a separate reward-shaping term. |
| Strengths | Tests whether executor fairness weights or action 7 affect trade-offs. Shows action 7 matters. |
| Weaknesses | Fairness weighting does not improve mean fairness over QMIX base; fair_w2 improves throughput and jamming but reduces fairness; fair_w3 degrades performance. |
| Role in final report | Fairness/coordination ablation supporting the choice of QMIX base with BALANCE_UNDERSERVED_IOT enabled. |

## Recommended Wording

- Use "QMIX-Hierarchical" for the final method.
- Use "hierarchical DDQN" for the strong DDQN baseline with the same 10-action wrapper.
- Use "flat DDQN" for the primitive 864-action baseline.
- Avoid saying QMIX solves fairness. Say it improves the jamming/fairness trade-off relative to hierarchical DDQN.
- Avoid saying QMIX has the highest throughput among all learned methods. It is the final MaDRL setting because it is multi-seed stable and better coordinated.
