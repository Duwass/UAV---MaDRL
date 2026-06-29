# MaDRL Formulation Draft

This draft describes the learning problem used by DDQN and QMIX in the current implementation. It is not final paper prose.

## 1. Agents and Cooperative Setting

Each UAV is treated as one learning agent:

\[
\mathcal{U}=\{1,\ldots,U\}.
\]

All agents share the same environment reward. There is no implementation-level local reward per UAV. This makes the problem cooperative and suitable for centralized training with decentralized or partially decentralized execution.

Implementation sources:

- `marl/qmix/qmix_trainer.py`;
- `marl/qmix/qmix_agent.py`;
- `marl/ddqn/ddqn_trainer.py`;
- `envs/uav_backscatter_env.py`.

## 2. Local Observation Space

For UAV \(u\), the local observation \(o^u_t\) is built by `UAVBackscatterEnv.get_local_observation`. It contains:

1. UAV normalized position and energy:
   - \(x^u_t/W\);
   - \(y^u_t/H\);
   - \(E^u_t/E^u_0\).
2. Nearest jammer features:
   - normalized relative jammer \(x\)-offset;
   - normalized relative jammer \(y\)-offset;
   - normalized nearest jammer distance.
3. Primary busy indicator.
4. For each IoT device:
   - relative \(x\)-offset from UAV to IoT;
   - relative \(y\)-offset from UAV to IoT;
   - normalized distance;
   - normalized queue length;
   - normalized IoT energy;
   - in-coverage indicator.

The implementation observation dimension is:

\[
d_o = 7 + 6N.
\]

For Scenario 4, \(N=15\), so \(d_o=97\).

## 3. Global State

QMIX uses a global state \(s_t\) from `UAVBackscatterEnv.get_global_state`. It contains:

- for each UAV: normalized \(x\), normalized \(y\), normalized energy;
- for each IoT: normalized \(x\), normalized \(y\), normalized queue, normalized energy;
- for each configured jammer: normalized \(x\), normalized \(y\);
- primary busy indicator;
- normalized current step.

The implementation global-state dimension is:

\[
d_s = 3U + 4N + 2J + 2.
\]

For Scenario 4, \(U=2\), \(N=15\), \(J=1\), so \(d_s=70\).

## 4. Flat DDQN Action Space

The original flat simulator action is:

\[
a^u_t = (m^u_t, i^u_t, c^u_t),
\]

where:

- \(m^u_t \in \{0,\ldots,8\}\) is one of nine movement commands;
- \(i^u_t \in \{0,\ldots,N\}\) is selected IoT index, with 0 meaning no target;
- \(c^u_t \in \{0,\ldots,5\}\) is communication mode.

The flat action dimension is:

\[
|\mathcal{A}_{\mathrm{flat}}|
=
9(N+1)6.
\]

For Scenario 4, \(N=15\), so:

\[
|\mathcal{A}_{\mathrm{flat}}|=9\times16\times6=864.
\]

Implementation functions:

- `encode_action`;
- `decode_action`;
- `_validate_decoded_action`;
- `get_action_mask`.

The flat DDQN result shows this interface is difficult for the tested DDQN prototype in heterogeneous Scenario 4.

## 5. Hierarchical 10-Action Abstraction

The hierarchical interface replaces the flat action with a high-level action:

\[
z^u_t \in \{0,\ldots,9\}.
\]

The 10 high-level actions are:

| id | high-level action | Intended behavior |
| --- | --- | --- |
| 0 | IDLE_SAFE | Idle unless a safe target is available. |
| 1 | SERVE_NEAREST_QUEUE | Serve nearest in-coverage IoT with queued packets. |
| 2 | SERVE_BEST_SINR | Serve queued IoT with strongest SINR score. |
| 3 | PRIORITIZE_BACKSCATTER_TYPE23 | Prefer Type 2/3 queued IoT and backscatter behavior. |
| 4 | PRIORITIZE_ACTIVE_TYPE1 | Prefer Type 1 active transmission when feasible. |
| 5 | HARVEST_LOW_ENERGY | Prefer low-energy devices and harvesting when busy. |
| 6 | AVOID_JAMMER | Move away from nearest jammer and use avoid-jammer mode. |
| 7 | BALANCE_UNDERSERVED_IOT | Prefer underserved IoT devices based on delivered-packet imbalance. |
| 8 | HYBRID_BALANCED | Weighted hybrid score over queue, SINR, type, fairness, distance, and jammer risk. |
| 9 | HIGH_QUEUE_PRIORITY | Prefer high queue pressure subject to feasibility. |

Implementation sources:

- `envs/hierarchical_action.py`;
- `envs/hierarchical_env.py`.

The executor maps high-level actions into original simulator actions using target-selection and mode-selection rules. It also records fallback counts and mode-use diagnostics.

## 6. Hierarchical Target Scoring

The executor uses normalized scores for queue, SINR, distance, energy need, type priority, underserved status, and jammer safety. The weighted target score used by `_select_by_score` is implementation-confirmed as:

\[
\begin{aligned}
\mathrm{score}(i)
=&
1.0\,\mathrm{coverage}(u,i)
+w_q\,\mathrm{queue}(i)
+w_\gamma\,\mathrm{sinr}(u,i)
+w_{\mathrm{type}}\,\mathrm{typePriority}(i)\\
&+w_{\mathrm{under}}\,\mathrm{underserved}(i)
+w_d(1-\mathrm{distance}(u,i))
-w_j(1-\mathrm{jammerSafety}(u,i))
+\mathrm{optional\ preference\ terms}.
\end{aligned}
\]

The optional terms depend on the selected high-level action. For example, BALANCE_UNDERSERVED_IOT adds an additional fairness-weighted underserved score, and HYBRID_BALANCED adds type/mode preference bonuses depending on the primary busy state.

This is a report-level rendering of procedural code and should be verified before final equation use.

## 7. Reward Design

The learning algorithms use the global environment reward:

\[
r_t
=
w_{\mathrm{thr}}S_t
+w_{\mathrm{avoid}}A_t
-w_{\mathrm{drop}}D_t
-w_{\mathrm{delay}}\bar{q}_t
-w_{\mathrm{uavE}}E^u_t
-w_{\mathrm{jam}}J_t
-w_{\mathrm{col}}C_t
-w_{\mathrm{unfair}}(1-F_t).
\]

There is no separate local UAV reward in the implementation.

Fairness enters in two ways:

1. environment reward includes an unfairness penalty \(1-F_t\);
2. the hierarchical executor uses BALANCE_UNDERSERVED_IOT and underserved scores to bias target selection.

Important distinction: BALANCE_UNDERSERVED_IOT affects action translation and target selection; it is not a separate reward component.

## 8. DDQN Formulation

The DDQN implementation is centralized-factorized:

- each UAV transition is stored separately;
- the reward is shared/global;
- the state can be global-only or global concatenated with the local observation;
- the final Phase 3 setting uses `concat_global_local`;
- UAV identity is encoded into the Q-network input;
- the Q-network outputs action values over either flat actions or high-level hierarchical actions depending on the training interface.

The DDQN target is the standard double-DQN target:

\[
y = r + \gamma(1-d)Q_{\mathrm{target}}(s',\arg\max_{a'}Q_{\mathrm{online}}(s',a')).
\]

This is implementation-confirmed in `marl/ddqn/ddqn_agent.py`.

## 9. QMIX Formulation

QMIX uses one decentralized agent network shared across UAVs:

\[
Q_u(o^u_t, z^u_t; \theta).
\]

The agent network optionally receives a one-hot agent id. It outputs Q-values over the 10 hierarchical actions.

The mixer combines selected per-agent Q-values using the global state:

\[
Q_{\mathrm{tot}}(s_t,\mathbf{z}_t)
=
f_{\mathrm{mix}}\left(Q_1,\ldots,Q_U,s_t;\phi\right).
\]

The implementation uses non-negative mixing weights generated by hypernetworks via absolute value, preserving the QMIX monotonicity condition:

\[
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_u} \ge 0.
\]

Implementation source: `marl/qmix/networks.py::QMixer`.

The TD target is implementation-confirmed as:

\[
y_t = r_t + \gamma(1-d_t)Q_{\mathrm{tot}}^{\mathrm{target}}(s_{t+1},\mathbf{z}_{t+1}^{*}),
\]

where double Q-learning selects next actions using the online agent network and evaluates them using the target agent network and target mixer:

\[
\mathbf{z}_{t+1}^{*}
=
\arg\max_{\mathbf{z}} Q_{\mathrm{online}}(o_{t+1},\mathbf{z}).
\]

The loss is a filled-mask MSE over episode batches:

\[
\mathcal{L}
=
\frac{\sum_{b,t} M_{b,t}
\left(Q_{\mathrm{tot}}(s_{b,t},\mathbf{z}_{b,t})-y_{b,t}\right)^2}
{\sum_{b,t}M_{b,t}},
\]

where \(M_{b,t}\) is the replay-buffer filled mask.

## 10. Action Masking

Both DDQN and QMIX apply action masks before greedy or epsilon-greedy selection. Invalid Q-values are filled with a large negative value. If all actions are invalid, the implementation falls back to action 0.

Implementation sources:

- `marl/ddqn/action_masking.py`;
- `marl/qmix/utils.py`;
- `envs/uav_backscatter_env.py::get_action_mask`;
- `envs/hierarchical_env.py::get_action_mask`.

## 11. Training and Execution Structure

QMIX training is centralized because:

- the replay batch contains local observations for all agents;
- the mixer receives global state;
- the team reward is shared;
- target networks and mixing networks are trained jointly.

Execution is decentralized at the learned high-level action-selection level because each UAV agent network can select a high-level action from its local observation and agent id. In the simulator, the rule-based hierarchical executor then maps high-level actions to original actions using environment state. The final paper should describe this carefully: the learned policy is decentralized over high-level actions, while the executor is a domain-informed action translator.

## 12. Why MaDRL Instead of Single-Agent DRL

Single-agent or centralized-factorized DDQN treats each UAV decision with a shared value function but does not explicitly learn a cooperative joint action-value decomposition. In contrast, QMIX learns how individual UAV utilities combine into a global team value conditioned on the global state. This is better aligned with:

- cooperative UAV service;
- duplicate-target avoidance;
- jamming-risk distribution;
- fairness across IoT devices;
- balancing active and backscatter opportunities across multiple UAVs.

The Phase 3 results support this: QMIX base has slightly lower mean throughput than the hierarchical DDQN single run, but improves jamming failure and fairness, which are coordination-sensitive metrics.
