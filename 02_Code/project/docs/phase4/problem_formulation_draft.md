# Problem Formulation Draft

This draft separates implementation-confirmed metrics from report-level mathematical abstraction. The final paper should preserve this distinction unless formulas are verified line-by-line against the simulator.

## 1. Decision Problem

At each frame \(t\), each UAV \(u\in\mathcal{U}\) selects an action that determines:

- UAV movement;
- selected IoT target, if any;
- communication mode.

The system objective is cooperative: all UAVs share a network-level reward and are evaluated using episode-level network metrics. The goal is to learn a policy that increases successful packet delivery while reducing packet drops, jamming failures, excessive queueing, collisions, energy consumption, and unfair service allocation.

## 2. Implementation Reward

The environment reward is implementation-confirmed in `envs/reward.py::compute_reward`. Let:

- \(S_t\): successful packets in frame \(t\);
- \(D_t\): dropped packets in frame \(t\);
- \(\bar{q}_t\): average IoT queue length;
- \(E^u_t\): UAV energy used in frame \(t\);
- \(J_t\): jamming-related failures in frame \(t\);
- \(C_t\): UAV collision count in frame \(t\);
- \(F_t\): Jain fairness index over cumulative delivered packets;
- \(A_t\): jammer-avoidance bonus.

The implementation reward is:

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

For Scenario 4, the calibrated config uses:

| Weight | Config key | Scenario 4 value |
| --- | --- | --- |
| \(w_{\mathrm{thr}}\) | `w_throughput` | 1.0 |
| \(w_{\mathrm{drop}}\) | `w_drop` | 2.0 |
| \(w_{\mathrm{delay}}\) | `w_delay` | 0.1 |
| \(w_{\mathrm{uavE}}\) | `w_uav_energy` | 0.01 |
| \(w_{\mathrm{jam}}\) | `w_jamming` | 1.0 |
| \(w_{\mathrm{col}}\) | `w_collision` | 5.0 |
| \(w_{\mathrm{unfair}}\) | `w_unfairness` | 0.2 |
| \(w_{\mathrm{avoid}}\) | `w_avoid_jammer` | 0.2 |

Note: IoT energy consumption is tracked as a metric but is not directly used in `compute_reward`; only UAV energy appears in the reward energy penalty.

## 3. Report-Level Optimization Objective

Report-level formulation, to verify against implementation:

\[
\max_{\pi}
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T-1}
\gamma^t r_t
\right],
\]

where \(\pi\) is the learned UAV policy and \(r_t\) is the shared environment reward above. This discounted-return objective matches the RL training setup for DDQN and QMIX, where both use \(\gamma=0.99\) in the final configurations.

For an intuitive network-optimization statement, the same objective can be described as:

\[
\max_\pi
\quad
\mathrm{Throughput}
-\lambda_1\mathrm{Drop}
-\lambda_2\mathrm{Jamming}
+\lambda_3\mathrm{Fairness}
-\lambda_4\mathrm{EnergyCost},
\]

where the \(\lambda\) terms are report-level weights. This compact form should not replace the implementation reward unless the final paper explicitly says it is a conceptual objective.

## 4. Constraints

### Movement Constraints

Each UAV movement action must belong to the nine-action movement set:

\[
m^u_t \in \{0,\ldots,8\}.
\]

UAV positions are clipped to the area:

\[
0 \le x^u_t \le W,\quad 0 \le y^u_t \le H.
\]

The altitude \(h^u\) is fixed in the current implementation.

### Service and Target Constraints

Each UAV action selects at most one IoT target per frame, or target index 0 for no target:

\[
i^u_t \in \{0,1,\ldots,N\}.
\]

Multiple UAVs may select the same IoT device. This is not forbidden, but duplicate target selection is counted through `duplicate_target_count`.

Communication modes requiring an IoT target are invalid if no target is selected.

### Coverage Constraint

For harvest, backscatter, and active modes, the selected IoT must be within the UAV coverage radius:

\[
d_{2D}(u,i) \le R_{\mathrm{cov}}.
\]

This is implementation-confirmed by `_validate_decoded_action` and `_apply_actions`.

### Queue Constraint

Backscatter and active transmission require a non-empty IoT queue:

\[
q_{i,t} > 0.
\]

If the selected queue is empty, the action is invalid and counted by `no_queue_selected_count`.

### Primary-Channel Constraints

Harvest and backscatter require the primary channel to be busy:

\[
b_t = 1.
\]

Active transmission is allowed when the primary channel is not busy, unless `allow_active_when_busy=true`. In Scenario 4, `allow_active_when_busy=false`, so:

\[
\mathrm{active\ valid} \Rightarrow b_t = 0.
\]

### Energy Constraints

Active transmission requires enough IoT energy:

\[
B_{i,t} \ge c_i^{\mathrm{act}}.
\]

UAV movement and communication consume UAV energy. The current implementation allows a UAV to consume up to its remaining energy, and an episode terminates when no UAV remains alive.

### Jamming Constraint or Penalty

Jamming is not a hard constraint. It affects SINR and is penalized through jamming-related transmission failures:

\[
J_t = \#\{\text{failed transmissions with SINR}<\theta \text{ and jammer interference}>N_0\}.
\]

This is implementation-confirmed by `_is_jamming_related_failure`.

### Action Masking

The implementation uses action masks to avoid intentionally selecting invalid actions in learning algorithms. For the flat interface, masks are generated from `_validate_decoded_action`. For the hierarchical interface, invalid high-level actions can be masked if:

- no jammer exists and AVOID_JAMMER is requested;
- no Type 2/3 devices exist and the Type 2/3 backscatter-priority action is requested;
- no Type 1 devices exist and the Type 1 active-priority action is requested;
- an action is explicitly disabled by config.

## 5. Metrics

### Throughput per Frame

Implementation form:

\[
\eta =
\frac{\sum_{i\in\mathcal{I}}D_i}{T_{\mathrm{ep}}},
\]

where \(D_i\) is total delivered packets for IoT \(i\), and \(T_{\mathrm{ep}}\) is the number of episode steps.

Implementation variable: `avg_throughput_per_frame`.

### Packet Drop Rate

Implementation form:

\[
\rho_{\mathrm{drop}}
=
\frac{\sum_i \mathrm{Dropped}_i}{\max(\sum_i \mathrm{Arrived}_i,1)}.
\]

Implementation variable: `packet_drop_rate`.

### Jamming Failure Rate

Implementation form:

\[
\rho_{\mathrm{jam}}
=
\frac{\mathrm{jamming\ failures}}{\max(\mathrm{transmission\ attempts},1)}.
\]

Implementation variable: `jamming_failure_rate`.

### Fairness

Implementation form:

\[
F =
\frac{(\sum_i D_i)^2}{N\sum_i D_i^2+\epsilon},
\]

where \(D_i\) is cumulative delivered packets for IoT \(i\), \(N\) is the number of IoT devices, and \(\epsilon=10^{-12}\) in the denominator. If no packets have been delivered, the implementation returns fairness 1.0.

Implementation variable/function: `fairness_index`, `_jain_fairness_index`.

### Energy Efficiency

Implementation form:

\[
\xi =
\frac{\sum_i D_i}{\max(E_{\mathrm{UAV}}+E_{\mathrm{IoT}},10^{-9})}.
\]

Implementation variable: `energy_efficiency`.

### Backscatter and Active Success Rates

Implementation forms:

\[
\rho_{\mathrm{bs}}
=
\frac{\mathrm{successful\ backscatter\ packets}}
{\max(\mathrm{attempted\ backscatter\ packets},1)},
\]

\[
\rho_{\mathrm{act}}
=
\frac{\mathrm{successful\ active\ packets}}
{\max(\mathrm{attempted\ active\ packets},1)}.
\]

Implementation variables: `backscatter_success_rate`, `active_success_rate`.

### Underserved IoT Score

Implementation form in the hierarchical executor:

\[
U_i =
\begin{cases}
1, & \max_k D_k \le 0,\\
1-\min\left(1,\frac{D_i}{\max_k D_k}\right), & \text{otherwise}.
\end{cases}
\]

This score is used in target selection for BALANCE_UNDERSERVED_IOT and HYBRID_BALANCED. It is not an environment-level metric unless logged indirectly through performance outcomes.

Implementation function: `HierarchicalActionExecutor.score_underserved`.

## 6. Final Problem Statement for the Report

A conservative final problem statement is:

Given a UAV-assisted RF-powered IoT network with mobile jamming, stochastic packet arrivals, energy-limited IoT devices, and active/backscatter/harvest communication options, learn a cooperative UAV policy that maximizes expected cumulative network reward while satisfying movement, coverage, channel-busy, queue, and energy feasibility constraints. The learned policy is evaluated by throughput/frame, packet drop rate, jamming failure rate, fairness, energy efficiency, active/backscatter success rates, and hierarchical executor fallback rate.
