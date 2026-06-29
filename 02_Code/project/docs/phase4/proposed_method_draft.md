# Proposed Method Draft

This draft provides the method-section foundation for the final report. It follows Direction A: use the implemented and evaluated Scenario 4 setting without changing simulation parameters or retraining.

## 1. Overview of the Proposed Approach

The proposed approach addresses cooperative multi-UAV control for UAV-assisted IoT service under mobile jamming and heterogeneous backscatter/active communication opportunities. Each UAV must decide how to move, which IoT device to serve, and which communication mode to use. In the original simulator, these decisions are encoded as a flat action over movement, selected IoT device, and communication mode.

Direct flat control becomes difficult in the final Scenario 4 setting because the primitive action space is large:

\[
|\mathcal{A}_{\mathrm{flat}}|
=9(N+1)6
=9(15+1)6
=864.
\]

The implemented research progression is:

1. Flat DDQN as a learning sanity baseline over the primitive action space.
2. Hierarchical DDQN as an action-abstraction test using a 10-action high-level wrapper.
3. QMIX with the same 10-action hierarchical wrapper as the final cooperative MaDRL method.

The final method is therefore best described as **QMIX-Hierarchical**: each UAV learns to select a high-level action, and a rule-based executor maps high-level actions to primitive simulator actions.

## 2. Flat DDQN Baseline

The flat DDQN baseline uses the original simulator action:

\[
a^u_t = (m^u_t, i^u_t, c^u_t),
\]

where:

- \(m^u_t \in \{0,\ldots,8\}\) is the UAV movement command;
- \(i^u_t \in \{0,\ldots,N\}\) is the selected IoT target, with 0 representing no target;
- \(c^u_t \in \{0,\ldots,5\}\) is the communication mode.

The communication modes implemented in `envs/uav_backscatter_env.py` are idle, harvest, backscatter, active, relay, and avoid-jammer. The report should focus on idle, harvest, backscatter, active, and avoid-jammer because relay is implemented but not central to the final Phase 3 results.

For Scenario 4, \(N=15\), yielding a flat action dimension of 864. This flat interface is a useful baseline because it preserves the original primitive action semantics. However, it is hard to learn because many actions are invalid, redundant, or low value in a given state. The flat DDQN result supports this: it reaches throughput/frame 0.3242 in Scenario 4, close to HTT-only 0.3278 and far below the best hierarchical and QMIX results.

Flat DDQN should therefore be presented as a baseline and diagnostic tool, not as the final proposed method.

## 3. Hierarchical Action Abstraction

The hierarchical interface replaces the flat primitive action with a high-level action:

\[
z^u_t \in \{0,\ldots,9\}.
\]

The implemented 10 high-level actions are:

| id | high-level action | Intended behavior |
| --- | --- | --- |
| 0 | IDLE_SAFE | Stay idle unless a safe transmission opportunity exists. |
| 1 | SERVE_NEAREST_QUEUE | Select the nearest in-coverage IoT device with queued packets. |
| 2 | SERVE_BEST_SINR | Select the queued in-coverage IoT device with best SINR score. |
| 3 | PRIORITIZE_BACKSCATTER_TYPE23 | Prefer Type 2/3 devices and backscatter behavior when feasible. |
| 4 | PRIORITIZE_ACTIVE_TYPE1 | Prefer Type 1 devices and active transmission when feasible. |
| 5 | HARVEST_LOW_ENERGY | Prefer low-energy devices and harvest behavior when the primary channel is busy. |
| 6 | AVOID_JAMMER | Move away from the nearest jammer using the avoid-jammer primitive mode. |
| 7 | BALANCE_UNDERSERVED_IOT | Prefer IoT devices that have received fewer delivered packets. |
| 8 | HYBRID_BALANCED | Use a weighted score over queue, SINR, type priority, underserved score, distance, and jammer safety. |
| 9 | HIGH_QUEUE_PRIORITY | Prefer high-queue devices subject to feasibility. |

Implementation source: `envs/hierarchical_action.py`.

### Executor Mapping

The hierarchical executor is rule-based, not learned. It receives one high-level action per UAV and returns one original flat action per UAV. It selects:

- a movement command;
- a target IoT device or no target;
- a primitive communication mode.

The executor uses current environment information, including UAV positions, IoT queues/energy/types, primary busy state, delivered-packet counters, jammers, and SINR-related scoring. This should be stated honestly in the final report. The learned policy selects high-level actions; the executor translates those actions using domain rules and simulator state.

The wrapper `HierarchicalUAVBackscatterEnv` preserves the original environment and changes the learning interface:

- the action dimension becomes 10;
- observations and global state are inherited from the base environment;
- action masks are computed over high-level actions;
- executor diagnostics such as fallback rate and high-level mode counts are logged.

### Why This Improves Learnability

The hierarchical interface reduces Scenario 4 action dimension from 864 to 10. It also removes many obviously invalid low-level combinations from the learner's direct control. The Phase 3 results show that this abstraction is decisive: hierarchical DDQN reaches throughput/frame 0.9710, compared with flat DDQN throughput/frame 0.3242.

The report should attribute this improvement primarily to action-interface design, not merely to a different neural-network architecture.

## 4. QMIX-Based MaDRL Method

The final method uses QMIX with the hierarchical action wrapper. Each UAV is one agent:

\[
u \in \mathcal{U}, \quad |\mathcal{U}|=2 \text{ in Scenario 4}.
\]

Each UAV agent observes a local observation \(o^u_t\) and selects one high-level action \(z^u_t\in\{0,\ldots,9\}\). For Scenario 4:

- local observation dimension: 97;
- global state dimension: 70;
- high-level action dimension: 10;
- number of agents: 2.

Implementation sources:

- `marl/qmix/qmix_trainer.py`;
- `marl/qmix/qmix_agent.py`;
- `marl/qmix/networks.py`.

### Local Observation and Global State

The local observation includes UAV position/energy, nearest-jammer features, primary busy state, and per-IoT relative position, distance, queue, energy, and coverage indicators.

The global state includes all UAV positions/energies, all IoT positions/queues/energies, jammer positions, primary busy state, and normalized time step. QMIX uses this global state only in the mixer during centralized training.

### Centralized Training and Decentralized High-Level Selection

QMIX supports centralized training because:

- the replay buffer stores joint episode data;
- the mixer receives global state;
- all agents share the global environment reward;
- the team action-value function is trained jointly.

Action selection is decentralized at the learned high-level policy level because each UAV's agent network maps its local observation and agent id to Q-values over the 10 high-level actions. A caveat is required: the rule-based executor then maps selected high-level actions to primitive actions using environment-level information. The final report should describe this as decentralized high-level action selection with a centralized/domain-informed executor, rather than fully decentralized primitive control.

### Mixing Network

Each agent network estimates:

\[
Q_u(o^u_t,z^u_t).
\]

The QMIX mixer estimates a joint action-value:

\[
Q_{\mathrm{tot}}(s_t,\mathbf{z}_t)
=
f_{\mathrm{mix}}(Q_1,\ldots,Q_U,s_t).
\]

The implementation uses hypernetworks to generate non-negative mixing weights through absolute value operations, supporting the monotonicity condition:

\[
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_u} \ge 0.
\]

This is appropriate for cooperative multi-UAV control because each UAV makes an individual high-level decision, while the global reward depends on joint outcomes such as duplicate service, jamming exposure, fairness, and mode balance.

## 5. Reward Design

The implementation uses a shared global reward:

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

Confirmed reward components:

- throughput reward for successful packet delivery;
- avoid-jammer bonus when the avoid-jammer action increases distance from the nearest jammer;
- packet-drop penalty;
- queue/delay proxy penalty;
- UAV energy-use penalty;
- jamming-failure penalty;
- collision penalty;
- unfairness penalty using \(1-F_t\), where \(F_t\) is Jain fairness over cumulative delivered packets.

Important detail: IoT energy consumption is tracked in metrics and affects active-transmission feasibility, but the direct reward energy penalty uses UAV energy only.

### Fairness and BALANCE_UNDERSERVED_IOT

Fairness appears in two places:

1. the environment reward penalizes unfairness through \(1-F_t\);
2. the hierarchical executor includes BALANCE_UNDERSERVED_IOT, which biases target selection toward devices with fewer delivered packets relative to the current maximum.

BALANCE_UNDERSERVED_IOT is not a separate reward-shaping term. It is a high-level action and target-selection rule. Phase 3.8 shows that disabling this action hurts throughput, fairness, and jamming behavior. However, the final report should not claim fairness is solved; QMIX base fairness remains moderate.

## 6. Training and Execution Pipeline

### Frame-Level Interaction

For each episode and frame:

1. The environment updates IoT packet arrivals.
2. The jammer moves.
3. Each UAV observes its local state.
4. The agent selects a high-level action with masked epsilon-greedy exploration during training.
5. The hierarchical executor maps high-level actions to primitive movement/target/mode actions.
6. The base environment applies movement, communication, energy updates, packet delivery, jamming checks, and reward calculation.
7. The transition is stored in replay.

### QMIX Training

QMIX stores episode-level transitions containing:

- local observations for all UAVs;
- global states;
- high-level actions;
- shared rewards;
- next observations and next global states;
- done flags;
- current and next action masks.

The agent networks and mixer are trained using a Double-Q style TD target, target networks, and a filled-mask MSE loss over sampled episode batches.

Final QMIX base settings include:

- hidden sizes: [128, 128];
- mixing embed dimension: 32;
- hypernetwork hidden dimension: 64;
- learning rate: 0.0005;
- discount factor: 0.99;
- batch size: 16;
- target update steps: 200;
- epsilon: 1.0 to 0.05 over 25000 steps;
- updates per episode: 1;
- double Q-learning: enabled.

### Evaluation Metrics

Final evaluation reports:

- reward;
- throughput/frame;
- drop rate;
- jamming failure rate;
- fairness;
- energy efficiency;
- backscatter success rate;
- active success rate;
- mode usage;
- executor fallback rate for hierarchical methods.

The final report should connect method claims to the existing Phase 3 publication artifacts, especially Tables 1-4 and Figures 1-6.
