# Method Architecture Notes

These notes are for diagrams to be drawn later. They describe content, arrows, captions, and caveats.

## 1. System Architecture Diagram

**Diagram title:** UAV-Assisted RF-Powered IoT Network under Mobile Jamming

**Components:**

- Two UAVs.
- Fifteen heterogeneous IoT devices.
- One mobile jammer.
- One RF/primary source or primary busy indicator.
- UAV coverage regions.
- Active transmission links.
- Backscatter-inspired/passive transmission links.
- Harvesting interaction under primary busy state.
- Jammer interference links.

**Arrows/relationships:**

- IoT to UAV active links when primary channel is not busy and IoT has enough energy.
- IoT to UAV backscatter-inspired links when primary channel is busy.
- RF/primary source to IoT devices as a conceptual busy/ambient RF source.
- Jammer to UAV receiver/interference region.
- UAV movement arrows over the service area.

**Caption draft:**

"System model for the evaluated Scenario 4 setting. Two UAVs serve fifteen heterogeneous IoT devices under one mobile jammer. Devices may be served through active transmission, backscatter-inspired passive transmission when the primary channel is busy, or energy harvesting. The jammer affects transmission success through SINR degradation."

**Caveats:**

- The implementation models backscatter through primary busy state and effective backscatter transmit-power factors; it does not explicitly model a full bistatic ambient-backscatter propagation chain.
- The final report should not use a diagram that implies unimplemented physical-layer details.

## 2. Hierarchical Action Architecture Diagram

**Diagram title:** Hierarchical 10-Action Interface for UAV Control

**Components:**

- Local/global state information available to the learner.
- High-level action selector.
- Ten high-level actions:
  - IDLE_SAFE;
  - SERVE_NEAREST_QUEUE;
  - SERVE_BEST_SINR;
  - PRIORITIZE_BACKSCATTER_TYPE23;
  - PRIORITIZE_ACTIVE_TYPE1;
  - HARVEST_LOW_ENERGY;
  - AVOID_JAMMER;
  - BALANCE_UNDERSERVED_IOT;
  - HYBRID_BALANCED;
  - HIGH_QUEUE_PRIORITY.
- Rule-based hierarchical executor.
- Primitive simulator action:
  - movement;
  - selected IoT target;
  - communication mode.
- Base UAV-backscatter environment.
- Reward and metrics feedback.

**Arrows/relationships:**

1. Observation/state goes to the learner.
2. Learner selects one high-level action per UAV.
3. Executor maps high-level action to primitive action.
4. Base environment applies primitive action.
5. Reward, next observation, and diagnostics return to learner/logging.

**Caption draft:**

"The hierarchical wrapper reduces the Scenario 4 action dimension from 864 primitive actions to 10 high-level strategy actions. A rule-based executor translates each selected high-level action into the original movement, IoT-target, and communication-mode tuple, preserving the original simulator while simplifying the learning interface."

**Caveats:**

- The executor is rule-based, not learned.
- The executor uses environment-level information for target and mode selection.
- This is a high-level action abstraction, not an end-to-end primitive-action policy.

## 3. QMIX Architecture Diagram

**Diagram title:** QMIX-Hierarchical Centralized Training with Decentralized High-Level Action Selection

**Components:**

- Local observation \(o^u_t\) for each UAV.
- Optional one-hot agent id.
- Shared per-agent Q-network.
- Action masks over 10 high-level actions.
- Per-agent selected action \(z^u_t\).
- Per-agent Q-values \(Q_u(o^u_t,z^u_t)\).
- Global state \(s_t\).
- QMIX mixer.
- Joint action-value \(Q_{\mathrm{tot}}\).
- Shared reward \(r_t\).
- Target networks and replay buffer for training.

**Arrows/relationships:**

- Each UAV local observation enters the shared agent network.
- Agent network outputs Q-values over 10 high-level actions.
- Action mask filters invalid high-level actions.
- Selected per-agent Q-values feed the mixer.
- Global state feeds the mixer hypernetworks.
- Mixer outputs \(Q_{\mathrm{tot}}\).
- TD loss trains agent networks and mixer using shared reward.
- During execution/evaluation, agent networks select high-level actions; executor maps them to primitive actions.

**Caption draft:**

"QMIX-Hierarchical uses a shared per-UAV action-value network for decentralized high-level action selection and a monotonic mixing network for centralized training. The mixer conditions on the global state and combines selected per-agent Q-values into a joint team value."

**Caveats:**

- CTDE should be phrased carefully: learned high-level action selection is decentralized, but the rule-based executor uses simulator state to translate actions.
- The mixer is used during training, not needed for greedy per-agent high-level action selection.

## 4. Results Story Diagram

**Diagram title:** From Flat Actions to Hierarchical QMIX

**Components:**

- Flat DDQN block with 864 actions.
- Hierarchical DDQN block with 10 actions.
- QMIX-Hierarchical block with 10 actions and multi-agent mixer.
- Key metrics:
  - flat DDQN throughput/frame 0.3242;
  - hierarchical DDQN throughput/frame 0.9710;
  - QMIX base throughput/frame 0.9604 +/- 0.0255;
  - QMIX base jamming failure 0.2056 +/- 0.0744;
  - QMIX base fairness 0.5260 +/- 0.0572.

**Arrows/relationships:**

- Flat DDQN to hierarchical DDQN: action abstraction.
- Hierarchical DDQN to QMIX: cooperative value decomposition.

**Caption draft:**

"Experimental progression in Scenario 4: flat DDQN exposes the primitive action-space bottleneck; the hierarchical action interface makes learning feasible; QMIX adds cooperative value decomposition and improves jamming/fairness coordination."

**Caveats:**

- Do not imply QMIX mean throughput exceeds hierarchical DDQN single-run throughput.
- Emphasize that QMIX is selected because it is stable across seeds and improves coordination-sensitive metrics.
