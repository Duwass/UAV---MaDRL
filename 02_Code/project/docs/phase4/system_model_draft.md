# System Model Draft

This draft reconstructs the system model from the current implementation. Equations marked "implementation-confirmed" are directly reflected in the code. Equations marked "report-level formulation, to verify against implementation" are compact mathematical descriptions of procedural simulator logic.

## 1. Network Overview

We consider a time-slotted UAV-assisted IoT network deployed over a bounded rectangular area of width \(W\) and height \(H\). The network contains:

- a set of UAVs \(\mathcal{U}=\{1,\ldots,U\}\);
- a set of IoT devices \(\mathcal{I}=\{1,\ldots,N\}\);
- a set of jammers \(\mathcal{J}=\{1,\ldots,J\}\);
- one or more RF/primary sources that define whether the primary channel is busy.

In each decision epoch \(t\), each UAV selects a movement command, an IoT target or no target, and a communication mode in the original simulator. In hierarchical experiments, each UAV instead selects one high-level strategy action, and the hierarchical executor maps it to the original simulator action.

The implementation treats each environment step as a frame-level decision. At each step:

1. IoT devices generate packet arrivals.
2. Jammers move.
3. UAV actions are decoded and applied.
4. Transmission, harvesting, movement energy, collisions, jamming failures, and reward are computed.
5. The primary channel state is updated for the next step unless the episode terminates.

Scenario 4, the main benchmark used in the final results, uses:

- \(U=2\) UAVs;
- \(N=15\) heterogeneous IoT devices;
- \(J=1\) jammer;
- one RF source;
- episode length \(T=200\);
- area \(500 \times 500\);
- frame length 10;
- primary busy probability 0.8;
- jammer mobility `chase_nearest_uav`.

Source: `configs/scenario_4_backscatter_types_calibrated.yaml`, `envs/uav_backscatter_env.py`.

## 2. Entities and State Variables

### UAVs

Each UAV \(u \in \mathcal{U}\) has:

- horizontal position \((x^u_t,y^u_t)\);
- altitude \(h^u\);
- residual energy \(E^u_t\);
- maximum speed \(v_{\max}\);
- coverage radius \(R_{\mathrm{cov}}\);
- current target and previous action diagnostics.

Implementation class: `envs/entities.py::UAV`.

### IoT Devices

Each IoT device \(i \in \mathcal{I}\) has:

- position \((x^i,y^i)\);
- type \(c_i \in \{1,2,3\}\) in Scenario 4;
- queue length \(q_{i,t}\);
- queue capacity \(Q_i^{\max}\);
- residual energy \(B_{i,t}\);
- energy capacity \(B_i^{\max}\);
- packet arrival probability \(p_i^{\mathrm{arr}}\);
- cumulative arrived, delivered, and dropped packet counters.

Implementation class: `envs/entities.py::IoTDevice`.

At each step, the implementation samples packet arrivals as:

\[
A_{i,t} \sim \mathrm{Binomial}(L, p_i^{\mathrm{arr}}),
\]

where \(L\) is `frame_length`. Packets exceeding queue capacity are counted as dropped. This is implementation-confirmed by `IoTDevice.generate_packets`.

### Jammers

Each jammer \(j \in \mathcal{J}\) has:

- position \((x^j_t,y^j_t)\);
- power \(P_j\);
- movement speed \(v_j\);
- effective radius \(R_j\);
- mobility mode.

The implementation supports `static`, `random_walk`, and `chase_nearest_uav`. In Scenario 4, the jammer moves toward the nearest live UAV and is clipped to the simulation area.

Implementation class: `envs/entities.py::Jammer`.

### RF/Primary Sources

Each RF source has:

- position \((x^r,y^r)\);
- transmit power \(P_r\);
- busy probability \(p_{\mathrm{busy}}\).

The primary channel busy state \(b_t\in\{0,1\}\) is true if any RF source is busy. The implementation samples each source independently using its busy probability.

Implementation class: `envs/entities.py::RFSource`; update logic: `UAVBackscatterEnv._update_primary_channel`.

## 3. Mobility Model

Each UAV chooses one of nine movement actions:

\[
\Delta \in \{(0,0),(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)\}.
\]

For nonzero movements, the direction is normalized and multiplied by the UAV maximum speed. The resulting position is clipped to the simulation area:

\[
x^u_{t+1} = \mathrm{clip}(x^u_t + \Delta_x v_{\max}, 0, W),
\quad
y^u_{t+1} = \mathrm{clip}(y^u_t + \Delta_y v_{\max}, 0, H).
\]

This is a report-level formulation of `envs/mobility_model.py::move_uav`. The implementation normalizes diagonal movement vectors before applying `max_speed`.

Movement energy is implementation-confirmed as:

\[
E^{u,\mathrm{move}}_t
= c_{\mathrm{move}}\sqrt{\Delta_x^2+\Delta_y^2},
\]

using the unnormalized movement direction stored in `MOVEMENT_DELTAS`. A stay action has zero movement energy; diagonal actions cost \(c_{\mathrm{move}}\sqrt{2}\).

## 4. Communication and Channel Model

### Distances and Path Loss

The implementation defines 2D and 3D distances as Euclidean distances. Path loss is:

\[
L(d) = \max(d,1)^\alpha,
\]

where \(\alpha\) is the path-loss exponent. Received power is:

\[
P_{\mathrm{rx}}(P,d)=\frac{\max(P,0)}{L(d)}.
\]

These equations are implementation-confirmed in `envs/channel_model.py`.

### Jammer Interference

Jammer \(j\) contributes interference at a receiver only if its 2D distance to the receiver is within the jammer radius:

\[
I_{j,r} =
\begin{cases}
\frac{P_j}{\max(d_{3D}(j,r),1)^\alpha}, & d_{2D}(j,r) \le R_j,\\
0, & \text{otherwise}.
\end{cases}
\]

This is implementation-confirmed by `jammer_interference`.

### SINR

For transmitter \(a\), receiver \(r\), and transmit power \(P_a\), SINR is:

\[
\mathrm{SINR}_{a,r,t}
=
\frac{P_{\mathrm{rx}}(P_a,d_{3D}(a,r))}
{N_0 + \sum_{j\in\mathcal{J}} I_{j,r}},
\]

where \(N_0\) is the noise power. This is implementation-confirmed by `compute_sinr`.

For active transmission, \(P_a\) is `tx_power_iot`. For backscatter transmission, the implementation uses an effective transmit power:

\[
P_i^{\mathrm{bs,eff}} = P_r \eta_i,
\]

where \(P_r\) is `tx_power_rf_source` and \(\eta_i\) is the IoT type-specific `backscatter_tx_power_factor`. This is an implementation-level abstraction; the current code does not explicitly model the RF-source-to-IoT incident link geometry in the backscatter SINR.

### Transmission Success Probability

The implementation uses a stochastic success probability based on SINR and threshold \(\theta\):

\[
p_{\mathrm{succ}}(\gamma)=
\begin{cases}
1, & \theta=0 \text{ and } \gamma>0,\\
0, & \theta=0 \text{ and } \gamma=0,\\
\min\left(1,\frac{\gamma}{\gamma+\theta}\right), & \gamma \ge \theta,\\
0.1\frac{\gamma}{\theta}, & 0 \le \gamma < \theta.
\end{cases}
\]

The simulator samples a Bernoulli trial using this probability. This is implementation-confirmed by `success_probability_from_sinr`, `_attempt_backscatter`, and `_attempt_active`.

### Jamming-Related Failure

A failed transmission is counted as jamming-related if:

\[
\mathrm{SINR}_{t}<\theta
\quad \text{and} \quad
\sum_{j\in\mathcal{J}} I_{j,r} > N_0.
\]

This is implementation-confirmed by `UAVBackscatterEnv._is_jamming_related_failure`.

## 5. Communication Modes

The original simulator supports six communication modes:

| Mode id | Mode name | Implementation constant | Main condition |
| --- | --- | --- | --- |
| 0 | idle | `MODE_IDLE` | Always valid. |
| 1 | harvest | `MODE_HARVEST` | Requires primary channel busy and a selected target. |
| 2 | backscatter | `MODE_BACKSCATTER` | Requires primary channel busy, target in coverage, and target queue \(q_i>0\). |
| 3 | active | `MODE_ACTIVE` | Requires target in coverage, queue \(q_i>0\), enough IoT energy, and primary channel not busy unless `allow_active_when_busy=true`. |
| 4 | relay | `MODE_RELAY` | Implemented as UAV communication-energy consumption only; not a main report mode unless explicitly discussed. |
| 5 | avoid jammer | `MODE_AVOID_JAMMER` | Requires at least one jammer; gives avoidance bonus if distance to nearest jammer increases. |

The paper should focus on idle, harvest, backscatter, active, and avoid-jammer behavior. Relay exists in the implementation but has not been central in the Phase 3 results.

## 6. Energy and Resource Model

### UAV Energy

UAV energy decreases from:

- movement energy;
- communication energy for harvest, backscatter, active, and relay modes.

UAV energy consumption is tracked and used in the reward through `uav_energy_used`. If all UAVs have zero energy, the episode terminates.

### IoT Energy

IoT energy is type dependent. Active transmission consumes IoT energy:

\[
B_{i,t+1}=B_{i,t}-c_i^{\mathrm{act}},
\]

when an active attempt is made and enough energy is available. Harvesting increases IoT energy up to capacity:

\[
B_{i,t+1}=\min(B_i^{\max}, B_{i,t}+h_i),
\]

when the primary channel is busy and the harvest mode is selected. Backscatter requires queued packets but does not consume IoT energy in the current implementation.

Implementation sources: `IoTDevice.harvest_energy`, `IoTDevice.can_active_transmit`, `UAVBackscatterEnv._active_energy_cost`, `UAVBackscatterEnv._harvested_energy`.

## 7. Throughput and Episode Metrics

Throughput is measured as delivered packets per environment step:

\[
\mathrm{Throughput/frame} =
\frac{\sum_i D_i}{T_{\mathrm{ep}}},
\]

where \(D_i\) is the total delivered packet count for IoT device \(i\), and \(T_{\mathrm{ep}}\) is the number of elapsed episode steps. This is implementation-confirmed by `avg_throughput_per_frame`.

Other implementation-confirmed metrics include:

- packet drop rate: total dropped packets divided by total arrived packets;
- jamming failure rate: jamming failures divided by transmission attempts;
- energy efficiency: total delivered packets divided by UAV plus IoT energy consumption;
- backscatter success rate: successful backscatter packets divided by attempted backscatter packets;
- active success rate: successful active packets divided by attempted active packets;
- Jain fairness over delivered packets per IoT device.

## 8. Assumptions and Scope

- The current system model is simulator-based, not a hardware RF deployment.
- Backscatter is represented by an effective transmit-power factor and primary-channel busy condition.
- Primary channel occupancy is stochastic and independent across RF sources.
- UAV altitude is fixed during an episode.
- IoT positions are sampled at reset and then static.
- Jamming affects receiver SINR and can be mobile.
- Collisions between UAVs are penalized but not treated as hard physical crashes.
- The hierarchical executor is rule-based and domain-informed; it is not a learned physical-layer controller.
