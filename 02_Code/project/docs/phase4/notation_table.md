# Notation Table

| Symbol | Meaning | Implementation variable/function if found | Notes / verification status |
| --- | --- | --- | --- |
| \(\mathcal{U}\) | Set of UAV agents | `network.num_uav`, `self.num_uav` | High confidence. |
| \(U\) | Number of UAVs | `num_uav` | Scenario 4 value: 2. |
| \(\mathcal{I}\) | Set of IoT devices | `network.num_iot`, `self.iot_devices` | High confidence. |
| \(N\) | Number of IoT devices | `num_iot` | Scenario 4 value: 15. |
| \(\mathcal{J}\) | Set of jammers | `self.jammers` | High confidence. |
| \(J\) | Number of configured jammers | `network.num_jammer`, `configured_num_jammer` | Scenario 4 value: 1. |
| \(t\) | Frame/decision step index | `current_step`, `step` | High confidence; one env step is one frame-level decision. |
| \(T\) | Episode length | `max_steps`, `episode_length` | Scenario 4 value: 200. |
| \(L\) | Packet-arrival frame length | `simulation.frame_length` | Scenario 4 value: 10. |
| \(W,H\) | Area width and height | `area_width`, `area_height` | Scenario 4 value: 500 x 500. |
| \((x^u_t,y^u_t,h^u)\) | UAV position and altitude | `UAV.x`, `UAV.y`, `UAV.h` | High confidence. |
| \(E^u_t\) | UAV residual energy | `UAV.energy` | High confidence. |
| \(R_{\mathrm{cov}}\) | UAV coverage radius | `UAV.coverage_radius`, `uav.coverage_radius` | High confidence. |
| \(v_{\max}\) | UAV maximum speed | `UAV.max_speed`, `uav.max_speed` | High confidence. |
| \((x^i,y^i)\) | IoT position | `IoTDevice.x`, `IoTDevice.y` | High confidence. |
| \(c_i\) | IoT device type | `IoTDevice.device_type` | Scenario 4 types are 1, 2, 3. |
| \(q_{i,t}\) | IoT queue length | `IoTDevice.queue` | High confidence. |
| \(Q_i^{\max}\) | IoT queue capacity | `IoTDevice.queue_capacity` | High confidence. |
| \(B_{i,t}\) | IoT residual energy | `IoTDevice.energy` | High confidence. |
| \(B_i^{\max}\) | IoT energy capacity | `IoTDevice.energy_capacity` | High confidence. |
| \(p_i^{\mathrm{arr}}\) | IoT packet arrival probability | `IoTDevice.packet_arrival_prob` | Used in binomial arrivals. |
| \(A_{i,t}\) | Packet arrivals at IoT \(i\) | `IoTDevice.generate_packets` | High confidence; sampled as binomial. |
| \(D_i\) | Cumulative delivered packets for IoT \(i\) | `delivered_per_iot`, `IoTDevice.total_delivered` | High confidence. |
| \(P_j\) | Jammer power | `Jammer.power` | High confidence. |
| \(R_j\) | Jammer radius | `Jammer.radius` | High confidence. |
| \(v_j\) | Jammer speed | `Jammer.speed` | High confidence. |
| \(b_t\) | Primary busy indicator | `primary_busy` | High confidence. |
| \(p_{\mathrm{busy}}\) | RF source busy probability | `RFSource.busy_prob`, `channel.primary_busy_prob` | High confidence. |
| \(m^u_t\) | UAV movement action | `movement_action` | Values 0-8. |
| \(i^u_t\) | Selected IoT index | `selected_iot_index` | 0 means no selected IoT. |
| \(c^u_t\) | Communication mode | `communication_mode` | Values 0-5. |
| \(a^u_t\) | Original flat action | `action_id`, `encode_action` | Flat encoded action. |
| \(z^u_t\) | Hierarchical high-level action | `high_action`, `HIGH_LEVEL_ACTION_NAMES` | Values 0-9. |
| \(\alpha\) | Path-loss exponent | `channel.path_loss_exponent` | High confidence. |
| \(N_0\) | Noise power | `channel.noise_power` | High confidence. |
| \(L(d)\) | Path loss | `path_loss(distance, path_loss_exponent)` | \(L(d)=\max(d,1)^\alpha\). |
| \(P_{\mathrm{rx}}\) | Received power | `received_power` | \(P/L(d)\). |
| \(I_{j,r}\) | Jammer interference at receiver | `jammer_interference` | Applies only inside jammer radius. |
| \(\gamma_{a,r,t}\) | SINR | `compute_sinr` | Avoid conflict with RL discount \(\gamma_{\mathrm{RL}}\). |
| \(\theta\) | SINR threshold | `channel.sinr_threshold` | High confidence. |
| \(p_{\mathrm{succ}}\) | Transmission success probability | `success_probability_from_sinr` | Stochastic, not hard threshold. |
| \(S_t\) | Successful packets in frame | `successful_packets` | Reward component. |
| \(D_t\) | Dropped packets in frame | `dropped_packets` | Reward component. |
| \(\bar{q}_t\) | Average queue length | `avg_queue_length`, `_average_queue_length` | Reward component. |
| \(J_t\) | Jamming failures in frame | `jamming_failures` | Reward component. |
| \(C_t\) | Collision count | `collision_count`, `_count_collisions` | Reward component. |
| \(F_t\) | Jain fairness index | `fairness_index`, `_jain_fairness_index` | High confidence. |
| \(r_t\) | Shared environment reward | `compute_reward` | High confidence. |
| \(\eta\) | Throughput per frame | `avg_throughput_per_frame` | High confidence. |
| \(\rho_{\mathrm{drop}}\) | Packet drop rate | `packet_drop_rate` | High confidence. |
| \(\rho_{\mathrm{jam}}\) | Jamming failure rate | `jamming_failure_rate` | High confidence. |
| \(\xi\) | Energy efficiency | `energy_efficiency` | Delivered packets per total energy consumption. |
| \(o^u_t\) | Local observation for UAV \(u\) | `get_local_observation` | Dimension \(7+6N\). |
| \(s_t\) | Global state | `get_global_state` | Dimension \(3U+4N+2J+2\). |
| \(Q_u\) | Per-agent action-value function | `AgentQNetwork` | QMIX shared agent network. |
| \(Q_{\mathrm{tot}}\) | Mixed team action-value | `QMixer.forward` | QMIX mixer output. |
| \(\gamma_{\mathrm{RL}}\) | RL discount factor | `qmix.gamma`, `ddqn.gamma` | Scenario 4 config value 0.99. |
| \(\epsilon\) | Epsilon-greedy exploration rate | `epsilon_start`, `epsilon_end`, `epsilon_decay_steps` | High confidence. |
| \(M_{b,t}\) | Filled timestep mask in episode replay | `filled` in `EpisodeReplayBuffer` batches | High confidence for QMIX. |
| \(U_i\) | Underserved IoT score | `score_underserved` | Executor-level score, not an environment metric. |
