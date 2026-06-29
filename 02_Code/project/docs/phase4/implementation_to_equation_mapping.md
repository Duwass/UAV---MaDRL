# Implementation to Equation Mapping

This file maps code-level objects to report-level mathematical concepts. Confidence reflects how directly the equation is implemented in code.

| File path | Class/function/variable | Mathematical concept | Confidence | Notes |
| --- | --- | --- | --- | --- |
| `envs/entities.py` | `UAV` | UAV agent state \((x^u,y^u,h^u,E^u,R_{\mathrm{cov}},v_{\max})\) | High | Direct dataclass fields. |
| `envs/entities.py` | `IoTDevice` | IoT state \((x^i,y^i,c_i,q_i,B_i)\) and packet counters | High | Direct dataclass fields. |
| `envs/entities.py` | `IoTDevice.generate_packets` | \(A_{i,t}\sim\mathrm{Binomial}(L,p_i^{\mathrm{arr}})\) and queue overflow drops | High | Directly uses `rng.binomial(frame_length, packet_arrival_prob)`. |
| `envs/entities.py` | `IoTDevice.harvest_energy` | \(B_{i,t+1}=\min(B_i^{\max},B_{i,t}+h_i)\) | High | Exact clipping to energy capacity. |
| `envs/entities.py` | `IoTDevice.can_active_transmit` | Active feasibility \(q_i>0\) and \(B_i\ge c_i^{\mathrm{act}}\) | High | Direct boolean condition. |
| `envs/entities.py` | `Jammer.move_random_walk` | Random-walk jammer mobility | High | Random angle and clipped position. |
| `envs/entities.py` | `Jammer.move_chase_nearest_uav` | Chase-nearest-UAV jammer mobility | High | Scenario 4 uses this. |
| `envs/entities.py` | `RFSource.is_busy` | Bernoulli primary busy state | High | Busy if `rng.random() < busy_prob`. |
| `envs/channel_model.py` | `distance_2d`, `distance_3d` | Euclidean distance | High | Direct functions. |
| `envs/channel_model.py` | `path_loss` | \(L(d)=\max(d,1)^\alpha\) | High | Direct function. |
| `envs/channel_model.py` | `received_power` | \(P_{\mathrm{rx}}=P/L(d)\) | High | Direct function. |
| `envs/channel_model.py` | `jammer_interference` | Jammer interference within radius | High | Zero outside 2D radius; otherwise received jammer power. |
| `envs/channel_model.py` | `compute_sinr` | \(\mathrm{SINR}=P_s/(N_0+\sum I_j)\) | High | Direct function. |
| `envs/channel_model.py` | `success_probability_from_sinr` | Stochastic packet success probability | High | Direct piecewise function. |
| `envs/mobility_model.py` | `MOVEMENT_DELTAS` | Nine UAV movement directions | High | Direct dictionary. |
| `envs/mobility_model.py` | `move_uav` | UAV position update and boundary clipping | High | Direction normalized before max-speed movement. |
| `envs/mobility_model.py` | `compute_movement_energy` | Movement energy \(c_{\mathrm{move}}\|\Delta\|\) | High | Uses unnormalized movement vector norm. |
| `envs/reward.py` | `compute_reward` | Shared reward weighted sum | High | Exact implementation reward. |
| `envs/uav_backscatter_env.py` | `MODE_IDLE`, `MODE_HARVEST`, `MODE_BACKSCATTER`, `MODE_ACTIVE`, `MODE_RELAY`, `MODE_AVOID_JAMMER` | Original communication mode set | High | Direct constants. |
| `envs/uav_backscatter_env.py` | `NUM_MOVEMENT_ACTIONS`, `NUM_COMMUNICATION_MODES` | Flat action factors | High | 9 movement actions, 6 modes. |
| `envs/uav_backscatter_env.py` | `encode_action`, `decode_action` | Flat action encoding \(9(N+1)6\) | High | Direct mapping. |
| `envs/uav_backscatter_env.py` | `get_local_observation` | UAV local observation \(o^u_t\) | High | Direct feature construction. |
| `envs/uav_backscatter_env.py` | `get_global_state` | Global state \(s_t\) | High | Direct feature construction. |
| `envs/uav_backscatter_env.py` | `get_action_mask`, `_validate_decoded_action` | Flat action feasibility mask | High | Direct validation logic. |
| `envs/uav_backscatter_env.py` | `_apply_actions` | Procedural transition and service model | High | Applies movement, communication modes, energy, attempts, and counters. |
| `envs/uav_backscatter_env.py` | `_attempt_backscatter` | Backscatter attempt, SINR, stochastic success, delivery | High | Uses effective RF-source power times backscatter factor. |
| `envs/uav_backscatter_env.py` | `_attempt_active` | Active attempt, SINR, stochastic success, delivery | High | Uses IoT transmit power. |
| `envs/uav_backscatter_env.py` | `_is_jamming_related_failure` | Jamming failure condition | High | SINR below threshold and interference above noise. |
| `envs/uav_backscatter_env.py` | `_episode_metrics` | Throughput/drop/jam/energy/success metrics | High | Direct source of logged metrics. |
| `envs/uav_backscatter_env.py` | `_jain_fairness_index` | Jain fairness over delivered packets | High | Direct formula with epsilon. |
| `envs/hierarchical_action.py` | `HIGH_LEVEL_ACTION_NAMES` | Hierarchical action set \(\{0,\ldots,9\}\) | High | Direct dictionary. |
| `envs/hierarchical_action.py` | `HierarchicalActionExecutor.build_original_actions` | High-level to original-action mapping | High | Direct executor entry point. |
| `envs/hierarchical_action.py` | `_select_by_score` | Weighted target scoring | Medium | Formula is procedural with optional terms; report equation should be described as representative. |
| `envs/hierarchical_action.py` | `_select_mode` | Mode selection policy | Medium | Procedural decision tree based on primary busy, type, energy, queue, and SINR score. |
| `envs/hierarchical_action.py` | `score_underserved` | Underserved IoT score | High | Direct formula using delivered-per-IoT counts. |
| `envs/hierarchical_action.py` | `score_jammer_safety` | Jammer safety score | Medium | Uses nearest jammer distance from UAV, not target-specific jammer distance. |
| `envs/hierarchical_env.py` | `HierarchicalUAVBackscatterEnv` | Hierarchical environment wrapper | High | Wraps base env without replacing original env. |
| `envs/hierarchical_env.py` | `get_action_mask` | Hierarchical action mask | High | Masks unavailable action categories and config-disabled actions. |
| `marl/ddqn/ddqn_agent.py` | `CentralizedFactorizedDDQNAgent` | DDQN agent with UAV-id-conditioned Q-network | High | Shared network over UAV ids. |
| `marl/ddqn/ddqn_agent.py` | `train_step` | Double-DQN TD target | High | Online selects next action, target evaluates. |
| `marl/ddqn/networks.py` | `DuelingQNetwork` | Dueling value/advantage decomposition | High | \(Q=V+A-\mathrm{mean}(A)\). |
| `marl/ddqn/ddqn_trainer.py` | `build_state` | DDQN state representation | High | `global_state` or `concat_global_local`. |
| `marl/ddqn/ddqn_trainer.py` | `RewardProcessor` usage | Processed reward for DDQN training | High | Logs raw reward while training can use processed reward. |
| `marl/qmix/networks.py` | `AgentQNetwork` | Per-agent local Q-network | High | Input is local observation plus optional one-hot agent id. |
| `marl/qmix/networks.py` | `QMixer` | QMIX monotonic value mixer | High | Hypernetwork weights use absolute value for non-negativity. |
| `marl/qmix/qmix_agent.py` | `select_actions` | Masked epsilon-greedy decentralized high-level action selection | High | Uses local observations and action masks. |
| `marl/qmix/qmix_agent.py` | `train_step` | QMIX TD target and filled-mask MSE | High | Direct implementation of double-Q target and mixer loss. |
| `marl/qmix/qmix_trainer.py` | `collect_episode` | Episode-level QMIX transition collection | High | Stores observations, global states, actions, rewards, masks, next states. |
| `marl/qmix/replay_buffer.py` | `EpisodeReplayBuffer` | Episode replay with padding/filled mask | High | Needed for QMIX sequence batches. |
| `marl/qmix/utils.py` | `apply_action_mask`, `masked_argmax`, `masked_epsilon_greedy` | Action masking and fallback behavior | High | Invalid actions receive -1e9; all-invalid fallback to 0. |
| `configs/scenario_4_backscatter_types_calibrated.yaml` | Scenario config values | Main benchmark parameters | High | Final Scenario 4 source config. |
| `configs/qmix_tuned/qmix_sc4_base.yaml` | QMIX base config | Final MaDRL setting | High | Seeds overridden by runner for multi-seed validation. |
