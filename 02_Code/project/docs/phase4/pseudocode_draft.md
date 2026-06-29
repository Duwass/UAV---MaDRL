# Pseudocode Draft

This pseudocode is report-facing and abstracts implementation details. It should not be treated as executable code.

## Algorithm 1: Hierarchical DDQN Training and Evaluation

```text
Input:
  Base UAV-backscatter environment E
  Hierarchical wrapper H with 10 high-level actions
  DDQN online network Q and target network Q-
  Replay buffer D
  Exploration schedule epsilon

Initialize:
  Wrap E with H
  Initialize Q and Q-
  Initialize replay buffer D

For each training episode:
  Reset H and receive initial observations

  For each frame t until terminal or max episode length:
    For each UAV u:
      Build state representation from global state and local observation
      Get high-level action mask from H
      Select high-level action z_u using masked epsilon-greedy DDQN

    H maps high-level actions z to primitive simulator actions
    E applies primitive actions and returns shared reward r, next observations, and metrics

    For each UAV u:
      Store transition (state_u, uav_id, z_u, r, next_state_u, done, mask_u, next_mask_u) in D

    If D contains enough transitions:
      Sample minibatch from D
      Compute Double-DQN target using Q and Q-
      Update Q by minimizing TD loss
      Periodically update Q-

  Log episode reward, throughput, drop rate, jamming failure, fairness, mode usage, and fallback diagnostics
  Periodically evaluate with epsilon = 0 and save best checkpoint

Output:
  Trained hierarchical DDQN policy and evaluation logs
```

Notes:

- In the implementation, hierarchical DDQN uses the same rule-based executor as QMIX.
- This algorithm is used as a strong sanity baseline, not the final MaDRL method.

## Algorithm 2: QMIX-Hierarchical Training

```text
Input:
  Base UAV-backscatter environment E
  Hierarchical wrapper H with 10 high-level actions
  Shared agent network Q_agent
  Target agent network Q_agent-
  QMIX mixer Q_mix
  Target mixer Q_mix-
  Episode replay buffer D
  Exploration schedule epsilon

Initialize:
  Wrap E with H
  Initialize Q_agent, Q_agent-, Q_mix, and Q_mix-
  Initialize episode replay buffer D

For each training episode:
  Reset H
  Initialize empty episode trajectory tau

  For each frame t until terminal or max episode length:
    For each UAV u:
      Obtain local observation o_u
      Obtain high-level action mask mask_u
      Compute Q_agent(o_u, agent_id_u)
      Select high-level action z_u using masked epsilon-greedy

    Execute joint high-level action z through H
    H maps z to primitive simulator actions
    E applies primitive actions and returns shared reward r
    Observe next local observations, next global state, next masks, done flag

    Append:
      observations, global state, actions, reward,
      next observations, next global state, done,
      action masks, next action masks
    to tau

  Store episode trajectory tau in D

  If D has at least min_replay_size episodes:
    Repeat updates_per_episode times:
      Sample an episode batch from D
      Use online agent network to select next masked actions
      Use target agent network and target mixer to evaluate next joint value
      Compute target y = r + gamma * (1 - done) * Q_tot_target
      Compute current Q_tot from selected agent Q-values and global state
      Minimize filled-mask MSE between Q_tot and y
      Periodically update target networks

  Periodically evaluate deterministically and save best checkpoint
  Log training reward, throughput, drop, jamming failure, fairness, epsilon, loss, Q_tot, mode usage, and fallback rate

Output:
  Trained QMIX-Hierarchical policy and evaluation logs
```

Notes:

- Final QMIX base uses updates_per_episode = 1.
- The shared reward is global; no local UAV reward is implemented.
- The mixer receives the global state only during training.

## Algorithm 3: QMIX-Hierarchical Execution and Evaluation

```text
Input:
  Trained QMIX agent network Q_agent
  Hierarchical wrapper H
  Evaluation episodes K

For each evaluation episode k = 1 to K:
  Reset H with evaluation seed

  For each frame t until terminal or max episode length:
    For each UAV u:
      Observe local observation o_u
      Obtain high-level action mask mask_u
      Compute Q_agent(o_u, agent_id_u)
      Select z_u = argmax over valid high-level actions

    H maps joint high-level actions z to primitive simulator actions
    E applies primitive actions
    Record reward and episode metrics

Aggregate across K episodes:
  reward
  throughput/frame
  drop rate
  jamming failure rate
  fairness
  energy efficiency
  backscatter success rate
  active success rate
  mode usage
  fallback rate

Output:
  Final evaluation CSV and aggregate metrics
```

Notes:

- Evaluation uses deterministic action selection, equivalent to epsilon = 0.
- The mixer is not required for greedy per-agent high-level action selection during evaluation, but it is part of the trained QMIX value-decomposition model.
- Primitive action execution still depends on the rule-based hierarchical executor.
