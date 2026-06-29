# Research Storyline

## Motivation

UAV-assisted IoT networks are useful when static infrastructure is unavailable, damaged, or energy constrained. UAVs can move toward demand, collect data from distributed IoT devices, and adapt coverage dynamically. In RF-powered ambient backscatter communication, low-power IoT devices can also exploit existing RF signals instead of relying only on active transmission. This is attractive for long-lived IoT deployments, but it creates a harder control problem: the UAV must decide where to move, which device to serve, whether to use active or backscatter communication, and how to respond to jamming.

## Problem Gap

Traditional heuristic policies can be strong when they exploit one clear rule, such as nearest-device service, greedy SINR, active-only transmission, or backscatter-only transmission. However, heterogeneous RF-powered backscatter networks require simultaneous decisions over mobility, target selection, communication mode, energy state, queue pressure, fairness, and jamming risk. A flat action representation that directly enumerates movement x IoT target x communication mode becomes large and difficult to learn. In Scenario 4, the flat action dimension is 864.

The Phase 3 results show this gap empirically. Flat DDQN reaches throughput/frame 0.3242 in Scenario 4, close to HTT-only 0.3278 and far below greedy nearest 0.8977 and backscatter-only 0.8522. This suggests that simply applying a neural value function to the flat action interface is not enough.

## Why UAV-IoT with Mobile Jamming is Challenging

The task is challenging because the UAVs must solve several coupled problems at once:

- Mobility affects coverage, SINR, and jammer exposure.
- Target selection affects throughput, fairness, and queue stability.
- Backscatter, active transmission, harvest, and idle modes have different energy and reliability trade-offs.
- Device heterogeneity means the best mode can depend on IoT type and local state.
- A mobile/chase jammer changes the risk landscape over time.
- Multiple UAVs must coordinate implicitly through a shared network objective.

These factors make a purely greedy or single-mode policy brittle. They also make the flat action interface hard for DDQN, because many flat actions are redundant, invalid, or poor in a given state.

## Why MaDRL is Suitable

The problem is cooperative multi-agent control. Each UAV has local observations and local actions, but the network objective is shared. Multi-agent deep reinforcement learning is suitable because it can learn decentralized action choices while training with centralized information. QMIX is a natural first MaDRL method because:

- actions are discrete after hierarchical abstraction;
- the task is cooperative;
- agents share a global reward;
- centralized training can use the global state;
- decentralized execution can use local UAV observations;
- the mixing network can learn how individual UAV action-values combine into a team value.

## Proposed Model Contribution

The central contribution is not only the use of QMIX. The key design is the hierarchical/factorized action interface:

- Original Scenario 4 flat action dimension: 864.
- Hierarchical high-level action dimension: 10.
- A rule-based executor maps each high-level action to the original simulator action.
- High-level actions encode useful strategies such as best-SINR service, Type 2/3 backscatter priority, Type 1 active priority, low-energy harvesting, jammer avoidance, underserved-device balancing, and hybrid balanced service.

This action interface makes the learning problem tractable while preserving the original simulator and action semantics. Hierarchical DDQN validates the action abstraction, and QMIX then adds cooperative multi-agent value decomposition.

## How Phase 3 Results Support the Contribution

Phase 3 supports the research story in four steps.

First, calibrated baselines establish nontrivial Scenario 4 behavior. Random reaches throughput/frame 0.1075, HTT-only reaches 0.3278, greedy SINR reaches 0.4783, backscatter-only reaches 0.8522, and greedy nearest reaches 0.8977.

Second, flat DDQN exposes the action-interface bottleneck. It reaches throughput/frame 0.3242, with jamming failure 0.6093 and fairness 0.1930. Reward scaling and dueling stabilization did not solve the flat interface problem.

Third, hierarchical DDQN demonstrates that action abstraction is essential. With the 10-action interface, hierarchical DDQN reaches throughput/frame 0.9710, drop 0.4761, jamming failure 0.4403, and fairness 0.4754.

Fourth, QMIX-Hierarchical improves cooperative behavior. Across seeds 42, 43, and 44, QMIX base reaches throughput/frame 0.9604 +/- 0.0255, jamming failure 0.2056 +/- 0.0744, fairness 0.5260 +/- 0.0572, and drop 0.4744 +/- 0.0054. Although hierarchical DDQN has a slightly higher single-run throughput than the QMIX multi-seed mean, QMIX gives a stronger jamming/fairness trade-off and is the final recommended MaDRL setting.

The fairness ablation further supports the design. Fairness weighting did not improve mean fairness enough to replace QMIX base, and disabling BALANCE_UNDERSERVED_IOT reduced throughput, fairness, and jamming performance. Therefore, the final report should argue that the hierarchical action interface is necessary, and that QMIX with the base executor is the best current cooperative MaDRL result.
