# Final Report Outline

## 1. Introduction

- Introduce UAV-assisted IoT networks under intentional RF jamming.
- Explain why RF-powered ambient backscatter communication is relevant for low-power IoT devices.
- State the challenge: UAV mobility, heterogeneous IoT device types, energy constraints, backscatter/active mode selection, and mobile jamming create a coupled multi-agent decision problem.
- Motivate MaDRL as a way to learn cooperative UAV policies under shared network-level objectives.
- Summarize the final contribution:
  - calibrated UAV-backscatter anti-jamming simulation scenarios;
  - evidence that flat DDQN struggles in heterogeneous Scenario 4;
  - a 10-action hierarchical/factorized action interface;
  - QMIX with the hierarchical wrapper as the final MaDRL prototype;
  - multi-seed validation and fairness/coordination ablation.
- Preview the key result: QMIX base achieves throughput/frame 0.9604 +/- 0.0255 across seeds 42, 43, and 44 in Scenario 4.

## 2. Related Work

- UAV-assisted IoT data collection and UAV trajectory/control optimization.
- Anti-jamming communication and adaptive channel/mode selection.
- Ambient backscatter communication and RF-powered IoT.
- Deep reinforcement learning for wireless resource allocation.
- Multi-agent reinforcement learning for cooperative control.
- Value-decomposition methods such as VDN/QMIX for cooperative discrete-action tasks.
- Hierarchical or action-abstracted reinforcement learning for large combinatorial action spaces.
- Gap to emphasize: few works jointly address cooperative UAV control, jamming avoidance, active/backscatter/harvest decisions, and heterogeneous RF-powered IoT devices with a learnable MaDRL interface.

## 3. System Model

- Define network entities:
  - UAV agents;
  - IoT devices;
  - jammer;
  - primary signal/channel activity when relevant.
- Describe Scenario 4 as the main benchmark:
  - 2 UAVs;
  - 15 heterogeneous IoT devices;
  - calibrated mobile/chase jammer;
  - active, backscatter, harvest, and idle modes.
- Define the communication modes:
  - active transmission;
  - backscatter transmission;
  - harvesting/support behavior;
  - idle/no communication.
- Describe RF-powered device constraints:
  - queue state;
  - energy state;
  - device type differences.
- Describe mobility and coverage assumptions.
- Define channel quality, SINR, jamming failure, and primary busy behavior using the simulator equations or implementation definitions.
- Clarify what is modeled and what is abstracted away.

## 4. Problem Formulation

- Define the cooperative decision process:
  - agents are UAVs;
  - local observations per UAV;
  - global state for centralized training;
  - shared network reward;
  - discrete high-level action set for hierarchical experiments.
- Define the original flat action:
  - movement action x selected IoT x communication mode;
  - Scenario 4 flat dimension: 864.
- Define the hierarchical high-level action space:
  - 10 high-level actions;
  - rule-based executor maps high-level actions to original simulator actions.
- Define objective metrics:
  - throughput/frame;
  - drop rate;
  - jamming failure;
  - fairness;
  - energy efficiency;
  - backscatter success;
  - active success;
  - fallback rate for the hierarchical executor.
- State the optimization goal:
  - maximize cooperative network utility while reducing jamming/drop and maintaining service fairness.
- Include any reward equation used by DDQN/QMIX if the final paper exposes reward design.

## 5. Proposed MaDRL Method

- Present the hierarchical action interface:
  - motivation for reducing the action space from 864 to 10;
  - high-level action list;
  - executor behavior and fallback design;
  - action masking.
- Present the DDQN sanity baseline:
  - flat DDQN;
  - hierarchical DDQN;
  - why DDQN is used as a learning sanity check rather than the final MaDRL method.
- Present QMIX-Hierarchical:
  - one agent per UAV;
  - shared agent network;
  - local observation inputs;
  - global state input to mixer;
  - monotonic mixing network;
  - double Q-learning;
  - target network update;
  - epsilon-greedy action masking.
- Explain centralized training and decentralized execution.
- Explain why QMIX is suitable for cooperative UAV control with discrete high-level actions.
- Note final selected setting:
  - QMIX base;
  - hierarchical action wrapper;
  - BALANCE_UNDERSERVED_IOT enabled;
  - updates_per_episode=1.

## 6. Experimental Setup

- List scenarios and their purpose, using `results/publication_tables/table5_simulation_scenarios_metrics.*`.
- Describe calibrated baselines:
  - random;
  - HTT-only;
  - backscatter-only;
  - greedy SINR;
  - greedy nearest.
- Describe learning methods:
  - flat DDQN;
  - hierarchical DDQN;
  - QMIX-Hierarchical;
  - fairness ablation variants.
- Describe training/evaluation seeds:
  - QMIX base seeds 42, 43, 44;
  - note single-run status for hierarchical DDQN and flat DDQN unless additional runs are added.
- Describe metrics and interpretation direction.
- Mention hardware only if relevant:
  - RTX 3060 Laptop GPU used for later training phases;
  - environment/action execution still contains CPU-side components.
- Point to reproducibility artifacts:
  - `results/publication_artifacts_index.md`;
  - source CSVs in `results/csv/`;
  - generation command `python scripts\generate_publication_artifacts.py`.

## 7. Results and Discussion

- Main Scenario 4 comparison:
  - use Table 1 and Figure 1;
  - report random 0.1075, flat DDQN 0.3242, hierarchical DDQN 0.9710, QMIX base 0.9604 +/- 0.0255 throughput/frame.
- Flat DDQN limitation:
  - use Table 2 and Figure 3;
  - explain why the flat 864-action interface is difficult.
- Hierarchical action effect:
  - show the improvement from flat DDQN to hierarchical DDQN;
  - emphasize action abstraction rather than only network architecture.
- QMIX coordination effect:
  - use Table 1, Table 3, Figure 2, and Figure 4;
  - compare QMIX jamming failure 0.2056 +/- 0.0744 and fairness 0.5260 +/- 0.0572 against hierarchical DDQN jam 0.4403 and fairness 0.4754.
- Multi-seed validation:
  - use Table 3 and Figure 4;
  - discuss throughput stability and residual fairness variation.
- Fairness/coordination ablation:
  - use Table 4, Figure 5, and Figure 6;
  - explain why fair_w2/fair_w3 are not selected despite some metric improvements;
  - highlight that disabling BALANCE_UNDERSERVED_IOT hurts results.
- Discuss trade-offs:
  - throughput versus fairness;
  - jamming reduction versus energy efficiency;
  - heuristic specialization versus learned coordination.

## 8. Limitations

- Results are simulator-based and depend on calibrated scenario assumptions.
- Hierarchical executor includes rule-based domain knowledge; it is not purely learned end-to-end.
- QMIX base is multi-seed, but flat DDQN and hierarchical DDQN are single-run in the current publication package.
- Fairness remains moderate and below the earlier strong threshold.
- No final MAPPO experiment is included yet.
- Real RF channel dynamics, sensing errors, and hardware constraints may require additional validation.
- The final paper should avoid claiming universal superiority beyond the tested scenarios.

## 9. Conclusion and Future Work

- Conclude that the flat action interface is the main bottleneck in Scenario 4.
- Conclude that hierarchical action abstraction is essential for learnability.
- Conclude that QMIX-Hierarchical is the strongest current MaDRL setting due to stable throughput and improved jamming/fairness trade-off.
- State the final recommended result: QMIX base with BALANCE_UNDERSERVED_IOT enabled.
- Future work:
  - MAPPO or other policy-gradient MaDRL methods;
  - reward-level fairness shaping;
  - learned executor or hierarchical policy refinement;
  - more seeds for DDQN/hierarchical DDQN;
  - real-channel or hardware-in-the-loop validation;
  - broader jammer behavior and mobility ablations.
