# Experimental Results Summary

## 1. Overview

This package aggregates existing Phase 3 result CSV files into publication-ready tables and figures. It does not rerun training or modify simulator/training code.

## 2. Scenario 4 Main Comparison

Scenario 4 is the main heterogeneous RF-powered ambient backscatter benchmark. The key throughput/frame values are:

- Random: 0.1075
- HTT-only: 0.3278
- Backscatter-only: 0.8522
- Greedy SINR: 0.4783
- Greedy nearest: 0.8977
- Flat DDQN: 0.3242
- Hierarchical DDQN: 0.9710
- QMIX base: 0.9604 +/- 0.0255

## 3. Flat DDQN Limitation

Flat DDQN uses the full movement x IoT target x communication-mode action interface with 864 actions in Scenario 4. It reaches throughput/frame 0.3242, close to HTT-only and far below the strongest heuristic baselines. This supports the conclusion that the flat action interface is a bottleneck for heterogeneous backscatter control.

## 4. Hierarchical Action Interface Effect

The hierarchical executor reduces the Scenario 4 action dimension from 864 to 10 high-level actions. Hierarchical DDQN reaches throughput/frame 0.9710, exceeding the calibrated heuristic baselines in this result set. This is the main action-abstraction result.

## 5. QMIX Multi-agent Coordination Effect

QMIX reuses the 10-action hierarchical interface and applies cooperative value decomposition across UAV agents. Across seeds 42, 43, and 44, QMIX base obtains throughput/frame 0.9604 +/- 0.0255, jamming failure 0.2056 +/- 0.0744, fairness 0.5260 +/- 0.0572, and drop rate 0.4744 +/- 0.0054. Relative to hierarchical DDQN, QMIX improves the jamming/fairness trade-off while retaining high throughput.

## 6. Multi-seed Stability

QMIX base is stable across three seeds: throughput/frame std is 0.0255. The mean throughput remains above greedy nearest (0.8977) and backscatter-only (0.8522).

## 7. Fairness Ablation

The fairness ablation shows that naive executor fairness weighting does not improve mean fairness over QMIX base:

- QMIX base fairness: 0.5260, throughput/frame: 0.9604, jam: 0.2056
- fair_w2 fairness: 0.5111, throughput/frame: 0.9883, jam: 0.1601
- fair_w3 fairness: 0.4933, throughput/frame: 0.8877, jam: 0.3123
- no_balance fairness: 0.4767, throughput/frame: 0.8439, jam: 0.3352

Disabling BALANCE_UNDERSERVED_IOT hurts throughput, fairness, and jamming relative to QMIX base, supporting the value of the high-level balance action. QMIX base remains the final recommended setting.

## 8. Recommended Figures and Tables for Final Report

- Table 1 and Figure 1 for the main Scenario 4 comparison.
- Table 2 and Figure 3 for the research progression from flat DDQN to hierarchical DDQN to QMIX.
- Table 3 and Figure 4 for multi-seed QMIX stability.
- Table 4, Figure 5, and Figure 6 for the fairness/coordination ablation.
- Table 5 for experimental setup and metric definitions.

## 9. Key Claims Supported by Data

- Hierarchical action abstraction reduces action dimension from 864 to 10.
- Flat DDQN is insufficient for heterogeneous backscatter Scenario 4.
- Hierarchical DDQN beats the heuristic baselines in Scenario 4.
- QMIX base provides stable multi-seed performance.
- QMIX improves the jamming/fairness trade-off relative to hierarchical DDQN.
- Disabling BALANCE_UNDERSERVED_IOT hurts fairness and throughput.
- Naive executor fairness weighting does not improve mean fairness, so QMIX base remains the final setting.

## 10. Cautions / Limitations

- QMIX base is reported as a three-seed aggregate, while hierarchical DDQN and flat DDQN are single-seed final evaluations unless otherwise noted.
- Scenario 4 results are tied to the calibrated simulator and the fixed hierarchical executor semantics.
- Fairness remains moderate; improving per-IoT service balance may require reward-level fairness shaping or more explicit coordination objectives.
