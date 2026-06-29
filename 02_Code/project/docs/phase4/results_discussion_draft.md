# Results and Discussion Draft

This draft is based only on existing Phase 3 publication artifacts. It should be converted into final paper prose after citations, diagrams, and final captions are prepared.

## 1. Overall Comparison in Scenario 4

Table 1, `results/publication_tables/table1_overall_scenario4_comparison.md`, summarizes the main Scenario 4 comparison. The evaluated Scenario 4 setting uses 2 UAVs, 15 IoT devices, 1 mobile jammer, frame length 10, and episode length 200.

The throughput/frame results show three distinct regimes:

- Random is weak, with throughput/frame 0.1075.
- Specialized heuristics are nontrivial: HTT-only reaches 0.3278, greedy SINR reaches 0.4783, backscatter-only reaches 0.8522, and greedy nearest reaches 0.8977.
- Learned hierarchical methods are much stronger than flat DDQN: flat DDQN reaches 0.3242, hierarchical DDQN reaches 0.9710, and QMIX base reaches 0.9604 +/- 0.0255.

The flat DDQN result is close to HTT-only and far below greedy nearest and backscatter-only. This supports the interpretation that the primitive 864-action interface is difficult for the tested DDQN prototype in heterogeneous Scenario 4. The result should not be written as a universal claim that flat-action RL cannot work, but it is strong evidence that the implemented flat DDQN baseline is insufficient in this setting.

Hierarchical DDQN dramatically improves throughput/frame to 0.9710. Because the main architectural change is the 10-action high-level wrapper, the safe conclusion is that action abstraction is the key learnability improvement in Scenario 4.

QMIX base obtains throughput/frame 0.9604 +/- 0.0255, which is slightly below the hierarchical DDQN single-run throughput of 0.9710. Therefore, the report should not claim that QMIX mean throughput beats hierarchical DDQN. Instead, the key QMIX result is that it preserves high throughput while improving coordination-sensitive metrics: jamming failure and fairness.

## 2. Algorithm Progression

Table 2, `results/publication_tables/table2_algorithm_progression.md`, presents the progression from flat DDQN to hierarchical DDQN to QMIX-Hierarchical.

The progression is:

| Method | Action interface | Action dimension | Throughput/frame | Jamming failure | Fairness |
| --- | --- | --- | --- | --- | --- |
| Flat DDQN | Flat movement x IoT x mode | 864 | 0.3242 | 0.6093 | 0.1930 |
| Flat DDQN + tuning | Flat movement x IoT x mode | 864 | 0.1203 | 0.5557 | 0.1374 |
| Hierarchical DDQN | High-level executor | 10 | 0.9710 | 0.4403 | 0.4754 |
| QMIX-Hierarchical | High-level executor | 10 | 0.9604 | 0.2056 | 0.5260 |

The flat DDQN + tuning row shows that reward scaling and dueling did not solve the primitive interface bottleneck. It should be used as supporting evidence only, not as the main flat-DDQN comparison.

The strongest transition is from flat DDQN to hierarchical DDQN. The action dimension drops from 864 to 10, and throughput/frame increases from 0.3242 to 0.9710. This is the central action-abstraction result.

QMIX-Hierarchical then changes the learning structure from centralized-factorized DDQN to cooperative value decomposition. Its main advantage is not mean throughput over hierarchical DDQN, but better coordination behavior:

- jamming failure improves from hierarchical DDQN 0.4403 to QMIX base 0.2056 +/- 0.0744;
- fairness improves from hierarchical DDQN 0.4754 to QMIX base 0.5260 +/- 0.0572;
- drop rate is similar or slightly better, 0.4761 for hierarchical DDQN versus 0.4744 +/- 0.0054 for QMIX base.

This supports selecting QMIX base as the final MaDRL setting.

## 3. QMIX Multi-Seed Stability

Table 3, `results/publication_tables/table3_qmix_multiseed.md`, reports QMIX base across seeds 42, 43, and 44.

Per-seed results:

| Seed | Throughput/frame | Reward | Drop | Jam | Fairness | Energy efficiency |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.9945 | -1475.8774 | 0.4670 | 0.2988 | 0.5981 | 3.7427 |
| 43 | 0.9332 | -1481.4517 | 0.4796 | 0.2012 | 0.5217 | 5.4559 |
| 44 | 0.9537 | -1455.1210 | 0.4766 | 0.1167 | 0.4581 | 7.7980 |

Aggregate results:

- throughput/frame: 0.9604 +/- 0.0255;
- reward: -1470.8167 +/- 11.3294;
- drop rate: 0.4744 +/- 0.0054;
- jamming failure: 0.2056 +/- 0.0744;
- fairness: 0.5260 +/- 0.0572;
- energy efficiency: 5.6655 +/- 1.6622;
- backscatter success: 0.7935 +/- 0.0747;
- active success: 0.7833 +/- 0.1068;
- fallback rate: 0.0262 +/- 0.0287.

The throughput result is stable across the three seeds, with a standard deviation of 0.0255 and all three seeds above 0.93 throughput/frame. Jamming failure and fairness vary more than drop rate, so the report should describe these as improved but still variable coordination metrics.

The number of seeds is three. This is acceptable for a focused validation result, but the final report should not imply exhaustive robustness across all seeds or scenarios.

## 4. Fairness Ablation and BALANCE_UNDERSERVED_IOT

Table 4, `results/publication_tables/table4_qmix_fairness_ablation.md`, compares QMIX base with fairness-related executor ablations.

| Variant | Throughput/frame mean +/- std | Drop mean +/- std | Jamming failure mean +/- std | Fairness mean +/- std |
| --- | --- | --- | --- | --- |
| QMIX base | 0.9604 +/- 0.0255 | 0.4744 +/- 0.0054 | 0.2056 +/- 0.0744 | 0.5260 +/- 0.0572 |
| fair_w2 | 0.9883 +/- 0.0350 | 0.4771 +/- 0.0180 | 0.1601 +/- 0.0501 | 0.5111 +/- 0.1478 |
| fair_w3 | 0.8877 +/- 0.0094 | 0.4938 +/- 0.0113 | 0.3123 +/- 0.0711 | 0.4933 +/- 0.1880 |
| no_balance_action | 0.8439 +/- 0.0473 | 0.4926 +/- 0.0065 | 0.3352 +/- 0.0598 | 0.4767 +/- 0.0401 |

The fair_w2 variant improves throughput and jamming failure relative to QMIX base, but its mean fairness is lower and its fairness standard deviation is much larger. The fair_w3 variant degrades throughput, jamming failure, drop rate, and mean fairness relative to QMIX base. These results do not support replacing QMIX base with stronger executor fairness weighting.

Disabling BALANCE_UNDERSERVED_IOT reduces throughput from 0.9604 to 0.8439, reduces fairness from 0.5260 to 0.4767, and increases jamming failure from 0.2056 to 0.3352. This supports the claim that the balance action contributes to fairness-aware coordination behavior.

Important wording: BALANCE_UNDERSERVED_IOT is an action/executor mechanism, not a separate reward-shaping term. The environment reward already contains an unfairness penalty. The final report should say that the fairness-aware action/executor design supports coordination, while avoiding the claim that fairness is solved.

## 5. Figures and Visual Trends

Figure 1, `results/publication_figures/fig1_overall_throughput_comparison.png`, shows the overall throughput comparison. It belongs in the main Scenario 4 results subsection and supports the flat-to-hierarchical action-abstraction story.

Figure 2, `results/publication_figures/fig2_drop_jam_fairness_comparison.png`, compares drop, jamming, and fairness for selected methods. It should be used to support the claim that QMIX improves jamming/fairness coordination relative to hierarchical DDQN while preserving high throughput.

Figure 3, `results/publication_figures/fig3_algorithm_progression.png`, visualizes the progression from flat DDQN to hierarchical DDQN to QMIX. It belongs in the algorithm progression subsection and should be paired with Table 2.

Figure 4, `results/publication_figures/fig4_qmix_multiseed_mean_std.png`, shows QMIX base mean/std across seeds for throughput, jamming failure, fairness, and drop. It belongs in the multi-seed stability subsection.

Figure 5, `results/publication_figures/fig5_fairness_ablation_tradeoff.png`, shows the fairness-throughput trade-off among QMIX base and fairness ablation variants. It belongs in the fairness ablation subsection and should be interpreted as evidence that fair_w2 is not the best fairness trade-off despite higher throughput and lower jam.

Figure 6, `results/publication_figures/fig6_fairness_ablation_bars.png`, shows grouped fairness ablation bars. It should be used to illustrate that disabling BALANCE_UNDERSERVED_IOT hurts throughput, fairness, and jamming.

## 6. Discussion and Interpretation

The results support the research storyline in three steps.

First, the flat action interface is a bottleneck. Scenario 4 requires choosing movement, target IoT device, and communication mode. The evaluated flat interface has 864 primitive actions. Flat DDQN reaches only 0.3242 throughput/frame, which is close to HTT-only and much weaker than the best heuristic and hierarchical policies.

Second, hierarchical abstraction makes learning feasible. Reducing the learner's direct action space from 864 primitive actions to 10 high-level actions improves throughput/frame to 0.9710 for hierarchical DDQN. This suggests that the action abstraction captures useful domain structure for heterogeneous active/backscatter service under mobile jamming.

Third, QMIX adds cooperative value decomposition. QMIX base does not improve mean throughput over the hierarchical DDQN single run, but it provides stronger jamming and fairness behavior across three seeds. This is why QMIX base is selected as the final MaDRL setting.

The jamming/fairness/throughput trade-off remains important. Backscatter-only and greedy nearest are strong throughput baselines, but they have low fairness values in Table 1. QMIX base retains high throughput while substantially reducing jamming failure relative to hierarchical DDQN and increasing fairness. However, fairness remains moderate at 0.5260 +/- 0.0572, so fairness should be presented as improved, not solved.

## 7. Limitations

- The final evaluated Scenario 4 setting uses 2 UAVs, 15 IoT devices, and 1 mobile jammer. Claims should not be generalized to larger networks without additional experiments.
- Flat DDQN and hierarchical DDQN are single-run results in the publication artifacts, while QMIX base is a three-seed aggregate. Direct statistical comparison should be phrased cautiously.
- The hierarchical executor is rule-based and uses environment-level information to translate high-level actions into primitive simulator actions.
- Backscatter is implemented as a simulator abstraction using primary busy state and effective backscatter transmit-power factors; it should not be over-described as a full physical ambient-backscatter model.
- Fairness remains moderate, and stronger fairness weighting did not improve the final trade-off.
- MAPPO is not evaluated and should appear only as future work.
