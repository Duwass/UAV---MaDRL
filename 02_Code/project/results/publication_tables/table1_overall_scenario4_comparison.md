| Method            | Type / Category     | Throughput/frame  | Drop rate         | Jamming failure   | Fairness          | Energy efficiency | Backscatter success | Active success    |
| ----------------- | ------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------------------- | ----------------- |
| Random            | Stochastic baseline | 0.1075            | 0.6084            | 0.6232            | 0.3530            | 0.3242            | 0.3817              | 0.3640            |
| HTT-only          | Heuristic baseline  | 0.3278            | 0.5823            | 0.4891            | 0.0937            | 0.4029            | 0.0000              | 0.4788            |
| Backscatter-only  | Heuristic baseline  | 0.8522            | 0.5294            | 0.5515            | 0.1719            | 10.7016           | 0.4258              | 0.0000            |
| Greedy SINR       | Heuristic baseline  | 0.4783            | 0.5575            | 0.7806            | 0.2385            | 1.7265            | 0.1879              | 0.2734            |
| Greedy nearest    | Heuristic baseline  | 0.8977            | 0.5222            | 0.5463            | 0.1836            | 7.2350            | 0.4302              | 0.6019            |
| Flat DDQN         | Flat DRL            | 0.3242            | 0.5846            | 0.6093            | 0.1930            | 0.9417            | 0.3899              | 0.5231            |
| Hierarchical DDQN | Hierarchical DRL    | 0.9710            | 0.4761            | 0.4403            | 0.4754            | 4.9295            | 0.5258              | 0.6817            |
| QMIX base         | MaDRL, hierarchical | 0.9604 +/- 0.0255 | 0.4744 +/- 0.0054 | 0.2056 +/- 0.0744 | 0.5260 +/- 0.0572 | 5.6655 +/- 1.6622 | 0.7935 +/- 0.0747   | 0.7833 +/- 0.1068 |
