# CTDE E4.2 Controlled Stability/Entropy Tuning Smoke

## 1. Scope

This is a controlled tuning smoke for the CTDE 3D path. It uses seed 42 only and a short horizon of 80 iterations x 30 rollout steps = 2400 collected transitions per variant.

This is not a final candidate rerun and is not sufficient for performance claims. It does not change reward terms, environment dynamics, actor/critic architecture, decentralized execution constraints, baselines, or action sanitizer behavior.

## 2. Variants

The training script does not expose CLI overrides for `normalize_advantage`, `entropy_coef`, or `max_grad_norm`, so temporary YAML configs were created under `results/ctde_stage_e42_configs/`. These files are run artifacts and should not be committed.

| Variant | normalize_advantage | max_grad_norm | entropy_coef | Purpose |
|---|---|---:|---:|---|
| neutral | false | null | 0.0 | Preserve neutral behavior while exporting E4.1 diagnostics. |
| stability | true | 1.0 | 0.0 | Inspect advantage normalization and gradient clipping diagnostics. |
| entropy_low | true | 1.0 | 0.001 | Inspect a small entropy coefficient with stability knobs. |
| entropy_mid | true | 1.0 | 0.005 | Inspect a larger controlled entropy coefficient with stability knobs. |

The `entropy_mid` value stayed at 0.005 because the expected entropy loss component is small relative to the observed smoke actor-loss scale.

## 3. Run Completion

| Variant | Exit code | Iterations | Transitions | Losses finite | Warning |
|---|---:|---:|---:|---|---|
| neutral | 0 | 80 | 2400 | true | null |
| stability | 0 | 80 | 2400 | true | null |
| entropy_low | 0 | 80 | 2400 | true | null |
| entropy_mid | 0 | 80 | 2400 | true | null |

Each run wrote `summary.json`, `metrics.jsonl`, `metrics.csv`, `config.yaml`, and `reproducibility.json`. Each `metrics.jsonl` file has 80 rows. No checkpoint file was present in the run directories.

## 4. Final Metrics and Diagnostics

| Variant | Eval return | Throughput/frame | Jamming failure | Fairness | Policy entropy | Movement entropy | Target entropy | Mode entropy | Advantage std | Actor grad norm | Critic grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | -0.626404 | 0.2 | 0.5 | 0.066667 | 6.959671 | 2.397563 | 2.770851 | 1.791258 | 2.360718 | 2.187141 | 111.212357 |
| stability | 1.525858 | 0.0 | 0.0 | 1.0 | 4.620594 | 1.533028 | 1.917848 | 1.169719 | 0.884118 | 5.865007 | 9.544075 |
| entropy_low | 1.525858 | 0.0 | 0.0 | 1.0 | 3.640624 | 1.224673 | 1.511651 | 0.904300 | 2.870542 | 22.982664 | 360.073853 |
| entropy_mid | 1.525858 | 0.0 | 0.0 | 1.0 | 4.651270 | 1.576142 | 1.891375 | 1.183754 | 0.950369 | 4.077834 | 3.148968 |

## 5. Final Action Concentration

| Variant | Eval top mode | Eval top movement | Eval no-target | Eval idle-mode | Eval sanitizer changed |
|---|---|---|---:|---:|---:|
| neutral | mode 2 (0.8) | movement 8 (0.5) | 0.0 | 0.0 | 0.0 |
| stability | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 | 0.0 |
| entropy_low | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 | 0.0 |
| entropy_mid | mode 5 (1.0) | movement 7 (1.0) | 0.0 | 0.0 | 0.0 |

## 6. Technical Observations Without Claim

All four variants completed with finite losses, no early stop, and no warning.

The neutral final row retained high policy entropy in the trainer diagnostics and had deterministic eval concentration at mode 2 and movement 8 in this short smoke.

The stability, entropy_low, and entropy_mid final rows had deterministic eval concentration at mode 5 and movement 7 in this short smoke. No final eval no-target, idle-mode, or sanitizer-change signal appeared in these final rows.

Entropy diagnostics differ across the nonzero entropy-coefficient variants. The entropy_low final row also had high actor and critic grad norm diagnostics compared with the other E4.2 final rows, so value/gradient scale should be inspected before using it for longer controlled runs.

These observations are diagnostic signals only. This smoke is not a CTDE-vs-baseline comparison and is not a final candidate result.

### First/Mid/Final Snapshot

| Variant | Row | Policy entropy | Advantage std | Actor grad norm | Critic grad norm | Eval top mode | Eval top movement | Eval throughput/frame |
|---|---:|---:|---:|---:|---:|---|---|---:|
| neutral | 1 | 6.956389 | 1.081306 | 2.628038 | 2.167944 | n/a | n/a | null |
| neutral | 40 | 6.955174 | 0.522858 | 1.324369 | 21.957479 | mode 5 (1.0) | movement 0 (0.6) | 0.0 |
| neutral | 80 | 6.959671 | 2.360718 | 2.187141 | 111.212357 | mode 2 (0.8) | movement 8 (0.5) | 0.2 |
| stability | 1 | 6.956389 | 1.081306 | 1.037668 | 2.167944 | n/a | n/a | null |
| stability | 40 | 5.034764 | 1.757861 | 4.519624 | 42.185898 | mode 5 (1.0) | movement 7 (1.0) | 0.0 |
| stability | 80 | 4.620594 | 0.884118 | 5.865007 | 9.544075 | mode 5 (1.0) | movement 7 (1.0) | 0.0 |
| entropy_low | 1 | 6.956389 | 1.081306 | 1.037678 | 2.167944 | n/a | n/a | null |
| entropy_low | 40 | 4.944225 | 1.757639 | 4.045315 | 42.207203 | mode 5 (1.0) | movement 7 (1.0) | 0.0 |
| entropy_low | 80 | 3.640624 | 2.870542 | 22.982664 | 360.073853 | mode 5 (1.0) | movement 7 (1.0) | 0.0 |
| entropy_mid | 1 | 6.956389 | 1.081306 | 1.037716 | 2.167944 | n/a | n/a | null |
| entropy_mid | 40 | 4.989006 | 2.659656 | 4.667893 | 96.137787 | mode 5 (1.0) | movement 7 (1.0) | 0.0 |
| entropy_mid | 80 | 4.651270 | 0.950369 | 4.077834 | 3.148968 | mode 5 (1.0) | movement 7 (1.0) | 0.0 |

## 7. Recommendation

Proceed to E4.3 multi-seed controlled diagnostic rerun with selected 1-2 variants only after the next stage prompt. Based on this smoke, reasonable candidates for E4.3 inspection are `stability` and `entropy_mid`, while `entropy_low` should be treated cautiously because its final gradient diagnostics were high in this single-seed smoke.

## 8. Claim Policy

No performance claim is made. Do not use this smoke as final evidence for any CTDE-vs-baseline ranking or final candidate statement. Do not use old 2D or hierarchical results as direct support for this CTDE stage.
