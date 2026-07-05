# CTDE E3 Training/Tuning Design

## 1. Scope

Stage E3 is design-only. It does not change training code, reward logic, actor or critic behavior, environment behavior, configs, baselines, checkpoints, or result files.

The purpose is to choose a safe tuning path after the E2 action-diagnostic rerun. This document makes no performance claim. The current CTDE candidate remains a debugging and tuning target.

## 2. Inputs

Documents inspected:

- `docs/ctde_e0_candidate_diagnosis.md`
- `docs/ctde_e2_diagnostic_rerun.md`
- `docs/ctde_d4_candidate_vs_baselines.md`
- `docs/ctde_controlled_evaluation_protocol.md`

E2 diagnostic directories inspected:

- `results/ctde_stage_e2_diag_seed42`
- `results/ctde_stage_e2_diag_seed43`
- `results/ctde_stage_e2_diag_seed44`

Code and config inspected:

- `marl/ctde/ctde_trainer.py`
- `marl/ctde/train_loop.py`
- `marl/ctde/factorized_policy.py`
- `marl/ctde/networks.py`
- `marl/ctde/rollout.py`
- `marl/ctde/evaluation.py`
- `marl/ctde/action_diagnostics.py`
- `configs/ctde/ctde_3d_base.yaml`

Run references:

- Current pre-E3 commit: `0bf779f76465467c83bfda872181dc19224f0d47`
- E2 run commit: `6fb7a1629ed4fbed4c030c838d56a6bce8e8046f`
- Seeds: `42`, `43`, `44`
- Horizon per seed: `200 iterations x 50 rollout steps = 10000 transitions`
- Batch size: `64`
- Eval interval: every `20` iterations

## 3. E2 Diagnosis Summary

| Seed | Final eval return | Final eval throughput/frame | Final rollout throughput/frame | Final eval top mode | Final eval top movement | Final eval no-target | Final eval idle | Final eval sanitizer changed |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 42 | -0.460000 | 0.000000 | 0.000000 | 3 / 1.000000 | 0 / 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 43 | -0.654142 | 0.000000 | 0.000000 | 3 / 1.000000 | 8 / 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 44 | 0.849000 | 0.400000 | 0.220000 | 2 / 1.000000 | 2 / 1.000000 | 0.000000 | 0.000000 | 0.000000 |

Key diagnosis points:

- Throughput is sparse in seeds `42` and `43`; both final rollout and final deterministic eval have zero throughput/frame.
- No-target and idle mode are not the primary final-row cause; final eval no-target and idle rates are `0.0` for all three seeds.
- Sanitizer fallback is low in the final rows; final eval sanitizer-changed rate is `0.0` for all seeds, and final rollout rates are `0.03`, `0.02`, and `0.0`.
- Final evaluation is deterministic and concentrated: top eval mode rate and top eval movement rate are both `1.0` for all seeds.
- Rollout uses stochastic epsilon exploration (`epsilon = 0.1`) and has a wider action mix than deterministic eval, especially seed `43`.
- Value and target scale need inspection. Final mean value/target magnitudes are large in all three E2 seeds, and E2 documented mid-run critic-loss spikes in seeds `43` and `44`.

## 4. Tuning Objectives

Priority order:

1. Stabilize training and value scale.
2. Lessen deterministic eval concentration.
3. Strengthen factorized action coordination across movement, target, and mode heads.
4. Preserve CTDE/DE constraints.
5. Later consider fairness/drop reward shaping only after basic transmission behavior and training diagnostics are reliable.

## 5. Candidate Tuning Options

| Option | Target issue | Expected effect to inspect | Risk | Requires code change | Recommended priority |
|---|---|---|---|---|---|
| Advantage normalization | Actor update scale from large centralized value targets | Advantage mean/std, actor loss scale, finite-loss continuity | Medium: can alter learning dynamics | Yes | 1 |
| Gradient clipping | Actor/critic gradient spikes and value-scale instability | Actor/critic grad norms, critic loss spikes, finite losses | Low to medium: clipping threshold must be visible | Partly present; needs config/log export | 1 |
| Entropy regularization per head | Deterministic head concentration | Movement/target/mode entropy, top mode rate, top movement rate | Medium: too much entropy can mask learning signal | Yes | 2 |
| Entropy coefficient schedule | Early exploration pressure versus later exploitation | Entropy trend by head, rollout/eval action mix | Medium to high: schedule can hide the root cause | Yes | 3 |
| Stochastic diagnostic eval | Separate deterministic eval behavior from sampled actor behavior | Prefixed stochastic eval metrics, action concentration gap | Low to medium: diagnostic-only interpretation needed | Yes | 2 |
| Learning rate adjustment | Oscillating actor/critic loss and value scale | Loss trend, value/target magnitude, grad norms | Medium: config-only sweep can be noisy | Config only | 3 |
| More iterations / longer rollout | Sparse transmission signal and short candidate horizon | Event frequency, rollout throughput, eval throughput, loss scale | Medium: higher runtime and possible drift without stability checks | Config/run only | 4 |
| Reward normalization | Large value/target magnitude | Target scale, return scale, critic loss scale | Medium to high: changes reward semantics | Yes | Defer |
| Reward shaping for fairness/drop | Later downstream metrics | Only after basic transmission and action diagnostics stabilize | High: can confound current diagnosis | Yes | Defer |

## 6. Recommended Minimal E4 Plan

### E4.1 Training Stability and Entropy Instrumentation

Goals:

- Implement advantage normalization if absent, behind an explicit config knob.
- Use the existing gradient clipping hook safely, expose a config knob if needed, and export actor/critic grad norms in metrics.
- Log entropy per factorized head: movement, target, and mode.
- Log total policy entropy in `metrics.jsonl`, `metrics.csv`, and `summary.json`.
- Keep reward, environment, action space, and baseline code unchanged.
- Add focused tests for metric export and default behavior.
- Run only a tiny verification run after implementation.

Notes from code inspection:

- `CTDETrainer` already computes aggregate entropy and returns `mean_entropy`.
- `CTDETrainer` already has a `grad_clip_norm` path, but the current base config does not set it.
- `train_loop.py` currently exports actor loss, critic loss, mean value, mean target, and mean advantage, but not aggregate entropy or grad norms.
- Advantage normalization is not currently visible in `ctde_trainer.py`.

### E4.2 Entropy Regularization Experiment Support

Goals:

- Add or verify an entropy coefficient config path for factorized actor heads.
- Compute entropy from movement, target, and mode heads without using global state in actor execution.
- Keep default behavior conservative; any nonzero coefficient belongs to an approved tuning stage.
- Export the coefficient and entropy metrics so runs can be audited.

### E4.3 Stochastic Diagnostic Evaluation

Goals:

- Keep deterministic evaluation as the primary evaluation path.
- Add an optional diagnostic stochastic-eval path with a clear prefix, for example `stochastic_eval_*`.
- Do not base any claim solely on stochastic diagnostic eval.
- Do not alter decentralized execution constraints.

### E4.4 Tuning Smoke and Diagnostic Rerun

Suggested order:

1. Run a tiny smoke after E4.1 instrumentation.
2. Run a medium diagnostic rerun on seeds `42`, `43`, and `44`.
3. Compare against E2 only as an internal technical signal.
4. Do not run matched baseline comparison until the tuned candidate is stable enough for a controlled protocol stage.

Fairness/drop reward shaping should not start before E4.1 through E4.4 are complete and reviewed.

## 7. CTDE/DE Safety Constraints

These constraints must hold for all E4 work:

- Actor action selection uses local observation only.
- Centralized critic may use global state only during training updates.
- Evaluation action selection must not use critic values, global state, environment action masks, or the old hierarchical executor.
- Diagnostics must not feed back into action selection.
- No hierarchical executor may be introduced into the CTDE candidate path.
- No environment action mask may be used for CTDE action selection except the existing local movement boundary mask.
- Tuning must not introduce hidden global or environment information into actor inputs.
- Stochastic diagnostic eval, if added, must remain clearly separate from primary deterministic eval.

## 8. Evaluation Protocol After Tuning

After an approved E4 implementation:

- Run smoke first.
- Run diagnostic seeds `42`, `43`, and `44` under the same accepted 3D environment and action interface.
- Verify output bundle completeness: `summary.json`, `metrics.jsonl`, `metrics.csv`, `config.yaml`, and `reproducibility.json`.
- Verify no unexpected checkpoint output.
- Compare with D4.6 baselines only after the tuned candidate has completed the diagnostic protocol.
- Do not compare against old 2D or hierarchical results as direct evidence.
- Make no performance claim until aligned reruns and a claim policy are approved.

## 9. Recommended Next Stage

Recommended next stage:

```text
Stage E4.1 - CTDE Training Stability and Entropy Instrumentation
```

E4.1 goals:

- Implement advantage normalization if absent.
- Implement or expose gradient clipping safely if absent from active config.
- Log entropy per factorized head.
- Log total policy entropy.
- Expose config knobs safely.
- Add tests.
- Run a tiny verification run.
- Make no reward change.
- Make no performance claim.

## 10. Claim Policy

- E3 makes no performance claim.
- E3 only designs the tuning path.
- The current CTDE candidate remains a debugging and tuning target.
- Any future claim requires tuned CTDE, aligned baselines, verified metric export, matched seeds/budgets, and an approved claim policy.
