# CTDE 3D Controlled Evaluation Protocol

## 1. Purpose

This protocol defines the controlled evaluation path for the CTDE 3D runner after the Stage A/B/C technical checks. Its purpose is to evaluate stability, logging, reproducibility, and later baseline alignment before any performance claim is made.

This document is a protocol only. It does not authorize a full experiment, does not create a paper result, and does not claim that CTDE outperforms any baseline.

The main CTDE path is the direct 3D environment runner with decentralized actor execution and centralized critic training. The old hierarchical executor may be used only as a baseline/reference in a later aligned protocol; it must not be treated as the primary CTDE implementation.

## 2. Current Verified Foundation

- Branch: `phase-ctde-foundation`
- Foundation commit: `dced01947d2182c67a636f17b019bdd5e582d8a5`
- CTDE config: `configs/ctde/ctde_3d_base.yaml`
- Environment config: `configs/scenario_4_3d_backscatter_types_calibrated.yaml`
- `obs_dim`: `114`
- `state_dim`: `89`
- `action_dim`: `1056`
- Movement actions: `11`
- Target options: `16`
- Modes: `6`
- Local actor input: local observation only.
- Centralized critic input: global state only during training.
- Evaluation action selection: actor only, no critic, no global state, no environment action mask, no hierarchical executor.
- Default output policy: no file writes unless `--save-dir` or explicit output save config is provided.
- Default checkpoint policy: `output.save_checkpoint=false`; the current guarded runner does not write checkpoints by default.
- No-overwrite guard: existing save directories fail unless `--overwrite` is explicitly provided.

## 3. Completed Controlled Checks

- EXP-Checkpoint: `124 passed` as recorded in the Stage D handoff.
- Stage A sanity smoke: pass.
- Stage B short-run pilot, seed `42`: pass.
- Stage C multi-seed short-run, seeds `42`, `43`, `44`: pass.

These checks are technical pipeline checks only. Return values from Stage A/B/C are not performance results and must not be used as evidence that CTDE is better or worse than any baseline.

## 4. Evaluation Stages

### Stage D1 - Medium Stability Run

Purpose:

- Check numerical stability over a longer budget than the short-run checks.
- Verify the existing output bundle remains complete.
- Keep the run small enough that failures can be inspected quickly.

Suggested settings:

- seeds: `42`
- `num_iterations`: `50`
- `rollout_steps`: `20`
- `batch_size`: `32`
- `eval_every`: `10`
- save-dir: `results/ctde_stage_d1_medium_seed42`

Pass criteria:

- Exit code `0`.
- `losses_finite=true`.
- `stopped_early=false`.
- `warning=null`, or the warning is explicitly explained.
- `summary.json`, `metrics.jsonl`, `metrics.csv`, `config.yaml`, and `reproducibility.json` are present.
- No unexpected checkpoint is created.
- Repo cleanliness remains at the expected baseline plus allowed result output.

Claim policy:

- Do not claim performance.
- Allowed wording: "Stage D1 completed and remained numerically stable."

### Stage D2 - Multi-Seed Medium Run

Purpose:

- Check consistency and seed variance after D1 passes.

Suggested settings:

- seeds: `42`, `43`, `44`
- `num_iterations`: `50`
- `rollout_steps`: `20`
- `batch_size`: `32`
- `eval_every`: `10`

Save dirs:

- `results/ctde_stage_d2_medium_seed42`
- `results/ctde_stage_d2_medium_seed43`
- `results/ctde_stage_d2_medium_seed44`

Pass criteria:

- Same technical criteria as D1 for every seed.
- Metrics may differ across seeds; record variance without interpreting it as better or worse performance.

Claim policy:

- Allowed wording: "runs complete and numerically stable."
- Do not use wording such as "outperform", "improved", "better", or "publication-grade".

### Stage D3 - Full CTDE Candidate Run

Purpose:

- Produce a CTDE candidate result for behavior analysis after D1/D2 are accepted.

Suggested settings:

- seeds: `42`, `43`, `44`
- longer episode/iteration budget than the medium run
- a reasonable `eval_every` value
- a separate save directory per seed

Current runner limitations:

- The CLI currently supports `--config`, `--smoke`, `--seed`, `--save-dir`, `--overwrite`, `--num-iterations`, `--rollout-steps`, `--batch-size`, and `--eval-every`.
- The CLI does not currently expose `eval_episodes` or `eval_max_steps`; those values come from the config unless the runner is extended later.
- Full candidate evaluation should not invent unsupported flags. If richer evaluation budgets are needed, add or approve a runner/config change in a separate stage.

Pass criteria:

- Finite losses.
- No early stop.
- Complete logs and reproducibility bundle.
- No checkpoint unless explicitly enabled in a later approved prompt.

Claim policy:

- A full CTDE candidate run alone is not enough for performance claims.
- Performance claims require matched baseline runs under the same protocol.

### Stage D4 - Baseline Alignment / Rerun Design

Purpose:

- Prepare a fair comparison design before any CTDE-vs-baseline statement.

Baselines to consider:

- Random baseline on the same accepted 3D environment.
- Greedy or heuristic 3D baseline if one exists or is separately implemented.
- Old hierarchical wrapper only as a baseline/reference, not as the primary CTDE path.
- Old QMIX/hierarchical results only if they are rerun or aligned to the same 3D environment/action interface.

Rules:

- If a baseline is not ported to the accepted 3D interface, create a separate design before comparing.
- Do not directly compare new 3D CTDE metrics with old 2D/hierarchical results as a performance claim.
- Use matched seeds and comparable budgets.

## 5. Metrics to Collect

Currently exported by the guarded CTDE runner:

- `actor_loss`
- `critic_loss`
- `mean_value`
- `mean_target`
- `mean_advantage`
- `transitions_collected`
- `buffer_size`
- `losses_finite`
- `episode_return`
- `rollout_return`
- `eval_mean_return`
- `eval_total_steps`

Minimum evaluation metrics to collect or verify before final comparison:

- reward / eval return
- throughput - `needs implementation/export check`
- packet drop - `needs implementation/export check`
- jamming failure - `needs implementation/export check`
- fairness - `needs implementation/export check`
- energy efficiency if available in the environment - `needs implementation/export check`
- backscatter success - `needs implementation/export check`
- active success - `needs implementation/export check`
- mode usage if available - `needs implementation/export check`

3D-specific metrics to collect or verify:

- `avg_uav_altitude` - `needs implementation/export check`
- `min_uav_altitude` - `needs implementation/export check`
- `max_uav_altitude` - `needs implementation/export check`
- `vertical_action_rate` - `needs implementation/export check`
- `avg_vertical_movement` - `needs implementation/export check`
- `avg_uav_iot_3d_distance` - `needs implementation/export check`
- `avg_uav_jammer_3d_distance` - `needs implementation/export check`
- `altitude_boundary_hits` - `needs implementation/export check`

Policy diagnostics to collect or verify:

- movement action distribution - `needs implementation/export check`
- target no-target rate - `needs implementation/export check`
- target distribution over IoT options - `needs implementation/export check`
- mode distribution - `needs implementation/export check`
- sanitizer rate, if available - `needs implementation/export check`

Do not fabricate metrics that are not present in output files. If a metric is not exported, mark it as a missing export requirement.

## 6. Result Directory Convention

Use this naming convention:

```text
results/ctde_stage_<stage>_<description>_seed<seed>
```

Examples:

```text
results/ctde_stage_d1_medium_seed42
results/ctde_stage_d2_medium_seed43
```

Rules:

- Do not use `--overwrite` unless the user explicitly asks for it.
- If a directory already exists, create a timestamped directory for that seed.
- Do not stage or commit result directories.
- Do not write results without `--save-dir` or an explicitly configured save directory.

## 7. Reproducibility Requirements

Every approved run with saving enabled must contain:

- `summary.json`
- `metrics.jsonl`
- `metrics.csv`
- `config.yaml`
- `reproducibility.json`

`reproducibility.json` must include:

- `seed`
- `branch`
- `commit`
- `config_path`
- `obs_dim`
- `state_dim`
- `action_dim`
- `timestamp`

The current runner also writes `test_status: "not_run_in_script"`. If pre-run test status is needed as evidence, run and report tests separately before the experiment.

## 8. Checkpoint Policy

Current state:

- `output.save_checkpoint=false`
- The guarded runner does not write checkpoints by default.

Rules:

- Do not enable checkpointing in D1 or D2.
- Enable checkpointing for a full candidate run only if a later prompt explicitly approves it and gives a separate save directory policy.
- Never stage or commit checkpoint files.
- Treat any unexpected checkpoint in D1/D2 as a failure requiring investigation.

## 9. Claim Policy

Allowed claims:

- "CTDE 3D pipeline runs successfully."
- "Controlled checks pass."
- "Losses are finite in short/medium runs."
- "Outputs are reproducible/logged."
- "Runs complete and remain numerically stable."

Forbidden claims before matched baseline evaluation:

- "CTDE outperforms baseline."
- "Fairness improved."
- "Packet drop reduced."
- "Anti-jamming performance is better."
- "Publication-grade result."

Performance claims are allowed only after:

- 3D baselines are run under the same protocol.
- Multi-seed summaries include mean/std.
- Metrics are verified and exported.
- Old 2D/hierarchical results are not mixed with new 3D CTDE as direct evidence.

## 10. Git Hygiene

Always keep these rules:

- Do not stage the deleted PDF:
  - `01_Papers_and_Docs/hoangtrungdung_thesis (1).pdf`
- Do not delete or stage `context_windows_for_codex`.
- Do not stage:
  - `results/`
  - checkpoints
  - `__pycache__/`
  - `.pyc`
  - `.pytest_cache/`
  - temporary outputs
- If tracked `.pyc` files are modified by Python runtime, restore only the targeted tracked `.pyc` files and report it.
- Do not run broad clean commands over the repo.

## 11. Immediate Next Command Recommendation

Recommended next stage:

- Stage D1 medium stability run, seed `42`.
- Do not run D2 or D3 before D1 passes.

Suggested PowerShell command, not executed as part of Stage D:

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -B scripts/train_ctde_3d.py --config configs/ctde/ctde_3d_base.yaml --seed 42 --save-dir results/ctde_stage_d1_medium_seed42 --num-iterations 50 --rollout-steps 20 --batch-size 32 --eval-every 10
```

Expected validation after D1:

- exit code `0`
- `summary.json`, `metrics.jsonl`, `metrics.csv`, `config.yaml`, and `reproducibility.json` present
- no unexpected checkpoint
- repo status has no modified source or tracked `.pyc`
- no performance claim
