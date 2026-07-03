# CTDE Guarded Short-Run Runner

This runner is a guarded short-run utility for CTDE smoke/debug runs on the accepted 3D environment foundation. It is not a final experiment pipeline and must not be used to claim performance.

## Default Behavior

By default the runner writes no files.

Smoke command:

```bash
python scripts/train_ctde_3d.py --smoke
```

The command prints a summary JSON to stdout and does not create results, checkpoints, or plots.

## Saving Debug Outputs

Outputs are written only when a user provides a save directory or configures an explicit output save directory.

Example:

```bash
python scripts/train_ctde_3d.py --smoke --save-dir <path>
```

If `<path>` already exists, the command fails unless `--overwrite` is provided.

Example with overwrite:

```bash
python scripts/train_ctde_3d.py --smoke --save-dir <path> --overwrite
```

Overwrite allows writing the standard output files into an existing directory. It does not delete unrelated files in that directory.

## Output Files

When saving is enabled, the runner writes:

- `summary.json`
- `metrics.jsonl`
- `metrics.csv`
- `config.yaml`
- `reproducibility.json`

The reproducibility file includes branch, commit, config path, obs/state/action dimensions, seed, timestamp, and a script-local test status marker.

## Safety Rules

- No checkpoint is written by default.
- No old results directory is used by default.
- No Stage B or multi-seed experiment is run by this runner configuration.
- Actor action selection remains decentralized and uses local observations.
- The utility is for guarded debugging and short-run protocol checks only.
