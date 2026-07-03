from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.ctde.experiment_io import (
    prepare_run_dir,
    save_config_copy,
    save_metrics_csv,
    save_metrics_jsonl,
    save_reproducibility_info,
    save_summary_json,
)
from marl.ctde.train_loop import DEFAULT_CONFIG_PATH, train_ctde_short_run


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run a minimal CTDE 3D smoke training loop.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to CTDE smoke YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Force tiny smoke settings.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--save-dir", default=None, help="Optional user-provided output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing save directory.")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override smoke.num_iterations.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override smoke.rollout_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override smoke.batch_size.")
    parser.add_argument("--eval-every", type=int, default=None, help="Override smoke.eval_every.")
    args = parser.parse_args(argv)

    config_path = _resolve_config_path(args.config)
    config = _load_config(config_path)
    _apply_cli_overrides(config, args)
    run_dir = _prepare_output_dir(config, args)

    summary = train_ctde_short_run(config)
    output_files = _write_outputs(config, summary, config_path, run_dir)
    if output_files:
        summary = dict(summary)
        summary["output_files"] = output_files
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _resolve_config_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    smoke = config.setdefault("smoke", {})
    if args.smoke:
        smoke.update(
            {
                "num_iterations": 2,
                "rollout_steps": 5,
                "batch_size": 4,
                "eval_every": 1,
                "eval_episodes": 1,
                "eval_max_steps": 5,
            }
        )
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.num_iterations is not None:
        smoke["num_iterations"] = int(args.num_iterations)
    if args.rollout_steps is not None:
        smoke["rollout_steps"] = int(args.rollout_steps)
    if args.batch_size is not None:
        smoke["batch_size"] = int(args.batch_size)
    if args.eval_every is not None:
        smoke["eval_every"] = int(args.eval_every)


def _prepare_output_dir(config: dict, args: argparse.Namespace) -> Path | None:
    output_cfg = dict(config.get("output", {}))
    save_dir = args.save_dir or output_cfg.get("save_dir")
    save_requested = args.save_dir is not None or bool(output_cfg.get("save_results", False))
    if not save_requested:
        return None
    if save_dir is None:
        raise ValueError("output.save_results=true requires output.save_dir or --save-dir.")
    return prepare_run_dir(save_dir, overwrite=bool(args.overwrite or output_cfg.get("overwrite", False)))


def _write_outputs(config: dict, summary: dict, config_path: Path, run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []
    output_cfg = dict(config.get("output", {}))
    files: list[str] = []
    files.append(str(save_summary_json(summary, run_dir / "summary.json")))
    metrics = list(summary.get("iteration_metrics", []))
    if bool(output_cfg.get("save_metrics_jsonl", True)):
        files.append(str(save_metrics_jsonl(metrics, run_dir / "metrics.jsonl")))
    if bool(output_cfg.get("save_metrics_csv", True)):
        files.append(str(save_metrics_csv(metrics, run_dir / "metrics.csv")))
    if bool(output_cfg.get("save_config", True)):
        files.append(str(save_config_copy(config, run_dir / "config.yaml")))
    if bool(output_cfg.get("save_reproducibility", True)):
        info = _reproducibility_info(config, summary, config_path)
        files.append(str(save_reproducibility_info(info, run_dir / "reproducibility.json")))
    return files


def _reproducibility_info(config: dict, summary: dict, config_path: Path) -> dict:
    return {
        "branch": _git_value(["git", "branch", "--show-current"]),
        "commit": _git_value(["git", "rev-parse", "HEAD"]),
        "config_path": str(config_path),
        "obs_dim": int(config.get("obs_dim", summary.get("obs_dim", 114))),
        "state_dim": int(config.get("state_dim", summary.get("state_dim", 89))),
        "action_dim": int(summary.get("action_dim", 1056)),
        "seed": int(config.get("seed", summary.get("seed", 0))),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_status": "not_run_in_script",
    }


def _git_value(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, cwd=ROOT.parents[1], check=True, capture_output=True, text=True)
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


if __name__ == "__main__":
    main()
