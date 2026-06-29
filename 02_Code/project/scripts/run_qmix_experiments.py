from __future__ import annotations

import argparse
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from marl.qmix.qmix_trainer import QMIXTrainer


SUMMARY_COLUMNS = [
    "config_name",
    "seed",
    "output_prefix",
    "status",
    "runtime_sec",
    "checkpoint_best",
    "checkpoint_latest",
    "train_log_path",
    "eval_log_path",
    "final_eval_path",
    "final_reward",
    "final_throughput_per_frame",
    "final_drop",
    "final_jam",
    "final_fairness",
    "final_energy_efficiency",
    "final_backscatter_success",
    "final_active_success",
    "final_mode_harvest",
    "final_mode_backscatter",
    "final_mode_active",
    "final_fallback_rate",
    "best_eval_throughput_per_frame",
    "best_eval_episode",
    "first50_loss_mean",
    "last50_loss_mean",
    "first50_train_throughput_mean",
    "last50_train_throughput_mean",
    "error",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run_config(base_config: dict[str, Any], config_name: str, seed: int, episodes: int | None = None) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("training", {})["seed"] = int(seed)
    if episodes is not None:
        config["training"]["total_episodes"] = int(episodes)
    base_prefix = str(config.setdefault("logging", {}).get("output_prefix", config_name))
    config["logging"]["output_prefix"] = f"{base_prefix}_seed{int(seed)}"
    return config


def write_run_config(config: dict[str, Any], output_prefix: str) -> Path:
    path = ROOT / "results" / "configs" / f"{output_prefix}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def summarize_logs(config_name: str, seed: int, output_prefix: str, runtime_sec: float, status: str, error: str = "") -> dict[str, Any]:
    csv_dir = ROOT / "results" / "csv"
    checkpoint_dir = ROOT / "results" / "checkpoints"
    train_path = csv_dir / f"{output_prefix}_train_log.csv"
    eval_path = csv_dir / f"{output_prefix}_eval_log.csv"
    final_path = csv_dir / f"{output_prefix}_final_eval.csv"
    best_path = checkpoint_dir / f"{output_prefix}_best.pt"
    latest_path = checkpoint_dir / f"{output_prefix}_latest.pt"
    row: dict[str, Any] = {
        "config_name": config_name,
        "seed": int(seed),
        "output_prefix": output_prefix,
        "status": status,
        "runtime_sec": float(runtime_sec),
        "checkpoint_best": str(best_path) if best_path.exists() else "",
        "checkpoint_latest": str(latest_path) if latest_path.exists() else "",
        "train_log_path": str(train_path) if train_path.exists() else "",
        "eval_log_path": str(eval_path) if eval_path.exists() else "",
        "final_eval_path": str(final_path) if final_path.exists() else "",
        "error": error,
    }

    final_df = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    row.update(_final_metrics(final_df))
    row.update(_eval_metrics(eval_df))
    row.update(_train_metrics(train_df))
    return {column: row.get(column, "") for column in SUMMARY_COLUMNS}


def _mean(df: pd.DataFrame, key: str) -> float:
    return float(df[key].mean()) if key in df.columns and not df.empty else 0.0


def _final_metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "final_reward": _mean(df, "total_reward"),
        "final_throughput_per_frame": _mean(df, "avg_throughput_per_frame"),
        "final_drop": _mean(df, "packet_drop_rate"),
        "final_jam": _mean(df, "jamming_failure_rate"),
        "final_fairness": _mean(df, "fairness_index"),
        "final_energy_efficiency": _mean(df, "energy_efficiency"),
        "final_backscatter_success": _mean(df, "backscatter_success_rate"),
        "final_active_success": _mean(df, "active_success_rate"),
        "final_mode_harvest": _mean(df, "mode_usage_harvest"),
        "final_mode_backscatter": _mean(df, "mode_usage_backscatter"),
        "final_mode_active": _mean(df, "mode_usage_active"),
        "final_fallback_rate": _mean(df, "fallback_rate"),
    }


def _eval_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "eval_avg_throughput_per_frame" not in df.columns:
        return {"best_eval_throughput_per_frame": 0.0, "best_eval_episode": 0}
    idx = df["eval_avg_throughput_per_frame"].idxmax()
    return {
        "best_eval_throughput_per_frame": float(df.loc[idx, "eval_avg_throughput_per_frame"]),
        "best_eval_episode": int(df.loc[idx, "episode"]) if "episode" in df.columns else 0,
    }


def _train_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "first50_loss_mean": 0.0,
            "last50_loss_mean": 0.0,
            "first50_train_throughput_mean": 0.0,
            "last50_train_throughput_mean": 0.0,
        }
    nonzero_loss = df[df.get("avg_loss", 0.0) > 0.0] if "avg_loss" in df.columns else pd.DataFrame()
    return {
        "first50_loss_mean": float(nonzero_loss.head(50)["avg_loss"].mean()) if not nonzero_loss.empty else 0.0,
        "last50_loss_mean": float(nonzero_loss.tail(50)["avg_loss"].mean()) if not nonzero_loss.empty else 0.0,
        "first50_train_throughput_mean": float(df.head(50)["train_throughput"].mean() / 200.0)
        if "train_throughput" in df.columns
        else 0.0,
        "last50_train_throughput_mean": float(df.tail(50)["train_throughput"].mean() / 200.0)
        if "train_throughput" in df.columns
        else 0.0,
    }


def save_summary(rows: list[dict[str, Any]], append: bool, output: str | Path | None = None) -> Path:
    path = Path(output) if output is not None else ROOT / "results" / "csv" / "qmix_experiment_summary.csv"
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if append and path.exists():
        old_df = pd.read_csv(path)
        out = pd.concat([old_df, new_df], ignore_index=True)
        out = out.drop_duplicates(subset=["config_name", "seed", "output_prefix"], keep="last")
    else:
        out = new_df
    out.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QMIX multi-seed experiments.")
    parser.add_argument("--configs", nargs="+", required=True, help="QMIX config YAML files.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Seeds to run.")
    parser.add_argument("--episodes", type=int, default=None, help="Override training.total_episodes.")
    parser.add_argument("--append", action="store_true", help="Append to existing experiment summary.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "csv" / "qmix_experiment_summary.csv"),
        help="Output summary CSV path.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for config_arg in args.configs:
        config_path = Path(config_arg)
        config_name = config_path.stem
        base_config = load_yaml(config_path)
        for seed in args.seeds:
            run_config = build_run_config(base_config, config_name, seed, args.episodes)
            output_prefix = str(run_config["logging"]["output_prefix"])
            run_config_path = write_run_config(run_config, output_prefix)
            start = time.time()
            status = "success"
            error = ""
            try:
                trainer = QMIXTrainer(run_config_path)
                trainer.train()
                best_checkpoint = ROOT / "results" / "checkpoints" / f"{output_prefix}_best.pt"
                final_df = trainer.evaluate_episodes(int(run_config["training"].get("eval_episodes", 30)), checkpoint=best_checkpoint)
                final_path = ROOT / "results" / "csv" / f"{output_prefix}_final_eval.csv"
                final_path.parent.mkdir(parents=True, exist_ok=True)
                final_df.to_csv(final_path, index=False)
            except Exception as exc:  # pragma: no cover - exercised only on failed experiment runs.
                status = "failed"
                error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"FAILED config={config_name} seed={seed}: {error}")
                traceback.print_exc()
            runtime = time.time() - start
            row = summarize_logs(config_name, seed, output_prefix, runtime, status, error)
            rows.append(row)
            save_summary(rows, append=args.append, output=args.output)
            print(
                f"finished config={config_name} seed={seed} status={status} "
                f"runtime_sec={runtime:.1f} throughput={row.get('final_throughput_per_frame', 0.0)}"
            )
    summary_path = save_summary(rows, append=args.append, output=args.output)
    print(f"Saved QMIX experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
