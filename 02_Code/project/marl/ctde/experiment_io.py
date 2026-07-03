from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def prepare_run_dir(save_dir: str | Path | None, overwrite: bool = False) -> Path | None:
    if save_dir is None:
        return None
    run_dir = Path(save_dir)
    if run_dir.exists() and not bool(overwrite):
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_summary_json(summary: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return output


def save_metrics_jsonl(metrics: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in metrics:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return output


def save_metrics_csv(metrics: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in metrics for key in row})
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    return output


def save_config_copy(config: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output


def save_reproducibility_info(info: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    return output
