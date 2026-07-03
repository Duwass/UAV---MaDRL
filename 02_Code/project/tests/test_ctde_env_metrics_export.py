from __future__ import annotations

import csv
import json

from marl.ctde.experiment_io import save_metrics_csv, save_metrics_jsonl
from marl.ctde.train_loop import ENV_METRIC_KEYS, _extract_env_metrics, train_ctde_short_run


def _tiny_config(seed: int = 901) -> dict:
    return {
        "env_config_path": "configs/scenario_4_3d_backscatter_types_calibrated.yaml",
        "obs_dim": 114,
        "state_dim": 89,
        "actor_hidden_dim": 32,
        "critic_hidden_dim": 32,
        "actor_lr": 1.0e-3,
        "critic_lr": 1.0e-3,
        "gamma": 0.99,
        "entropy_coef": 0.0,
        "seed": seed,
        "replay_capacity": 16,
        "smoke": {
            "num_iterations": 1,
            "rollout_steps": 3,
            "batch_size": 2,
            "eval_every": 1,
            "eval_episodes": 1,
            "eval_max_steps": 2,
            "epsilon": 0.1,
        },
        "output": {
            "save_results": False,
            "save_checkpoint": False,
        },
    }


def test_train_loop_exports_env_metrics_in_rows_and_summary():
    summary = train_ctde_short_run(_tiny_config())

    row = summary["iteration_metrics"][0]
    expected_row_keys = {
        "rollout_total_throughput",
        "rollout_packet_drop_rate",
        "rollout_fairness_index",
        "rollout_avg_uav_altitude",
        "eval_total_throughput",
        "eval_packet_drop_rate",
        "eval_fairness_index",
        "eval_avg_uav_altitude",
    }
    assert expected_row_keys.issubset(row)
    assert row["rollout_avg_uav_altitude"] is not None
    assert row["eval_avg_uav_altitude"] is not None

    expected_summary_keys = {
        "rollout_total_throughput",
        "rollout_avg_uav_altitude",
        "eval_total_throughput",
        "eval_avg_uav_altitude",
    }
    assert expected_summary_keys.issubset(summary)
    assert summary["rollout_avg_uav_altitude"] is not None
    assert summary["eval_avg_uav_altitude"] is not None


def test_train_loop_preserves_existing_summary_fields():
    summary = train_ctde_short_run(_tiny_config(seed=902))

    assert summary["updates"] == 1
    assert summary["transitions_collected"] == 3
    assert summary["eval_mean_return"] is not None
    assert summary["losses_finite"] is True


def test_extract_env_metrics_handles_missing_values():
    missing = _extract_env_metrics({}, "rollout_")
    assert set(missing) == {f"rollout_{key}" for key in ENV_METRIC_KEYS}
    assert all(value is None for value in missing.values())

    mixed = _extract_env_metrics({"total_throughput": 3, "fairness_index": "bad"}, "eval_")
    assert mixed["eval_total_throughput"] == 3.0
    assert mixed["eval_fairness_index"] is None


def test_metrics_writers_accept_exported_env_metric_keys(tmp_path):
    rows = [
        {
            "iteration": 1,
            "eval_mean_return": 1.25,
            "rollout_total_throughput": 2.0,
            "eval_fairness_index": 0.5,
        }
    ]

    jsonl_path = save_metrics_jsonl(rows, tmp_path / "metrics.jsonl")
    csv_path = save_metrics_csv(rows, tmp_path / "metrics.csv")

    saved_row = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert saved_row["rollout_total_throughput"] == 2.0
    assert saved_row["eval_fairness_index"] == 0.5

    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "rollout_total_throughput" in header
    assert "eval_fairness_index" in header
