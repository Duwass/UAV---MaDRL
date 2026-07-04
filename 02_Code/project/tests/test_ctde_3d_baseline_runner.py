from __future__ import annotations

import csv
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde import baseline_evaluation
from marl.ctde.baseline_evaluation import run_baseline_evaluation, select_actions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
CRITICAL_EVAL_METRICS = {
    "eval_avg_throughput_per_frame",
    "eval_packet_drop_rate",
    "eval_fairness_index",
    "eval_avg_uav_altitude",
}


@pytest.mark.parametrize("policy", ["random", "idle", "nearest"])
def test_baseline_tiny_run_writes_output_bundle(policy: str, tmp_path: Path):
    save_dir = tmp_path / policy

    summary = run_baseline_evaluation(
        env_config_path=ENV_CONFIG,
        policy=policy,
        seed=42,
        save_dir=save_dir,
        num_iterations=2,
        steps_per_iteration=3,
        test_status="pytest",
    )

    expected_files = {
        "summary.json",
        "metrics.jsonl",
        "metrics.csv",
        "config.yaml",
        "reproducibility.json",
    }
    assert expected_files == {path.name for path in save_dir.iterdir()}
    assert len(summary["output_files"]) == 5

    saved_summary = json.loads((save_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["policy"] == policy
    assert saved_summary["seed"] == 42
    assert saved_summary["total_steps"] == 6
    assert CRITICAL_EVAL_METRICS.issubset(saved_summary)
    assert saved_summary["warning"] is None

    jsonl_rows = (save_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_rows) == 2
    row = json.loads(jsonl_rows[-1])
    assert row["policy"] == policy
    assert CRITICAL_EVAL_METRICS.issubset(row)

    with (save_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert CRITICAL_EVAL_METRICS.issubset(set(header))

    repro = json.loads((save_dir / "reproducibility.json").read_text(encoding="utf-8"))
    assert repro["policy"] == policy
    assert repro["seed"] == 42
    assert repro["branch"]
    assert repro["commit"]
    assert repro["obs_dim"] == 114
    assert repro["state_dim"] == 89
    assert repro["action_dim"] == 1056
    assert repro["test_status"] == "pytest"


def test_baseline_runner_no_save_dir_writes_no_files(tmp_path: Path):
    before = set(tmp_path.iterdir())

    summary = run_baseline_evaluation(
        env_config_path=ENV_CONFIG,
        policy="idle",
        seed=43,
        save_dir=None,
        num_iterations=1,
        steps_per_iteration=2,
    )

    assert summary["output_files"] == []
    assert set(tmp_path.iterdir()) == before


def test_baseline_runner_no_overwrite_guard(tmp_path: Path):
    save_dir = tmp_path / "existing"
    save_dir.mkdir()

    with pytest.raises(FileExistsError):
        run_baseline_evaluation(
            env_config_path=ENV_CONFIG,
            policy="random",
            seed=44,
            save_dir=save_dir,
            num_iterations=1,
            steps_per_iteration=2,
        )

    assert not (save_dir / "summary.json").exists()


def test_baseline_policy_actions_are_in_valid_range():
    env = UAVBackscatterEnv(ENV_CONFIG)
    try:
        observations, _ = env.reset(seed=45)
        rng = np.random.default_rng(45)
        for policy in ("random", "idle", "nearest"):
            actions = select_actions(policy, observations, env, rng)
            assert len(actions) == env.num_uav
            assert all(0 <= int(action) < env.action_size for action in actions)
    finally:
        env.close()


def test_baseline_runner_does_not_import_wrapper_executor():
    cli = importlib.import_module("scripts.evaluate_ctde_3d_baselines")
    sources = [inspect.getsource(baseline_evaluation), inspect.getsource(cli)]
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for source in sources for item in forbidden)
