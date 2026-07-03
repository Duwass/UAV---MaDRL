from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from marl.ctde import experiment_io
from marl.ctde.experiment_io import prepare_run_dir
from marl.ctde.train_loop import train_ctde_short_run
from marl.ctde import train_loop as ctde_train_loop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ctde" / "ctde_3d_base.yaml"


def _tiny_config(seed: int = 701) -> dict:
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
            "num_iterations": 2,
            "rollout_steps": 3,
            "batch_size": 2,
            "eval_every": 0,
            "eval_episodes": 1,
            "eval_max_steps": 2,
            "epsilon": 0.1,
        },
        "output": {
            "save_results": False,
            "save_checkpoint": False,
            "overwrite": False,
        },
    }


def _results_snapshot() -> set[str]:
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return set()
    return {str(path.relative_to(results_dir)) for path in results_dir.rglob("*")}


def _script_command(*args: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "train_ctde_3d.py"), *args]


def test_train_loop_returns_iteration_metrics():
    summary = train_ctde_short_run(_tiny_config())

    assert summary["updates"] == 2
    assert len(summary["iteration_metrics"]) == 2
    row = summary["iteration_metrics"][0]
    required = {
        "iteration",
        "transitions_collected",
        "buffer_size",
        "actor_loss",
        "critic_loss",
        "mean_value",
        "mean_target",
        "mean_advantage",
        "losses_finite",
        "episode_return",
    }
    assert required.issubset(row)


def test_train_loop_eval_every_runs():
    cfg = _tiny_config(seed=702)
    cfg["smoke"]["eval_every"] = 1
    summary = train_ctde_short_run(cfg)

    assert summary["iteration_metrics"][0]["eval_mean_return"] is not None
    assert summary["iteration_metrics"][0]["eval_total_steps"] > 0
    assert summary["eval_total_steps"] > 0


def test_train_loop_no_save_dir_writes_no_files(tmp_path):
    before_results = _results_snapshot()
    before_tmp = set(tmp_path.iterdir())

    train_ctde_short_run(_tiny_config(seed=703))

    assert _results_snapshot() == before_results
    assert set(tmp_path.iterdir()) == before_tmp


def test_prepare_run_dir_no_overwrite(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()

    with pytest.raises(FileExistsError):
        prepare_run_dir(run_dir, overwrite=False)


def test_prepare_run_dir_overwrite_allowed_or_clean_behavior(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    marker = run_dir / "keep.txt"
    marker.write_text("kept", encoding="utf-8")

    prepared = prepare_run_dir(run_dir, overwrite=True)

    assert prepared == run_dir
    assert marker.exists()


def test_script_no_save_dir_prints_summary_only():
    before_results = _results_snapshot()
    completed = subprocess.run(
        _script_command(
            "--config",
            str(CONFIG_PATH),
            "--smoke",
            "--seed",
            "704",
            "--num-iterations",
            "1",
            "--rollout-steps",
            "2",
            "--batch-size",
            "2",
            "--eval-every",
            "1",
        ),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["updates"] == 1
    assert "iteration_metrics" in summary
    assert _results_snapshot() == before_results


def test_script_save_dir_writes_expected_files_tmp(tmp_path):
    save_dir = tmp_path / "ctde_run"
    completed = subprocess.run(
        _script_command(
            "--config",
            str(CONFIG_PATH),
            "--smoke",
            "--seed",
            "705",
            "--num-iterations",
            "1",
            "--rollout-steps",
            "2",
            "--batch-size",
            "2",
            "--eval-every",
            "1",
            "--save-dir",
            str(save_dir),
        ),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["updates"] == 1
    assert (save_dir / "summary.json").exists()
    assert (save_dir / "metrics.jsonl").exists()
    assert (save_dir / "metrics.csv").exists()
    assert (save_dir / "config.yaml").exists()
    assert (save_dir / "reproducibility.json").exists()


def test_script_existing_save_dir_without_overwrite_fails(tmp_path):
    save_dir = tmp_path / "existing"
    save_dir.mkdir()

    completed = subprocess.run(
        _script_command("--config", str(CONFIG_PATH), "--smoke", "--save-dir", str(save_dir)),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not (save_dir / "summary.json").exists()


def test_guarded_runner_does_not_import_hierarchical():
    script = importlib.import_module("scripts.train_ctde_3d")
    sources = [
        inspect.getsource(ctde_train_loop),
        inspect.getsource(script),
        inspect.getsource(experiment_io),
    ]
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for source in sources for item in forbidden)


def test_guarded_runner_does_not_use_env_action_mask():
    script = importlib.import_module("scripts.train_ctde_3d")
    sources = [
        inspect.getsource(ctde_train_loop),
        inspect.getsource(script),
        inspect.getsource(experiment_io),
    ]
    assert all("get_action_mask" not in source for source in sources)


def test_existing_ctde_smoke_tests_still_pass():
    summary = train_ctde_short_run(_tiny_config(seed=706))
    assert summary["losses_finite"]
