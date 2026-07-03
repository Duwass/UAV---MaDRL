from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde.evaluation import evaluate_decentralized_policy
from marl.ctde.networks import FactorizedActor
from marl.ctde.train_loop import train_ctde_smoke
from marl.ctde import train_loop as ctde_train_loop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ctde" / "ctde_3d_base.yaml"


def _tiny_config(seed: int = 123) -> dict:
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
            "epsilon": 0.1,
            "eval_episodes": 1,
            "eval_max_steps": 2,
        },
        "output": {
            "save_results": False,
            "save_checkpoint": False,
        },
    }


def _results_snapshot() -> set[str]:
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return set()
    return {str(path.relative_to(results_dir)) for path in results_dir.rglob("*")}


def test_ctde_smoke_train_loop_runs():
    summary = train_ctde_smoke(_tiny_config())
    assert summary["updates"] == 1
    assert summary["last_actor_loss"] is not None
    assert summary["last_critic_loss"] is not None


def test_ctde_smoke_losses_are_finite():
    summary = train_ctde_smoke(_tiny_config(seed=124))
    assert summary["losses_finite"]
    assert np.isfinite(summary["last_actor_loss"])
    assert np.isfinite(summary["last_critic_loss"])


def test_ctde_smoke_collects_transitions():
    summary = train_ctde_smoke(_tiny_config(seed=125))
    assert summary["transitions_collected"] > 0


def test_ctde_smoke_does_not_write_results_by_default():
    before = _results_snapshot()
    train_ctde_smoke(_tiny_config(seed=126))
    after = _results_snapshot()
    assert after == before


def test_ctde_config_loads():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["obs_dim"] == 114
    assert cfg["state_dim"] == 89
    assert cfg["output"]["save_results"] is False
    assert cfg["output"]["save_checkpoint"] is False


def test_train_script_smoke_imports():
    module = importlib.import_module("scripts.train_ctde_3d")
    assert hasattr(module, "main")


def test_train_script_smoke_cli_tmp():
    before = _results_snapshot()
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_ctde_3d.py"),
            "--config",
            str(CONFIG_PATH),
            "--smoke",
            "--seed",
            "127",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["updates"] > 0
    assert _results_snapshot() == before


def test_train_loop_does_not_import_hierarchical():
    source = inspect.getsource(ctde_train_loop)
    assert "hierarchical" not in source


def test_train_loop_does_not_use_env_action_mask():
    source = inspect.getsource(ctde_train_loop)
    assert "get_action_mask" not in source


def test_decentralized_evaluation_still_clean_after_smoke_training():
    train_ctde_smoke(_tiny_config(seed=128))
    env = UAVBackscatterEnv(CONFIG_PATH.parents[1] / "scenario_4_3d_backscatter_types_calibrated.yaml")
    actor = FactorizedActor()
    summary = evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2)
    assert summary["total_steps"] == 2


def test_existing_ctde_tests_still_pass():
    summary = train_ctde_smoke(_tiny_config(seed=129))
    assert summary["losses_finite"]
