from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde import action_diagnostics, evaluation as ctde_evaluation, rollout as ctde_rollout, train_loop as ctde_train_loop
from marl.ctde.action_diagnostics import prefix_action_diagnostics, summarize_action_diagnostics
from marl.ctde.evaluation import evaluate_decentralized_policy
from marl.ctde.networks import FactorizedActor
from marl.ctde.rollout import collect_ctde_rollout
from marl.ctde.train_loop import train_ctde_short_run
from marl.ctde.utils import (
    MODE_ACTIVE,
    MODE_BACKSCATTER,
    MODE_HARVEST,
    MODE_IDLE,
    MOVE_DOWN,
    MOVE_UP,
    FactorizedAction,
    sanitize_factorized_action,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"


def _tiny_config(seed: int = 1101) -> dict:
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


def test_action_diagnostics_helper_rates_from_factorized_actions():
    raw = [
        FactorizedAction(0, 0, MODE_IDLE),
        FactorizedAction(MOVE_UP, 0, MODE_BACKSCATTER),
        FactorizedAction(MOVE_DOWN, 2, MODE_ACTIVE),
        FactorizedAction(3, 1, MODE_HARVEST),
    ]
    sanitized = [sanitize_factorized_action(action) for action in raw]

    diagnostics = summarize_action_diagnostics(
        sanitized,
        raw_actions=raw,
        agent_ids=[0, 1, 0, 1],
        num_agents=2,
        deterministic=False,
        epsilon=0.1,
    )

    assert diagnostics["action_count"] == 4
    assert diagnostics["policy_deterministic"] is False
    assert diagnostics["policy_epsilon"] == 0.1
    assert diagnostics["movement_stay_rate"] == pytest.approx(0.25)
    assert diagnostics["movement_up_rate"] == pytest.approx(0.25)
    assert diagnostics["movement_down_rate"] == pytest.approx(0.25)
    assert diagnostics["vertical_action_rate_from_actions"] == pytest.approx(0.5)
    assert diagnostics["no_target_rate"] == pytest.approx(0.5)
    assert diagnostics["target_selected_rate"] == pytest.approx(0.5)
    assert diagnostics["idle_mode_rate"] == pytest.approx(0.5)
    assert diagnostics["target_required_no_target_rate"] == pytest.approx(0.25)
    assert diagnostics["sanitizer_changed_rate"] == pytest.approx(0.25)
    assert diagnostics["mode_0_rate"] == pytest.approx(0.5)
    assert diagnostics["target_0_rate"] == pytest.approx(0.5)
    assert diagnostics["agent1_sanitizer_changed_rate"] == pytest.approx(0.5)


def test_action_diagnostics_helper_handles_empty_actions():
    diagnostics = summarize_action_diagnostics([], num_agents=2, deterministic=True, epsilon=0.0)

    assert diagnostics["action_count"] == 0
    assert diagnostics["policy_deterministic"] is True
    assert diagnostics["movement_stay_rate"] is None
    assert diagnostics["agent0_action_count"] == 0
    assert diagnostics["agent0_idle_mode_rate"] is None

    prefixed = prefix_action_diagnostics(diagnostics, "rollout_", num_agents=2)
    assert prefixed["rollout_action_count"] == 0
    assert prefixed["rollout_policy_deterministic"] is True
    assert prefixed["rollout_movement_0_rate"] is None


def test_collect_rollout_returns_action_diagnostics():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()

    summary = collect_ctde_rollout(env, actor, max_steps=2, rng=np.random.default_rng(1))

    diagnostics = summary["rollout_action_diagnostics"]
    assert diagnostics["rollout_action_count"] == 4
    assert "rollout_movement_stay_rate" in diagnostics
    assert "rollout_mode_0_rate" in diagnostics
    assert "rollout_no_target_rate" in diagnostics
    assert "rollout_sanitizer_changed_rate" in diagnostics


def test_evaluation_returns_action_diagnostics():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()

    summary = evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2, deterministic=True)

    diagnostics = summary["eval_action_diagnostics"]
    assert diagnostics["eval_action_count"] == 4
    assert diagnostics["eval_policy_deterministic"] is True
    assert diagnostics["eval_policy_epsilon"] == 0.0
    assert "eval_idle_mode_rate" in diagnostics
    assert "eval_agent0_no_target_rate" in diagnostics


def test_train_loop_exports_action_diagnostics_in_rows_and_summary():
    summary = train_ctde_short_run(_tiny_config())
    row = summary["iteration_metrics"][0]

    expected = {
        "rollout_action_count",
        "rollout_movement_stay_rate",
        "rollout_mode_0_rate",
        "rollout_no_target_rate",
        "rollout_sanitizer_changed_rate",
        "eval_action_count",
        "eval_policy_deterministic",
        "eval_idle_mode_rate",
        "eval_agent0_no_target_rate",
    }
    assert expected.issubset(row)
    assert row["rollout_action_count"] == 6
    assert row["eval_action_count"] == 4
    assert row["eval_policy_deterministic"] is True

    assert expected.issubset(summary)
    assert summary["transitions_collected"] == 6
    assert summary["losses_finite"] is True
    assert summary["eval_mean_return"] is not None


def test_action_diagnostics_modules_do_not_import_hierarchical():
    sources = [
        inspect.getsource(action_diagnostics),
        inspect.getsource(ctde_rollout),
        inspect.getsource(ctde_evaluation),
        inspect.getsource(ctde_train_loop),
    ]
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for source in sources for item in forbidden)
