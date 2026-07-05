from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde import evaluation as ctde_evaluation, train_loop as ctde_train_loop
from marl.ctde.evaluation import evaluate_decentralized_policy
from marl.ctde.experiment_io import save_metrics_jsonl
from marl.ctde.networks import FactorizedActor
from marl.ctde.train_loop import train_ctde_short_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"


class FixedLogitActor:
    def __call__(self, obs: torch.Tensor):
        movement = torch.zeros(1, 11)
        movement[0, 3] = 4.0
        movement[0, 4] = 2.0
        target = torch.zeros(1, 16)
        target[0, 2] = 5.0
        target[0, 1] = 3.0
        mode = torch.zeros(1, 6)
        mode[0, 2] = 6.0
        mode[0, 3] = 4.0
        return {
            "movement_logits": movement,
            "target_logits": target,
            "mode_logits": mode,
        }


def _tiny_config(seed: int = 1501, diagnostic: bool = False) -> dict:
    config = {
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
    if diagnostic:
        config["evaluation"] = {
            "diagnostic_stochastic_eval": True,
            "diagnostic_stochastic_episodes": 1,
            "diagnostic_epsilon_eval": True,
            "diagnostic_epsilon": 0.05,
            "diagnostic_epsilon_episodes": 1,
        }
    return config


def test_primary_eval_exports_confidence_raw_sanitized_and_diversity_keys():
    env = UAVBackscatterEnv(CONFIG_3D)
    summary = evaluate_decentralized_policy(env, FixedLogitActor(), num_episodes=1, max_steps=2)
    diagnostics = summary["eval_action_diagnostics"]

    expected = {
        "eval_movement_top1_prob",
        "eval_movement_top2_prob",
        "eval_movement_top1_top2_margin",
        "eval_target_top1_prob",
        "eval_mode_top1_prob",
        "eval_raw_no_target_rate",
        "eval_sanitized_no_target_rate",
        "eval_raw_idle_mode_rate",
        "eval_sanitized_idle_mode_rate",
        "eval_unique_joint_action_count",
        "eval_joint_action_top1_rate",
        "eval_agent0_unique_joint_action_count",
        "eval_agent0_joint_action_top1_rate",
    }
    assert expected.issubset(diagnostics)
    assert diagnostics["eval_policy_deterministic"] is True
    assert diagnostics["eval_policy_selection_mode"] == "epsilon_argmax"
    assert 0.0 <= diagnostics["eval_movement_top2_prob"] <= diagnostics["eval_movement_top1_prob"] <= 1.0
    assert 0.0 <= diagnostics["eval_joint_action_top1_rate"] <= 1.0


def test_diagnostic_stochastic_eval_uses_prefixed_distribution_sampling_keys():
    env = UAVBackscatterEnv(CONFIG_3D)
    summary = evaluate_decentralized_policy(
        env,
        FixedLogitActor(),
        num_episodes=1,
        max_steps=2,
        deterministic=False,
        epsilon=0.0,
        selection_mode="stochastic",
        diagnostic_prefix="diag_stoch_eval_",
        rng=np.random.default_rng(7),
    )
    diagnostics = summary["diag_stoch_eval_action_diagnostics"]

    assert diagnostics["diag_stoch_eval_policy_deterministic"] is False
    assert diagnostics["diag_stoch_eval_policy_selection_mode"] == "stochastic"
    assert "diag_stoch_eval_mode_top1_prob" in diagnostics
    assert "diag_stoch_eval_unique_joint_action_count" in diagnostics
    assert 0.0 <= diagnostics["diag_stoch_eval_joint_action_top1_rate"] <= 1.0


def test_diagnostic_epsilon_eval_uses_prefixed_epsilon_keys():
    env = UAVBackscatterEnv(CONFIG_3D)
    summary = evaluate_decentralized_policy(
        env,
        FixedLogitActor(),
        num_episodes=1,
        max_steps=2,
        deterministic=False,
        epsilon=0.05,
        selection_mode="epsilon_argmax",
        diagnostic_prefix="diag_epsilon_eval_",
        rng=np.random.default_rng(8),
    )
    diagnostics = summary["diag_epsilon_eval_action_diagnostics"]

    assert diagnostics["diag_epsilon_eval_policy_deterministic"] is False
    assert diagnostics["diag_epsilon_eval_policy_epsilon"] == 0.05
    assert diagnostics["diag_epsilon_eval_policy_selection_mode"] == "epsilon_argmax"
    assert "diag_epsilon_eval_raw_no_target_rate" in diagnostics
    assert "diag_epsilon_eval_joint_action_top1_rate" in diagnostics


def test_train_loop_default_keeps_diagnostic_eval_disabled():
    summary = train_ctde_short_run(_tiny_config(seed=1502, diagnostic=False))
    row = summary["iteration_metrics"][0]

    assert "eval_movement_top1_prob" in row
    assert "diag_stoch_eval_mean_return" not in row
    assert "diag_epsilon_eval_mean_return" not in row
    assert "diag_stoch_eval_mean_return" not in summary


def test_train_loop_exports_diagnostic_eval_rows_summary_and_jsonl(tmp_path):
    summary = train_ctde_short_run(_tiny_config(seed=1503, diagnostic=True))
    row = summary["iteration_metrics"][0]

    expected = {
        "diag_stoch_eval_mean_return",
        "diag_stoch_eval_avg_throughput_per_frame",
        "diag_stoch_eval_mode_top1_prob",
        "diag_stoch_eval_joint_action_top1_rate",
        "diag_epsilon_eval_mean_return",
        "diag_epsilon_eval_avg_throughput_per_frame",
        "diag_epsilon_eval_mode_top1_prob",
        "diag_epsilon_eval_joint_action_top1_rate",
    }
    assert expected.issubset(row)
    assert expected.issubset(summary)

    path = save_metrics_jsonl(summary["iteration_metrics"], tmp_path / "metrics.jsonl")
    saved_row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert expected.issubset(saved_row)


def test_evaluation_diagnostics_modules_do_not_import_hierarchical():
    sources = [inspect.getsource(ctde_evaluation), inspect.getsource(ctde_train_loop)]
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for source in sources for item in forbidden)
