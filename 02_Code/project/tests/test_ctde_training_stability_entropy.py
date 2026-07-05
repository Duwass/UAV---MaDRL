from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn

from marl.ctde import ctde_trainer, train_loop as ctde_train_loop
from marl.ctde.ctde_trainer import CTDETrainer
from marl.ctde.networks import CentralizedVCritic, FactorizedActor
from marl.ctde.train_loop import train_ctde_short_run


OBS_DIM = 114
STATE_DIM = 89
NUM_AGENTS = 2
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConstantCritic(nn.Module):
    def __init__(self, current_value: float = 1.0, next_value: float = 1.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.current_value = float(current_value)
        self.next_value = float(next_value)
        self.calls = 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = self.current_value if self.calls % 2 == 1 else self.next_value
        return self.bias + torch.full((state.shape[0], 1), value, dtype=state.dtype, device=state.device)


def _fake_batch(batch_size: int = 4, seed: int = 321) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "obs": rng.normal(size=(batch_size, NUM_AGENTS, OBS_DIM)).astype(np.float32),
        "state": rng.normal(size=(batch_size, STATE_DIM)).astype(np.float32),
        "movement_actions": rng.integers(0, 11, size=(batch_size, NUM_AGENTS), dtype=np.int64),
        "target_actions": rng.integers(0, 16, size=(batch_size, NUM_AGENTS), dtype=np.int64),
        "mode_actions": rng.integers(0, 6, size=(batch_size, NUM_AGENTS), dtype=np.int64),
        "flat_actions": np.zeros((batch_size, NUM_AGENTS), dtype=np.int64),
        "reward": np.linspace(0.0, 1.0, batch_size, dtype=np.float32),
        "next_obs": rng.normal(size=(batch_size, NUM_AGENTS, OBS_DIM)).astype(np.float32),
        "next_state": rng.normal(size=(batch_size, STATE_DIM)).astype(np.float32),
        "done": np.zeros(batch_size, dtype=bool),
    }


def _trainer(
    *,
    actor: nn.Module | None = None,
    critic: nn.Module | None = None,
    **kwargs,
) -> CTDETrainer:
    actor = actor if actor is not None else FactorizedActor()
    critic = critic if critic is not None else CentralizedVCritic()
    return CTDETrainer(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1.0e-3),
        **kwargs,
    )


def _tiny_config(seed: int = 1401) -> dict:
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
        "normalize_advantage": False,
        "max_grad_norm": None,
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


def test_default_neutral_update_returns_finite_entropy_diagnostics():
    metrics = _trainer().update(_fake_batch())

    expected_old = {"actor_loss", "critic_loss", "mean_advantage", "mean_value", "mean_target"}
    expected_new = {
        "movement_entropy",
        "target_entropy",
        "mode_entropy",
        "policy_entropy_total",
        "entropy_coef",
        "entropy_loss_component",
        "advantage_mean",
        "advantage_std",
        "advantage_abs_mean",
        "advantage_normalized",
        "actor_grad_norm",
        "critic_grad_norm",
        "max_grad_norm",
        "grad_clipping_enabled",
    }
    assert expected_old.issubset(metrics)
    assert expected_new.issubset(metrics)
    assert metrics["entropy_coef"] == 0.0
    assert metrics["advantage_normalized"] is False
    assert metrics["max_grad_norm"] is None
    assert metrics["grad_clipping_enabled"] is False

    finite_keys = expected_old | (expected_new - {"advantage_normalized", "max_grad_norm", "grad_clipping_enabled"})
    assert all(np.isfinite(metrics[key]) for key in finite_keys)


def test_positive_entropy_coef_produces_finite_entropy_loss_component():
    metrics = _trainer(entropy_coef=0.01).update(_fake_batch())

    assert metrics["entropy_coef"] == 0.01
    assert np.isfinite(metrics["entropy_loss_component"])
    assert metrics["entropy_loss_component"] <= 0.0
    assert np.isfinite(metrics["actor_loss"])


def test_normalize_advantage_works_with_nonconstant_advantage():
    metrics = _trainer(normalize_advantage=True).update(_fake_batch())

    assert metrics["advantage_normalized"] is True
    assert metrics["advantage_std"] > 0.0
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["advantage_abs_mean"])


def test_normalize_advantage_handles_near_constant_advantage():
    batch = _fake_batch()
    batch["reward"] = np.full(batch["reward"].shape, 0.25, dtype=np.float32)
    critic = ConstantCritic(current_value=1.0, next_value=1.0)
    metrics = _trainer(critic=critic, normalize_advantage=True).update(batch)

    assert metrics["advantage_normalized"] is True
    assert metrics["advantage_std"] == 0.0
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])


def test_max_grad_norm_exports_enabled_grad_diagnostics():
    metrics = _trainer(max_grad_norm=0.5).update(_fake_batch())

    assert metrics["grad_clipping_enabled"] is True
    assert metrics["max_grad_norm"] == 0.5
    assert np.isfinite(metrics["actor_grad_norm"])
    assert np.isfinite(metrics["critic_grad_norm"])


def test_train_loop_exports_training_diagnostics_and_preserves_existing_diagnostics():
    summary = train_ctde_short_run(_tiny_config())
    row = summary["iteration_metrics"][0]

    training_keys = {
        "movement_entropy",
        "target_entropy",
        "mode_entropy",
        "policy_entropy_total",
        "entropy_coef",
        "entropy_loss_component",
        "advantage_mean",
        "advantage_std",
        "advantage_abs_mean",
        "advantage_normalized",
        "actor_grad_norm",
        "critic_grad_norm",
        "max_grad_norm",
        "grad_clipping_enabled",
    }
    assert training_keys.issubset(row)
    assert training_keys.issubset(summary)
    assert row["advantage_normalized"] is False
    assert row["grad_clipping_enabled"] is False
    assert np.isfinite(row["policy_entropy_total"])
    assert np.isfinite(summary["policy_entropy_total"])

    assert "rollout_total_throughput" in row
    assert "eval_total_throughput" in row
    assert "rollout_action_count" in row
    assert "eval_action_count" in row
    assert summary["losses_finite"] is True


def test_training_stability_entropy_path_does_not_import_hierarchical():
    sources = [inspect.getsource(ctde_trainer), inspect.getsource(ctde_train_loop)]
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]

    assert all(item not in source for source in sources for item in forbidden)
