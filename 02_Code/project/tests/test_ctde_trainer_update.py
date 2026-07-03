from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde.ctde_trainer import CTDETrainer
from marl.ctde.evaluation import evaluate_decentralized_policy
from marl.ctde.networks import CentralizedVCritic, FactorizedActor
from marl.ctde.replay_buffer import CTDEReplayBuffer
from marl.ctde.rollout import collect_ctde_rollout
from marl.ctde import ctde_trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
OBS_DIM = 114
STATE_DIM = 89
NUM_AGENTS = 2


class RecordingActor(FactorizedActor):
    def __init__(self):
        super().__init__()
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_shapes.append(tuple(obs.shape))
        return super().forward(obs)


class RecordingCritic(CentralizedVCritic):
    def __init__(self):
        super().__init__()
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append(tuple(state.shape))
        return super().forward(state)


class ConstantCritic(nn.Module):
    def __init__(self, current_value: float = 2.0, next_value: float = 5.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.current_value = float(current_value)
        self.next_value = float(next_value)
        self.calls = 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = self.current_value if self.calls % 2 == 1 else self.next_value
        return self.bias + torch.full((state.shape[0], 1), value, dtype=state.dtype, device=state.device)


def _fake_batch(batch_size: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    movement = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)[:batch_size]
    target = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)[:batch_size]
    mode = np.asarray([[0, 1], [2, 3], [4, 5], [1, 2]], dtype=np.int64)[:batch_size]
    return {
        "obs": rng.normal(size=(batch_size, NUM_AGENTS, OBS_DIM)).astype(np.float32),
        "state": rng.normal(size=(batch_size, STATE_DIM)).astype(np.float32),
        "movement_actions": movement,
        "target_actions": target,
        "mode_actions": mode,
        "flat_actions": np.zeros((batch_size, NUM_AGENTS), dtype=np.int64),
        "reward": np.linspace(0.0, 1.0, batch_size, dtype=np.float32),
        "next_obs": rng.normal(size=(batch_size, NUM_AGENTS, OBS_DIM)).astype(np.float32),
        "next_state": rng.normal(size=(batch_size, STATE_DIM)).astype(np.float32),
        "done": np.asarray([False, True, False, True], dtype=bool)[:batch_size],
    }


def _trainer(actor: nn.Module | None = None, critic: nn.Module | None = None) -> CTDETrainer:
    actor = actor if actor is not None else FactorizedActor()
    critic = critic if critic is not None else CentralizedVCritic()
    return CTDETrainer(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1.0e-3),
    )


def test_ctde_trainer_compute_losses_returns_required_keys():
    trainer = _trainer()
    losses = trainer.compute_losses(_fake_batch())
    required = {"critic_loss", "actor_loss", "mean_value", "mean_target", "mean_advantage"}
    assert required.issubset(losses)


def test_ctde_trainer_update_changes_actor_or_critic_params():
    actor = FactorizedActor()
    critic = CentralizedVCritic()
    trainer = _trainer(actor, critic)
    before_actor = [param.detach().clone() for param in actor.parameters()]
    before_critic = [param.detach().clone() for param in critic.parameters()]
    trainer.update(_fake_batch())
    actor_changed = any(not torch.allclose(before, after) for before, after in zip(before_actor, actor.parameters()))
    critic_changed = any(not torch.allclose(before, after) for before, after in zip(before_critic, critic.parameters()))
    assert actor_changed
    assert critic_changed


def test_critic_uses_global_state_shape():
    critic = RecordingCritic()
    trainer = _trainer(critic=critic)
    trainer.compute_losses(_fake_batch(batch_size=3))
    assert (3, STATE_DIM) in critic.seen_shapes


def test_actor_uses_only_local_obs_shape():
    actor = RecordingActor()
    trainer = _trainer(actor=actor)
    trainer.compute_losses(_fake_batch(batch_size=3))
    assert actor.seen_shapes == [(3 * NUM_AGENTS, OBS_DIM)]


def test_done_bool_converted_for_target():
    critic = ConstantCritic(current_value=2.0, next_value=5.0)
    trainer = _trainer(critic=critic)
    batch = _fake_batch(batch_size=2)
    batch["reward"] = np.asarray([1.0, 1.0], dtype=np.float32)
    batch["done"] = np.asarray([False, True], dtype=bool)
    losses = trainer.compute_losses(batch)
    expected_targets = np.asarray([1.0 + trainer.gamma * 5.0, 1.0], dtype=np.float32)
    assert np.isclose(float(losses["mean_target"].detach()), float(expected_targets.mean()))


def test_actor_loss_detaches_critic_advantage():
    actor = FactorizedActor()
    critic = CentralizedVCritic()
    trainer = _trainer(actor, critic)
    losses = trainer.compute_losses(_fake_batch())
    trainer.actor_optimizer.zero_grad(set_to_none=True)
    trainer.critic_optimizer.zero_grad(set_to_none=True)
    losses["actor_loss"].backward()
    assert all(param.grad is None for param in critic.parameters())
    assert any(param.grad is not None for param in actor.parameters())


def test_critic_loss_does_not_update_actor_when_backward_separate():
    actor = FactorizedActor()
    critic = CentralizedVCritic()
    trainer = _trainer(actor, critic)
    losses = trainer.compute_losses(_fake_batch())
    trainer.actor_optimizer.zero_grad(set_to_none=True)
    trainer.critic_optimizer.zero_grad(set_to_none=True)
    losses["critic_loss"].backward()
    assert all(param.grad is None for param in actor.parameters())
    assert any(param.grad is not None for param in critic.parameters())


def test_trainer_update_returns_float_metrics():
    trainer = _trainer()
    metrics = trainer.update(_fake_batch())
    for value in metrics.values():
        assert isinstance(value, float)


def test_trainer_no_env_or_hierarchical_imports():
    source = inspect.getsource(ctde_trainer)
    forbidden = ["hierarchical", "get_action_mask", "env.step", "env.get_global_state"]
    assert all(item not in source for item in forbidden)


def test_rollout_buffer_sample_can_feed_trainer_update():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()
    critic = CentralizedVCritic()
    buffer = CTDEReplayBuffer(capacity=8, seed=1)
    collect_ctde_rollout(env, actor, buffer=buffer, max_steps=3, rng=np.random.default_rng(1))
    batch = buffer.sample(2)
    trainer = _trainer(actor, critic)
    metrics = trainer.update(batch)
    assert isinstance(metrics["critic_loss"], float)
    assert isinstance(metrics["actor_loss"], float)


def test_decentralized_evaluation_still_passes_after_trainer_added():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()
    summary = evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2)
    assert summary["total_steps"] == 2
