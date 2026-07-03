from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from envs.uav_backscatter_env import UAVBackscatterEnv, load_config
from marl.ctde.ctde_trainer import CTDETrainer
from marl.ctde.evaluation import evaluate_decentralized_policy
from marl.ctde.networks import CentralizedVCritic, FactorizedActor
from marl.ctde.replay_buffer import CTDEReplayBuffer
from marl.ctde.rollout import collect_ctde_rollout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ctde" / "ctde_3d_base.yaml"


def train_ctde_smoke(config: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Run a tiny CTDE smoke loop, not a full experiment.

    Training actions are selected by the decentralized actor from local
    observations. The centralized critic uses global state only in the update.
    No files, results, or checkpoints are written by this function.
    """
    cfg = _load_ctde_config(config)
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env_cfg = load_config(_resolve_project_path(cfg.get("env_config_path")))
    env_cfg.setdefault("simulation", {})["seed"] = seed
    env = UAVBackscatterEnv(env_cfg)

    actor = FactorizedActor(
        obs_dim=int(cfg.get("obs_dim", 114)),
        hidden_dim=int(cfg.get("actor_hidden_dim", 128)),
    )
    critic = CentralizedVCritic(
        state_dim=int(cfg.get("state_dim", 89)),
        hidden_dim=int(cfg.get("critic_hidden_dim", 128)),
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(cfg.get("actor_lr", 1.0e-3)))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.get("critic_lr", 1.0e-3)))
    trainer = CTDETrainer(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=float(cfg.get("gamma", 0.99)),
        entropy_coef=float(cfg.get("entropy_coef", 0.0)),
        grad_clip_norm=cfg.get("grad_clip_norm"),
    )
    buffer = CTDEReplayBuffer(int(cfg.get("replay_capacity", 128)), seed=seed)

    smoke_cfg = dict(cfg.get("smoke", {}))
    num_iterations = int(smoke_cfg.get("num_iterations", 2))
    rollout_steps = int(smoke_cfg.get("rollout_steps", 5))
    batch_size = int(smoke_cfg.get("batch_size", 4))
    epsilon = float(smoke_cfg.get("epsilon", 0.1))

    metrics_history: list[dict[str, float]] = []
    transitions_collected = 0
    losses_finite = True
    for _ in range(num_iterations):
        rollout_summary = collect_ctde_rollout(
            env,
            actor,
            buffer=buffer,
            max_steps=rollout_steps,
            epsilon=epsilon,
            rng=rng,
            use_movement_mask=True,
        )
        transitions_collected += int(rollout_summary["transitions_collected"])
        if len(buffer) <= 0:
            continue
        batch = buffer.sample(min(batch_size, len(buffer)), rng=rng)
        metrics = trainer.update(batch)
        metrics_history.append(metrics)
        losses_finite = losses_finite and bool(
            np.isfinite(metrics["actor_loss"]) and np.isfinite(metrics["critic_loss"])
        )

    eval_summary: dict[str, Any] = {}
    eval_episodes = int(smoke_cfg.get("eval_episodes", 1))
    if eval_episodes > 0:
        eval_summary = evaluate_decentralized_policy(
            env,
            actor,
            num_episodes=eval_episodes,
            max_steps=int(smoke_cfg.get("eval_max_steps", 5)),
            deterministic=True,
            rng=rng,
        )

    last_metrics = metrics_history[-1] if metrics_history else {}
    return {
        "iterations": int(num_iterations),
        "updates": int(len(metrics_history)),
        "transitions_collected": int(transitions_collected),
        "last_actor_loss": _float_or_none(last_metrics.get("actor_loss")),
        "last_critic_loss": _float_or_none(last_metrics.get("critic_loss")),
        "losses_finite": bool(losses_finite and bool(metrics_history)),
        "eval_mean_return": _float_or_none(eval_summary.get("mean_return")),
        "eval_total_steps": int(eval_summary.get("total_steps", 0)) if eval_summary else 0,
    }


def _load_ctde_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        path = DEFAULT_CONFIG_PATH
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if isinstance(config, dict):
        return deepcopy(config)
    path = _resolve_project_path(config)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: str | Path | None) -> Path:
    if path is None:
        return PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
