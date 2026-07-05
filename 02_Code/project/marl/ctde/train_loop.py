from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from envs.uav_backscatter_env import UAVBackscatterEnv, load_config
from marl.ctde.action_diagnostics import prefix_action_diagnostics, prefixed_action_diagnostic_keys
from marl.ctde.ctde_trainer import CTDETrainer
from marl.ctde.evaluation import evaluate_decentralized_policy
from marl.ctde.networks import CentralizedVCritic, FactorizedActor
from marl.ctde.replay_buffer import CTDEReplayBuffer
from marl.ctde.rollout import collect_ctde_rollout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ctde" / "ctde_3d_base.yaml"
ENV_METRIC_KEYS: tuple[str, ...] = (
    "total_throughput",
    "avg_throughput_per_frame",
    "packet_drop_rate",
    "jamming_failure_rate",
    "jammed_transmission_rate",
    "fairness_index",
    "energy_efficiency",
    "backscatter_success_rate",
    "active_success_rate",
    "avg_uav_altitude",
    "min_uav_altitude",
    "max_uav_altitude",
    "vertical_action_rate",
    "avg_vertical_movement",
    "avg_uav_iot_3d_distance",
    "avg_uav_jammer_3d_distance",
    "altitude_boundary_hits",
)
TRAINING_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "mean_entropy",
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
)


def train_ctde_smoke(config: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Run a tiny CTDE smoke loop, not a full experiment.

    Training actions are selected by the decentralized actor from local
    observations. The centralized critic uses global state only in the update.
    No files, results, or checkpoints are written by this function.
    """
    return train_ctde_short_run(config)


def train_ctde_short_run(config: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Run a guarded short CTDE loop and return per-iteration metrics."""
    cfg = _load_ctde_config(config)
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env_cfg = load_config(_resolve_project_path(cfg.get("env_config_path")))
    env_cfg.setdefault("simulation", {})["seed"] = seed
    env = UAVBackscatterEnv(env_cfg)
    num_agents = int(cfg.get("num_agents", getattr(env, "num_uav", 2)))

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
        max_grad_norm=_configured_max_grad_norm(cfg),
        normalize_advantage=bool(cfg.get("normalize_advantage", False)),
    )
    buffer = CTDEReplayBuffer(int(cfg.get("replay_capacity", 128)), seed=seed)

    smoke_cfg = dict(cfg.get("smoke", {}))
    num_iterations = int(smoke_cfg.get("num_iterations", 2))
    rollout_steps = int(smoke_cfg.get("rollout_steps", 5))
    batch_size = int(smoke_cfg.get("batch_size", 4))
    epsilon = float(smoke_cfg.get("epsilon", 0.1))
    eval_every = int(smoke_cfg.get("eval_every", 0))
    eval_episodes = int(smoke_cfg.get("eval_episodes", 1))
    eval_max_steps = int(smoke_cfg.get("eval_max_steps", 5))
    evaluation_cfg = dict(cfg.get("evaluation", {}))
    diagnostic_stochastic_eval = bool(evaluation_cfg.get("diagnostic_stochastic_eval", False))
    diagnostic_stochastic_episodes = int(evaluation_cfg.get("diagnostic_stochastic_episodes", eval_episodes))
    diagnostic_epsilon_eval = bool(evaluation_cfg.get("diagnostic_epsilon_eval", False))
    diagnostic_epsilon = float(evaluation_cfg.get("diagnostic_epsilon", 0.05))
    diagnostic_epsilon_episodes = int(evaluation_cfg.get("diagnostic_epsilon_episodes", eval_episodes))

    metrics_history: list[dict[str, float]] = []
    transitions_collected = 0
    losses_finite = True
    stopped_early = False
    warning: str | None = None
    last_eval_summary: dict[str, Any] = {}
    last_diag_stoch_eval_summary: dict[str, Any] = {}
    last_diag_epsilon_eval_summary: dict[str, Any] = {}
    for iteration in range(1, num_iterations + 1):
        rollout_summary = collect_ctde_rollout(
            env,
            actor,
            buffer=buffer,
            max_steps=rollout_steps,
            epsilon=epsilon,
            rng=rng,
            use_movement_mask=True,
        )
        rollout_env_metrics = _extract_env_metrics(rollout_summary.get("episode_metrics"), "rollout_")
        rollout_action_metrics = prefix_action_diagnostics(rollout_summary.get("action_diagnostics"), "rollout_", num_agents=num_agents)
        transitions_collected += int(rollout_summary["transitions_collected"])
        row: dict[str, Any] = {
            "iteration": int(iteration),
            "transitions_collected": int(transitions_collected),
            "rollout_transitions": int(rollout_summary["transitions_collected"]),
            "buffer_size": int(len(buffer)),
            "actor_loss": None,
            "critic_loss": None,
            "mean_value": None,
            "mean_target": None,
            "mean_advantage": None,
            "losses_finite": False,
            "episode_return": float(rollout_summary.get("episode_return", 0.0)),
            "rollout_return": float(rollout_summary.get("episode_return", 0.0)),
            "eval_mean_return": None,
            "eval_total_steps": 0,
        }
        row.update({key: None for key in TRAINING_DIAGNOSTIC_KEYS})
        row.update(rollout_env_metrics)
        row.update(rollout_action_metrics)
        row.update(_extract_env_metrics(None, "eval_"))
        row.update(prefix_action_diagnostics(None, "eval_", num_agents=num_agents))
        if len(buffer) <= 0:
            metrics_history.append(row)
            continue
        batch = buffer.sample(min(batch_size, len(buffer)), rng=rng)
        update_metrics = trainer.update(batch)
        row.update({key: _metric_value_or_none(value) for key, value in update_metrics.items()})
        row["losses_finite"] = _metrics_are_finite(row, ("actor_loss", "critic_loss", "mean_value", "mean_target", "mean_advantage"))
        losses_finite = losses_finite and bool(row["losses_finite"])

        should_eval = eval_every > 0 and eval_episodes > 0 and iteration % eval_every == 0
        if should_eval:
            last_eval_summary = evaluate_decentralized_policy(
                env,
                actor,
                num_episodes=eval_episodes,
                max_steps=eval_max_steps,
                deterministic=True,
                rng=rng,
            )
            row["eval_mean_return"] = _float_or_none(last_eval_summary.get("mean_return"))
            row["eval_total_steps"] = int(last_eval_summary.get("total_steps", 0))
            row.update(_extract_env_metrics(last_eval_summary.get("last_episode_metrics"), "eval_"))
            row.update(prefix_action_diagnostics(last_eval_summary.get("action_diagnostics"), "eval_", num_agents=num_agents))
            if diagnostic_stochastic_eval and diagnostic_stochastic_episodes > 0:
                last_diag_stoch_eval_summary = evaluate_decentralized_policy(
                    env,
                    actor,
                    num_episodes=diagnostic_stochastic_episodes,
                    max_steps=eval_max_steps,
                    deterministic=False,
                    epsilon=0.0,
                    selection_mode="stochastic",
                    diagnostic_prefix="diag_stoch_eval_",
                    rng=np.random.default_rng(seed * 100000 + iteration * 10 + 1),
                )
                row.update(_prefixed_eval_summary_fields(last_diag_stoch_eval_summary, "diag_stoch_eval_"))
                row.update(_extract_env_metrics(last_diag_stoch_eval_summary.get("last_episode_metrics"), "diag_stoch_eval_"))
                row.update(prefix_action_diagnostics(last_diag_stoch_eval_summary.get("action_diagnostics"), "diag_stoch_eval_", num_agents=num_agents))
            if diagnostic_epsilon_eval and diagnostic_epsilon_episodes > 0:
                last_diag_epsilon_eval_summary = evaluate_decentralized_policy(
                    env,
                    actor,
                    num_episodes=diagnostic_epsilon_episodes,
                    max_steps=eval_max_steps,
                    deterministic=False,
                    epsilon=diagnostic_epsilon,
                    selection_mode="epsilon_argmax",
                    diagnostic_prefix="diag_epsilon_eval_",
                    rng=np.random.default_rng(seed * 100000 + iteration * 10 + 2),
                )
                row.update(_prefixed_eval_summary_fields(last_diag_epsilon_eval_summary, "diag_epsilon_eval_"))
                row.update(_extract_env_metrics(last_diag_epsilon_eval_summary.get("last_episode_metrics"), "diag_epsilon_eval_"))
                row.update(prefix_action_diagnostics(last_diag_epsilon_eval_summary.get("action_diagnostics"), "diag_epsilon_eval_", num_agents=num_agents))

        metrics_history.append(row)
        if not row["losses_finite"]:
            stopped_early = True
            warning = "non_finite_loss"
            break

    if not last_eval_summary and eval_every <= 0 and eval_episodes > 0:
        last_eval_summary = evaluate_decentralized_policy(
            env,
            actor,
            num_episodes=eval_episodes,
            max_steps=eval_max_steps,
            deterministic=True,
            rng=rng,
        )

    last_metrics = metrics_history[-1] if metrics_history else {}
    summary = {
        "iterations": int(num_iterations),
        "updates": int(sum(1 for row in metrics_history if row.get("actor_loss") is not None)),
        "transitions_collected": int(transitions_collected),
        "last_actor_loss": _float_or_none(last_metrics.get("actor_loss")),
        "last_critic_loss": _float_or_none(last_metrics.get("critic_loss")),
        "losses_finite": bool(losses_finite and bool(metrics_history)),
        "eval_mean_return": _float_or_none(last_eval_summary.get("mean_return")),
        "eval_total_steps": int(last_eval_summary.get("total_steps", 0)) if last_eval_summary else 0,
        "iteration_metrics": metrics_history,
        "metrics": metrics_history,
        "stopped_early": bool(stopped_early),
        "warning": warning,
        "seed": int(seed),
        "obs_dim": int(cfg.get("obs_dim", 114)),
        "state_dim": int(cfg.get("state_dim", 89)),
        "action_dim": 1056,
    }
    summary.update({key: last_metrics.get(key) for key in _prefixed_env_metric_names("rollout_")})
    summary.update({key: last_metrics.get(key) for key in prefixed_action_diagnostic_keys("rollout_", num_agents=num_agents)})
    summary.update({key: last_metrics.get(key) for key in TRAINING_DIAGNOSTIC_KEYS})
    summary.update(_extract_env_metrics(last_eval_summary.get("last_episode_metrics"), "eval_"))
    summary.update(prefix_action_diagnostics(last_eval_summary.get("action_diagnostics"), "eval_", num_agents=num_agents))
    if last_diag_stoch_eval_summary:
        summary.update(_prefixed_eval_summary_fields(last_diag_stoch_eval_summary, "diag_stoch_eval_"))
        summary.update(_extract_env_metrics(last_diag_stoch_eval_summary.get("last_episode_metrics"), "diag_stoch_eval_"))
        summary.update(prefix_action_diagnostics(last_diag_stoch_eval_summary.get("action_diagnostics"), "diag_stoch_eval_", num_agents=num_agents))
    if last_diag_epsilon_eval_summary:
        summary.update(_prefixed_eval_summary_fields(last_diag_epsilon_eval_summary, "diag_epsilon_eval_"))
        summary.update(_extract_env_metrics(last_diag_epsilon_eval_summary.get("last_episode_metrics"), "diag_epsilon_eval_"))
        summary.update(prefix_action_diagnostics(last_diag_epsilon_eval_summary.get("action_diagnostics"), "diag_epsilon_eval_", num_agents=num_agents))
    return summary


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


def _metric_value_or_none(value: Any) -> float | bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return float(value)


def _configured_max_grad_norm(cfg: dict[str, Any]) -> float | None:
    value = cfg.get("max_grad_norm")
    if value is None:
        value = cfg.get("grad_clip_norm")
    if value is None:
        return None
    value = float(value)
    return value if value > 0.0 else None


def _metrics_are_finite(values: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(values.get(key) is not None and bool(np.isfinite(values[key])) for key in keys)


def _prefixed_env_metric_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{key}" for key in ENV_METRIC_KEYS)


def _prefixed_eval_summary_fields(source: dict[str, Any], prefix: str) -> dict[str, float | int | None]:
    return {
        f"{prefix}mean_return": _safe_float_or_none(source.get("mean_return")),
        f"{prefix}total_steps": int(source.get("total_steps", 0)) if source else 0,
    }


def _extract_env_metrics(source: Any, prefix: str) -> dict[str, float | None]:
    metrics = source if isinstance(source, dict) else {}
    extracted: dict[str, float | None] = {}
    for key in ENV_METRIC_KEYS:
        value = metrics.get(key)
        extracted[f"{prefix}{key}"] = _safe_float_or_none(value)
    return extracted


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
