from __future__ import annotations

import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from envs.uav_backscatter_env import MODE_BACKSCATTER, MODE_HARVEST, MODE_IDLE, UAVBackscatterEnv, encode_action, load_config
from marl.ctde.experiment_io import (
    prepare_run_dir,
    save_config_copy,
    save_metrics_csv,
    save_metrics_jsonl,
    save_reproducibility_info,
    save_summary_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]
POLICIES: tuple[str, ...] = ("random", "idle", "nearest")
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


def run_baseline_evaluation(
    env_config_path: str | Path,
    policy: str,
    seed: int,
    num_iterations: int,
    steps_per_iteration: int,
    save_dir: str | Path | None = None,
    overwrite: bool = False,
    test_status: str = "not_run_in_script",
) -> dict[str, Any]:
    """Run a no-training 3D baseline evaluation and optionally save a CTDE-style bundle."""
    policy_name = _validate_policy(policy)
    num_iterations = _positive_int(num_iterations, "num_iterations")
    steps_per_iteration = _positive_int(steps_per_iteration, "steps_per_iteration")
    run_dir = prepare_run_dir(save_dir, overwrite=overwrite)

    resolved_config_path = _resolve_project_path(env_config_path)
    env_cfg = load_config(resolved_config_path)
    env_cfg = deepcopy(env_cfg)
    env_cfg.setdefault("simulation", {})["seed"] = int(seed)

    rng = np.random.default_rng(int(seed))
    env = UAVBackscatterEnv(env_cfg)
    try:
        _ensure_3d_base_env(env)
        metrics = _run_iterations(env, policy_name, int(seed), num_iterations, steps_per_iteration, rng)
        summary = _build_summary(env, policy_name, int(seed), num_iterations, steps_per_iteration, metrics)
    finally:
        env.close()

    output_files: list[str] = []
    if run_dir is not None:
        output_files = _write_outputs(
            summary=summary,
            metrics=metrics,
            env_config=env_cfg,
            env_config_path=resolved_config_path,
            run_dir=run_dir,
            test_status=test_status,
        )
    summary = dict(summary)
    summary["output_files"] = output_files
    return summary


def select_actions(policy: str, observations: np.ndarray, env: UAVBackscatterEnv, rng: np.random.Generator) -> list[int]:
    policy_name = _validate_policy(policy)
    if policy_name == "random":
        return [_random_action(env, uav_id, rng) for uav_id in range(env.num_uav)]
    if policy_name == "idle":
        return [_idle_action(env) for _ in range(env.num_uav)]
    return [_nearest_observation_action(np.asarray(observations[uav_id], dtype=np.float32), env) for uav_id in range(env.num_uav)]


def _run_iterations(
    env: UAVBackscatterEnv,
    policy: str,
    seed: int,
    num_iterations: int,
    steps_per_iteration: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_steps = 0
    for iteration in range(1, num_iterations + 1):
        observations, info = env.reset(seed=seed + iteration - 1)
        episode_return = 0.0
        steps = 0
        terminated = False
        truncated = False
        while steps < steps_per_iteration and not (terminated or truncated):
            actions = select_actions(policy, observations, env, rng)
            observations, reward, terminated, truncated, info = env.step(actions)
            episode_return += float(reward)
            steps += 1
        cumulative_steps += steps
        row: dict[str, Any] = {
            "iteration": int(iteration),
            "policy": policy,
            "seed": int(seed),
            "steps": int(steps),
            "cumulative_steps": int(cumulative_steps),
            "episode_return": float(episode_return),
            "rollout_return": float(episode_return),
            "eval_mean_return": float(episode_return),
            "eval_total_steps": int(steps),
        }
        row.update(_extract_env_metrics(info.get("episode_metrics"), "eval_"))
        rows.append(row)
    return rows


def _build_summary(
    env: UAVBackscatterEnv,
    policy: str,
    seed: int,
    num_iterations: int,
    steps_per_iteration: int,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    last_metrics = metrics[-1] if metrics else {}
    returns = [float(row.get("episode_return", 0.0)) for row in metrics]
    summary: dict[str, Any] = {
        "policy": policy,
        "seed": int(seed),
        "iterations": int(num_iterations),
        "num_iterations": int(num_iterations),
        "steps_per_iteration": int(steps_per_iteration),
        "total_steps": int(sum(int(row.get("steps", 0)) for row in metrics)),
        "eval_mean_return": float(np.mean(returns)) if returns else None,
        "eval_total_steps": int(sum(int(row.get("eval_total_steps", 0)) for row in metrics)),
        "obs_dim": int(env.get_obs_dim()),
        "state_dim": int(env.get_state_dim()),
        "action_dim": int(env.get_action_dim()),
        "iteration_metrics": metrics,
        "metrics": metrics,
        "stopped_early": False,
        "warning": None,
    }
    summary.update({key: last_metrics.get(key) for key in _prefixed_env_metric_names("eval_")})
    return summary


def _write_outputs(
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    env_config: dict[str, Any],
    env_config_path: Path,
    run_dir: Path,
    test_status: str,
) -> list[str]:
    files: list[str] = []
    files.append(str(save_summary_json(summary, run_dir / "summary.json")))
    files.append(str(save_metrics_jsonl(metrics, run_dir / "metrics.jsonl")))
    files.append(str(save_metrics_csv(metrics, run_dir / "metrics.csv")))
    files.append(str(save_config_copy(_bundle_config(summary, env_config, env_config_path), run_dir / "config.yaml")))
    files.append(str(save_reproducibility_info(_reproducibility_info(summary, env_config_path, test_status), run_dir / "reproducibility.json")))
    return files


def _bundle_config(summary: dict[str, Any], env_config: dict[str, Any], env_config_path: Path) -> dict[str, Any]:
    return {
        "env_config_path": str(env_config_path),
        "policy": summary["policy"],
        "seed": int(summary["seed"]),
        "num_iterations": int(summary["num_iterations"]),
        "steps_per_iteration": int(summary["steps_per_iteration"]),
        "env_config": env_config,
    }


def _reproducibility_info(summary: dict[str, Any], env_config_path: Path, test_status: str) -> dict[str, Any]:
    return {
        "policy": summary["policy"],
        "seed": int(summary["seed"]),
        "branch": _git_value(["git", "branch", "--show-current"]),
        "commit": _git_value(["git", "rev-parse", "HEAD"]),
        "env_config_path": str(env_config_path),
        "obs_dim": int(summary["obs_dim"]),
        "state_dim": int(summary["state_dim"]),
        "action_dim": int(summary["action_dim"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_status": str(test_status),
    }


def _random_action(env: UAVBackscatterEnv, uav_id: int, rng: np.random.Generator) -> int:
    if bool(getattr(env, "action_masking_cfg", {}).get("enabled", False)):
        valid_actions = np.flatnonzero(env.get_action_mask(uav_id))
        if valid_actions.size > 0:
            return int(rng.choice(valid_actions))
    return int(rng.integers(0, env.action_size))


def _idle_action(env: UAVBackscatterEnv) -> int:
    return encode_action(0, 0, MODE_IDLE, env.num_iot, env.num_movement_actions)


def _nearest_observation_action(local_obs: np.ndarray, env: UAVBackscatterEnv) -> int:
    base_len = 9
    block_len = 7
    if local_obs.shape[0] < base_len + env.num_iot * block_len:
        return _idle_action(env)

    primary_busy = bool(local_obs[8] >= 0.5)
    best_target = 0
    best_distance = float("inf")
    best_queue = 0.0
    best_in_coverage = False
    for iot_id in range(env.num_iot):
        start = base_len + iot_id * block_len
        distance_3d = float(local_obs[start + 3])
        queue = float(local_obs[start + 4])
        in_coverage = bool(local_obs[start + 6] >= 0.5)
        prefer = queue > 0.0
        current_rank = (0 if prefer else 1, distance_3d)
        best_rank = (0 if best_queue > 0.0 else 1, best_distance)
        if current_rank < best_rank:
            best_target = iot_id + 1
            best_distance = distance_3d
            best_queue = queue
            best_in_coverage = in_coverage

    if best_target <= 0 or not best_in_coverage or not primary_busy:
        return _idle_action(env)
    mode = MODE_BACKSCATTER if best_queue > 0.0 else MODE_HARVEST
    return encode_action(0, best_target, mode, env.num_iot, env.num_movement_actions)


def _extract_env_metrics(source: Any, prefix: str) -> dict[str, float | None]:
    metrics = source if isinstance(source, dict) else {}
    return {f"{prefix}{key}": _safe_float_or_none(metrics.get(key)) for key in ENV_METRIC_KEYS}


def _prefixed_env_metric_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{key}" for key in ENV_METRIC_KEYS)


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_3d_base_env(env: UAVBackscatterEnv) -> None:
    expected_obs_dim = 9 + int(env.num_iot) * 7
    if int(env.get_obs_dim()) != expected_obs_dim:
        raise ValueError(f"Expected 3D base env obs_dim={expected_obs_dim}, got {env.get_obs_dim()}.")
    expected_action_dim = int(env.num_movement_actions) * (int(env.num_iot) + 1) * 6
    if int(env.get_action_dim()) != expected_action_dim:
        raise ValueError(f"Unexpected action_dim={env.get_action_dim()}, expected {expected_action_dim}.")


def _resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _validate_policy(policy: str) -> str:
    policy_name = str(policy)
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy '{policy_name}'. Valid policies: {', '.join(POLICIES)}")
    return policy_name


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return result


def _git_value(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None
