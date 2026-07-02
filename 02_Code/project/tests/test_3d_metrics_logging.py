from __future__ import annotations

from pathlib import Path

import pytest

from envs.mobility_model import MOVE_DOWN, MOVE_UP
from envs.uav_backscatter_env import MODE_IDLE, UAVBackscatterEnv, encode_action, load_config
from scripts import plot_3d_trajectories, plot_altitude_over_time
from scripts.run_baseline import run_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
CONFIG_2D = PROJECT_ROOT / "configs" / "scenario_4_backscatter_types_calibrated.yaml"
METRIC_KEYS_3D = {
    "avg_uav_altitude",
    "min_uav_altitude",
    "max_uav_altitude",
    "vertical_action_rate",
    "avg_vertical_movement",
    "avg_uav_iot_3d_distance",
    "avg_uav_jammer_3d_distance",
    "altitude_boundary_hits",
}


def _short_config(path: Path) -> dict:
    cfg = load_config(path)
    cfg["simulation"]["max_steps"] = 3
    cfg["simulation"]["episode_length"] = 3
    cfg.setdefault("evaluation", {})["save_csv"] = False
    return cfg


def _idle_action(env: UAVBackscatterEnv, movement: int = 0) -> int:
    return encode_action(movement, 0, MODE_IDLE, env.num_iot, env.num_movement_actions)


def test_3d_metrics_exist_after_reset_step_or_episode():
    env = UAVBackscatterEnv(_short_config(CONFIG_3D))
    _, info = env.reset(seed=123)

    assert METRIC_KEYS_3D.issubset(info["episode_metrics"])

    _, _, _, _, step_info = env.step([_idle_action(env) for _ in range(env.num_uav)])
    assert METRIC_KEYS_3D.issubset(step_info["episode_metrics"])
    assert METRIC_KEYS_3D.issubset(step_info["frame_metrics"])


def test_3d_metrics_values_reasonable():
    env = UAVBackscatterEnv(_short_config(CONFIG_3D))
    env.reset(seed=123)
    actions = [_idle_action(env, MOVE_UP) for _ in range(env.num_uav)]
    _, _, _, _, info = env.step(actions)
    metrics = info["episode_metrics"]

    assert env.uav_altitude_min <= metrics["min_uav_altitude"] <= metrics["avg_uav_altitude"] <= metrics["max_uav_altitude"]
    assert metrics["max_uav_altitude"] <= env.uav_altitude_max
    assert metrics["avg_uav_iot_3d_distance"] >= 0.0
    assert metrics["avg_uav_jammer_3d_distance"] >= 0.0
    assert 0.0 <= metrics["vertical_action_rate"] <= 1.0


def test_vertical_action_rate_counts_up_down():
    env = UAVBackscatterEnv(_short_config(CONFIG_3D))
    env.reset(seed=123)
    actions = [_idle_action(env, MOVE_UP), _idle_action(env, MOVE_DOWN)]

    _, _, _, _, info = env.step(actions)
    metrics = info["episode_metrics"]

    assert metrics["vertical_action_rate"] == pytest.approx(1.0)
    assert metrics["avg_vertical_movement"] == pytest.approx(env.uav_vertical_step)


def test_altitude_boundary_hits_counts_clip_or_mask():
    env = UAVBackscatterEnv(_short_config(CONFIG_3D))
    env.reset(seed=123)
    env.uavs[0].z = env.uav_altitude_min
    env.uavs[1].z = env.uav_altitude_max
    actions = [_idle_action(env, MOVE_DOWN), _idle_action(env, MOVE_UP)]

    _, _, _, _, info = env.step(actions)

    assert info["episode_metrics"]["altitude_boundary_hits"] == pytest.approx(2.0)
    assert info["frame_metrics"]["altitude_boundary_hits"] == pytest.approx(2.0)


def test_legacy_2d_metrics_no_crash():
    env = UAVBackscatterEnv(_short_config(CONFIG_2D))
    _, info = env.reset(seed=123)
    _, _, _, _, step_info = env.step([_idle_action(env) for _ in range(env.num_uav)])

    assert not (METRIC_KEYS_3D & set(info["episode_metrics"]))
    assert not (METRIC_KEYS_3D & set(step_info["episode_metrics"]))


def test_evaluation_summary_contains_3d_metrics():
    cfg = _short_config(CONFIG_3D)
    episodes_df, frames_df = run_policy(
        cfg,
        "random",
        episodes=1,
        scenario_name="pytest_3d_metrics",
        save_csv=False,
    )

    assert METRIC_KEYS_3D.issubset(episodes_df.columns)
    assert METRIC_KEYS_3D.issubset(frames_df.columns)


def test_plot_3d_trajectory_script_imports_or_runs_tmp(tmp_path):
    output = tmp_path / "trajectory.png"

    result = plot_3d_trajectories.main(["--config", str(CONFIG_3D), "--steps", "2", "--output", str(output)])

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_altitude_script_imports_or_runs_tmp(tmp_path):
    output = tmp_path / "altitude.png"

    result = plot_altitude_over_time.main(["--config", str(CONFIG_3D), "--steps", "2", "--output", str(output)])

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
