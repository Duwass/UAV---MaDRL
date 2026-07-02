from __future__ import annotations

from pathlib import Path

import pytest

from envs.hierarchical_env import HierarchicalUAVBackscatterEnv
from envs.mobility_model import MOVE_DOWN, MOVE_UP, MOVEMENT_DELTAS
from envs.uav_backscatter_env import (
    MODE_IDLE,
    NUM_COMMUNICATION_MODES,
    decode_action,
    encode_action,
    load_config,
    UAVBackscatterEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
CONFIG_2D = PROJECT_ROOT / "configs" / "scenario_4_backscatter_types_calibrated.yaml"


def _idle_action(env: UAVBackscatterEnv, movement: int = 0) -> int:
    return encode_action(movement, 0, MODE_IDLE, env.num_iot, env.num_movement_actions)


def _step_first_uav(env: UAVBackscatterEnv, first_action: int):
    actions = [_idle_action(env) for _ in range(env.num_uav)]
    actions[0] = first_action
    return env.step(actions)


def test_movement_primitives_include_up_down():
    assert len(MOVEMENT_DELTAS) == 11
    assert MOVEMENT_DELTAS[MOVE_UP] == (0.0, 0.0, 1.0)
    assert MOVEMENT_DELTAS[MOVE_DOWN] == (0.0, 0.0, -1.0)
    assert all(MOVEMENT_DELTAS[idx][2] == 0.0 for idx in range(9))


def test_3d_flat_action_dim_is_1056_for_scenario4_3d():
    env = UAVBackscatterEnv(CONFIG_3D)
    assert env.num_movement_actions == 11
    assert env.action_size == 11 * (15 + 1) * NUM_COMMUNICATION_MODES
    assert env.action_size == 1056


def test_legacy_action_dim_still_valid_or_reported():
    env = UAVBackscatterEnv(CONFIG_2D)
    assert env.num_movement_actions == 9
    assert env.action_size == 9 * (15 + 1) * NUM_COMMUNICATION_MODES
    assert env.action_size == 864


def test_uav_move_up_increases_altitude():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)
    before = env.uavs[0].z

    _step_first_uav(env, _idle_action(env, MOVE_UP))

    assert env.uavs[0].z == pytest.approx(before + env.uav_vertical_step)


def test_uav_move_down_decreases_altitude():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)
    env.uavs[0].z = env.uav_initial_altitude
    before = env.uavs[0].z

    _step_first_uav(env, _idle_action(env, MOVE_DOWN))

    assert env.uavs[0].z == pytest.approx(before - env.uav_vertical_step)


def test_altitude_clips_at_min():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)
    env.uavs[0].z = env.uav_altitude_min

    _step_first_uav(env, _idle_action(env, MOVE_DOWN))

    assert env.uavs[0].z == pytest.approx(env.uav_altitude_min)
    assert env.get_action_mask(0)[_idle_action(env, MOVE_DOWN)] == 0


def test_altitude_clips_at_max():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)
    env.uavs[0].z = env.uav_altitude_max

    _step_first_uav(env, _idle_action(env, MOVE_UP))

    assert env.uavs[0].z == pytest.approx(env.uav_altitude_max)
    assert env.get_action_mask(0)[_idle_action(env, MOVE_UP)] == 0


def test_flat_action_encode_decode_with_11_movements():
    num_iot = 15
    action_id = encode_action(MOVE_DOWN, num_iot, MODE_IDLE, num_iot, num_movement_actions=11)
    movement, selected_iot, mode = decode_action(action_id, num_iot, num_movement_actions=11)

    assert movement == MOVE_DOWN
    assert selected_iot == num_iot
    assert mode == MODE_IDLE


def test_3d_env_step_with_up_down_no_crash():
    env = UAVBackscatterEnv(CONFIG_3D)
    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, step_info = _step_first_uav(env, _idle_action(env, MOVE_UP))
    next_obs_2, reward_2, terminated_2, truncated_2, step_info_2 = _step_first_uav(env, _idle_action(env, MOVE_DOWN))

    assert obs.shape == (env.num_uav, env.observation_dim)
    assert next_obs.shape == (env.num_uav, env.observation_dim)
    assert next_obs_2.shape == (env.num_uav, env.observation_dim)
    assert "global_state" in info
    assert isinstance(reward, float)
    assert isinstance(reward_2, float)
    assert isinstance(terminated, bool)
    assert isinstance(terminated_2, bool)
    assert isinstance(truncated, bool)
    assert isinstance(truncated_2, bool)
    assert "global_state" in step_info
    assert "global_state" in step_info_2


def test_hierarchical_wrapper_3d_no_crash():
    cfg = load_config(CONFIG_3D)
    base_env = UAVBackscatterEnv(cfg)
    env = HierarchicalUAVBackscatterEnv(base_env)
    obs, info = env.reset(seed=123)
    mask = env.get_action_mask(0)
    next_obs, reward, terminated, truncated, step_info = env.step([0 for _ in range(env.num_uav)])

    assert env.action_size == 10
    assert mask.shape == (10,)
    assert obs.shape == (env.num_uav, env.observation_dim)
    assert next_obs.shape == (env.num_uav, env.observation_dim)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "hierarchical_action" in step_info
