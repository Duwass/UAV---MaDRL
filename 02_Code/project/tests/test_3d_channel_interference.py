from __future__ import annotations

from pathlib import Path

import pytest

from envs.channel_model import distance_3d, jammer_interference, received_power
from envs.uav_backscatter_env import MODE_ACTIVE, MODE_IDLE, UAVBackscatterEnv, encode_action


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
CONFIG_2D = PROJECT_ROOT / "configs" / "scenario_4_backscatter_types_calibrated.yaml"


def _idle_action(env: UAVBackscatterEnv) -> int:
    return encode_action(0, 0, MODE_IDLE, env.num_iot)


def test_path_loss_or_received_power_changes_with_altitude():
    flat_distance = distance_3d((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    raised_distance = distance_3d((0.0, 0.0, 0.0), (10.0, 0.0, 10.0))

    assert raised_distance > flat_distance
    assert received_power(1.0, raised_distance, 2.2) < received_power(1.0, flat_distance, 2.2)


def test_sinr_changes_when_uav_altitude_changes():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)
    env.jammers = []

    uav = env.uavs[0]
    iot = env.iot_devices[0]
    uav.x = iot.x
    uav.y = iot.y

    uav.z = iot.z
    low_altitude_sinr = env._compute_sinr(float(env.channel_cfg.get("tx_power_iot", 0.1)), iot, uav)

    uav.z = iot.z + 100.0
    high_altitude_sinr = env._compute_sinr(float(env.channel_cfg.get("tx_power_iot", 0.1)), iot, uav)

    assert high_altitude_sinr < low_altitude_sinr


def test_jammer_interference_changes_with_jammer_altitude():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)

    uav = env.uavs[0]
    jammer = env.jammers[0]
    jammer.x = uav.x
    jammer.y = uav.y
    jammer.radius = 1000.0
    jammer.is_active = True

    jammer.z = uav.z
    same_altitude = env._jammer_interference(jammer, uav)

    jammer.z = uav.z + 300.0
    higher_altitude = env._jammer_interference(jammer, uav)

    assert higher_altitude < same_altitude
    assert jammer_interference(jammer, uav, 2.2, "3d", "3d") == pytest.approx(higher_altitude)


def test_coverage_uses_3d_distance_in_3d_config():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)

    uav = env.uavs[0]
    iot = env.iot_devices[0]
    uav.x = iot.x
    uav.y = iot.y
    uav.z = iot.z + 100.0
    uav.coverage_radius = 50.0
    iot.queue = 1
    iot.energy = 10.0
    env.primary_busy = False

    active_action = encode_action(0, iot.id + 1, MODE_ACTIVE, env.num_iot)
    assert env._coverage_distance(uav, iot) == pytest.approx(100.0)
    assert env._in_coverage(uav, iot) is False
    assert env.get_action_mask(uav.id)[active_action] == 0

    uav.z = iot.z
    assert env._in_coverage(uav, iot) is True
    assert env.get_action_mask(uav.id)[active_action] == 1


def test_legacy_2d_config_still_works():
    env = UAVBackscatterEnv(CONFIG_2D)
    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, step_info = env.step([_idle_action(env) for _ in range(env.num_uav)])

    assert obs.shape == (env.num_uav, env.observation_dim)
    assert next_obs.shape == (env.num_uav, env.observation_dim)
    assert "global_state" in info
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "global_state" in step_info


def test_3d_config_reset_step_no_crash():
    env = UAVBackscatterEnv(CONFIG_3D)
    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, step_info = env.step([_idle_action(env) for _ in range(env.num_uav)])

    assert obs.shape == (env.num_uav, env.observation_dim)
    assert next_obs.shape == (env.num_uav, env.observation_dim)
    assert "global_state" in info
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "global_state" in step_info
