from __future__ import annotations

from pathlib import Path

import pytest

from envs.channel_model import distance_2d, distance_3d, horizontal_distance_2d
from envs.entities import IoTDevice, Jammer, UAV
from envs.uav_backscatter_env import UAVBackscatterEnv, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_uav_position_is_3d():
    uav = UAV(id=0, x=10.0, y=20.0, z=30.0, energy=100.0, coverage_radius=50.0, max_speed=5.0)
    assert uav.position() == (10.0, 20.0, 30.0)
    assert uav.h == 30.0

    legacy_uav = UAV(id=1, x=10.0, y=20.0, h=40.0, energy=100.0, coverage_radius=50.0, max_speed=5.0)
    assert legacy_uav.position() == (10.0, 20.0, 40.0)
    legacy_uav.h = 45.0
    assert legacy_uav.z == 45.0


def test_iot_position_is_3d_and_ground_altitude():
    iot = IoTDevice(
        id=0,
        x=1.0,
        y=2.0,
        device_type=1,
        queue=0,
        queue_capacity=10,
        energy=1.0,
        energy_capacity=10.0,
        packet_arrival_prob=0.0,
    )
    assert iot.position() == (1.0, 2.0, 0.0)


def test_jammer_position_is_3d():
    jammer = Jammer(id=0, x=3.0, y=4.0, power=1.0, speed=2.0, radius=100.0, mobility="static", z=5.0)
    assert jammer.position() == (3.0, 4.0, 5.0)
    assert jammer.h == 5.0


def test_distance_3d_accepts_2d_and_3d_inputs():
    assert distance_3d((0.0, 0.0), (3.0, 4.0, 12.0)) == pytest.approx(13.0)
    assert distance_3d((0.0, 0.0), (0.0, 0.0), min_distance=1.0) == pytest.approx(1.0)


def test_horizontal_distance_2d_ignores_z():
    assert horizontal_distance_2d((0.0, 0.0, 100.0), (3.0, 4.0, -20.0)) == pytest.approx(5.0)
    assert distance_2d((0.0, 0.0, 100.0), (3.0, 4.0, -20.0)) == pytest.approx(5.0)


def test_3d_config_loads_and_env_resets():
    env = UAVBackscatterEnv(PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml")
    obs, info = env.reset(seed=123)

    assert obs.shape == (env.num_uav, env.observation_dim)
    assert "global_state" in info
    assert all(len(uav.position()) == 3 and uav.z == pytest.approx(100.0) for uav in env.uavs)
    assert all(iot.z == pytest.approx(0.0) for iot in env.iot_devices)
    assert all(rf_source.z == pytest.approx(0.0) for rf_source in env.rf_sources)
    assert all(jammer.z == pytest.approx(100.0) for jammer in env.jammers)


def test_legacy_2d_config_still_resets():
    env = UAVBackscatterEnv(PROJECT_ROOT / "configs" / "scenario_4_backscatter_types_calibrated.yaml")
    obs, info = env.reset(seed=123)

    assert obs.shape == (env.num_uav, env.observation_dim)
    assert "global_state" in info
    assert all(len(uav.position()) == 3 and uav.z == pytest.approx(100.0) for uav in env.uavs)
    assert all(iot.position()[2] == pytest.approx(0.0) for iot in env.iot_devices)
    assert all(jammer.position()[2] == pytest.approx(0.0) for jammer in env.jammers)


def test_configured_2d_and_3d_initial_positions_are_normalized():
    cfg = load_config(PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml")
    cfg["network"]["num_iot"] = 2
    cfg["network"]["num_jammer"] = 2
    cfg["network"]["num_rf_sources"] = 2
    cfg["iot"]["device_types"] = [1, 2]
    cfg["iot"]["altitude"] = 1.5
    cfg["iot"]["initial_positions"] = [[10.0, 20.0], [30.0, 40.0, 5.0]]
    cfg["jammer"]["altitude"] = 7.5
    cfg["jammer"]["initial_positions"] = [[100.0, 110.0], [120.0, 130.0, 15.0]]
    cfg["rf_source"]["altitude"] = 2.5
    cfg["rf_source"]["initial_positions"] = [[200.0, 210.0], [220.0, 230.0, 9.0]]

    env = UAVBackscatterEnv(cfg)
    env.reset(seed=123)

    assert env.iot_devices[0].position() == (10.0, 20.0, 1.5)
    assert env.iot_devices[1].position() == (30.0, 40.0, 5.0)
    assert env.jammers[0].position() == (100.0, 110.0, 7.5)
    assert env.jammers[1].position() == (120.0, 130.0, 15.0)
    assert env.rf_sources[0].position() == (200.0, 210.0, 2.5)
    assert env.rf_sources[1].position() == (220.0, 230.0, 9.0)
