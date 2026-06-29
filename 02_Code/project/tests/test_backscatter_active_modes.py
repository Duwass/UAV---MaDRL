from __future__ import annotations

from envs.uav_backscatter_env import MODE_ACTIVE, MODE_BACKSCATTER, UAVBackscatterEnv, encode_action
from tests.conftest import co_locate_first_iot, make_test_config


def test_backscatter_valid_during_busy_channel():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 1.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=3)
    co_locate_first_iot(env)
    iot = env.iot_devices[0]
    iot.queue = 2
    action = encode_action(0, 1, MODE_BACKSCATTER, env.num_iot)
    env.step([action])
    assert iot.queue == 1


def test_active_valid_when_energy_is_enough():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 0.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=3)
    co_locate_first_iot(env)
    iot = env.iot_devices[0]
    iot.queue = 2
    iot.energy = 2.0
    action = encode_action(0, 1, MODE_ACTIVE, env.num_iot)
    env.step([action])
    assert iot.queue == 0
    assert iot.energy == 1.0


def test_active_invalid_when_energy_is_insufficient():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 0.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=3)
    co_locate_first_iot(env)
    iot = env.iot_devices[0]
    iot.queue = 2
    iot.energy = 0.0
    action = encode_action(0, 1, MODE_ACTIVE, env.num_iot)
    env.step([action])
    assert iot.queue == 2
    assert iot.energy == 0.0

