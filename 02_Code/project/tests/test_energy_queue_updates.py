from __future__ import annotations

import pytest

from envs.uav_backscatter_env import MODE_ACTIVE, MODE_HARVEST, UAVBackscatterEnv, encode_action
from tests.conftest import co_locate_first_iot, make_test_config


def test_harvest_does_not_exceed_capacity():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 1.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=1)
    co_locate_first_iot(env)
    iot = env.iot_devices[0]
    iot.energy_capacity = 10.0
    iot.energy = 9.5
    action = encode_action(0, 1, MODE_HARVEST, env.num_iot)
    env.step([action])
    assert iot.energy <= iot.energy_capacity
    assert iot.energy == pytest.approx(10.0)


def test_active_transmission_consumes_energy():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 0.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=1)
    co_locate_first_iot(env)
    iot = env.iot_devices[0]
    iot.queue = 5
    iot.energy = 5.0
    action = encode_action(0, 1, MODE_ACTIVE, env.num_iot)
    env.step([action])
    assert iot.energy == pytest.approx(4.0)


def test_packet_queue_decreases_after_successful_transmission():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 0.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=1)
    co_locate_first_iot(env)
    iot = env.iot_devices[0]
    iot.queue = 5
    iot.energy = 5.0
    action = encode_action(0, 1, MODE_ACTIVE, env.num_iot)
    env.step([action])
    assert iot.queue == 3


def test_packet_queue_never_negative():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 0.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=1)
    co_locate_first_iot(env)
    iot = env.iot_devices[0]
    iot.queue = 1
    iot.energy = 5.0
    action = encode_action(0, 1, MODE_ACTIVE, env.num_iot)
    env.step([action])
    assert iot.queue == 0

