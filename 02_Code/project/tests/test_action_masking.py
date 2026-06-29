from __future__ import annotations

from envs.uav_backscatter_env import MODE_ACTIVE, MODE_BACKSCATTER, MODE_HARVEST, UAVBackscatterEnv, encode_action
from tests.conftest import co_locate_first_iot, make_test_config


def test_action_mask_blocks_backscatter_when_idle():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 0.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=4)
    co_locate_first_iot(env)
    env.iot_devices[0].queue = 2
    mask = env.get_action_mask(0)
    assert mask[encode_action(0, 1, MODE_BACKSCATTER, env.num_iot)] == 0


def test_action_mask_blocks_active_when_busy():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 1.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=4)
    co_locate_first_iot(env)
    env.iot_devices[0].queue = 2
    env.iot_devices[0].energy = 5
    mask = env.get_action_mask(0)
    assert mask[encode_action(0, 1, MODE_ACTIVE, env.num_iot)] == 0


def test_action_mask_allows_harvest_when_busy_in_coverage():
    cfg = make_test_config()
    cfg["channel"]["primary_busy_prob"] = 1.0
    env = UAVBackscatterEnv(cfg)
    env.reset(seed=4)
    co_locate_first_iot(env)
    mask = env.get_action_mask(0)
    assert mask[encode_action(0, 1, MODE_HARVEST, env.num_iot)] == 1

