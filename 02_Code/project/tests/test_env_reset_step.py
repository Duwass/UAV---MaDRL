from __future__ import annotations

import numpy as np

from envs.uav_backscatter_env import MODE_IDLE, UAVBackscatterEnv, encode_action
from tests.conftest import make_test_config


def test_env_reset_returns_valid_observation():
    cfg = make_test_config()
    env = UAVBackscatterEnv(cfg)
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (cfg["network"]["num_uav"], env.observation_dim)
    assert "global_state" in info


def test_env_step_returns_correct_types():
    cfg = make_test_config()
    env = UAVBackscatterEnv(cfg)
    obs, _ = env.reset(seed=123)
    action = encode_action(0, 0, MODE_IDLE, env.num_iot)
    next_obs, reward, terminated, truncated, info = env.step([action])
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_truncates_at_max_steps():
    cfg = make_test_config()
    cfg["simulation"]["max_steps"] = 3
    env = UAVBackscatterEnv(cfg)
    obs, _ = env.reset(seed=123)
    action = encode_action(0, 0, MODE_IDLE, env.num_iot)
    truncated = False
    steps = 0
    while not truncated:
        obs, reward, terminated, truncated, info = env.step([action])
        steps += 1
    assert steps == 3
    assert truncated is True

