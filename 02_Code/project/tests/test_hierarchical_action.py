import numpy as np

from envs.hierarchical_action import (
    HIGH_LEVEL_ACTION_NAMES,
    HierarchicalActionExecutor,
)
from envs.uav_backscatter_env import UAVBackscatterEnv
from tests.conftest import make_test_config


def test_executor_returns_valid_original_actions():
    env = UAVBackscatterEnv(make_test_config())
    env.reset(seed=123)
    executor = HierarchicalActionExecutor(env)
    actions = executor.build_original_actions([1 for _ in range(env.num_uav)])
    assert len(actions) == env.num_uav
    for action in actions:
        assert 0 <= action < env.action_size


def test_executor_action_names_are_fixed():
    executor = HierarchicalActionExecutor(env=None)
    assert executor.num_high_level_actions() == 10
    assert executor.decode_high_level_action(0) == "IDLE_SAFE"
    assert set(HIGH_LEVEL_ACTION_NAMES) == set(range(10))
