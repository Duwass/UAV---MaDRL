import numpy as np

from envs.hierarchical_env import HierarchicalUAVBackscatterEnv
from envs.uav_backscatter_env import UAVBackscatterEnv
from tests.conftest import make_test_config


def test_disabled_high_level_action_is_masked_out():
    env = HierarchicalUAVBackscatterEnv(
        UAVBackscatterEnv(make_test_config()),
        hierarchical_actions_config={"disabled_actions": [7]},
    )
    env.reset(seed=123)
    mask = env.get_action_mask(0)
    assert mask.shape == (env.action_size,)
    assert mask[7] == 0
    assert np.any(mask)


def test_disabled_balance_action_falls_back_safely_on_manual_step():
    env = HierarchicalUAVBackscatterEnv(
        UAVBackscatterEnv(make_test_config()),
        hierarchical_actions_config={"disabled_actions": [7]},
    )
    env.reset(seed=123)
    _, _, _, _, info = env.step([7 for _ in range(env.num_uav)])
    names = info["hierarchical_action"]["high_level_action_names"]
    assert "BALANCE_UNDERSERVED_IOT" not in names

