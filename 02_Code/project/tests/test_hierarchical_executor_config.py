from envs.hierarchical_action import HierarchicalActionExecutor
from envs.uav_backscatter_env import UAVBackscatterEnv
from tests.conftest import make_test_config


def test_executor_accepts_fairness_weights_and_preserves_defaults():
    default_executor = HierarchicalActionExecutor(env=None)
    assert default_executor.fairness_weight == 1.0
    assert default_executor.underserved_weight == 1.0
    assert default_executor.repeat_target_penalty == 0.0
    assert default_executor.queue_weight == 2.0
    assert default_executor.sinr_weight == 1.5
    assert default_executor.jammer_risk_weight == 1.5

    configured = HierarchicalActionExecutor(
        env=None,
        config={
            "fairness_weight": 3.0,
            "underserved_weight": 2.0,
            "repeat_target_penalty": 0.5,
            "queue_weight": 4.0,
            "sinr_weight": 1.0,
            "jammer_risk_weight": 2.5,
        },
    )
    assert configured.fairness_weight == 3.0
    assert configured.underserved_weight == 2.0
    assert configured.repeat_target_penalty == 0.5
    assert configured.queue_weight == 4.0
    assert configured.sinr_weight == 1.0
    assert configured.jammer_risk_weight == 2.5


def test_executor_configured_weights_still_return_valid_actions():
    env = UAVBackscatterEnv(make_test_config())
    env.reset(seed=123)
    executor = HierarchicalActionExecutor(env, {"fairness_weight": 3.0, "underserved_weight": 3.0})
    actions = executor.build_original_actions([7 for _ in range(env.num_uav)])
    assert len(actions) == env.num_uav
    assert all(0 <= action < env.action_size for action in actions)

