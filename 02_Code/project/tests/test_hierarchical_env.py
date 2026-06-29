import yaml

from envs.hierarchical_env import HierarchicalUAVBackscatterEnv
from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ddqn.ddqn_trainer import DDQNTrainer
from tests.conftest import make_test_config


def test_hierarchical_env_reset_step_and_mask():
    base_env = UAVBackscatterEnv(make_test_config())
    env = HierarchicalUAVBackscatterEnv(base_env)
    obs, info = env.reset(seed=123)
    assert obs.shape[0] == env.num_uav
    mask = env.get_action_mask(0)
    assert mask.shape == (env.action_size,)
    assert env.action_size == 10
    obs, reward, terminated, truncated, info = env.step([1 for _ in range(env.num_uav)])
    assert obs.shape[0] == env.num_uav
    assert isinstance(reward, float)
    assert "hierarchical_action" in info
    assert "fallback_rate" in info["episode_metrics"]


def test_ddqn_trainer_initializes_hierarchical_interface(tmp_path):
    config = {
        "env_config": "configs/scenario_0_no_jammer_calibrated.yaml",
        "training_interface": {"type": "hierarchical"},
        "training": {
            "total_episodes": 1,
            "max_steps_per_episode": 2,
            "eval_interval_episodes": 1,
            "eval_episodes": 1,
            "save_interval_episodes": 1,
            "seed": 123,
        },
        "ddqn": {
            "state_type": "concat_global_local",
            "network_type": "standard",
            "hidden_sizes": [16],
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 4,
            "replay_capacity": 100,
            "min_replay_size": 4,
            "target_update_steps": 10,
            "gradient_clip_norm": 10.0,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 100,
            "device": "cpu",
        },
        "logging": {
            "output_prefix": "hier_pytest_smoke",
            "save_frame_logs": False,
            "save_checkpoints": True,
        },
    }
    config_path = tmp_path / "hier_pytest_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    trainer = DDQNTrainer(config_path)
    assert trainer.action_dim == 10
    train_df, eval_df = trainer.train()
    assert len(train_df) == 1
    assert len(eval_df) >= 1
