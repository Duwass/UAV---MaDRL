import yaml

from marl.qmix.qmix_trainer import QMIXTrainer


def test_qmix_trainer_initializes_hierarchical_env_and_smoke_trains(tmp_path):
    config = {
        "env_config": "configs/scenario_4_backscatter_types_calibrated.yaml",
        "training_interface": {"type": "hierarchical"},
        "training": {
            "total_episodes": 1,
            "max_steps_per_episode": 2,
            "eval_interval_episodes": 1,
            "eval_episodes": 1,
            "save_interval_episodes": 1,
            "seed": 123,
        },
        "qmix": {
            "device": "cpu",
            "use_agent_id": True,
            "agent_hidden_sizes": [16],
            "mixing_embed_dim": 8,
            "hypernet_hidden_dim": 16,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 1,
            "replay_capacity": 4,
            "min_replay_size": 1,
            "target_update_steps": 5,
            "gradient_clip_norm": 10.0,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 10,
            "double_q": True,
        },
        "logging": {"output_prefix": "qmix_pytest_smoke", "save_checkpoints": True},
    }
    config_path = tmp_path / "qmix_pytest_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    trainer = QMIXTrainer(config_path)
    assert trainer.action_dim == 10
    assert trainer.n_agents == 2
    train_df, eval_df = trainer.train()
    assert len(train_df) == 1
    assert len(eval_df) == 1

