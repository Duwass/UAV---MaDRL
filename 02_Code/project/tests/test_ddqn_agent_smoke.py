from __future__ import annotations

import subprocess
import sys
import yaml

import numpy as np

from marl.ddqn.ddqn_agent import CentralizedFactorizedDDQNAgent
from marl.ddqn.replay_buffer import ReplayBuffer


def test_ddqn_agent_train_step_save_load(tmp_path):
    state_dim = 6
    action_dim = 4
    agent = CentralizedFactorizedDDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_uav=2,
        hidden_sizes=[16],
        batch_size=4,
        target_update_steps=2,
        seed=7,
    )
    buffer = ReplayBuffer(capacity=20, seed=7)
    for idx in range(8):
        mask = np.array([1, 0, 1, 1], dtype=np.float32)
        buffer.push(
            state=np.random.default_rng(idx).random(state_dim).astype(np.float32),
            uav_id=idx % 2,
            action=0 if idx % 2 == 0 else 2,
            reward=1.0,
            next_state=np.random.default_rng(idx + 10).random(state_dim).astype(np.float32),
            done=False,
            action_mask=mask,
            next_action_mask=mask,
        )
    metrics = agent.train_step(buffer.sample(4))
    assert metrics["loss"] >= 0.0
    action = agent.select_action(np.zeros(state_dim, dtype=np.float32), 0, np.array([0, 1, 0, 0]), deterministic=True)
    assert action == 1

    checkpoint = tmp_path / "agent.pt"
    agent.save_checkpoint(checkpoint)
    new_agent = CentralizedFactorizedDDQNAgent(state_dim, action_dim, 2, hidden_sizes=[16], seed=8)
    new_agent.load_checkpoint(checkpoint)
    assert new_agent.train_steps == agent.train_steps


def test_train_ddqn_script_smoke(tmp_path):
    config = {
        "env_config": "configs/scenario_0_no_jammer_calibrated.yaml",
        "training": {
            "total_episodes": 2,
            "max_steps_per_episode": 5,
            "eval_interval_episodes": 1,
            "eval_episodes": 1,
            "save_interval_episodes": 1,
            "seed": 123,
        },
        "ddqn": {
            "state_type": "concat_global_local",
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
        },
        "logging": {
            "output_prefix": "ddqn_pytest_smoke",
            "save_frame_logs": False,
            "save_checkpoints": True,
        },
    }
    config_path = tmp_path / "ddqn_pytest_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_ddqn.py",
            "--config",
            str(config_path),
            "--episodes",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Saved train log" in result.stdout
