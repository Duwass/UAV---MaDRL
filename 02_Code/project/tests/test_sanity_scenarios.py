from __future__ import annotations

from pathlib import Path

from envs.uav_backscatter_env import MODE_IDLE, UAVBackscatterEnv, encode_action


def test_short_episode_for_each_config_without_crashing():
    root = Path(__file__).resolve().parents[1]
    for config_path in sorted((root / "configs").glob("scenario_*.yaml")):
        env = UAVBackscatterEnv(config_path)
        obs, info = env.reset(seed=7)
        action = encode_action(0, 0, MODE_IDLE, env.num_iot)
        terminated = False
        truncated = False
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step([action] * env.num_uav)
            assert obs.shape == (env.num_uav, env.observation_dim)
            if terminated or truncated:
                break
        env.close()

