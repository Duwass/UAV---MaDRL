import numpy as np

from marl.qmix.replay_buffer import EpisodeReplayBuffer


def make_episode(length: int):
    return {
        "observations": np.zeros((length, 2, 4), dtype=np.float32),
        "global_states": np.zeros((length, 9), dtype=np.float32),
        "actions": np.zeros((length, 2), dtype=np.int64),
        "rewards": np.ones(length, dtype=np.float32),
        "next_observations": np.zeros((length, 2, 4), dtype=np.float32),
        "next_global_states": np.zeros((length, 9), dtype=np.float32),
        "dones": np.zeros(length, dtype=np.float32),
        "action_masks": np.ones((length, 2, 10), dtype=np.float32),
        "next_action_masks": np.ones((length, 2, 10), dtype=np.float32),
    }


def test_episode_replay_buffer_push_sample_and_padding():
    buffer = EpisodeReplayBuffer(capacity=4, seed=123)
    buffer.push_episode(make_episode(3))
    buffer.push_episode(make_episode(5))
    batch = buffer.sample(2)
    assert len(buffer) == 2
    assert batch["observations"].shape == (2, 5, 2, 4)
    assert batch["filled"].shape == (2, 5)
    assert batch["filled"].sum() == 8

