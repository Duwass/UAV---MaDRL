from __future__ import annotations

import numpy as np

from marl.ddqn.replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_sample_shapes():
    buffer = ReplayBuffer(capacity=10, seed=1)
    for idx in range(6):
        buffer.push(
            state=np.ones(4, dtype=np.float32) * idx,
            uav_id=idx % 2,
            action=idx,
            reward=float(idx),
            next_state=np.ones(4, dtype=np.float32) * (idx + 1),
            done=False,
            action_mask=np.ones(5, dtype=np.float32),
            next_action_mask=np.ones(5, dtype=np.float32),
        )
    batch = buffer.sample(4)
    assert len(buffer) == 6
    assert batch["state"].shape == (4, 4)
    assert batch["uav_id"].shape == (4,)
    assert batch["action"].shape == (4,)
    assert batch["action_mask"].shape == (4, 5)
    assert batch["next_action_mask"].shape == (4, 5)

