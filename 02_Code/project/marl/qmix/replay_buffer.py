from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class EpisodeReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.buffer: deque[dict[str, np.ndarray]] = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def push_episode(self, episode_data: dict[str, Any]) -> None:
        required = [
            "observations",
            "global_states",
            "actions",
            "rewards",
            "next_observations",
            "next_global_states",
            "dones",
            "action_masks",
            "next_action_masks",
        ]
        missing = [key for key in required if key not in episode_data]
        if missing:
            raise KeyError(f"Episode data missing keys: {missing}")
        self.buffer.append({key: np.asarray(episode_data[key]) for key in required})

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not self.buffer:
            raise ValueError("Cannot sample from an empty EpisodeReplayBuffer.")
        size = min(int(batch_size), len(self.buffer))
        indices = self.rng.choice(len(self.buffer), size=size, replace=False)
        episodes = [self.buffer[int(idx)] for idx in indices]
        max_t = max(int(ep["rewards"].shape[0]) for ep in episodes)
        sample: dict[str, np.ndarray] = {}

        for key in episodes[0]:
            shape = (size, max_t) + tuple(episodes[0][key].shape[1:])
            dtype = episodes[0][key].dtype
            sample[key] = np.zeros(shape, dtype=dtype)
        sample["filled"] = np.zeros((size, max_t), dtype=np.float32)

        for batch_idx, ep in enumerate(episodes):
            length = int(ep["rewards"].shape[0])
            for key, value in ep.items():
                sample[key][batch_idx, :length] = value
            sample["filled"][batch_idx, :length] = 1.0
        return sample

    def __len__(self) -> int:
        return len(self.buffer)

