from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    uav_id: int
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.buffer: deque[Transition] = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        uav_id: int,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                uav_id=int(uav_id),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                action_mask=np.asarray(action_mask, dtype=np.float32),
                next_action_mask=np.asarray(next_action_mask, dtype=np.float32),
            )
        )

    def sample(self, batch_size: int) -> dict[str, Any]:
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample batch_size={batch_size} from replay size={len(self.buffer)}")
        indices = self.rng.choice(len(self.buffer), size=int(batch_size), replace=False)
        batch = [self.buffer[int(idx)] for idx in indices]
        return {
            "state": np.stack([item.state for item in batch]),
            "uav_id": np.asarray([item.uav_id for item in batch], dtype=np.int64),
            "action": np.asarray([item.action for item in batch], dtype=np.int64),
            "reward": np.asarray([item.reward for item in batch], dtype=np.float32),
            "next_state": np.stack([item.next_state for item in batch]),
            "done": np.asarray([item.done for item in batch], dtype=np.float32),
            "action_mask": np.stack([item.action_mask for item in batch]),
            "next_action_mask": np.stack([item.next_action_mask for item in batch]),
        }

    def __len__(self) -> int:
        return len(self.buffer)

