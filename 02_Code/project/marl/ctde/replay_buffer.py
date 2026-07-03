from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CTDETransition:
    obs: np.ndarray
    state: np.ndarray
    movement_actions: np.ndarray
    target_actions: np.ndarray
    mode_actions: np.ndarray
    flat_actions: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_state: np.ndarray
    done: bool
    movement_masks: np.ndarray | None = None
    info: dict[str, Any] | None = None


class CTDEReplayBuffer:
    """Stores local obs for actor separation and global state for future centralized critic training."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.buffer: deque[CTDETransition] = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        movement_actions: np.ndarray,
        target_actions: np.ndarray,
        mode_actions: np.ndarray,
        flat_actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        movement_masks: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.buffer.append(
            CTDETransition(
                obs=np.asarray(obs, dtype=np.float32),
                state=np.asarray(state, dtype=np.float32),
                movement_actions=np.asarray(movement_actions, dtype=np.int64),
                target_actions=np.asarray(target_actions, dtype=np.int64),
                mode_actions=np.asarray(mode_actions, dtype=np.int64),
                flat_actions=np.asarray(flat_actions, dtype=np.int64),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                movement_masks=None if movement_masks is None else np.asarray(movement_masks, dtype=bool),
                info=None if info is None else dict(info),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample batch_size={batch_size} from replay size={len(self.buffer)}")
        generator = rng if rng is not None else self.rng
        indices = generator.choice(len(self.buffer), size=int(batch_size), replace=False)
        batch = [self.buffer[int(idx)] for idx in indices]
        sample = {
            "obs": np.stack([item.obs for item in batch]),
            "state": np.stack([item.state for item in batch]),
            "movement_actions": np.stack([item.movement_actions for item in batch]),
            "target_actions": np.stack([item.target_actions for item in batch]),
            "mode_actions": np.stack([item.mode_actions for item in batch]),
            "flat_actions": np.stack([item.flat_actions for item in batch]),
            "reward": np.asarray([item.reward for item in batch], dtype=np.float32),
            "next_obs": np.stack([item.next_obs for item in batch]),
            "next_state": np.stack([item.next_state for item in batch]),
            "done": np.asarray([item.done for item in batch], dtype=bool),
        }
        if all(item.movement_masks is not None for item in batch):
            sample["movement_masks"] = np.stack([item.movement_masks for item in batch])
        return sample

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
