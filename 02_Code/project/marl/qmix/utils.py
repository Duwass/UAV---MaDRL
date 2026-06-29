from __future__ import annotations

from typing import Any

import numpy as np
import torch


def apply_action_mask(q_values: torch.Tensor, action_mask: torch.Tensor, invalid_value: float = -1.0e9) -> torch.Tensor:
    mask = action_mask.to(device=q_values.device, dtype=torch.bool)
    return q_values.masked_fill(~mask, invalid_value)


def masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor, invalid_value: float = -1.0e9) -> torch.Tensor:
    masked = apply_action_mask(q_values, action_mask, invalid_value=invalid_value)
    all_invalid = action_mask.to(device=q_values.device).sum(dim=-1) <= 0
    actions = masked.argmax(dim=-1)
    return torch.where(all_invalid, torch.zeros_like(actions), actions)


def masked_epsilon_greedy(
    q_values: np.ndarray | torch.Tensor,
    action_mask: np.ndarray | torch.Tensor,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if isinstance(q_values, torch.Tensor):
        q_np = q_values.detach().cpu().numpy()
    else:
        q_np = np.asarray(q_values)
    if isinstance(action_mask, torch.Tensor):
        mask_np = action_mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(action_mask)

    if q_np.ndim == 1:
        q_np = q_np[None, :]
        mask_np = mask_np[None, :]

    actions = np.zeros(q_np.shape[0], dtype=np.int64)
    fallback_count = 0
    for idx in range(q_np.shape[0]):
        valid = np.flatnonzero(mask_np[idx] > 0)
        if valid.size == 0:
            actions[idx] = 0
            fallback_count += 1
            continue
        if rng.random() < float(epsilon):
            actions[idx] = int(rng.choice(valid))
        else:
            masked_q = np.full(q_np.shape[1], -1.0e9, dtype=np.float32)
            masked_q[valid] = q_np[idx, valid]
            actions[idx] = int(masked_q.argmax())
    return actions, fallback_count


def to_numpy_batch(batch: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in batch.items()}

