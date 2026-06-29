from __future__ import annotations

import warnings

import numpy as np
import torch


ALL_ZERO_MASK_WARNINGS = 0


def apply_action_mask(q_values: torch.Tensor, mask: torch.Tensor, invalid_value: float = -1.0e9) -> torch.Tensor:
    mask_bool = mask.to(dtype=torch.bool, device=q_values.device)
    return q_values.masked_fill(~mask_bool, invalid_value)


def masked_argmax(q_values: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> int:
    global ALL_ZERO_MASK_WARNINGS
    if isinstance(q_values, torch.Tensor):
        q_np = q_values.detach().cpu().numpy()
    else:
        q_np = np.asarray(q_values)
    mask_np = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask, dtype=np.int8)
    valid = np.flatnonzero(mask_np > 0)
    if valid.size == 0:
        ALL_ZERO_MASK_WARNINGS += 1
        warnings.warn("Action mask is all zeros; falling back to action 0.", RuntimeWarning, stacklevel=2)
        return 0
    return int(valid[np.argmax(q_np[valid])])


def masked_epsilon_greedy(
    q_values: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    global ALL_ZERO_MASK_WARNINGS
    mask_np = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask, dtype=np.int8)
    valid = np.flatnonzero(mask_np > 0)
    if valid.size == 0:
        ALL_ZERO_MASK_WARNINGS += 1
        warnings.warn("Action mask is all zeros; falling back to action 0.", RuntimeWarning, stacklevel=2)
        return 0
    if rng.random() < float(epsilon):
        return int(rng.choice(valid))
    return masked_argmax(q_values, mask_np)

