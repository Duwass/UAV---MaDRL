from __future__ import annotations

import numpy as np

from marl.ctde.utils import FactorizedAction, sanitize_factorized_action


def select_factorized_action_from_logits(
    movement_logits: np.ndarray,
    target_logits: np.ndarray,
    mode_logits: np.ndarray,
    movement_mask: np.ndarray | None = None,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FactorizedAction:
    rng = rng if rng is not None else np.random.default_rng()
    movement = _select_index(movement_logits, epsilon, rng, movement_mask)
    target = _select_index(target_logits, epsilon, rng)
    mode = _select_index(mode_logits, epsilon, rng)
    return sanitize_factorized_action(FactorizedAction(movement, target, mode))


def _select_index(
    logits: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> int:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("logits must be a 1D array.")
    if mask is None:
        valid = np.arange(values.shape[0])
    else:
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != values.shape:
            raise ValueError("mask shape must match logits shape.")
        valid = np.flatnonzero(mask_values)
    if valid.size == 0:
        return 0
    if float(epsilon) > 0.0 and rng.random() < float(epsilon):
        return int(rng.choice(valid))
    best_valid_offset = int(np.argmax(values[valid]))
    return int(valid[best_valid_offset])
