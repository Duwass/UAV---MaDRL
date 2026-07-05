from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl.ctde.utils import FactorizedAction, sanitize_factorized_action


@dataclass(frozen=True)
class FactorizedActionDecision:
    raw_action: FactorizedAction
    action: FactorizedAction
    head_diagnostics: dict[str, Any] | None = None

    @property
    def sanitizer_changed(self) -> bool:
        return self.raw_action != self.action


def select_factorized_action_from_logits(
    movement_logits: np.ndarray,
    target_logits: np.ndarray,
    mode_logits: np.ndarray,
    movement_mask: np.ndarray | None = None,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    selection_mode: str = "epsilon_argmax",
    temperature: float = 1.0,
) -> FactorizedAction:
    return select_factorized_action_decision(
        movement_logits,
        target_logits,
        mode_logits,
        movement_mask=movement_mask,
        epsilon=epsilon,
        rng=rng,
        selection_mode=selection_mode,
        temperature=temperature,
    ).action


def select_factorized_action_decision(
    movement_logits: np.ndarray,
    target_logits: np.ndarray,
    mode_logits: np.ndarray,
    movement_mask: np.ndarray | None = None,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    selection_mode: str = "epsilon_argmax",
    temperature: float = 1.0,
) -> FactorizedActionDecision:
    rng = rng if rng is not None else np.random.default_rng()
    movement, movement_diag = _select_index_with_diagnostics(
        movement_logits,
        epsilon,
        rng,
        movement_mask,
        selection_mode,
        temperature,
    )
    target, target_diag = _select_index_with_diagnostics(
        target_logits,
        epsilon,
        rng,
        None,
        selection_mode,
        temperature,
    )
    mode, mode_diag = _select_index_with_diagnostics(
        mode_logits,
        epsilon,
        rng,
        None,
        selection_mode,
        temperature,
    )
    raw_action = FactorizedAction(movement, target, mode)
    return FactorizedActionDecision(
        raw_action=raw_action,
        action=sanitize_factorized_action(raw_action),
        head_diagnostics={
            "movement_top1_prob": movement_diag["top1_prob"],
            "movement_top2_prob": movement_diag["top2_prob"],
            "movement_top1_top2_margin": movement_diag["top1_top2_margin"],
            "target_top1_prob": target_diag["top1_prob"],
            "target_top2_prob": target_diag["top2_prob"],
            "target_top1_top2_margin": target_diag["top1_top2_margin"],
            "mode_top1_prob": mode_diag["top1_prob"],
            "mode_top2_prob": mode_diag["top2_prob"],
            "mode_top1_top2_margin": mode_diag["top1_top2_margin"],
        },
    )


def _select_index(
    logits: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> int:
    index, _ = _select_index_with_diagnostics(logits, epsilon, rng, mask, "epsilon_argmax", 1.0)
    return index


def _select_index_with_diagnostics(
    logits: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    mask: np.ndarray | None,
    selection_mode: str,
    temperature: float,
) -> tuple[int, dict[str, float | None]]:
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
        return 0, {"top1_prob": None, "top2_prob": None, "top1_top2_margin": None}
    probabilities = _valid_probabilities(values, valid, temperature)
    diagnostics = _probability_diagnostics(probabilities)
    mode = str(selection_mode)
    if mode == "stochastic":
        return int(rng.choice(valid, p=probabilities)), diagnostics
    if mode != "epsilon_argmax":
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    if float(epsilon) > 0.0 and rng.random() < float(epsilon):
        return int(rng.choice(valid)), diagnostics
    best_valid_offset = int(np.argmax(values[valid]))
    return int(valid[best_valid_offset]), diagnostics


def _valid_probabilities(values: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1.0e-8)
    valid_logits = values[valid].astype(np.float64) / safe_temperature
    shifted = valid_logits - np.max(valid_logits)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(valid.shape[0], 1.0 / float(valid.shape[0]), dtype=np.float64)
    return exp_values / total


def _probability_diagnostics(probabilities: np.ndarray) -> dict[str, float | None]:
    if probabilities.size <= 0:
        return {"top1_prob": None, "top2_prob": None, "top1_top2_margin": None}
    ordered = np.sort(probabilities)[::-1]
    top1 = float(ordered[0])
    top2 = float(ordered[1]) if ordered.size > 1 else 0.0
    return {
        "top1_prob": top1,
        "top2_prob": top2,
        "top1_top2_margin": float(top1 - top2),
    }
