from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from marl.ctde.utils import (
    DEFAULT_NUM_MODES,
    DEFAULT_NUM_MOVEMENT_ACTIONS,
    DEFAULT_NUM_TARGETS,
    MOVE_DOWN,
    MOVE_UP,
    NO_TARGET_INDEX,
    TARGET_REQUIRED_MODES,
    FactorizedAction,
)


BASE_ACTION_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "action_count",
    "policy_deterministic",
    "policy_epsilon",
    "movement_stay_rate",
    "movement_up_rate",
    "movement_down_rate",
    "vertical_action_rate_from_actions",
    "no_target_rate",
    "target_selected_rate",
    "idle_mode_rate",
    "target_required_no_target_rate",
    "sanitizer_changed_rate",
)


def summarize_action_diagnostics(
    actions: Sequence[FactorizedAction],
    raw_actions: Sequence[FactorizedAction] | None = None,
    agent_ids: Sequence[int] | None = None,
    num_agents: int | None = None,
    deterministic: bool | None = None,
    epsilon: float | None = None,
    movement_count: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
    target_count: int = DEFAULT_NUM_TARGETS,
    mode_count: int = DEFAULT_NUM_MODES,
) -> dict[str, Any]:
    """Summarize selected CTDE factorized actions without affecting behavior."""
    selected = list(actions)
    raw = list(raw_actions) if raw_actions is not None else None
    if raw is not None and len(raw) != len(selected):
        raise ValueError("raw_actions length must match actions length.")
    ids = _normalize_agent_ids(agent_ids, len(selected))
    inferred_agents = max(ids) + 1 if ids else 0
    agent_count = int(num_agents if num_agents is not None else inferred_agents)
    agent_count = max(agent_count, inferred_agents)

    action_count = len(selected)
    diagnostics: dict[str, Any] = {
        "action_count": int(action_count),
        "policy_deterministic": None if deterministic is None else bool(deterministic),
        "policy_epsilon": None if epsilon is None else float(epsilon),
    }
    diagnostics.update(_rate_fields(selected, raw, action_count, int(movement_count), int(target_count), int(mode_count)))
    diagnostics.update(_per_agent_fields(selected, raw, ids, agent_count))
    return diagnostics


def prefixed_action_diagnostic_keys(
    prefix: str,
    num_agents: int = 2,
    movement_count: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
    target_count: int = DEFAULT_NUM_TARGETS,
    mode_count: int = DEFAULT_NUM_MODES,
) -> tuple[str, ...]:
    keys = list(BASE_ACTION_DIAGNOSTIC_KEYS)
    keys.extend(f"movement_{idx}_rate" for idx in range(int(movement_count)))
    keys.extend(f"mode_{idx}_rate" for idx in range(int(mode_count)))
    keys.extend(f"target_{idx}_rate" for idx in range(int(target_count)))
    for agent_id in range(int(num_agents)):
        keys.extend(
            (
                f"agent{agent_id}_action_count",
                f"agent{agent_id}_idle_mode_rate",
                f"agent{agent_id}_no_target_rate",
                f"agent{agent_id}_target_selected_rate",
                f"agent{agent_id}_target_required_no_target_rate",
                f"agent{agent_id}_sanitizer_changed_rate",
                f"agent{agent_id}_vertical_action_rate_from_actions",
            )
        )
    return tuple(f"{prefix}{key}" for key in keys)


def prefix_action_diagnostics(
    diagnostics: dict[str, Any] | None,
    prefix: str,
    num_agents: int = 2,
    movement_count: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
    target_count: int = DEFAULT_NUM_TARGETS,
    mode_count: int = DEFAULT_NUM_MODES,
) -> dict[str, Any]:
    source = diagnostics if isinstance(diagnostics, dict) else {}
    return {
        key: _safe_value(source.get(key.removeprefix(prefix)))
        for key in prefixed_action_diagnostic_keys(prefix, num_agents, movement_count, target_count, mode_count)
    }


def _rate_fields(
    actions: list[FactorizedAction],
    raw_actions: list[FactorizedAction] | None,
    action_count: int,
    movement_count: int,
    target_count: int,
    mode_count: int,
) -> dict[str, Any]:
    if action_count <= 0:
        empty: dict[str, Any] = {
            "movement_stay_rate": None,
            "movement_up_rate": None,
            "movement_down_rate": None,
            "vertical_action_rate_from_actions": None,
            "no_target_rate": None,
            "target_selected_rate": None,
            "idle_mode_rate": None,
            "target_required_no_target_rate": None,
            "sanitizer_changed_rate": None if raw_actions is None else 0.0,
        }
        empty.update({f"movement_{idx}_rate": None for idx in range(movement_count)})
        empty.update({f"mode_{idx}_rate": None for idx in range(mode_count)})
        empty.update({f"target_{idx}_rate": None for idx in range(target_count)})
        return empty

    movements = Counter(int(action.movement) for action in actions)
    modes = Counter(int(action.mode) for action in actions)
    targets = Counter(int(action.target) for action in actions)
    raw_source = raw_actions if raw_actions is not None else actions
    target_required_no_target = sum(
        1
        for action in raw_source
        if int(action.mode) in TARGET_REQUIRED_MODES and int(action.target) == NO_TARGET_INDEX
    )
    changed = None
    if raw_actions is not None:
        changed = sum(1 for raw, sanitized in zip(raw_actions, actions) if raw != sanitized)

    fields: dict[str, Any] = {
        "movement_stay_rate": _rate(movements.get(0, 0), action_count),
        "movement_up_rate": _rate(movements.get(MOVE_UP, 0), action_count),
        "movement_down_rate": _rate(movements.get(MOVE_DOWN, 0), action_count),
        "vertical_action_rate_from_actions": _rate(movements.get(MOVE_UP, 0) + movements.get(MOVE_DOWN, 0), action_count),
        "no_target_rate": _rate(targets.get(NO_TARGET_INDEX, 0), action_count),
        "target_selected_rate": _rate(action_count - targets.get(NO_TARGET_INDEX, 0), action_count),
        "idle_mode_rate": _rate(modes.get(0, 0), action_count),
        "target_required_no_target_rate": _rate(target_required_no_target, action_count),
        "sanitizer_changed_rate": None if changed is None else _rate(changed, action_count),
    }
    fields.update({f"movement_{idx}_rate": _rate(movements.get(idx, 0), action_count) for idx in range(movement_count)})
    fields.update({f"mode_{idx}_rate": _rate(modes.get(idx, 0), action_count) for idx in range(mode_count)})
    fields.update({f"target_{idx}_rate": _rate(targets.get(idx, 0), action_count) for idx in range(target_count)})
    return fields


def _per_agent_fields(
    actions: list[FactorizedAction],
    raw_actions: list[FactorizedAction] | None,
    agent_ids: list[int],
    num_agents: int,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    raw_source = raw_actions if raw_actions is not None else actions
    for agent_id in range(int(num_agents)):
        idxs = [idx for idx, current in enumerate(agent_ids) if current == agent_id]
        count = len(idxs)
        agent_actions = [actions[idx] for idx in idxs]
        agent_raw = [raw_source[idx] for idx in idxs]
        changed = None
        if raw_actions is not None:
            changed = sum(1 for idx in idxs if raw_actions[idx] != actions[idx])
        idle_count = sum(1 for action in agent_actions if int(action.mode) == 0)
        no_target_count = sum(1 for action in agent_actions if int(action.target) == NO_TARGET_INDEX)
        vertical_count = sum(1 for action in agent_actions if int(action.movement) in (MOVE_UP, MOVE_DOWN))
        target_required_no_target = sum(
            1
            for action in agent_raw
            if int(action.mode) in TARGET_REQUIRED_MODES and int(action.target) == NO_TARGET_INDEX
        )
        fields.update(
            {
                f"agent{agent_id}_action_count": int(count),
                f"agent{agent_id}_idle_mode_rate": _rate_or_none(idle_count, count),
                f"agent{agent_id}_no_target_rate": _rate_or_none(no_target_count, count),
                f"agent{agent_id}_target_selected_rate": _rate_or_none(count - no_target_count, count),
                f"agent{agent_id}_target_required_no_target_rate": _rate_or_none(target_required_no_target, count),
                f"agent{agent_id}_sanitizer_changed_rate": None if changed is None else _rate_or_none(changed, count),
                f"agent{agent_id}_vertical_action_rate_from_actions": _rate_or_none(vertical_count, count),
            }
        )
    return fields


def _normalize_agent_ids(agent_ids: Sequence[int] | None, length: int) -> list[int]:
    if agent_ids is None:
        return [0 for _ in range(length)]
    ids = [int(value) for value in agent_ids]
    if len(ids) != int(length):
        raise ValueError("agent_ids length must match actions length.")
    return ids


def _rate_or_none(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return _rate(numerator, denominator)


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value
