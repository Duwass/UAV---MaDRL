from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_NUM_MOVEMENT_ACTIONS = 11
DEFAULT_NUM_IOT = 15
DEFAULT_NUM_TARGETS = DEFAULT_NUM_IOT + 1
DEFAULT_NUM_MODES = 6
NO_TARGET_INDEX = 0

MODE_IDLE = 0
MODE_HARVEST = 1
MODE_BACKSCATTER = 2
MODE_ACTIVE = 3
MODE_RELAY = 4
MODE_AVOID_JAMMER = 5

MOVE_UP = 9
MOVE_DOWN = 10
TARGET_REQUIRED_MODES = frozenset({MODE_BACKSCATTER, MODE_ACTIVE, MODE_RELAY})


@dataclass(frozen=True)
class FactorizedAction:
    movement: int
    target: int
    mode: int


def flat_action_dim(
    num_iot: int = DEFAULT_NUM_IOT,
    num_movement_actions: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
) -> int:
    return int(num_movement_actions) * (int(num_iot) + 1) * DEFAULT_NUM_MODES


def encode_factorized_action(
    movement: int | FactorizedAction,
    target: int | None = None,
    mode: int | None = None,
    num_iot: int = DEFAULT_NUM_IOT,
    num_movement_actions: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
) -> int:
    if isinstance(movement, FactorizedAction):
        action = movement
        movement = action.movement
        target = action.target
        mode = action.mode
    if target is None or mode is None:
        raise ValueError("target and mode are required when movement is not a FactorizedAction.")

    movement = int(movement)
    target = int(target)
    mode = int(mode)
    num_iot = int(num_iot)
    num_movement_actions = int(num_movement_actions)
    if not 0 <= movement < num_movement_actions:
        raise ValueError(f"movement must be in [0, {num_movement_actions - 1}], got {movement}")
    if not 0 <= target <= num_iot:
        raise ValueError(f"target must be in [0, {num_iot}], got {target}")
    if not 0 <= mode < DEFAULT_NUM_MODES:
        raise ValueError(f"mode must be in [0, {DEFAULT_NUM_MODES - 1}], got {mode}")
    return int(((movement * (num_iot + 1)) + target) * DEFAULT_NUM_MODES + mode)


def decode_flat_action(
    flat_action: int,
    num_iot: int = DEFAULT_NUM_IOT,
    num_movement_actions: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
) -> FactorizedAction:
    flat_action = int(flat_action)
    num_iot = int(num_iot)
    num_movement_actions = int(num_movement_actions)
    if flat_action < 0 or flat_action >= flat_action_dim(num_iot, num_movement_actions):
        return FactorizedAction(MODE_IDLE, NO_TARGET_INDEX, MODE_IDLE)
    mode = flat_action % DEFAULT_NUM_MODES
    quotient = flat_action // DEFAULT_NUM_MODES
    target = quotient % (num_iot + 1)
    movement = quotient // (num_iot + 1)
    return FactorizedAction(int(movement), int(target), int(mode))


def sanitize_factorized_action(action: FactorizedAction) -> FactorizedAction:
    movement = int(action.movement)
    target = int(action.target)
    mode = int(action.mode)

    if mode == MODE_IDLE:
        return FactorizedAction(movement, NO_TARGET_INDEX, MODE_IDLE)
    if mode in TARGET_REQUIRED_MODES and target == NO_TARGET_INDEX:
        return FactorizedAction(movement, NO_TARGET_INDEX, MODE_IDLE)
    return FactorizedAction(movement, target, mode)


def build_local_movement_mask_from_obs(
    obs: np.ndarray,
    movement_count: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
    eps: float = 1.0e-6,
) -> np.ndarray:
    values = np.asarray(obs, dtype=np.float32)
    if values.shape[0] < 3:
        raise ValueError("local observation must contain own x/y/z at indexes 0, 1, and 2.")
    mask = np.ones(int(movement_count), dtype=bool)
    z_norm = float(values[2])
    if mask.shape[0] > MOVE_UP and z_norm >= 1.0 - float(eps):
        mask[MOVE_UP] = False
    if mask.shape[0] > MOVE_DOWN and z_norm <= 0.0 + float(eps):
        mask[MOVE_DOWN] = False
    return mask
