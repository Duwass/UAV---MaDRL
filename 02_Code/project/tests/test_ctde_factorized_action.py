from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from envs.uav_backscatter_env import UAVBackscatterEnv, encode_action
from marl.ctde import factorized_policy
from marl.ctde.factorized_policy import select_factorized_action_from_logits
from marl.ctde.utils import (
    DEFAULT_NUM_IOT,
    DEFAULT_NUM_MODES,
    DEFAULT_NUM_MOVEMENT_ACTIONS,
    MODE_ACTIVE,
    MODE_BACKSCATTER,
    MODE_IDLE,
    MODE_RELAY,
    MOVE_DOWN,
    MOVE_UP,
    NO_TARGET_INDEX,
    FactorizedAction,
    build_local_movement_mask_from_obs,
    decode_flat_action,
    encode_factorized_action,
    flat_action_dim,
    sanitize_factorized_action,
)
from marl.ctde import utils as ctde_utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"


def test_factorized_encode_decode_roundtrip_3d():
    combinations = [
        (0, 0, MODE_IDLE),
        (1, 1, MODE_BACKSCATTER),
        (MOVE_UP, 7, MODE_ACTIVE),
        (MOVE_DOWN, DEFAULT_NUM_IOT, MODE_RELAY),
        (DEFAULT_NUM_MOVEMENT_ACTIONS - 1, DEFAULT_NUM_IOT, DEFAULT_NUM_MODES - 1),
    ]
    for movement, target, mode in combinations:
        flat = encode_factorized_action(movement, target, mode)
        assert decode_flat_action(flat) == FactorizedAction(movement, target, mode)


def test_factorized_flat_action_range_3d():
    for movement in range(DEFAULT_NUM_MOVEMENT_ACTIONS):
        for target in range(DEFAULT_NUM_IOT + 1):
            for mode in range(DEFAULT_NUM_MODES):
                flat = encode_factorized_action(movement, target, mode)
                assert 0 <= flat < 1056
    assert flat_action_dim() == 1056


def test_factorized_encode_matches_env_helper():
    flat = encode_factorized_action(MOVE_DOWN, DEFAULT_NUM_IOT, MODE_ACTIVE)
    env_flat = encode_action(MOVE_DOWN, DEFAULT_NUM_IOT, MODE_ACTIVE, DEFAULT_NUM_IOT, DEFAULT_NUM_MOVEMENT_ACTIONS)
    assert flat == env_flat


def test_sanitize_idle_forces_no_target():
    action = sanitize_factorized_action(FactorizedAction(3, 5, MODE_IDLE))
    assert action == FactorizedAction(3, NO_TARGET_INDEX, MODE_IDLE)


def test_sanitize_target_required_no_target_falls_back_idle():
    for mode in (MODE_BACKSCATTER, MODE_ACTIVE, MODE_RELAY):
        action = sanitize_factorized_action(FactorizedAction(2, NO_TARGET_INDEX, mode))
        assert action == FactorizedAction(2, NO_TARGET_INDEX, MODE_IDLE)


def test_local_movement_mask_shape():
    obs = np.zeros(114, dtype=np.float32)
    mask = build_local_movement_mask_from_obs(obs)
    assert mask.shape == (DEFAULT_NUM_MOVEMENT_ACTIONS,)
    assert mask.dtype == np.bool_


def test_local_movement_mask_blocks_up_at_z_max():
    obs = np.zeros(114, dtype=np.float32)
    obs[2] = 1.0
    mask = build_local_movement_mask_from_obs(obs)
    assert not bool(mask[MOVE_UP])
    assert bool(mask[MOVE_DOWN])


def test_local_movement_mask_blocks_down_at_z_min():
    obs = np.zeros(114, dtype=np.float32)
    obs[2] = 0.0
    mask = build_local_movement_mask_from_obs(obs)
    assert bool(mask[MOVE_UP])
    assert not bool(mask[MOVE_DOWN])


def test_select_factorized_action_greedy_no_mask():
    action = select_factorized_action_from_logits(
        movement_logits=np.asarray([0.0, 5.0, 1.0]),
        target_logits=np.asarray([0.0, 1.0, 7.0]),
        mode_logits=np.asarray([0.0, 2.0, 1.0]),
        epsilon=0.0,
    )
    assert action == FactorizedAction(1, 2, 1)


def test_select_factorized_action_applies_movement_mask():
    movement_logits = np.zeros(DEFAULT_NUM_MOVEMENT_ACTIONS, dtype=np.float32)
    movement_logits[MOVE_UP] = 99.0
    movement_logits[1] = 10.0
    movement_mask = np.ones(DEFAULT_NUM_MOVEMENT_ACTIONS, dtype=bool)
    movement_mask[MOVE_UP] = False
    action = select_factorized_action_from_logits(
        movement_logits=movement_logits,
        target_logits=np.asarray([0.0, 3.0]),
        mode_logits=np.asarray([0.0, 1.0]),
        movement_mask=movement_mask,
    )
    assert action.movement == 1
    assert action.target == 1
    assert action.mode == 1


def test_select_factorized_action_signature_has_no_env_or_state():
    forbidden = {"env", "state", "global_state", "critic", "executor"}
    params = set(inspect.signature(select_factorized_action_from_logits).parameters)
    assert forbidden.isdisjoint(params)


def test_ctde_utils_do_not_import_hierarchical_executor():
    source = inspect.getsource(ctde_utils) + inspect.getsource(factorized_policy)
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for item in forbidden)


def test_legacy_3d_env_still_passes_basic_dims():
    env = UAVBackscatterEnv(CONFIG_3D)
    assert env.get_obs_dim() == 114
    assert env.get_state_dim() == 89
    assert env.get_action_dim() == 1056
