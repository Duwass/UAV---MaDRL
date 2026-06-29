from __future__ import annotations

from typing import Any

from envs.channel_model import distance_2d
from envs.entities import IoTDevice, UAV
from envs.uav_backscatter_env import MODE_IDLE, decode_action, encode_action


def unwrap_env(env: Any) -> Any:
    return getattr(env, "base_env", env)


def format_actions(actions: list[int], observations: Any, env: Any | None = None) -> list[int] | dict[str, int]:
    if isinstance(observations, dict):
        agents = list(observations.keys())
        return {agent: int(actions[idx]) for idx, agent in enumerate(agents)}
    return [int(action) for action in actions]


def movement_toward(uav: UAV, target: IoTDevice, tolerance: float = 1.0) -> int:
    dx = target.x - uav.x
    dy = target.y - uav.y
    east = dx > tolerance
    west = dx < -tolerance
    north = dy > tolerance
    south = dy < -tolerance
    if north and east:
        return 5
    if north and west:
        return 6
    if south and east:
        return 7
    if south and west:
        return 8
    if north:
        return 1
    if south:
        return 2
    if east:
        return 3
    if west:
        return 4
    return 0


def closest_target(uav: UAV, candidates: list[IoTDevice], taken: set[int] | None = None) -> IoTDevice | None:
    if not candidates:
        return None
    taken = taken or set()
    untaken = [iot for iot in candidates if iot.id not in taken]
    pool = untaken if untaken else candidates
    return min(pool, key=lambda iot: distance_2d(uav, iot))


def mask_enabled(env: Any) -> bool:
    base_env = unwrap_env(env)
    return bool(getattr(base_env, "action_masking_cfg", {}).get("enabled", False))


def masked_fallback_action(env: Any, uav_id: int, proposed_action: int) -> int:
    base_env = unwrap_env(env)
    if not mask_enabled(base_env):
        return int(proposed_action)
    mask = base_env.get_action_mask(uav_id)
    if 0 <= int(proposed_action) < len(mask) and mask[int(proposed_action)] == 1:
        return int(proposed_action)

    movement, selected_iot_index, _ = decode_action(int(proposed_action), base_env.num_iot)
    for mode in (MODE_IDLE,):
        candidate = encode_action(movement, selected_iot_index, mode, base_env.num_iot)
        if mask[candidate] == 1:
            base_env.record_policy_fallback()
            return candidate
    for iot in sorted(base_env.iot_devices, key=lambda item: distance_2d(base_env.uavs[uav_id], item)):
        candidate = encode_action(movement, iot.id + 1, MODE_IDLE, base_env.num_iot)
        if mask[candidate] == 1:
            base_env.record_policy_fallback()
            return candidate
    candidate = encode_action(0, 0, MODE_IDLE, base_env.num_iot)
    base_env.record_policy_fallback()
    return candidate if mask[candidate] == 1 else int(mask.nonzero()[0][0])
