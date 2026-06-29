from __future__ import annotations

from typing import Any

from baselines import closest_target, format_actions, masked_fallback_action, movement_toward, unwrap_env
from envs.channel_model import distance_2d
from envs.uav_backscatter_env import MODE_ACTIVE, MODE_HARVEST, MODE_IDLE, encode_action


class HTTOnlyPolicy:
    """Harvest-then-transmit baseline: harvest in busy frames, actively transmit in idle frames."""

    def act(self, observations: Any, env: Any | None = None):
        base_env = unwrap_env(env)
        actions: list[int] = []
        taken: set[int] = set()
        for uav_id, uav in enumerate(base_env.uavs):
            if base_env.primary_busy:
                candidates = sorted(base_env.iot_devices, key=lambda iot: (iot.energy, distance_2d(uav, iot)))
            else:
                candidates = [
                    iot
                    for iot in base_env.iot_devices
                    if iot.queue > 0 and iot.can_active_transmit(base_env._active_energy_cost(iot))
                ]
            target = closest_target(uav, list(candidates), taken) or closest_target(uav, base_env.iot_devices, taken)
            if target is None:
                actions.append(masked_fallback_action(base_env, uav_id, encode_action(0, 0, MODE_IDLE, base_env.num_iot)))
                continue
            taken.add(target.id)
            in_coverage = distance_2d(uav, target) <= uav.coverage_radius
            movement = 0 if in_coverage else movement_toward(uav, target)
            mode = MODE_IDLE
            if in_coverage:
                if base_env.primary_busy:
                    mode = MODE_HARVEST
                elif target.can_active_transmit(base_env._active_energy_cost(target)):
                    mode = MODE_ACTIVE
            action = encode_action(movement, target.id + 1, mode, base_env.num_iot)
            actions.append(masked_fallback_action(base_env, uav_id, action))
        return format_actions(actions, observations, env)


Policy = HTTOnlyPolicy
