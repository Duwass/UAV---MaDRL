from __future__ import annotations

from typing import Any

from baselines import closest_target, format_actions, in_coverage, masked_fallback_action, movement_toward, unwrap_env
from envs.uav_backscatter_env import MODE_ACTIVE, MODE_BACKSCATTER, MODE_HARVEST, MODE_IDLE, encode_action


class GreedyNearestPolicy:
    def act(self, observations: Any, env: Any | None = None):
        base_env = unwrap_env(env)
        actions: list[int] = []
        taken: set[int] = set()
        for uav_id, uav in enumerate(base_env.uavs):
            queued = [iot for iot in base_env.iot_devices if iot.queue > 0]
            target = closest_target(uav, queued, taken) or closest_target(uav, base_env.iot_devices, taken)
            if target is None:
                actions.append(masked_fallback_action(base_env, uav_id, encode_action(0, 0, MODE_IDLE, base_env.num_iot)))
                continue
            taken.add(target.id)
            covered = in_coverage(base_env, uav, target)
            movement = 0 if covered else movement_toward(uav, target)
            mode = MODE_IDLE
            if covered:
                if base_env.primary_busy:
                    mode = MODE_BACKSCATTER if target.queue > 0 else MODE_HARVEST
                elif target.can_active_transmit(base_env._active_energy_cost(target)):
                    mode = MODE_ACTIVE
            action = encode_action(movement, target.id + 1, mode, base_env.num_iot)
            actions.append(masked_fallback_action(base_env, uav_id, action))
        return format_actions(actions, observations, env)


Policy = GreedyNearestPolicy
