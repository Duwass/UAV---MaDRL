from __future__ import annotations

from typing import Any

from baselines import closest_target, format_actions, in_coverage, masked_fallback_action, movement_toward, unwrap_env
from envs.channel_model import compute_sinr
from envs.uav_backscatter_env import MODE_ACTIVE, MODE_BACKSCATTER, MODE_HARVEST, MODE_IDLE, encode_action


class GreedySINRPolicy:
    def act(self, observations: Any, env: Any | None = None):
        base_env = unwrap_env(env)
        actions: list[int] = []
        taken: set[int] = set()
        for uav_id, uav in enumerate(base_env.uavs):
            target = self._select_target(base_env, uav, taken)
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

    def _select_target(self, base_env: Any, uav: Any, taken: set[int]):
        if base_env.primary_busy:
            candidates = [iot for iot in base_env.iot_devices if iot.queue > 0]
            tx_power = float(base_env.channel_cfg.get("tx_power_rf_source", 1.0)) * float(
                base_env.channel_cfg.get("backscatter_tx_power_factor", 0.02)
            )
        else:
            candidates = [
                iot
                for iot in base_env.iot_devices
                if iot.queue > 0 and iot.can_active_transmit(base_env._active_energy_cost(iot))
            ]
            tx_power = float(base_env.channel_cfg.get("tx_power_iot", 0.1))
        if not candidates:
            return closest_target(uav, base_env.iot_devices, taken)
        untaken = [iot for iot in candidates if iot.id not in taken]
        pool = untaken if untaken else candidates
        return max(
            pool,
            key=lambda iot: compute_sinr(
                tx_power,
                iot,
                uav,
                base_env.jammers,
                float(base_env.channel_cfg.get("noise_power", 1.0e-9)),
                float(base_env.channel_cfg.get("path_loss_exponent", 2.2)),
                base_env._channel_distance_mode() if hasattr(base_env, "_channel_distance_mode") else "3d",
                base_env._jammer_influence_distance_mode() if hasattr(base_env, "_jammer_influence_distance_mode") else "horizontal",
            ),
        )


Policy = GreedySINRPolicy
