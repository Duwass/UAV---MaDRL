from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.channel_model import compute_sinr, distance_2d
from envs.mobility_model import MOVEMENT_DELTAS
from envs.uav_backscatter_env import (
    MODE_ACTIVE,
    MODE_AVOID_JAMMER,
    MODE_BACKSCATTER,
    MODE_HARVEST,
    MODE_IDLE,
    encode_action,
)


IDLE_SAFE = 0
SERVE_NEAREST_QUEUE = 1
SERVE_BEST_SINR = 2
PRIORITIZE_BACKSCATTER_TYPE23 = 3
PRIORITIZE_ACTIVE_TYPE1 = 4
HARVEST_LOW_ENERGY = 5
AVOID_JAMMER = 6
BALANCE_UNDERSERVED_IOT = 7
HYBRID_BALANCED = 8
HIGH_QUEUE_PRIORITY = 9


HIGH_LEVEL_ACTION_NAMES = {
    IDLE_SAFE: "IDLE_SAFE",
    SERVE_NEAREST_QUEUE: "SERVE_NEAREST_QUEUE",
    SERVE_BEST_SINR: "SERVE_BEST_SINR",
    PRIORITIZE_BACKSCATTER_TYPE23: "PRIORITIZE_BACKSCATTER_TYPE23",
    PRIORITIZE_ACTIVE_TYPE1: "PRIORITIZE_ACTIVE_TYPE1",
    HARVEST_LOW_ENERGY: "HARVEST_LOW_ENERGY",
    AVOID_JAMMER: "AVOID_JAMMER",
    BALANCE_UNDERSERVED_IOT: "BALANCE_UNDERSERVED_IOT",
    HYBRID_BALANCED: "HYBRID_BALANCED",
    HIGH_QUEUE_PRIORITY: "HIGH_QUEUE_PRIORITY",
}


@dataclass
class ExecutorDecision:
    original_action: int
    selected_iot: int
    selected_mode: int
    fallback: bool


class HierarchicalActionExecutor:
    def __init__(self, env, config: dict[str, Any] | None = None):
        self.env = env
        self.config = dict(config or {})
        self.fairness_weight = float(self.config.get("fairness_weight", 1.0))
        self.underserved_weight = float(self.config.get("underserved_weight", 1.0))
        self.repeat_target_penalty = float(self.config.get("repeat_target_penalty", 0.0))
        self.jammer_risk_weight = float(self.config.get("jammer_risk_weight", 1.5))
        self.queue_weight = float(self.config.get("queue_weight", 2.0))
        self.sinr_weight = float(self.config.get("sinr_weight", 1.5))
        self.distance_weight = float(self.config.get("distance_weight", 1.0))
        self.type_priority_weight = float(self.config.get("type_priority_weight", 1.0))
        self.last_diagnostics: dict[str, Any] = {}

    def num_high_level_actions(self) -> int:
        return len(HIGH_LEVEL_ACTION_NAMES)

    def decode_high_level_action(self, high_action_id: int) -> str:
        return HIGH_LEVEL_ACTION_NAMES.get(int(high_action_id), "IDLE_SAFE")

    def build_original_actions(self, high_level_actions: list[int]) -> list[int]:
        actions: list[int] = []
        fallback_count = 0
        selected_iot_per_uav: list[int] = []
        selected_mode_per_uav: list[int] = []
        high_counts = {idx: 0 for idx in range(self.num_high_level_actions())}
        mode_counts = {"idle": 0, "harvest": 0, "backscatter": 0, "active": 0, "avoid_jammer": 0}

        normalized = list(high_level_actions)
        if len(normalized) < self.env.num_uav:
            normalized.extend([IDLE_SAFE] * (self.env.num_uav - len(normalized)))

        for uav_id, high_action in enumerate(normalized[: self.env.num_uav]):
            high_action = int(high_action)
            if high_action not in HIGH_LEVEL_ACTION_NAMES:
                high_action = IDLE_SAFE
                fallback_count += 1
            high_counts[high_action] += 1
            decision = self._build_action_for_uav(uav_id, high_action)
            actions.append(decision.original_action)
            selected_iot_per_uav.append(decision.selected_iot)
            selected_mode_per_uav.append(decision.selected_mode)
            fallback_count += int(decision.fallback)
            mode_counts[self._mode_name(decision.selected_mode)] += 1

        self.last_diagnostics = {
            "high_level_action_names": [self.decode_high_level_action(action) for action in normalized[: self.env.num_uav]],
            "original_actions": actions,
            "fallback_count": float(fallback_count),
            "fallback_rate": float(fallback_count / max(self.env.num_uav, 1)),
            "selected_iot_per_uav": selected_iot_per_uav,
            "selected_mode_per_uav": selected_mode_per_uav,
            "high_level_action_counts": high_counts,
            "high_level_mode_usage": mode_counts,
        }
        return actions

    def _build_action_for_uav(self, uav_id: int, high_action: int) -> ExecutorDecision:
        uav = self.env.uavs[int(uav_id)]
        fallback = False
        if high_action == AVOID_JAMMER and self.env.jammers:
            movement = self._movement_away_from_jammer(uav)
            action = encode_action(movement, 0, MODE_AVOID_JAMMER, self.env.num_iot)
            return ExecutorDecision(action, -1, MODE_AVOID_JAMMER, False)
        if high_action == IDLE_SAFE:
            safe_target = self._select_best_sinr_target(uav, require_queue=True)
            if safe_target is not None and self.score_jammer_safety(uav, safe_target) > 0.5:
                mode = self._select_mode(uav, safe_target, safe_only=True)
                if mode != MODE_IDLE:
                    return self._decision(uav, safe_target, mode, fallback=False)
            return ExecutorDecision(encode_action(0, 0, MODE_IDLE, self.env.num_iot), -1, MODE_IDLE, False)

        target = self._select_target(uav, high_action)
        if target is None:
            target = self._fallback_target(uav)
            fallback = True
        if target is None:
            return ExecutorDecision(encode_action(0, 0, MODE_IDLE, self.env.num_iot), -1, MODE_IDLE, True)

        mode = self._select_mode(uav, target, high_action=high_action)
        if not self._is_mode_valid_now(uav, target, mode):
            mode = self._best_feasible_mode(uav, target)
            fallback = True
        return self._decision(uav, target, mode, fallback=fallback)

    def _decision(self, uav, target, mode: int, fallback: bool) -> ExecutorDecision:
        if mode in (MODE_IDLE, MODE_AVOID_JAMMER):
            selected = 0
        else:
            selected = int(target.id) + 1
        movement = 0 if mode not in (MODE_IDLE, MODE_AVOID_JAMMER) and self._in_coverage(uav, target) else self._movement_toward(uav, target)
        if mode == MODE_IDLE:
            selected = 0
        action = encode_action(movement, selected, mode, self.env.num_iot)
        return ExecutorDecision(action, int(target.id) if target is not None else -1, int(mode), bool(fallback))

    def _select_target(self, uav, high_action: int):
        if high_action == SERVE_NEAREST_QUEUE:
            return self._select_nearest_queue_target(uav)
        if high_action == SERVE_BEST_SINR:
            return self._select_best_sinr_target(uav, require_queue=True)
        if high_action == PRIORITIZE_BACKSCATTER_TYPE23:
            return self._select_by_score(uav, preferred_types={2, 3}, require_queue=True, prefer_queue=True)
        if high_action == PRIORITIZE_ACTIVE_TYPE1:
            return self._select_active_type1(uav)
        if high_action == HARVEST_LOW_ENERGY:
            return self._select_by_score(uav, preferred_types={2, 3}, require_queue=False, prefer_low_energy=True)
        if high_action == BALANCE_UNDERSERVED_IOT:
            return self._select_by_score(uav, require_queue=True, prefer_underserved=True)
        if high_action == HYBRID_BALANCED:
            return self._select_by_score(uav, preferred_types={1, 2, 3}, require_queue=True, hybrid=True)
        if high_action == HIGH_QUEUE_PRIORITY:
            return self._select_by_score(uav, require_queue=True, prefer_queue=True)
        return None

    def _select_nearest_queue_target(self, uav):
        candidates = self._candidates(uav, require_queue=True, require_coverage=True)
        if not candidates:
            return None
        return min(candidates, key=lambda iot: distance_2d(uav, iot))

    def _select_best_sinr_target(self, uav, require_queue: bool):
        candidates = self._candidates(uav, require_queue=require_queue, require_coverage=True)
        if not candidates:
            return None
        return max(candidates, key=lambda iot: self.score_sinr(uav, iot))

    def _select_active_type1(self, uav):
        candidates = [
            iot
            for iot in self._candidates(uav, require_queue=True, require_coverage=True)
            if int(iot.device_type) == 1 and iot.can_active_transmit(self.env._active_energy_cost(iot))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda iot: self.score_queue(iot) + self.score_sinr(uav, iot))

    def _select_by_score(
        self,
        uav,
        preferred_types: set[int] | None = None,
        require_queue: bool = True,
        prefer_queue: bool = False,
        prefer_low_energy: bool = False,
        prefer_underserved: bool = False,
        hybrid: bool = False,
    ):
        candidates = self._candidates(uav, require_queue=require_queue, require_coverage=True)
        if not candidates and prefer_low_energy:
            candidates = self._candidates(uav, require_queue=False, require_coverage=True)
        if not candidates:
            return None

        def score(iot) -> float:
            value = 1.0 * self._in_coverage_score(uav, iot)
            value += self.queue_weight * self.score_queue(iot)
            value += self.sinr_weight * self.score_sinr(uav, iot)
            value += self.type_priority_weight * self.score_type_priority(iot, preferred_types or set())
            value += self.underserved_weight * self.score_underserved(iot)
            value += self.distance_weight * (1.0 - self.score_distance(uav, iot))
            value -= self.jammer_risk_weight * (1.0 - self.score_jammer_safety(uav, iot))
            if prefer_queue:
                value += self.queue_weight * self.score_queue(iot)
            if prefer_low_energy:
                value += 2.0 * self.score_energy_need(iot)
            if prefer_underserved:
                value += self.fairness_weight * self.score_underserved(iot)
            if hybrid:
                if self.env.primary_busy and int(iot.device_type) in {2, 3}:
                    value += 1.0
                if (not self.env.primary_busy) and int(iot.device_type) == 1:
                    value += 1.0
            if any(uav.current_target == iot.id for uav in self.env.uavs):
                value -= self.repeat_target_penalty
            return float(value)

        return max(candidates, key=score)

    def _fallback_target(self, uav):
        target = self._select_best_sinr_target(uav, require_queue=True)
        if target is not None:
            return target
        return self._select_nearest_queue_target(uav)

    def _select_mode(self, uav, target, safe_only: bool = False, high_action: int | None = None) -> int:
        if target is None:
            return MODE_IDLE
        active_cost = self.env._active_energy_cost(target)
        active_allowed = (not self.env.primary_busy) or bool(self.env.channel_cfg.get("allow_active_when_busy", False))
        safe_sinr = self.score_sinr(uav, target) >= 0.25

        if self.env.primary_busy:
            if high_action == HARVEST_LOW_ENERGY and self.score_energy_need(target) > 0.35:
                return MODE_HARVEST
            if target.queue > 0:
                return MODE_BACKSCATTER
            return MODE_HARVEST

        if active_allowed and target.queue > 0 and target.can_active_transmit(active_cost):
            if int(target.device_type) == 1 or safe_sinr or high_action in (PRIORITIZE_ACTIVE_TYPE1, HYBRID_BALANCED):
                return MODE_ACTIVE
        if safe_only:
            return MODE_IDLE
        return MODE_IDLE

    def _best_feasible_mode(self, uav, target) -> int:
        for mode in (MODE_ACTIVE, MODE_BACKSCATTER, MODE_HARVEST, MODE_IDLE):
            if self._is_mode_valid_now(uav, target, mode):
                return mode
        return MODE_IDLE

    def _is_mode_valid_now(self, uav, target, mode: int) -> bool:
        if mode == MODE_IDLE:
            return True
        if target is None or not self._in_coverage(uav, target):
            return False
        if mode == MODE_HARVEST:
            return bool(self.env.primary_busy)
        if mode == MODE_BACKSCATTER:
            return bool(self.env.primary_busy and target.queue > 0)
        if mode == MODE_ACTIVE:
            active_allowed = (not self.env.primary_busy) or bool(self.env.channel_cfg.get("allow_active_when_busy", False))
            return bool(active_allowed and target.can_active_transmit(self.env._active_energy_cost(target)))
        return False

    def _candidates(self, uav, require_queue: bool, require_coverage: bool):
        candidates = []
        for iot in self.env.iot_devices:
            if require_queue and iot.queue <= 0:
                continue
            if require_coverage and not self._in_coverage(uav, iot):
                continue
            candidates.append(iot)
        return candidates

    def score_queue(self, iot) -> float:
        return float(iot.queue / max(iot.queue_capacity, 1))

    def score_sinr(self, uav, iot) -> float:
        tx_power = float(self.env.channel_cfg.get("tx_power_iot", 0.1))
        sinr = compute_sinr(
            tx_power,
            iot,
            uav,
            self.env.jammers,
            float(self.env.channel_cfg.get("noise_power", 1.0e-9)),
            float(self.env.channel_cfg.get("path_loss_exponent", 2.2)),
        )
        threshold = float(self.env.channel_cfg.get("sinr_threshold", 1.0))
        return float(sinr / max(sinr + threshold, 1.0e-12))

    def score_distance(self, uav, iot) -> float:
        return float(min(1.0, distance_2d(uav, iot) / max(uav.coverage_radius, 1.0)))

    def score_energy_need(self, iot) -> float:
        return float(1.0 - min(1.0, iot.energy / max(iot.energy_capacity, 1.0e-12)))

    def score_type_priority(self, iot, preferred_types: set[int]) -> float:
        return 1.0 if int(iot.device_type) in preferred_types else 0.0

    def score_underserved(self, iot) -> float:
        delivered = self.env.delivered_per_iot
        if delivered.size == 0:
            return 0.0
        max_delivered = float(np.max(delivered))
        if max_delivered <= 0.0:
            return 1.0
        return float(1.0 - min(1.0, delivered[int(iot.id)] / max_delivered))

    def score_jammer_safety(self, uav, iot) -> float:
        if not self.env.jammers:
            return 1.0
        nearest = min(distance_2d(uav, jammer) for jammer in self.env.jammers)
        radius = max(max(float(jammer.radius) for jammer in self.env.jammers), 1.0)
        return float(min(1.0, nearest / radius))

    def _in_coverage(self, uav, iot) -> bool:
        return distance_2d(uav, iot) <= float(uav.coverage_radius)

    def _in_coverage_score(self, uav, iot) -> float:
        return 1.0 if self._in_coverage(uav, iot) else 0.0

    def _movement_toward(self, uav, target) -> int:
        if target is None:
            return 0
        return self._movement_for_vector(float(target.x - uav.x), float(target.y - uav.y))

    def _movement_away_from_jammer(self, uav) -> int:
        if not self.env.jammers:
            return 0
        jammer = min(self.env.jammers, key=lambda item: distance_2d(uav, item))
        return self._movement_for_vector(float(uav.x - jammer.x), float(uav.y - jammer.y))

    def _movement_for_vector(self, dx: float, dy: float) -> int:
        if abs(dx) < 1.0e-9 and abs(dy) < 1.0e-9:
            return 0
        desired = np.asarray([dx, dy], dtype=np.float64)
        desired /= max(float(np.linalg.norm(desired)), 1.0e-12)
        best_action = 0
        best_dot = -np.inf
        for action, delta in MOVEMENT_DELTAS.items():
            vector = np.asarray(delta, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                continue
            dot = float(np.dot(desired, vector / norm))
            if dot > best_dot:
                best_dot = dot
                best_action = int(action)
        return best_action

    def _mode_name(self, mode: int) -> str:
        return {
            MODE_IDLE: "idle",
            MODE_HARVEST: "harvest",
            MODE_BACKSCATTER: "backscatter",
            MODE_ACTIVE: "active",
            MODE_AVOID_JAMMER: "avoid_jammer",
        }.get(int(mode), "idle")
