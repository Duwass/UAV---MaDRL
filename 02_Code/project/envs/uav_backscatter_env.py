from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from envs.channel_model import (
    compute_sinr,
    distance_2d,
    distance_3d,
    jammer_interference,
    received_power,
    success_probability_from_sinr,
)
from envs.entities import IoTDevice, Jammer, RFSource, UAV
from envs.mobility_model import compute_movement_energy, move_uav
from envs.reward import compute_reward


MODE_IDLE = 0
MODE_HARVEST = 1
MODE_BACKSCATTER = 2
MODE_ACTIVE = 3
MODE_RELAY = 4
MODE_AVOID_JAMMER = 5
NUM_MOVEMENT_ACTIONS = 9
NUM_COMMUNICATION_MODES = 6


def load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return deepcopy(config)


def encode_action(movement_action: int, selected_iot_index: int, communication_mode: int, num_iot: int) -> int:
    movement_action = int(movement_action)
    selected_iot_index = int(selected_iot_index)
    communication_mode = int(communication_mode)
    if not 0 <= movement_action < NUM_MOVEMENT_ACTIONS:
        raise ValueError(f"movement_action must be in [0, 8], got {movement_action}")
    if not 0 <= selected_iot_index <= int(num_iot):
        raise ValueError(f"selected_iot_index must be in [0, {num_iot}], got {selected_iot_index}")
    if not 0 <= communication_mode < NUM_COMMUNICATION_MODES:
        raise ValueError(f"communication_mode must be in [0, 5], got {communication_mode}")
    return int(((movement_action * (int(num_iot) + 1)) + selected_iot_index) * NUM_COMMUNICATION_MODES + communication_mode)


def decode_action(action_id: int, num_iot: int) -> tuple[int, int, int]:
    action_id = int(action_id)
    total_actions = NUM_MOVEMENT_ACTIONS * (int(num_iot) + 1) * NUM_COMMUNICATION_MODES
    if action_id < 0 or action_id >= total_actions:
        return MODE_IDLE, 0, MODE_IDLE
    communication_mode = action_id % NUM_COMMUNICATION_MODES
    action_id //= NUM_COMMUNICATION_MODES
    selected_iot_index = action_id % (int(num_iot) + 1)
    movement_action = action_id // (int(num_iot) + 1)
    return int(movement_action), int(selected_iot_index), int(communication_mode)


class UAVBackscatterEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: str | Path | dict[str, Any]):
        super().__init__()
        self.config = load_config(config)
        self.sim_cfg = self.config.get("simulation", {})
        self.net_cfg = self.config.get("network", {})
        self.iot_cfg = self.config.get("iot", {})
        self.uav_cfg = self.config.get("uav", {})
        self.channel_cfg = self.config.get("channel", {})
        self.jammer_cfg = self.config.get("jammer", {})
        self.reward_cfg = self.config.get("reward", {})
        self.action_masking_cfg = self.config.get("action_masking", {})

        self.area_width = float(self.sim_cfg.get("area_width", 500.0))
        self.area_height = float(self.sim_cfg.get("area_height", 500.0))
        self.frame_length = int(self.sim_cfg.get("frame_length", 10))
        self.max_steps = int(self.sim_cfg.get("max_steps", self.sim_cfg.get("episode_length", 200)))
        self.base_seed = int(self.sim_cfg.get("seed", 42))

        self.num_uav = int(self.net_cfg.get("num_uav", 1))
        self.num_iot = int(self.net_cfg.get("num_iot", 1))
        self.configured_num_jammer = int(self.net_cfg.get("num_jammer", 0))
        self.num_rf_sources = int(self.net_cfg.get("num_rf_sources", 1))

        self.action_size = NUM_MOVEMENT_ACTIONS * (self.num_iot + 1) * NUM_COMMUNICATION_MODES
        self.observation_dim = 7 + self.num_iot * 6
        self.global_state_dim = self.num_uav * 3 + self.num_iot * 4 + self.configured_num_jammer * 3 + 2
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_uav, self.observation_dim),
            dtype=np.float32,
        )
        self.single_observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.state_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.global_state_dim,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(self.base_seed)
        self.current_step = 0
        self.primary_busy = False
        self.uavs: list[UAV] = []
        self.iot_devices: list[IoTDevice] = []
        self.jammers: list[Jammer] = []
        self.rf_sources: list[RFSource] = []
        self.delivered_per_iot = np.zeros(self.num_iot, dtype=np.float64)
        self.per_uav_served_packets = np.zeros(self.num_uav, dtype=np.float64)
        self.episode_totals: dict[str, float] = {}
        self.last_frame_metrics: dict[str, float] = {}
        self.last_reward_breakdown: dict[str, float] = {}
        self.pending_policy_fallbacks = 0.0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None:
            seed = self.base_seed
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.primary_busy = False
        self._init_entities()
        self._reset_episode_totals()
        self.pending_policy_fallbacks = 0.0
        self._update_primary_channel()
        observations = self._get_all_observations()
        return observations, self._build_info()

    def step(self, actions: list[int] | dict[str, int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_index = self.current_step
        arrivals, dropped_packets = self._update_iot_arrivals()
        self._update_jammers()
        for jammer in self.jammers:
            if jammer.is_active:
                self.episode_totals["jammer_active_steps"] += 1.0
            else:
                self.episode_totals["jammer_inactive_steps"] += 1.0
        action_metrics = self._apply_actions(actions)
        collision_count = self._count_collisions()
        avg_queue_length = self._average_queue_length()
        fairness = self._jain_fairness_index()

        components = {
            "successful_packets": action_metrics["successful_packets"],
            "dropped_packets": dropped_packets,
            "avg_queue_length": avg_queue_length,
            "uav_energy_used": action_metrics["uav_energy_used"],
            "jamming_failures": action_metrics["jamming_failures"],
            "collision_count": collision_count,
            "unfairness_penalty": 1.0 - fairness,
            "avoidance_bonus": action_metrics["avoidance_bonus"],
        }
        reward, reward_breakdown = compute_reward(components, self.reward_cfg)
        self.last_reward_breakdown = reward_breakdown

        self.episode_totals["steps"] += 1.0
        self.episode_totals["total_reward"] += reward
        self.episode_totals["total_throughput"] += action_metrics["successful_packets"]
        self.episode_totals["total_arrived"] += arrivals
        self.episode_totals["total_dropped"] += dropped_packets
        self.episode_totals["sum_queue_length"] += avg_queue_length
        self.episode_totals["uav_energy_consumption"] += action_metrics["uav_energy_used"]
        self.episode_totals["iot_energy_consumption"] += action_metrics["iot_energy_used"]
        self.episode_totals["jamming_failures"] += action_metrics["jamming_failures"]
        self.episode_totals["transmission_attempts"] += action_metrics["transmission_attempts"]
        self.episode_totals["collision_count"] += collision_count
        self.episode_totals["duplicate_target_count"] += action_metrics["duplicate_target_count"]
        metric_accumulators = [
            "invalid_action_count",
            "actions_processed",
            "out_of_coverage_action_count",
            "no_queue_selected_count",
            "insufficient_energy_count",
            "busy_mode_invalid_count",
            "jammed_transmission_count",
            "successful_backscatter_packets",
            "successful_active_packets",
            "attempted_backscatter_packets",
            "attempted_active_packets",
            "harvested_energy_total",
            "mode_usage_idle",
            "mode_usage_harvest",
            "mode_usage_backscatter",
            "mode_usage_active",
            "mode_usage_relay",
            "mode_usage_avoid_jammer",
            "greedy_fallback_count",
        ]
        for key in metric_accumulators:
            self.episode_totals[key] += action_metrics.get(key, 0.0)

        self.last_frame_metrics = {
            "step": float(step_index),
            "primary_busy": float(self.primary_busy),
            "successful_packets": float(action_metrics["successful_packets"]),
            "dropped_packets": float(dropped_packets),
            "avg_queue_length": avg_queue_length,
            "total_iot_energy": float(sum(iot.energy for iot in self.iot_devices)),
            "total_uav_energy": float(sum(uav.energy for uav in self.uavs)),
            "jamming_failures": float(action_metrics["jamming_failures"]),
            "collision_count": float(collision_count),
            "duplicate_target_count": float(action_metrics["duplicate_target_count"]),
            "invalid_action_rate": float(action_metrics["invalid_action_count"] / max(action_metrics["actions_processed"], 1.0)),
            "out_of_coverage_action_rate": float(
                action_metrics["out_of_coverage_action_count"] / max(action_metrics["actions_processed"], 1.0)
            ),
            "jammed_transmission_rate": float(
                action_metrics["jammed_transmission_count"] / max(action_metrics["transmission_attempts"], 1.0)
            ),
            **self._diagnostic_frame_metrics(action_metrics),
            **reward_breakdown,
        }

        self.current_step += 1
        terminated = not any(uav.is_alive() for uav in self.uavs)
        truncated = self.current_step >= self.max_steps
        if not terminated and not truncated:
            self._update_primary_channel()
        observations = self._get_all_observations()
        return observations, float(reward), bool(terminated), bool(truncated), self._build_info()

    def render(self) -> None:
        print(
            f"step={self.current_step} primary_busy={self.primary_busy} "
            f"throughput={self.episode_totals.get('total_throughput', 0):.0f}"
        )

    def close(self) -> None:
        return

    def get_global_state(self) -> np.ndarray:
        values: list[float] = []
        initial_uav_energy = float(self.uav_cfg.get("initial_energy", 1000.0))
        for uav in self.uavs:
            values.extend([uav.x / self.area_width, uav.y / self.area_height, uav.energy / max(initial_uav_energy, 1.0e-9)])
        for iot in self.iot_devices:
            values.extend(
                [
                    iot.x / self.area_width,
                    iot.y / self.area_height,
                    iot.queue / max(iot.queue_capacity, 1),
                    iot.energy / max(iot.energy_capacity, 1.0e-9),
                ]
            )
        for idx in range(self.configured_num_jammer):
            if idx < len(self.jammers):
                jammer = self.jammers[idx]
                values.extend([
                    jammer.x / self.area_width,
                    jammer.y / self.area_height,
                    jammer.energy / max(jammer.energy_capacity, 1.0e-9),
                ])
            else:
                values.extend([0.0, 0.0, 0.0])
        values.extend([float(self.primary_busy), self.current_step / max(self.max_steps, 1)])
        return np.asarray(values, dtype=np.float32)

    def get_local_observation(self, uav_id: int) -> np.ndarray:
        uav = self.uavs[int(uav_id)]
        diag = float(np.hypot(self.area_width, self.area_height))
        initial_uav_energy = float(self.uav_cfg.get("initial_energy", 1000.0))
        if self.jammers:
            nearest = min(self.jammers, key=lambda jammer: distance_2d(uav, jammer))
            jammer_dx = (nearest.x - uav.x) / self.area_width
            jammer_dy = (nearest.y - uav.y) / self.area_height
            jammer_distance = distance_2d(uav, nearest) / diag
        else:
            jammer_dx = 0.0
            jammer_dy = 0.0
            jammer_distance = 1.0

        values: list[float] = [
            uav.x / self.area_width,
            uav.y / self.area_height,
            uav.energy / max(initial_uav_energy, 1.0e-9),
            jammer_dx,
            jammer_dy,
            jammer_distance,
            float(self.primary_busy),
        ]
        for iot in self.iot_devices:
            dist = distance_2d(uav, iot)
            values.extend(
                [
                    (iot.x - uav.x) / self.area_width,
                    (iot.y - uav.y) / self.area_height,
                    dist / diag,
                    iot.queue / max(iot.queue_capacity, 1),
                    iot.energy / max(iot.energy_capacity, 1.0e-9),
                    float(dist <= uav.coverage_radius),
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def get_action_space(self) -> spaces.Discrete:
        return self.action_space

    def get_observation_space(self) -> spaces.Box:
        return self.observation_space

    def record_policy_fallback(self, count: float = 1.0) -> None:
        self.pending_policy_fallbacks += float(count)

    def get_action_mask(self, uav_id: int) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.int8)
        for action_id in range(self.action_size):
            movement_action, selected_iot_index, communication_mode = decode_action(action_id, self.num_iot)
            valid, _ = self._validate_decoded_action(int(uav_id), movement_action, selected_iot_index, communication_mode)
            mask[action_id] = 1 if valid else 0
        return mask

    def _validate_decoded_action(
        self,
        uav_id: int,
        movement_action: int,
        selected_iot_index: int,
        communication_mode: int,
    ) -> tuple[bool, str]:
        if not 0 <= int(movement_action) < NUM_MOVEMENT_ACTIONS:
            return False, "invalid_action"
        if not 0 <= int(selected_iot_index) <= self.num_iot:
            return False, "invalid_action"
        if not 0 <= int(communication_mode) < NUM_COMMUNICATION_MODES:
            return False, "invalid_action"
        if communication_mode == MODE_IDLE:
            return True, "valid"
        if communication_mode == MODE_RELAY:
            return True, "valid"
        if communication_mode == MODE_AVOID_JAMMER:
            return (True, "valid") if self.jammers else (False, "avoid_without_jammer")
        if selected_iot_index == 0:
            return False, "missing_target"

        uav = self.uavs[int(uav_id)]
        target = self.iot_devices[selected_iot_index - 1]
        if distance_2d(uav, target) > uav.coverage_radius:
            return False, "out_of_coverage"
        if communication_mode == MODE_HARVEST:
            return (True, "valid") if self.primary_busy else (False, "busy_mode_invalid")
        if communication_mode == MODE_BACKSCATTER:
            if not self.primary_busy:
                return False, "busy_mode_invalid"
            if target.queue <= 0:
                return False, "no_queue"
            return True, "valid"
        if communication_mode == MODE_ACTIVE:
            active_allowed = (not self.primary_busy) or bool(self.channel_cfg.get("allow_active_when_busy", False))
            if not active_allowed:
                return False, "busy_mode_invalid"
            if target.queue <= 0:
                return False, "no_queue"
            if not target.can_active_transmit(self._active_energy_cost(target)):
                return False, "insufficient_energy"
            return True, "valid"
        return False, "invalid_action"

    def _init_entities(self) -> None:
        self.uavs = []
        for idx in range(self.num_uav):
            x = (idx + 1) * self.area_width / (self.num_uav + 1)
            y = 0.2 * self.area_height
            self.uavs.append(
                UAV(
                    id=idx,
                    x=float(x),
                    y=float(y),
                    h=float(self.uav_cfg.get("altitude", 100.0)),
                    energy=float(self.uav_cfg.get("initial_energy", 1000.0)),
                    coverage_radius=float(self.uav_cfg.get("coverage_radius", 180.0)),
                    max_speed=float(self.uav_cfg.get("max_speed", 20.0)),
                )
            )

        device_types = list(self.iot_cfg.get("device_types", []))
        self.iot_devices = []
        for idx in range(self.num_iot):
            device_type = int(device_types[idx]) if idx < len(device_types) else 1
            energy_min = float(self.iot_cfg.get("initial_energy_min", 0.0))
            energy_max = float(self.iot_cfg.get("initial_energy_max", 5.0))
            if device_type == 3:
                energy_max = min(energy_max, max(energy_min, float(self.iot_cfg.get("energy_capacity", 10.0)) * 0.3))
            type_cfg = self._iot_type_cfg(device_type)
            energy_min = float(type_cfg.get("initial_energy_min", energy_min))
            energy_max = float(type_cfg.get("initial_energy_max", energy_max))
            energy_capacity = float(type_cfg.get("energy_capacity", self.iot_cfg.get("energy_capacity", 10.0)))
            queue_capacity = int(type_cfg.get("queue_capacity", self.iot_cfg.get("queue_capacity", 10)))
            packet_arrival_prob = float(type_cfg.get("packet_arrival_prob", self.iot_cfg.get("packet_arrival_prob", 0.5)))
            initial_energy = float(self.rng.uniform(energy_min, energy_max))
            self.iot_devices.append(
                IoTDevice(
                    id=idx,
                    x=float(self.rng.uniform(0.1 * self.area_width, 0.9 * self.area_width)),
                    y=float(self.rng.uniform(0.1 * self.area_height, 0.9 * self.area_height)),
                    device_type=device_type,
                    queue=0,
                    queue_capacity=queue_capacity,
                    energy=initial_energy,
                    energy_capacity=energy_capacity,
                    packet_arrival_prob=packet_arrival_prob,
                )
            )

        self.rf_sources = []
        for idx in range(self.num_rf_sources):
            self.rf_sources.append(
                RFSource(
                    id=idx,
                    x=float((idx + 1) * self.area_width / (self.num_rf_sources + 1)),
                    y=float(0.5 * self.area_height),
                    tx_power=float(self.channel_cfg.get("tx_power_rf_source", 1.0)),
                    busy_prob=float(self.channel_cfg.get("primary_busy_prob", 0.5)),
                )
            )

        self.jammers = []
        jammer_enabled = bool(self.jammer_cfg.get("enabled", True))
        if jammer_enabled:
            initial_positions = self.jammer_cfg.get("initial_positions", [])
            for idx in range(self.configured_num_jammer):
                if idx < len(initial_positions):
                    x, y = initial_positions[idx]
                else:
                    x = self.area_width * (0.5 + 0.15 * (idx % 2))
                    y = self.area_height * 0.5
                initial_energy = float(self.jammer_cfg.get("initial_energy", 0.0))
                energy_threshold = float(self.jammer_cfg.get("energy_threshold", 5.0))
                self.jammers.append(
                    Jammer(
                        id=idx,
                        x=float(x),
                        y=float(y),
                        power=float(self.jammer_cfg.get("power", 1.0)),
                        speed=float(self.jammer_cfg.get("speed", 10.0)),
                        radius=float(self.jammer_cfg.get("radius", 200.0)),
                        mobility=str(self.jammer_cfg.get("mobility", "random_walk")),
                        energy=initial_energy,
                        energy_capacity=float(self.jammer_cfg.get("energy_capacity", 100.0)),
                        is_active=initial_energy >= energy_threshold,
                    )
                )

    def _reset_episode_totals(self) -> None:
        self.delivered_per_iot = np.zeros(self.num_iot, dtype=np.float64)
        self.per_uav_served_packets = np.zeros(self.num_uav, dtype=np.float64)
        self.episode_totals = {
            "steps": 0.0,
            "actions_processed": 0.0,
            "total_reward": 0.0,
            "total_throughput": 0.0,
            "total_arrived": 0.0,
            "total_dropped": 0.0,
            "sum_queue_length": 0.0,
            "uav_energy_consumption": 0.0,
            "iot_energy_consumption": 0.0,
            "jamming_failures": 0.0,
            "transmission_attempts": 0.0,
            "collision_count": 0.0,
            "duplicate_target_count": 0.0,
            "invalid_action_count": 0.0,
            "out_of_coverage_action_count": 0.0,
            "no_queue_selected_count": 0.0,
            "insufficient_energy_count": 0.0,
            "busy_mode_invalid_count": 0.0,
            "jammed_transmission_count": 0.0,
            "successful_backscatter_packets": 0.0,
            "successful_active_packets": 0.0,
            "attempted_backscatter_packets": 0.0,
            "attempted_active_packets": 0.0,
            "harvested_energy_total": 0.0,
            "mode_usage_idle": 0.0,
            "mode_usage_harvest": 0.0,
            "mode_usage_backscatter": 0.0,
            "mode_usage_active": 0.0,
            "mode_usage_relay": 0.0,
            "mode_usage_avoid_jammer": 0.0,
            "greedy_fallback_count": 0.0,
            "jammer_harvested_energy_total": 0.0,
            "jammer_active_steps": 0.0,
            "jammer_inactive_steps": 0.0,
        }
        self.last_frame_metrics = {}
        self.last_reward_breakdown = {}

    def _apply_actions(self, actions: list[int] | dict[str, int]) -> dict[str, float]:
        action_list = self._normalize_actions(actions)
        metrics = {
            "successful_packets": 0.0,
            "uav_energy_used": 0.0,
            "iot_energy_used": 0.0,
            "jamming_failures": 0.0,
            "transmission_attempts": 0.0,
            "duplicate_target_count": 0.0,
            "invalid_action_count": 0.0,
            "actions_processed": 0.0,
            "out_of_coverage_action_count": 0.0,
            "no_queue_selected_count": 0.0,
            "insufficient_energy_count": 0.0,
            "busy_mode_invalid_count": 0.0,
            "jammed_transmission_count": 0.0,
            "successful_backscatter_packets": 0.0,
            "successful_active_packets": 0.0,
            "attempted_backscatter_packets": 0.0,
            "attempted_active_packets": 0.0,
            "harvested_energy_total": 0.0,
            "mode_usage_idle": 0.0,
            "mode_usage_harvest": 0.0,
            "mode_usage_backscatter": 0.0,
            "mode_usage_active": 0.0,
            "mode_usage_relay": 0.0,
            "mode_usage_avoid_jammer": 0.0,
            "greedy_fallback_count": 0.0,
            "avoidance_bonus": 0.0,
        }
        metrics["greedy_fallback_count"] = float(self.pending_policy_fallbacks)
        self.pending_policy_fallbacks = 0.0
        selected_targets: list[int] = []

        for uav, action_id in zip(self.uavs, action_list):
            if not uav.is_alive():
                continue
            metrics["actions_processed"] += 1.0
            valid_action = 0 <= int(action_id) < self.action_size
            movement_action, selected_iot_index, communication_mode = decode_action(int(action_id), self.num_iot)
            self._record_mode_usage(metrics, communication_mode)
            if not valid_action:
                self._record_invalid_action(metrics, "invalid_action")

            before_jammer_distance = self._nearest_jammer_distance(uav)
            movement_energy = compute_movement_energy(movement_action, float(self.uav_cfg.get("movement_energy_cost", 0.1)))
            move_uav(uav, movement_action, uav.max_speed, self.area_width, self.area_height)
            metrics["uav_energy_used"] += uav.consume_energy(movement_energy)
            after_jammer_distance = self._nearest_jammer_distance(uav)

            uav.last_action = int(action_id)
            target = self.iot_devices[selected_iot_index - 1] if selected_iot_index > 0 else None
            uav.current_target = target.id if target is not None else None

            if communication_mode == MODE_AVOID_JAMMER:
                if not self.jammers:
                    self._record_invalid_action(metrics, "avoid_without_jammer")
                    continue
                if after_jammer_distance > before_jammer_distance:
                    metrics["avoidance_bonus"] += 1.0
                continue
            if communication_mode == MODE_IDLE:
                continue
            if communication_mode == MODE_RELAY:
                metrics["uav_energy_used"] += uav.consume_energy(float(self.uav_cfg.get("communication_energy_cost", 0.05)))
                continue
            if target is None:
                self._record_invalid_action(metrics, "missing_target")
                continue

            selected_targets.append(target.id)
            if distance_2d(uav, target) > uav.coverage_radius:
                self._record_invalid_action(metrics, "out_of_coverage")
                continue

            if communication_mode == MODE_HARVEST:
                if not self.primary_busy:
                    self._record_invalid_action(metrics, "busy_mode_invalid")
                    continue
                metrics["uav_energy_used"] += uav.consume_energy(float(self.uav_cfg.get("communication_energy_cost", 0.05)))
                metrics["harvested_energy_total"] += target.harvest_energy(self._harvested_energy(target))
            elif communication_mode == MODE_BACKSCATTER:
                if not self.primary_busy:
                    self._record_invalid_action(metrics, "busy_mode_invalid")
                    continue
                if not target.can_backscatter():
                    self._record_invalid_action(metrics, "no_queue")
                    continue
                metrics["uav_energy_used"] += uav.consume_energy(float(self.uav_cfg.get("communication_energy_cost", 0.05)))
                metrics["transmission_attempts"] += 1.0
                metrics["attempted_backscatter_packets"] += min(target.queue, self._backscatter_packets_per_slot(target))
                delivered, jammed = self._attempt_backscatter(target, uav)
                self.per_uav_served_packets[uav.id] += delivered
                metrics["successful_packets"] += delivered
                metrics["successful_backscatter_packets"] += delivered
                metrics["jamming_failures"] += float(jammed)
                metrics["jammed_transmission_count"] += float(jammed)
            elif communication_mode == MODE_ACTIVE:
                active_cost = self._active_energy_cost(target)
                active_allowed = (not self.primary_busy) or bool(self.channel_cfg.get("allow_active_when_busy", False))
                if not active_allowed:
                    self._record_invalid_action(metrics, "busy_mode_invalid")
                    continue
                if target.queue <= 0:
                    self._record_invalid_action(metrics, "no_queue")
                    continue
                if not target.can_active_transmit(active_cost):
                    self._record_invalid_action(metrics, "insufficient_energy")
                    continue
                metrics["uav_energy_used"] += uav.consume_energy(float(self.uav_cfg.get("communication_energy_cost", 0.05)))
                target.energy = max(0.0, target.energy - active_cost)
                metrics["iot_energy_used"] += active_cost
                metrics["transmission_attempts"] += 1.0
                metrics["attempted_active_packets"] += min(target.queue, self._active_packets_per_slot(target))
                delivered, jammed = self._attempt_active(target, uav)
                self.per_uav_served_packets[uav.id] += delivered
                metrics["successful_packets"] += delivered
                metrics["successful_active_packets"] += delivered
                metrics["jamming_failures"] += float(jammed)
                metrics["jammed_transmission_count"] += float(jammed)
            else:
                self._record_invalid_action(metrics, "invalid_action")

        metrics["duplicate_target_count"] = float(len(selected_targets) - len(set(selected_targets)))
        return metrics

    def _attempt_backscatter(self, iot: IoTDevice, uav: UAV) -> tuple[int, bool]:
        effective_tx_power = float(self.channel_cfg.get("tx_power_rf_source", 1.0)) * self._backscatter_tx_power_factor(iot)
        sinr = compute_sinr(
            effective_tx_power,
            iot,
            uav,
            self.jammers,
            float(self.channel_cfg.get("noise_power", 1.0e-9)),
            float(self.channel_cfg.get("path_loss_exponent", 2.2)),
        )
        success_prob = success_probability_from_sinr(sinr, float(self.channel_cfg.get("sinr_threshold", 1.0)))
        if self.rng.random() < success_prob:
            delivered = iot.remove_packets(self._backscatter_packets_per_slot(iot))
            self.delivered_per_iot[iot.id] += delivered
            return delivered, False
        return 0, self._is_jamming_related_failure(sinr, uav)

    def _attempt_active(self, iot: IoTDevice, uav: UAV) -> tuple[int, bool]:
        sinr = compute_sinr(
            float(self.channel_cfg.get("tx_power_iot", 0.1)),
            iot,
            uav,
            self.jammers,
            float(self.channel_cfg.get("noise_power", 1.0e-9)),
            float(self.channel_cfg.get("path_loss_exponent", 2.2)),
        )
        success_prob = success_probability_from_sinr(sinr, float(self.channel_cfg.get("sinr_threshold", 1.0)))
        if self.rng.random() < success_prob:
            delivered = iot.remove_packets(self._active_packets_per_slot(iot))
            self.delivered_per_iot[iot.id] += delivered
            return delivered, False
        return 0, self._is_jamming_related_failure(sinr, uav)

    def _is_jamming_related_failure(self, sinr: float, receiver: UAV) -> bool:
        if not self.jammers:
            return False
        threshold = float(self.channel_cfg.get("sinr_threshold", 1.0))
        noise_power = float(self.channel_cfg.get("noise_power", 1.0e-9))
        exponent = float(self.channel_cfg.get("path_loss_exponent", 2.2))
        interference = sum(jammer_interference(jammer, receiver, exponent) for jammer in self.jammers)
        return bool(sinr < threshold and interference > noise_power)

    def _active_energy_cost(self, iot: IoTDevice) -> float:
        base_cost = float(self.iot_cfg.get("active_energy_cost_per_slot", 1.0))
        type_cfg = self._iot_type_cfg(iot.device_type)
        if "active_energy_cost_per_slot" in type_cfg:
            return float(type_cfg["active_energy_cost_per_slot"])
        if iot.device_type == 2:
            return 1.25 * base_cost
        if iot.device_type == 3:
            return 1.50 * base_cost
        return base_cost

    def _backscatter_packets_per_slot(self, iot: IoTDevice) -> int:
        value = self._iot_type_cfg(iot.device_type).get("backscatter_packets_per_slot", self.iot_cfg.get("backscatter_packets_per_slot", 1))
        return max(1, int(round(float(value))))

    def _active_packets_per_slot(self, iot: IoTDevice) -> int:
        value = self._iot_type_cfg(iot.device_type).get("active_packets_per_slot", self.iot_cfg.get("active_packets_per_slot", 2))
        return max(1, int(round(float(value))))

    def _harvested_energy(self, iot: IoTDevice) -> float:
        return float(self._iot_type_cfg(iot.device_type).get("harvested_energy_per_busy_slot", self.iot_cfg.get("harvested_energy_per_busy_slot", 1.0)))

    def _backscatter_tx_power_factor(self, iot: IoTDevice) -> float:
        return float(self._iot_type_cfg(iot.device_type).get("backscatter_tx_power_factor", self.channel_cfg.get("backscatter_tx_power_factor", 0.02)))

    # ── Jammer helpers (mirror _harvested_energy / _active_energy_cost pattern) ──

    def _jammer_harvest_amount(self, jammer: Jammer, rf_source: RFSource) -> float:
        """Physics-based RF harvest: uses received_power() so closer jammer = more energy.
        Unlike IoTDevice (static, fixed value), jammer is mobile so distance matters."""
        if not self.primary_busy:
            return 0.0
        raw_power = received_power(
            rf_source.tx_power,
            distance_3d(rf_source, jammer),
            float(self.channel_cfg.get("path_loss_exponent", 2.2)),
        )
        efficiency = float(self.jammer_cfg.get("harvest_efficiency", 0.5))
        return raw_power * efficiency

    def _jam_energy_cost(self) -> float:
        return float(self.jammer_cfg.get("jam_energy_cost", 0.5))

    def _jammer_energy_threshold(self) -> float:
        return float(self.jammer_cfg.get("energy_threshold", 5.0))

    def _iot_type_cfg(self, device_type: int) -> dict[str, Any]:
        by_type = self.iot_cfg.get("type_params", {})
        if not isinstance(by_type, dict):
            return {}
        return dict(by_type.get(int(device_type), by_type.get(str(device_type), {})))

    def _record_mode_usage(self, metrics: dict[str, float], communication_mode: int) -> None:
        key_by_mode = {
            MODE_IDLE: "mode_usage_idle",
            MODE_HARVEST: "mode_usage_harvest",
            MODE_BACKSCATTER: "mode_usage_backscatter",
            MODE_ACTIVE: "mode_usage_active",
            MODE_RELAY: "mode_usage_relay",
            MODE_AVOID_JAMMER: "mode_usage_avoid_jammer",
        }
        metrics[key_by_mode.get(int(communication_mode), "mode_usage_idle")] += 1.0

    def _record_invalid_action(self, metrics: dict[str, float], reason: str) -> None:
        metrics["invalid_action_count"] += 1.0
        if reason == "out_of_coverage":
            metrics["out_of_coverage_action_count"] += 1.0
        elif reason == "no_queue":
            metrics["no_queue_selected_count"] += 1.0
        elif reason == "insufficient_energy":
            metrics["insufficient_energy_count"] += 1.0
        elif reason == "busy_mode_invalid":
            metrics["busy_mode_invalid_count"] += 1.0

    def _update_iot_arrivals(self) -> tuple[int, int]:
        arrivals = 0
        dropped = 0
        for iot in self.iot_devices:
            new_arrivals, new_dropped = iot.generate_packets(self.rng, self.frame_length)
            arrivals += new_arrivals
            dropped += new_dropped
        return arrivals, dropped

    def _update_jammers(self) -> None:
        path_loss_exp = float(self.channel_cfg.get("path_loss_exponent", 2.2))
        for jammer in self.jammers:
            for rf_source in self.rf_sources:
                jammer.harvest_energy(self._jammer_harvest_amount(jammer, rf_source))
            if jammer.is_active:
                jammer.consume_jam_energy(self._jam_energy_cost())
            if jammer.is_active:
                if jammer.energy < self._jammer_energy_threshold():
                    jammer.is_active = False
            else:
                resume_threshold = float(self.jammer_cfg.get("resume_threshold", 30.0))
                if jammer.energy >= resume_threshold:
                    jammer.is_active = True
            if not jammer.is_active and self.rf_sources:
                jammer.measured_rssi = float(sum(
                    received_power(
                        rf.tx_power,
                        max(float(np.sqrt(
                            (jammer.x - rf.x) ** 2 + (jammer.y - rf.y) ** 2
                        )), 1.0),
                        path_loss_exp,
                    )
                    for rf in self.rf_sources
                ))
                jammer.move_hill_climbing(self.rng, self.area_width, self.area_height)
            else:
                jammer.step(self.uavs, self.rng, self.area_width, self.area_height)

    def _update_primary_channel(self) -> None:
        if not self.rf_sources:
            self.primary_busy = False
            return
        self.primary_busy = any(rf_source.is_busy(self.rng) for rf_source in self.rf_sources)

    def _count_collisions(self) -> int:
        collision_distance = float(self.uav_cfg.get("collision_distance", 20.0))
        collisions = 0
        for i in range(len(self.uavs)):
            for j in range(i + 1, len(self.uavs)):
                if distance_2d(self.uavs[i], self.uavs[j]) < collision_distance:
                    collisions += 1
        return collisions

    def _compute_metrics(self) -> dict[str, float]:
        return self._episode_metrics()

    def _build_info(self) -> dict[str, Any]:
        return {
            "step": self.current_step,
            "primary_busy": self.primary_busy,
            "global_state": self.get_global_state(),
            "frame_metrics": deepcopy(self.last_frame_metrics),
            "episode_metrics": self._episode_metrics(),
            "reward_breakdown": deepcopy(self.last_reward_breakdown),
        }

    def _episode_metrics(self) -> dict[str, float]:
        steps = max(self.episode_totals.get("steps", 0.0), 1.0)
        total_arrived = float(sum(iot.total_arrived for iot in self.iot_devices))
        total_delivered = float(sum(iot.total_delivered for iot in self.iot_devices))
        total_dropped = float(sum(iot.total_dropped for iot in self.iot_devices))
        avg_throughput_per_frame = total_delivered / steps
        avg_queue_length = self.episode_totals.get("sum_queue_length", 0.0) / steps
        total_energy = self.episode_totals.get("uav_energy_consumption", 0.0) + self.episode_totals.get("iot_energy_consumption", 0.0)
        delay_rate_floor = 1.0 / steps
        actions_processed = self.episode_totals.get("actions_processed", 0.0)
        transmission_attempts = self.episode_totals.get("transmission_attempts", 0.0)
        attempted_backscatter = self.episode_totals.get("attempted_backscatter_packets", 0.0)
        attempted_active = self.episode_totals.get("attempted_active_packets", 0.0)
        metrics = {
            "total_reward": float(self.episode_totals.get("total_reward", 0.0)),
            "total_throughput": total_delivered,
            "avg_throughput_per_frame": float(avg_throughput_per_frame),
            "packet_delivery_ratio": total_delivered / max(total_arrived, 1.0),
            "packet_drop_rate": total_dropped / max(total_arrived, 1.0),
            "avg_queue_length": float(avg_queue_length),
            "avg_delay_proxy": float(avg_queue_length / max(avg_throughput_per_frame, delay_rate_floor)),
            "energy_efficiency": float(total_delivered / max(total_energy, 1.0e-9)),
            "uav_energy_consumption": float(self.episode_totals.get("uav_energy_consumption", 0.0)),
            "iot_energy_consumption": float(self.episode_totals.get("iot_energy_consumption", 0.0)),
            "total_energy_consumption": float(total_energy),
            "jamming_failure_rate": float(
                self.episode_totals.get("jamming_failures", 0.0) / max(self.episode_totals.get("transmission_attempts", 0.0), 1.0)
            ),
            "collision_count": float(self.episode_totals.get("collision_count", 0.0)),
            "fairness_index": float(self._jain_fairness_index()),
            "duplicate_target_count": float(self.episode_totals.get("duplicate_target_count", 0.0)),
            "invalid_action_count": float(self.episode_totals.get("invalid_action_count", 0.0)),
            "invalid_action_rate": float(self.episode_totals.get("invalid_action_count", 0.0) / max(actions_processed, 1.0)),
            "out_of_coverage_action_count": float(self.episode_totals.get("out_of_coverage_action_count", 0.0)),
            "out_of_coverage_action_rate": float(self.episode_totals.get("out_of_coverage_action_count", 0.0) / max(actions_processed, 1.0)),
            "no_queue_selected_count": float(self.episode_totals.get("no_queue_selected_count", 0.0)),
            "insufficient_energy_count": float(self.episode_totals.get("insufficient_energy_count", 0.0)),
            "busy_mode_invalid_count": float(self.episode_totals.get("busy_mode_invalid_count", 0.0)),
            "jammed_transmission_count": float(self.episode_totals.get("jammed_transmission_count", 0.0)),
            "jammed_transmission_rate": float(self.episode_totals.get("jammed_transmission_count", 0.0) / max(transmission_attempts, 1.0)),
            "successful_backscatter_packets": float(self.episode_totals.get("successful_backscatter_packets", 0.0)),
            "successful_active_packets": float(self.episode_totals.get("successful_active_packets", 0.0)),
            "attempted_backscatter_packets": float(attempted_backscatter),
            "attempted_active_packets": float(attempted_active),
            "backscatter_success_rate": float(self.episode_totals.get("successful_backscatter_packets", 0.0) / max(attempted_backscatter, 1.0)),
            "active_success_rate": float(self.episode_totals.get("successful_active_packets", 0.0) / max(attempted_active, 1.0)),
            "harvested_energy_total": float(self.episode_totals.get("harvested_energy_total", 0.0)),
            "mode_usage_idle": float(self.episode_totals.get("mode_usage_idle", 0.0)),
            "mode_usage_harvest": float(self.episode_totals.get("mode_usage_harvest", 0.0)),
            "mode_usage_backscatter": float(self.episode_totals.get("mode_usage_backscatter", 0.0)),
            "mode_usage_active": float(self.episode_totals.get("mode_usage_active", 0.0)),
            "mode_usage_relay": float(self.episode_totals.get("mode_usage_relay", 0.0)),
            "mode_usage_avoid_jammer": float(self.episode_totals.get("mode_usage_avoid_jammer", 0.0)),
            "greedy_fallback_count": float(self.episode_totals.get("greedy_fallback_count", 0.0)),
            "jammer_active_steps": float(self.episode_totals.get("jammer_active_steps", 0.0)),
            "jammer_inactive_steps": float(self.episode_totals.get("jammer_inactive_steps", 0.0)),
            "jammer_harvested_energy_total": float(self.episode_totals.get("jammer_harvested_energy_total", 0.0)),
        }
        metrics.update(self._flatten_named_values("per_uav_served_packets_uav", self.per_uav_served_packets))
        metrics.update(self._flatten_named_values("per_iot_delivered_packets_iot", self.delivered_per_iot))
        return metrics

    def _diagnostic_frame_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        attempted_backscatter = metrics.get("attempted_backscatter_packets", 0.0)
        attempted_active = metrics.get("attempted_active_packets", 0.0)
        frame = {
            "invalid_action_count": float(metrics.get("invalid_action_count", 0.0)),
            "out_of_coverage_action_count": float(metrics.get("out_of_coverage_action_count", 0.0)),
            "no_queue_selected_count": float(metrics.get("no_queue_selected_count", 0.0)),
            "insufficient_energy_count": float(metrics.get("insufficient_energy_count", 0.0)),
            "busy_mode_invalid_count": float(metrics.get("busy_mode_invalid_count", 0.0)),
            "jammed_transmission_count": float(metrics.get("jammed_transmission_count", 0.0)),
            "successful_backscatter_packets": float(metrics.get("successful_backscatter_packets", 0.0)),
            "successful_active_packets": float(metrics.get("successful_active_packets", 0.0)),
            "attempted_backscatter_packets": float(attempted_backscatter),
            "attempted_active_packets": float(attempted_active),
            "backscatter_success_rate": float(metrics.get("successful_backscatter_packets", 0.0) / max(attempted_backscatter, 1.0)),
            "active_success_rate": float(metrics.get("successful_active_packets", 0.0) / max(attempted_active, 1.0)),
            "harvested_energy_total": float(metrics.get("harvested_energy_total", 0.0)),
            "mode_usage_idle": float(metrics.get("mode_usage_idle", 0.0)),
            "mode_usage_harvest": float(metrics.get("mode_usage_harvest", 0.0)),
            "mode_usage_backscatter": float(metrics.get("mode_usage_backscatter", 0.0)),
            "mode_usage_active": float(metrics.get("mode_usage_active", 0.0)),
            "mode_usage_relay": float(metrics.get("mode_usage_relay", 0.0)),
            "mode_usage_avoid_jammer": float(metrics.get("mode_usage_avoid_jammer", 0.0)),
            "greedy_fallback_count": float(metrics.get("greedy_fallback_count", 0.0)),
        }
        frame.update(self._flatten_named_values("per_uav_served_packets_uav", self.per_uav_served_packets))
        frame.update(self._flatten_named_values("per_iot_delivered_packets_iot", self.delivered_per_iot))
        return frame

    def _flatten_named_values(self, prefix: str, values: np.ndarray) -> dict[str, float]:
        return {f"{prefix}_{idx}": float(value) for idx, value in enumerate(values)}

    def _get_all_observations(self) -> np.ndarray:
        return np.stack([self.get_local_observation(uav_id) for uav_id in range(self.num_uav)], axis=0)

    def _normalize_actions(self, actions: list[int] | dict[str, int]) -> list[int]:
        if isinstance(actions, dict):
            return [int(actions.get(f"uav_{idx}", 0)) for idx in range(self.num_uav)]
        action_list = list(actions)
        if len(action_list) < self.num_uav:
            action_list.extend([0] * (self.num_uav - len(action_list)))
        return [int(action) for action in action_list[: self.num_uav]]

    def _average_queue_length(self) -> float:
        if not self.iot_devices:
            return 0.0
        return float(np.mean([iot.queue for iot in self.iot_devices]))

    def _jain_fairness_index(self) -> float:
        x = self.delivered_per_iot.astype(np.float64)
        if x.size == 0 or np.sum(x) <= 0.0:
            return 1.0
        numerator = float(np.sum(x) ** 2)
        denominator = float(x.size * np.sum(x**2) + 1.0e-12)
        return numerator / denominator

    def _nearest_jammer_distance(self, uav: UAV) -> float:
        if not self.jammers:
            return float(np.hypot(self.area_width, self.area_height))
        return min(distance_2d(uav, jammer) for jammer in self.jammers)
