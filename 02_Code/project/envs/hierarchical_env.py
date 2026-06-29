from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.hierarchical_action import (
    AVOID_JAMMER,
    HIGH_LEVEL_ACTION_NAMES,
    IDLE_SAFE,
    PRIORITIZE_ACTIVE_TYPE1,
    PRIORITIZE_BACKSCATTER_TYPE23,
    HierarchicalActionExecutor,
)


class HierarchicalUAVBackscatterEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_env,
        executor_config: dict[str, Any] | None = None,
        hierarchical_actions_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.base_env = base_env
        self.executor = HierarchicalActionExecutor(base_env, executor_config)
        self.hierarchical_actions_config = dict(hierarchical_actions_config or {})
        self.num_uav = int(base_env.num_uav)
        self.num_iot = int(base_env.num_iot)
        self.action_size = self.executor.num_high_level_actions()
        self.disabled_actions = {
            int(action)
            for action in self.hierarchical_actions_config.get("disabled_actions", [])
            if 0 <= int(action) < self.action_size
        }
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = base_env.observation_space
        self.single_observation_space = base_env.single_observation_space
        self.state_space = base_env.state_space
        self._episode_high_action_counts = np.zeros(self.action_size, dtype=np.float64)
        self._episode_fallback_count = 0.0
        self._episode_executor_mode_counts = {"idle": 0.0, "harvest": 0.0, "backscatter": 0.0, "active": 0.0, "avoid_jammer": 0.0}

    def reset(self, *args, **kwargs):
        self._episode_high_action_counts = np.zeros(self.action_size, dtype=np.float64)
        self._episode_fallback_count = 0.0
        self._episode_executor_mode_counts = {"idle": 0.0, "harvest": 0.0, "backscatter": 0.0, "active": 0.0, "avoid_jammer": 0.0}
        observations, info = self.base_env.reset(*args, **kwargs)
        return observations, self._augment_info(info)

    def step(self, high_level_actions):
        high_actions = self._normalize_high_actions(high_level_actions)
        original_actions = self.executor.build_original_actions(high_actions)
        self._record_executor_diagnostics()
        observations, reward, terminated, truncated, info = self.base_env.step(original_actions)
        return observations, reward, terminated, truncated, self._augment_info(info)

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()

    def get_global_state(self) -> np.ndarray:
        return self.base_env.get_global_state()

    def get_local_observation(self, uav_id: int) -> np.ndarray:
        return self.base_env.get_local_observation(uav_id)

    def get_action_mask(self, uav_id: int) -> np.ndarray:
        mask = np.ones(self.action_size, dtype=np.int8)
        if not self.base_env.jammers:
            mask[AVOID_JAMMER] = 0
        device_types = {int(iot.device_type) for iot in self.base_env.iot_devices}
        if not ({2, 3} & device_types):
            mask[PRIORITIZE_BACKSCATTER_TYPE23] = 0
        if 1 not in device_types:
            mask[PRIORITIZE_ACTIVE_TYPE1] = 0
        for action_id in self.disabled_actions:
            mask[int(action_id)] = 0
        if not np.any(mask):
            fallback = self._fallback_action_id()
            mask[fallback] = 1
            warnings.warn(
                "All hierarchical actions were masked out; falling back to action "
                f"{fallback} ({HIGH_LEVEL_ACTION_NAMES.get(fallback, 'UNKNOWN')}).",
                RuntimeWarning,
                stacklevel=2,
            )
        return mask

    def _normalize_high_actions(self, actions) -> list[int]:
        if isinstance(actions, dict):
            action_list = [int(actions.get(f"uav_{idx}", 0)) for idx in range(self.num_uav)]
        else:
            action_list = list(actions)
        if len(action_list) < self.num_uav:
            action_list.extend([0] * (self.num_uav - len(action_list)))
        normalized = [int(np.clip(action, 0, self.action_size - 1)) for action in action_list[: self.num_uav]]
        fallback = self._fallback_action_id()
        return [fallback if action in self.disabled_actions else action for action in normalized]

    def _fallback_action_id(self) -> int:
        if IDLE_SAFE not in self.disabled_actions:
            return IDLE_SAFE
        for action_id in range(self.action_size):
            if action_id not in self.disabled_actions:
                return action_id
        return IDLE_SAFE

    def _record_executor_diagnostics(self) -> None:
        diag = self.executor.last_diagnostics
        for action_id, count in diag.get("high_level_action_counts", {}).items():
            self._episode_high_action_counts[int(action_id)] += float(count)
        self._episode_fallback_count += float(diag.get("fallback_count", 0.0))
        for mode, count in diag.get("high_level_mode_usage", {}).items():
            self._episode_executor_mode_counts[mode] = self._episode_executor_mode_counts.get(mode, 0.0) + float(count)

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        out = dict(info)
        diag = deepcopy(self.executor.last_diagnostics)
        out["hierarchical_action"] = diag
        metrics = dict(out.get("episode_metrics", {}))
        actions = max(float(self.base_env.episode_totals.get("actions_processed", 0.0)), 1.0)
        for idx, count in enumerate(self._episode_high_action_counts):
            metrics[f"high_level_action_{idx}_count"] = float(count)
        metrics["fallback_count"] = float(self._episode_fallback_count)
        metrics["fallback_rate"] = float(self._episode_fallback_count / actions)
        metrics["executor_idle_count"] = float(self._episode_executor_mode_counts.get("idle", 0.0))
        metrics["executor_harvest_count"] = float(self._episode_executor_mode_counts.get("harvest", 0.0))
        metrics["executor_backscatter_count"] = float(self._episode_executor_mode_counts.get("backscatter", 0.0))
        metrics["executor_active_count"] = float(self._episode_executor_mode_counts.get("active", 0.0))
        metrics["executor_avoid_jammer_count"] = float(self._episode_executor_mode_counts.get("avoid_jammer", 0.0))
        out["episode_metrics"] = metrics
        return out

    def __getattr__(self, name: str):
        return getattr(self.base_env, name)
