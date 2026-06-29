from __future__ import annotations

from typing import Any

import numpy as np

from baselines import format_actions, unwrap_env


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, observations: Any, env: Any | None = None):
        base_env = unwrap_env(env)
        num_uav = len(observations) if isinstance(observations, dict) else base_env.num_uav
        use_mask = bool(getattr(base_env, "action_masking_cfg", {}).get("enabled", False)) and bool(
            getattr(base_env, "action_masking_cfg", {}).get("random_uses_mask", False)
        )
        actions = []
        for uav_id in range(num_uav):
            if use_mask:
                valid_actions = np.flatnonzero(base_env.get_action_mask(uav_id))
                actions.append(int(self.rng.choice(valid_actions)))
            else:
                actions.append(int(self.rng.integers(0, base_env.action_size)))
        return format_actions(actions, observations, env)


Policy = RandomPolicy
