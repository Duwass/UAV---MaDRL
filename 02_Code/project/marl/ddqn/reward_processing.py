from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardProcessor:
    enabled: bool = False
    mode: str = "none"
    scale: float = 1.0
    clip_rewards: bool = False
    clip_min: float = -10.0
    clip_max: float = 10.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "RewardProcessor":
        cfg = dict(config or {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            mode=str(cfg.get("mode", "none")).lower(),
            scale=float(cfg.get("scale", 1.0)),
            clip_rewards=bool(cfg.get("clip_rewards", False)),
            clip_min=float(cfg.get("clip_min", -10.0)),
            clip_max=float(cfg.get("clip_max", 10.0)),
        )

    def process(self, reward: float) -> float:
        if not self.enabled or self.mode == "none":
            value = float(reward)
        elif self.mode == "scale":
            value = float(reward) / max(self.scale, 1.0e-12)
        elif self.mode == "clip":
            value = float(reward) / max(self.scale, 1.0e-12)
            value = float(np.clip(value, self.clip_min, self.clip_max))
        else:
            raise ValueError(f"Unsupported reward_processing.mode={self.mode}")

        if self.clip_rewards and self.mode != "clip":
            value = float(np.clip(value, self.clip_min, self.clip_max))
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "scale": self.scale,
            "clip_rewards": self.clip_rewards,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }
