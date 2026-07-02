from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.mobility_model import MOVE_DOWN, MOVE_UP
from envs.uav_backscatter_env import MODE_IDLE, UAVBackscatterEnv, encode_action


DEFAULT_3D_CONFIG = ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"


def _positions(entities: list[Any]) -> list[list[float]]:
    return [[float(entity.x), float(entity.y), float(entity.z)] for entity in entities]


def _idle_movement_action(env: UAVBackscatterEnv, movement: int = 0) -> int:
    return encode_action(movement, 0, MODE_IDLE, env.num_iot, env.num_movement_actions)


def _rollout_actions(env: UAVBackscatterEnv, step: int, rng: np.random.Generator, action_mode: str) -> list[int]:
    if action_mode == "random":
        return [int(rng.integers(0, env.action_size)) for _ in range(env.num_uav)]
    if action_mode == "vertical" and env.num_movement_actions > MOVE_DOWN:
        movement = MOVE_UP if step % 4 in (0, 1) else MOVE_DOWN
        return [_idle_movement_action(env, movement) for _ in range(env.num_uav)]
    return [_idle_movement_action(env, 0) for _ in range(env.num_uav)]


def collect_rollout(
    config: str | Path | dict[str, Any] = DEFAULT_3D_CONFIG,
    steps: int = 30,
    seed: int = 123,
    action_mode: str = "vertical",
) -> dict[str, Any]:
    env = UAVBackscatterEnv(config)
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    data: dict[str, Any] = {
        "uav_positions": [_positions(env.uavs)],
        "jammer_positions": [_positions(env.jammers)],
        "iot_positions": _positions(env.iot_devices),
        "episode_metrics": [dict(info.get("episode_metrics", {}))],
        "action_mode": action_mode,
    }

    for step in range(max(0, int(steps))):
        actions = _rollout_actions(env, step, rng, action_mode)
        _, _, terminated, truncated, info = env.step(actions)
        data["uav_positions"].append(_positions(env.uavs))
        data["jammer_positions"].append(_positions(env.jammers))
        data["episode_metrics"].append(dict(info.get("episode_metrics", {})))
        if terminated or truncated:
            break

    env.close()
    data["steps"] = list(range(len(data["uav_positions"])))
    return data


def load_trajectory_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_trajectory_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output
