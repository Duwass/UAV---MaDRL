from __future__ import annotations

import numpy as np

from envs.entities import UAV


MOVEMENT_DELTAS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, 1.0),
    2: (0.0, -1.0),
    3: (1.0, 0.0),
    4: (-1.0, 0.0),
    5: (1.0, 1.0),
    6: (-1.0, 1.0),
    7: (1.0, -1.0),
    8: (-1.0, -1.0),
}


def move_uav(uav: UAV, action: int, max_speed: float, area_width: float, area_height: float) -> None:
    dx, dy = MOVEMENT_DELTAS.get(int(action), (0.0, 0.0))
    norm = float(np.hypot(dx, dy))
    if norm > 0.0:
        dx = dx / norm * max_speed
        dy = dy / norm * max_speed
    uav.x = float(np.clip(uav.x + dx, 0.0, area_width))
    uav.y = float(np.clip(uav.y + dy, 0.0, area_height))


def compute_movement_energy(action: int, movement_energy_cost: float) -> float:
    dx, dy = MOVEMENT_DELTAS.get(int(action), (0.0, 0.0))
    distance_factor = float(np.hypot(dx, dy))
    return float(max(0.0, movement_energy_cost) * distance_factor)

