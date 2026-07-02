from __future__ import annotations

import numpy as np

from envs.entities import UAV


MOVE_STAY = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_EAST = 3
MOVE_WEST = 4
MOVE_NORTHEAST = 5
MOVE_NORTHWEST = 6
MOVE_SOUTHEAST = 7
MOVE_SOUTHWEST = 8
MOVE_UP = 9
MOVE_DOWN = 10

MOVEMENT_DELTAS: dict[int, tuple[float, float, float]] = {
    MOVE_STAY: (0.0, 0.0, 0.0),
    MOVE_NORTH: (0.0, 1.0, 0.0),
    MOVE_SOUTH: (0.0, -1.0, 0.0),
    MOVE_EAST: (1.0, 0.0, 0.0),
    MOVE_WEST: (-1.0, 0.0, 0.0),
    MOVE_NORTHEAST: (1.0, 1.0, 0.0),
    MOVE_NORTHWEST: (-1.0, 1.0, 0.0),
    MOVE_SOUTHEAST: (1.0, -1.0, 0.0),
    MOVE_SOUTHWEST: (-1.0, -1.0, 0.0),
    MOVE_UP: (0.0, 0.0, 1.0),
    MOVE_DOWN: (0.0, 0.0, -1.0),
}

LEGACY_MOVEMENT_ACTIONS = 9
MOVEMENT_ACTIONS_3D = len(MOVEMENT_DELTAS)


def movement_delta(action: int) -> tuple[float, float, float]:
    return MOVEMENT_DELTAS.get(int(action), (0.0, 0.0, 0.0))


def move_uav(
    uav: UAV,
    action: int,
    horizontal_step: float,
    area_width: float,
    area_height: float,
    vertical_step: float = 0.0,
    altitude_min: float | None = None,
    altitude_max: float | None = None,
) -> None:
    dx, dy, dz = movement_delta(action)
    norm = float(np.hypot(dx, dy))
    if norm > 0.0:
        dx = dx / norm * horizontal_step
        dy = dy / norm * horizontal_step
    dz = dz * vertical_step
    uav.x = float(np.clip(uav.x + dx, 0.0, area_width))
    uav.y = float(np.clip(uav.y + dy, 0.0, area_height))
    if altitude_min is None:
        altitude_min = float(getattr(uav, "z", getattr(uav, "h", 0.0)))
    if altitude_max is None:
        altitude_max = float(getattr(uav, "z", getattr(uav, "h", 0.0)))
    uav.z = float(np.clip(uav.z + dz, float(altitude_min), float(altitude_max)))


def compute_movement_energy(action: int, movement_energy_cost: float) -> float:
    dx, dy, dz = movement_delta(action)
    distance_factor = float(np.sqrt(dx**2 + dy**2 + dz**2))
    return float(max(0.0, movement_energy_cost) * distance_factor)
