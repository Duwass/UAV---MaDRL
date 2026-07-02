from __future__ import annotations

import numpy as np


def _xyz(entity: object) -> tuple[float, float, float]:
    if isinstance(entity, (tuple, list, np.ndarray)):
        if len(entity) == 2:
            return float(entity[0]), float(entity[1]), 0.0
        return float(entity[0]), float(entity[1]), float(entity[2])
    return float(getattr(entity, "x")), float(getattr(entity, "y")), float(getattr(entity, "z", getattr(entity, "h", 0.0)))


def horizontal_distance_2d(a: object, b: object, min_distance: float = 0.0) -> float:
    ax, ay, _ = _xyz(a)
    bx, by, _ = _xyz(b)
    return float(max(np.hypot(ax - bx, ay - by), float(min_distance)))


def distance_2d(a: object, b: object, min_distance: float = 0.0) -> float:
    return horizontal_distance_2d(a, b, min_distance)


def distance_3d(a: object, b: object, min_distance: float = 0.0) -> float:
    ax, ay, az = _xyz(a)
    bx, by, bz = _xyz(b)
    return float(max(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2), float(min_distance)))


def select_distance(a: object, b: object, mode: str = "3d", min_distance: float = 0.0) -> float:
    normalized = str(mode).lower().replace("-", "_")
    if normalized in {"3d", "euclidean_3d"}:
        return distance_3d(a, b, min_distance)
    if normalized in {"2d", "horizontal", "horizontal_2d"}:
        return horizontal_distance_2d(a, b, min_distance)
    raise ValueError(f"Unsupported distance mode: {mode}")


def path_loss(distance: float, path_loss_exponent: float) -> float:
    return float(max(distance, 1.0) ** path_loss_exponent)


def received_power(tx_power: float, distance: float, path_loss_exponent: float) -> float:
    return float(max(tx_power, 0.0) / path_loss(distance, path_loss_exponent))


def jammer_interference(
    jammer: object,
    receiver: object,
    path_loss_exponent: float,
    distance_mode: str = "3d",
    influence_distance_mode: str = "horizontal",
) -> float:
    if not bool(getattr(jammer, "is_active", True)):
        return 0.0
    if select_distance(jammer, receiver, influence_distance_mode) > float(getattr(jammer, "radius", np.inf)):
        return 0.0
    return received_power(float(getattr(jammer, "power")), select_distance(jammer, receiver, distance_mode), path_loss_exponent)


def compute_sinr(
    tx_power: float,
    transmitter: object,
    receiver: object,
    jammers: list[object],
    noise_power: float,
    path_loss_exponent: float,
    distance_mode: str = "3d",
    jammer_influence_distance_mode: str = "horizontal",
) -> float:
    signal_power = received_power(tx_power, select_distance(transmitter, receiver, distance_mode), path_loss_exponent)
    total_interference = sum(
        jammer_interference(jammer, receiver, path_loss_exponent, distance_mode, jammer_influence_distance_mode)
        for jammer in jammers
    )
    denominator = max(float(noise_power) + total_interference, 1.0e-18)
    return float(signal_power / denominator)


def success_probability_from_sinr(sinr: float, threshold: float) -> float:
    sinr = max(float(sinr), 0.0)
    threshold = max(float(threshold), 0.0)
    if threshold == 0.0:
        return 1.0 if sinr > 0.0 else 0.0
    if sinr >= threshold:
        return float(min(1.0, sinr / (sinr + threshold)))
    return float(0.1 * sinr / threshold)

