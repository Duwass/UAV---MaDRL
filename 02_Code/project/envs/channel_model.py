from __future__ import annotations

import numpy as np


def _xyh(entity: object) -> tuple[float, float, float]:
    if isinstance(entity, (tuple, list, np.ndarray)):
        if len(entity) == 2:
            return float(entity[0]), float(entity[1]), 0.0
        return float(entity[0]), float(entity[1]), float(entity[2])
    return float(getattr(entity, "x")), float(getattr(entity, "y")), float(getattr(entity, "h", 0.0))


def distance_2d(a: object, b: object) -> float:
    ax, ay, _ = _xyh(a)
    bx, by, _ = _xyh(b)
    return float(np.hypot(ax - bx, ay - by))


def distance_3d(a: object, b: object) -> float:
    ax, ay, ah = _xyh(a)
    bx, by, bh = _xyh(b)
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (ah - bh) ** 2))


def path_loss(distance: float, path_loss_exponent: float) -> float:
    # Keep the reference distance at 1 m to avoid singular received power.
    return float(max(distance, 1.0) ** path_loss_exponent)


def received_power(tx_power: float, distance: float, path_loss_exponent: float) -> float:
    return float(max(tx_power, 0.0) / path_loss(distance, path_loss_exponent))


def jammer_interference(jammer: object, receiver: object, path_loss_exponent: float) -> float:
    if distance_2d(jammer, receiver) > float(getattr(jammer, "radius", np.inf)):
        return 0.0
    return received_power(float(getattr(jammer, "power")), distance_3d(jammer, receiver), path_loss_exponent)


def compute_sinr(
    tx_power: float,
    transmitter: object,
    receiver: object,
    jammers: list[object],
    noise_power: float,
    path_loss_exponent: float,
) -> float:
    signal_power = received_power(tx_power, distance_3d(transmitter, receiver), path_loss_exponent)
    total_interference = sum(jammer_interference(jammer, receiver, path_loss_exponent) for jammer in jammers)
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

