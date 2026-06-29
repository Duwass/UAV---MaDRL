from __future__ import annotations


def compute_reward(components: dict[str, float], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    throughput = float(components.get("successful_packets", 0.0))
    dropped = float(components.get("dropped_packets", 0.0))
    avg_queue = float(components.get("avg_queue_length", 0.0))
    uav_energy = float(components.get("uav_energy_used", 0.0))
    jamming_failures = float(components.get("jamming_failures", 0.0))
    collisions = float(components.get("collision_count", 0.0))
    unfairness = float(components.get("unfairness_penalty", 0.0))
    avoidance_bonus = float(components.get("avoidance_bonus", 0.0))

    reward_throughput = float(weights.get("w_throughput", 1.0)) * throughput
    penalty_drop = float(weights.get("w_drop", 2.0)) * dropped
    penalty_delay = float(weights.get("w_delay", 0.1)) * avg_queue
    penalty_energy = float(weights.get("w_uav_energy", 0.01)) * uav_energy
    penalty_jamming = float(weights.get("w_jamming", 1.0)) * jamming_failures
    penalty_collision = float(weights.get("w_collision", 5.0)) * collisions
    penalty_unfairness = float(weights.get("w_unfairness", 0.2)) * unfairness
    reward_avoidance = float(weights.get("w_avoid_jammer", 0.0)) * avoidance_bonus

    total = (
        reward_throughput
        + reward_avoidance
        - penalty_drop
        - penalty_delay
        - penalty_energy
        - penalty_jamming
        - penalty_collision
        - penalty_unfairness
    )
    breakdown = {
        "reward_total": total,
        "reward_throughput": reward_throughput,
        "reward_avoidance": reward_avoidance,
        "penalty_drop": penalty_drop,
        "penalty_delay": penalty_delay,
        "penalty_energy": penalty_energy,
        "penalty_jamming": penalty_jamming,
        "penalty_collision": penalty_collision,
        "penalty_unfairness": penalty_unfairness,
    }
    return float(total), breakdown

