from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _clip_position(x: float, y: float, area_width: float, area_height: float) -> tuple[float, float]:
    return float(np.clip(x, 0.0, area_width)), float(np.clip(y, 0.0, area_height))


@dataclass(init=False)
class UAV:
    id: int
    x: float
    y: float
    z: float
    energy: float
    coverage_radius: float
    max_speed: float
    current_target: int | None = None
    last_action: int | None = None

    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        z: float | None = None,
        energy: float = 0.0,
        coverage_radius: float = 0.0,
        max_speed: float = 0.0,
        current_target: int | None = None,
        last_action: int | None = None,
        h: float | None = None,
    ):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)
        self.z = float(h if z is None and h is not None else (0.0 if z is None else z))
        self.energy = float(energy)
        self.coverage_radius = float(coverage_radius)
        self.max_speed = float(max_speed)
        self.current_target = current_target
        self.last_action = last_action

    @property
    def h(self) -> float:
        return self.z

    @h.setter
    def h(self, value: float) -> None:
        self.z = float(value)

    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def distance_to(self, entity: object) -> float:
        ex = float(getattr(entity, "x"))
        ey = float(getattr(entity, "y"))
        ez = float(getattr(entity, "z", getattr(entity, "h", 0.0)))
        return float(np.sqrt((self.x - ex) ** 2 + (self.y - ey) ** 2 + (self.z - ez) ** 2))

    def consume_energy(self, amount: float) -> float:
        amount = max(0.0, float(amount))
        consumed = min(self.energy, amount)
        self.energy -= consumed
        return consumed

    def is_alive(self) -> bool:
        return self.energy > 0.0


@dataclass
class IoTDevice:
    id: int
    x: float
    y: float
    device_type: int
    queue: int
    queue_capacity: int
    energy: float
    energy_capacity: float
    packet_arrival_prob: float
    total_arrived: int = 0
    total_delivered: int = 0
    total_dropped: int = 0
    z: float = 0.0

    @property
    def h(self) -> float:
        return self.z

    @h.setter
    def h(self, value: float) -> None:
        self.z = float(value)

    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def generate_packets(self, rng: np.random.Generator, frame_length: int) -> tuple[int, int]:
        arrivals = int(rng.binomial(int(frame_length), float(self.packet_arrival_prob)))
        dropped = self.add_packets(arrivals)
        self.total_arrived += arrivals
        self.total_dropped += dropped
        return arrivals, dropped

    def harvest_energy(self, amount: float) -> float:
        amount = max(0.0, float(amount))
        before = self.energy
        self.energy = min(self.energy_capacity, self.energy + amount)
        return self.energy - before

    def can_backscatter(self) -> bool:
        return self.queue > 0

    def can_active_transmit(self, active_energy_cost: float) -> bool:
        return self.queue > 0 and self.energy >= active_energy_cost

    def remove_packets(self, num_packets: int) -> int:
        removed = min(max(0, int(num_packets)), self.queue)
        self.queue -= removed
        self.total_delivered += removed
        return removed

    def add_packets(self, num_packets: int) -> int:
        num_packets = max(0, int(num_packets))
        accepted = min(num_packets, self.queue_capacity - self.queue)
        self.queue += accepted
        return num_packets - accepted


@dataclass
class Jammer:
    id: int
    x: float
    y: float
    power: float
    speed: float
    radius: float
    mobility: str
    energy: float = 0.0
    energy_capacity: float = 100.0
    is_active: bool = True
    measured_rssi: float = 0.0 
    last_rssi: float = 0.0
    last_dx: float = 0.0 
    last_dy: float = 0.0
    z: float = 0.0

    @property
    def h(self) -> float:
        return self.z

    @h.setter
    def h(self, value: float) -> None:
        self.z = float(value)

    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def harvest_energy(self, amount: float) -> float:
        amount = max(0.0, float(amount))
        before = self.energy
        self.energy = min(self.energy_capacity, self.energy + amount)
        return self.energy - before

    def consume_jam_energy(self, amount: float) -> float:
        amount = max(0.0, float(amount))
        consumed = min(self.energy, amount)
        self.energy -= consumed
        return consumed

    def move_hill_climbing(
        self,
        rng: np.random.Generator,
        area_width: float,
        area_height: float,
    ) -> None:
        if self.measured_rssi >= self.last_rssi:
            if self.last_dx == 0.0 and self.last_dy == 0.0:
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                self.last_dx = float(np.cos(angle))
                self.last_dy = float(np.sin(angle))
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            self.last_dx = float(np.cos(angle))
            self.last_dy = float(np.sin(angle))
        self.last_rssi = self.measured_rssi
        self.x = float(np.clip(self.x + self.speed * self.last_dx, 0.0, area_width))
        self.y = float(np.clip(self.y + self.speed * self.last_dy, 0.0, area_height))

    def move_toward_stronger_signal(
        self,
        rf_sources: list,
        path_loss_exponent: float,
        area_width: float,
        area_height: float,
    ) -> None:
        if not rf_sources:
            return
        best_power = -1.0
        best_cos, best_sin = 0.0, 0.0
        for k in range(8):
            angle = k * 2.0 * np.pi / 8.0
            nx = self.x + self.speed * np.cos(angle)
            ny = self.y + self.speed * np.sin(angle)
            total_power = 0.0
            for rf in rf_sources:
                rx = float(getattr(rf, "x"))
                ry = float(getattr(rf, "y"))
                rh = float(getattr(rf, "h", 0.0))
                dist = float(np.sqrt((nx - rx) ** 2 + (ny - ry) ** 2 + rh ** 2))
                dist = max(dist, 1.0)
                total_power += float(getattr(rf, "tx_power", 1.0)) / (dist ** path_loss_exponent)
            if total_power > best_power:
                best_power = total_power
                best_cos = np.cos(angle)
                best_sin = np.sin(angle)
        self.x = float(np.clip(self.x + self.speed * best_cos, 0.0, area_width))
        self.y = float(np.clip(self.y + self.speed * best_sin, 0.0, area_height))

    def move_random_walk(self, rng: np.random.Generator, area_width: float, area_height: float) -> None:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        self.x += self.speed * np.cos(angle)
        self.y += self.speed * np.sin(angle)
        self.x, self.y = _clip_position(self.x, self.y, area_width, area_height)

    def move_chase_nearest_uav(self, uavs: Sequence[UAV], area_width: float, area_height: float) -> None:
        live_uavs = [uav for uav in uavs if uav.is_alive()]
        if not live_uavs:
            return
        nearest = min(live_uavs, key=lambda uav: (uav.x - self.x) ** 2 + (uav.y - self.y) ** 2)
        dx = nearest.x - self.x
        dy = nearest.y - self.y
        dist = float(np.hypot(dx, dy))
        if dist > 1.0e-12:
            step = min(self.speed, dist)
            self.x += step * dx / dist
            self.y += step * dy / dist
        self.x, self.y = _clip_position(self.x, self.y, area_width, area_height)

    def step(
        self,
        uavs: Sequence[UAV],
        rng: np.random.Generator,
        area_width: float,
        area_height: float,
    ) -> None:
        if self.mobility == "random_walk":
            self.move_random_walk(rng, area_width, area_height)
        elif self.mobility == "chase_nearest_uav":
            self.move_chase_nearest_uav(uavs, area_width, area_height)
        elif self.mobility == "static":
            return
        else:
            self.move_random_walk(rng, area_width, area_height)


@dataclass
class RFSource:
    id: int
    x: float
    y: float
    tx_power: float
    busy_prob: float
    z: float = 0.0

    @property
    def h(self) -> float:
        return self.z

    @h.setter
    def h(self, value: float) -> None:
        self.z = float(value)

    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def is_busy(self, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.busy_prob)

