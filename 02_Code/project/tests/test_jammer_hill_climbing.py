from __future__ import annotations

import numpy as np
import pytest

from envs.entities import Jammer


def make_jammer(x: float = 250.0, y: float = 250.0, speed: float = 10.0) -> Jammer:
    return Jammer(
        id=0, x=x, y=y,
        power=1.0, speed=speed, radius=200.0, mobility="random_walk",
    )


def rssi_at(jammer: Jammer, src_x: float, src_y: float,
            tx_power: float = 1.0, alpha: float = 2.2) -> float:
    dist = max(float(np.hypot(jammer.x - src_x, jammer.y - src_y)), 1.0)
    return tx_power / dist ** alpha


RNG = np.random.default_rng(42)

class TestHillClimbingInterface:
    def test_fields_exist_on_dataclass(self):
        j = make_jammer()
        assert hasattr(j, "measured_rssi")
        assert hasattr(j, "last_rssi")
        assert hasattr(j, "last_dx")
        assert hasattr(j, "last_dy")

    def test_initial_field_values_are_zero(self):
        j = make_jammer()
        assert j.measured_rssi == 0.0
        assert j.last_rssi == 0.0
        assert j.last_dx == 0.0
        assert j.last_dy == 0.0

    def test_jammer_moves_on_first_call(self):
        j = make_jammer(x=250.0, y=250.0)
        j.measured_rssi = 0.5
        j.move_hill_climbing(RNG, 500.0, 500.0)
        assert (j.x, j.y) != (250.0, 250.0), "Jammer should have moved"

    def test_last_rssi_updated_after_move(self):
        j = make_jammer()
        j.measured_rssi = 0.42
        j.move_hill_climbing(RNG, 500.0, 500.0)
        assert j.last_rssi == pytest.approx(0.42)

    def test_position_stays_within_bounds(self):
        j = make_jammer(x=0.0, y=0.0, speed=50.0)
        j.measured_rssi = 1.0
        for _ in range(20):
            j.move_hill_climbing(RNG, 500.0, 500.0)
        assert 0.0 <= j.x <= 500.0
        assert 0.0 <= j.y <= 500.0


class TestHillClimbingDirection:
    def test_keeps_direction_when_rssi_improves(self):
        j = make_jammer()
        j.measured_rssi = 0.1
        j.move_hill_climbing(RNG, 500.0, 500.0)
        dx_first = j.last_dx
        dy_first = j.last_dy

        j.measured_rssi = 0.5
        j.move_hill_climbing(RNG, 500.0, 500.0)
        assert j.last_dx == pytest.approx(dx_first, abs=1e-9)
        assert j.last_dy == pytest.approx(dy_first, abs=1e-9)

    def test_changes_direction_when_rssi_drops(self):
        rng = np.random.default_rng(0)
        changed = False
        for seed in range(50):
            rng_i = np.random.default_rng(seed)
            j = make_jammer()
            j.measured_rssi = 0.5
            j.move_hill_climbing(rng_i, 500.0, 500.0)
            dx_before, dy_before = j.last_dx, j.last_dy

            j.measured_rssi = 0.1      # signal dropped
            j.move_hill_climbing(rng_i, 500.0, 500.0)
            if (j.last_dx, j.last_dy) != (dx_before, dy_before):
                changed = True
                break
        assert changed, "Direction should change when RSSI drops"

    def test_direction_vector_is_unit_length(self):
        j = make_jammer()
        j.measured_rssi = 1.0
        j.move_hill_climbing(RNG, 500.0, 500.0)
        length = np.hypot(j.last_dx, j.last_dy)
        assert length == pytest.approx(1.0, abs=1e-9)


class TestHillClimbingConvergence:
    def test_jammer_moves_closer_to_rf_source_over_many_steps(self):
        src_x, src_y = 300.0, 300.0
        j = make_jammer(x=100.0, y=100.0, speed=5.0)
        initial_dist = float(np.hypot(j.x - src_x, j.y - src_y))

        rng = np.random.default_rng(7)
        for _ in range(200):
            j.measured_rssi = rssi_at(j, src_x, src_y)
            j.move_hill_climbing(rng, 500.0, 500.0)

        final_dist = float(np.hypot(j.x - src_x, j.y - src_y))
        assert final_dist < initial_dist, (
            f"Jammer should get closer to RF source: "
            f"initial_dist={initial_dist:.1f}, final_dist={final_dist:.1f}"
        )

    def test_jammer_inactive_then_active_cycle(self):
        j = make_jammer(x=200.0, y=200.0)
        j.is_active = False
        src_x, src_y = 250.0, 250.0
        rng = np.random.default_rng(99)

        for _ in range(10):
            j.measured_rssi = rssi_at(j, src_x, src_y)
            j.move_hill_climbing(rng, 500.0, 500.0)
        dist = float(np.hypot(j.x - src_x, j.y - src_y))
        assert dist < float(np.hypot(200.0 - src_x, 200.0 - src_y))

    def test_jammer_hysteresis_activation_logic(self):
        from envs.uav_backscatter_env import UAVBackscatterEnv
        
        cfg = {
            "simulation": {"area_width": 500, "area_height": 500, "max_steps": 10, "seed": 42},
            "network": {"num_uav": 1, "num_iot": 1, "num_jammer": 1, "num_rf_sources": 1},
            "uav": {"altitude": 100, "initial_energy": 1000, "coverage_radius": 180, "max_speed": 20},
            "iot": {"device_types": [1], "queue_capacity": 10, "energy_capacity": 10, "packet_arrival_prob": 0.5},
            "channel": {"tx_power_rf_source": 1.0, "primary_busy_prob": 0.5, "path_loss_exponent": 2.2, "noise_power": 1e-9, "sinr_threshold": 1.0},
            "jammer": {
                "enabled": True, "power": 1.0, "speed": 10.0, "radius": 200.0, "mobility": "static",
                "initial_energy": 10.0, "energy_capacity": 50.0, "energy_threshold": 5.0,
                "resume_threshold": 30.0, "harvest_efficiency": 0.5, "jam_energy_cost": 2.0
            },
            "reward": {},
        }
        
        env = UAVBackscatterEnv(cfg)
        env.reset()
        j = env.jammers[0]
        assert j.is_active is True
        env.step([0])
        assert j.is_active is True
        env.step([0])
        assert j.is_active is True
        env.step([0])
        assert j.is_active is False
        
        j.energy = 15.0
        env._update_jammers()
        assert j.is_active is False, "Should remain inactive while energy (15) < resume_threshold (30)"
        
        # Let's set energy to 31.0 (>= resume 30.0) and trigger env update.
        # It must now turn back ACTIVE.
        j.energy = 31.0
        env._update_jammers()
        assert j.is_active is True, "Should resume activity once energy (31) >= resume_threshold (30)"

