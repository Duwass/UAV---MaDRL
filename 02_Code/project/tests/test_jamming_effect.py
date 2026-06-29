from __future__ import annotations

from types import SimpleNamespace

from envs.channel_model import compute_sinr


def test_sinr_decreases_when_jammer_power_increases():
    transmitter = SimpleNamespace(x=0.0, y=0.0, h=0.0)
    receiver = SimpleNamespace(x=100.0, y=0.0, h=0.0)
    weak_jammer = SimpleNamespace(x=50.0, y=0.0, h=0.0, power=0.1, radius=1000.0)
    strong_jammer = SimpleNamespace(x=50.0, y=0.0, h=0.0, power=10.0, radius=1000.0)
    weak_sinr = compute_sinr(0.1, transmitter, receiver, [weak_jammer], 1.0e-9, 2.2)
    strong_sinr = compute_sinr(0.1, transmitter, receiver, [strong_jammer], 1.0e-9, 2.2)
    assert strong_sinr < weak_sinr


def test_sinr_decreases_when_jammer_is_closer():
    transmitter = SimpleNamespace(x=0.0, y=0.0, h=0.0)
    receiver = SimpleNamespace(x=100.0, y=0.0, h=0.0)
    far_jammer = SimpleNamespace(x=500.0, y=0.0, h=0.0, power=1.0, radius=1000.0)
    close_jammer = SimpleNamespace(x=101.0, y=0.0, h=0.0, power=1.0, radius=1000.0)
    far_sinr = compute_sinr(0.1, transmitter, receiver, [far_jammer], 1.0e-9, 2.2)
    close_sinr = compute_sinr(0.1, transmitter, receiver, [close_jammer], 1.0e-9, 2.2)
    assert close_sinr < far_sinr

