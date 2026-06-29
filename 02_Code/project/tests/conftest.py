from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.uav_backscatter_env import load_config


def make_test_config() -> dict:
    cfg = deepcopy(load_config(PROJECT_ROOT / "configs" / "scenario_0_no_jammer.yaml"))
    cfg["simulation"]["max_steps"] = 4
    cfg["simulation"]["episode_length"] = 4
    cfg["network"]["num_uav"] = 1
    cfg["network"]["num_iot"] = 1
    cfg["network"]["num_jammer"] = 0
    cfg["iot"]["device_types"] = [1]
    cfg["iot"]["packet_arrival_prob"] = 0.0
    cfg["iot"]["initial_energy_min"] = 5
    cfg["iot"]["initial_energy_max"] = 5
    cfg["uav"]["coverage_radius"] = 1000
    cfg["channel"]["sinr_threshold"] = 0.0
    cfg["jammer"]["enabled"] = False
    return cfg


def co_locate_first_iot(env) -> None:
    env.iot_devices[0].x = env.uavs[0].x
    env.iot_devices[0].y = env.uavs[0].y

