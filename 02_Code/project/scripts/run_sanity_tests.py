from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.uav_backscatter_env import load_config
from run_baseline import run_policy


def base_config() -> dict:
    cfg = load_config(ROOT / "configs" / "scenario_0_no_jammer.yaml")
    cfg["simulation"]["max_steps"] = 40
    cfg["simulation"]["episode_length"] = 40
    cfg["uav"]["coverage_radius"] = 1000
    cfg["iot"]["energy_capacity"] = 100
    cfg["iot"]["initial_energy_min"] = 50
    cfg["iot"]["initial_energy_max"] = 50
    cfg["iot"]["packet_arrival_prob"] = 0.8
    cfg["channel"]["sinr_threshold"] = 0.5
    return cfg


def run_metric(cfg: dict, policy_name: str, episodes: int = 3) -> dict:
    episodes_df, _ = run_policy(
        config=cfg,
        policy_name=policy_name,
        episodes=episodes,
        scenario_name="sanity",
        save_csv=False,
    )
    return episodes_df.mean(numeric_only=True).to_dict()


def report(name: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} - {detail}")
    return passed


def main() -> None:
    results: list[bool] = []

    cfg = base_config()
    cfg["iot"]["packet_arrival_prob"] = 0.0
    no_traffic = run_metric(cfg, "random", episodes=2)
    results.append(report("No traffic", no_traffic["total_throughput"] == 0, f"throughput={no_traffic['total_throughput']:.3f}"))

    no_jam_cfg = base_config()
    no_jam_cfg["channel"]["primary_busy_prob"] = 0.0
    strong_jam_cfg = deepcopy(no_jam_cfg)
    strong_jam_cfg["network"]["num_jammer"] = 1
    strong_jam_cfg["jammer"]["enabled"] = True
    strong_jam_cfg["jammer"]["mobility"] = "static"
    strong_jam_cfg["jammer"]["power"] = 1.0e9
    strong_jam_cfg["jammer"]["radius"] = 1000
    no_jam = run_metric(no_jam_cfg, "htt_only", episodes=3)
    strong_jam = run_metric(strong_jam_cfg, "htt_only", episodes=3)
    results.append(
        report(
            "No jammer vs strong jammer",
            no_jam["total_throughput"] > strong_jam["total_throughput"],
            f"no_jammer={no_jam['total_throughput']:.3f}, strong_jammer={strong_jam['total_throughput']:.3f}",
        )
    )

    high_energy_cfg = base_config()
    high_energy_cfg["channel"]["primary_busy_prob"] = 0.0
    low_energy_cfg = deepcopy(high_energy_cfg)
    low_energy_cfg["iot"]["energy_capacity"] = 0
    low_energy_cfg["iot"]["initial_energy_min"] = 0
    low_energy_cfg["iot"]["initial_energy_max"] = 0
    high_energy = run_metric(high_energy_cfg, "htt_only", episodes=2)
    low_energy = run_metric(low_energy_cfg, "htt_only", episodes=2)
    results.append(
        report(
            "Infinite/high energy",
            high_energy["total_throughput"] > low_energy["total_throughput"],
            f"high_energy={high_energy['total_throughput']:.3f}, zero_energy={low_energy['total_throughput']:.3f}",
        )
    )

    no_busy_cfg = base_config()
    no_busy_cfg["network"]["num_rf_sources"] = 0
    no_busy_cfg["channel"]["primary_busy_prob"] = 0.0
    no_busy = run_metric(no_busy_cfg, "backscatter_only", episodes=2)
    results.append(
        report("No RF source/no busy channel", no_busy["total_throughput"] == 0, f"throughput={no_busy['total_throughput']:.3f}")
    )

    queue_full_cfg = base_config()
    queue_full_cfg["simulation"]["max_steps"] = 8
    queue_full_cfg["iot"]["queue_capacity"] = 1
    queue_full_cfg["iot"]["packet_arrival_prob"] = 1.0
    queue_full = run_metric(queue_full_cfg, "random", episodes=2)
    results.append(report("Queue full", queue_full["packet_drop_rate"] > 0, f"drop_rate={queue_full['packet_drop_rate']:.3f}"))

    mild_jam_cfg = base_config()
    mild_jam_cfg["channel"]["primary_busy_prob"] = 0.0
    severe_jam_cfg = deepcopy(mild_jam_cfg)
    severe_jam_cfg["network"]["num_jammer"] = 1
    severe_jam_cfg["jammer"]["enabled"] = True
    severe_jam_cfg["jammer"]["power"] = 1.0e9
    severe_jam_cfg["jammer"]["radius"] = 1000
    mild_jam = run_metric(mild_jam_cfg, "htt_only", episodes=2)
    severe_jam = run_metric(severe_jam_cfg, "htt_only", episodes=2)
    results.append(
        report(
            "Very strong jammer",
            severe_jam["jamming_failure_rate"] > mild_jam["jamming_failure_rate"],
            f"mild={mild_jam['jamming_failure_rate']:.3f}, severe={severe_jam['jamming_failure_rate']:.3f}",
        )
    )

    backscatter_cfg = base_config()
    backscatter_cfg["channel"]["primary_busy_prob"] = 1.0
    active_cfg = base_config()
    active_cfg["channel"]["primary_busy_prob"] = 0.0
    backscatter = run_metric(backscatter_cfg, "backscatter_only", episodes=2)
    active = run_metric(active_cfg, "htt_only", episodes=2)
    results.append(
        report(
            "Backscatter-only low energy",
            backscatter["iot_energy_consumption"] <= active["iot_energy_consumption"],
            f"backscatter_iot_energy={backscatter['iot_energy_consumption']:.3f}, active_iot_energy={active['iot_energy_consumption']:.3f}",
        )
    )
    results.append(
        report(
            "Active-only higher throughput with energy",
            active["total_throughput"] > backscatter["total_throughput"],
            f"active={active['total_throughput']:.3f}, backscatter={backscatter['total_throughput']:.3f}",
        )
    )

    if not all(results):
        raise SystemExit(1)
    print("All sanity tests passed.")


if __name__ == "__main__":
    main()

