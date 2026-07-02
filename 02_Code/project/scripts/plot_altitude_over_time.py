from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trajectory_3d_utils import DEFAULT_3D_CONFIG, collect_rollout, load_trajectory_json, save_trajectory_json


def default_output_path() -> Path:
    return Path(tempfile.gettempdir()) / "uav_altitude_over_time.png"


def plot_altitude_over_time(data: dict, output: str | Path) -> Path:
    uav_positions = np.asarray(data.get("uav_positions", []), dtype=float)
    if uav_positions.ndim != 3 or uav_positions.shape[1] == 0:
        raise ValueError("Trajectory data must contain uav_positions with shape [time, uav, xyz].")
    steps = np.asarray(data.get("steps", list(range(uav_positions.shape[0]))), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for uav_id in range(uav_positions.shape[1]):
        ax.plot(steps, uav_positions[:, uav_id, 2], linewidth=1.8, label=f"UAV {uav_id}")
    ax.set_xlabel("step")
    ax.set_ylabel("altitude z")
    ax.set_title("UAV Altitude Over Time")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Plot UAV altitude over time from a short rollout or trajectory JSON.")
    parser.add_argument("--config", default=str(DEFAULT_3D_CONFIG), help="3D scenario YAML used when no trajectory JSON is provided.")
    parser.add_argument("--trajectory-json", default=None, help="Optional trajectory JSON from trajectory_3d_utils.")
    parser.add_argument("--save-trajectory-json", default=None, help="Optional path to save the collected rollout JSON.")
    parser.add_argument("--steps", type=int, default=30, help="Short rollout length when collecting from env.")
    parser.add_argument("--seed", type=int, default=123, help="Rollout seed.")
    parser.add_argument("--action-mode", choices=["vertical", "random", "idle"], default="vertical")
    parser.add_argument("--output", default=str(default_output_path()), help="PNG output path.")
    args = parser.parse_args(argv)

    if args.trajectory_json:
        data = load_trajectory_json(args.trajectory_json)
    else:
        data = collect_rollout(args.config, steps=args.steps, seed=args.seed, action_mode=args.action_mode)
        if args.save_trajectory_json:
            save_trajectory_json(data, args.save_trajectory_json)
    output = plot_altitude_over_time(data, args.output)
    print(f"Saved altitude plot to {output}")
    return output


if __name__ == "__main__":
    main()
