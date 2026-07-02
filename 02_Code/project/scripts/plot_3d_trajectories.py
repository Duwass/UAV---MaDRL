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
    return Path(tempfile.gettempdir()) / "uav_3d_trajectory.png"


def plot_3d_trajectories(data: dict, output: str | Path) -> Path:
    uav_positions = np.asarray(data.get("uav_positions", []), dtype=float)
    iot_positions = np.asarray(data.get("iot_positions", []), dtype=float)
    jammer_steps = data.get("jammer_positions", [])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if uav_positions.ndim == 3 and uav_positions.shape[1] > 0:
        for uav_id in range(uav_positions.shape[1]):
            path = uav_positions[:, uav_id, :]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=1.8, label=f"UAV {uav_id}")
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], marker="o", s=24)
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], marker="^", s=32)

    if iot_positions.ndim == 2 and iot_positions.shape[0] > 0:
        ax.scatter(iot_positions[:, 0], iot_positions[:, 1], iot_positions[:, 2], marker="x", s=28, label="IoT")

    if jammer_steps and jammer_steps[0]:
        jammer_positions = np.asarray(jammer_steps, dtype=float)
        for jammer_id in range(jammer_positions.shape[1]):
            path = jammer_positions[:, jammer_id, :]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], linestyle="--", linewidth=1.4, label=f"Jammer {jammer_id}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D UAV Trajectories")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Plot 3D UAV trajectories from a short rollout or trajectory JSON.")
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
    output = plot_3d_trajectories(data, args.output)
    print(f"Saved 3D trajectory plot to {output}")
    return output


if __name__ == "__main__":
    main()
