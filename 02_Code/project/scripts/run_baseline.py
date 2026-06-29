from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.backscatter_only import BackscatterOnlyPolicy
from baselines.greedy_nearest import GreedyNearestPolicy
from baselines.greedy_sinr import GreedySINRPolicy
from baselines.htt_only import HTTOnlyPolicy
from baselines.random_policy import RandomPolicy
from envs.uav_backscatter_env import UAVBackscatterEnv, load_config


POLICIES = {
    "random": RandomPolicy,
    "greedy_nearest": GreedyNearestPolicy,
    "greedy_sinr": GreedySINRPolicy,
    "htt_only": HTTOnlyPolicy,
    "backscatter_only": BackscatterOnlyPolicy,
}


def make_policy(policy_name: str, seed: int = 0):
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy '{policy_name}'. Valid policies: {sorted(POLICIES)}")
    policy_cls = POLICIES[policy_name]
    if policy_name == "random":
        return policy_cls(seed=seed)
    return policy_cls()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def run_policy(
    config: str | Path | dict[str, Any],
    policy_name: str,
    episodes: int | None = None,
    scenario_name: str | None = None,
    save_csv: bool = True,
    output_dir: str | Path | None = None,
    seed_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config)
    if scenario_name is None:
        scenario_name = Path(config).stem if isinstance(config, (str, Path)) else "custom_scenario"
    if episodes is None:
        episodes = int(cfg.get("evaluation", {}).get("num_episodes", 20))

    env = UAVBackscatterEnv(cfg)
    policy = make_policy(policy_name, seed=int(cfg.get("simulation", {}).get("seed", 42)) + seed_offset)
    episode_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    base_seed = int(cfg.get("simulation", {}).get("seed", 42))

    for episode_id in range(int(episodes)):
        obs, info = env.reset(seed=base_seed + seed_offset + episode_id)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            actions = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(actions)
            frame_metrics = dict(info.get("frame_metrics", {}))
            if frame_metrics:
                frame_metrics.update(
                    {
                        "episode_id": episode_id,
                        "policy_name": policy_name,
                        "scenario_name": scenario_name,
                    }
                )
                frame_rows.append(frame_metrics)

        episode_metrics = dict(info.get("episode_metrics", {}))
        episode_metrics.update(
            {
                "episode_id": episode_id,
                "policy_name": policy_name,
                "scenario_name": scenario_name,
            }
        )
        episode_rows.append(episode_metrics)

    env.close()
    episodes_df = pd.DataFrame(episode_rows)
    frames_df = pd.DataFrame(frame_rows)

    if save_csv and bool(cfg.get("evaluation", {}).get("save_csv", True)):
        out_dir = Path(output_dir) if output_dir is not None else ROOT / "results" / "csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{_safe_name(scenario_name)}_{_safe_name(policy_name)}"
        episodes_df.to_csv(out_dir / f"{prefix}_episodes.csv", index=False)
        frames_df.to_csv(out_dir / f"{prefix}_frames.csv", index=False)

    return episodes_df, frames_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one baseline policy on one UAV backscatter scenario.")
    parser.add_argument("--config", required=True, help="Path to scenario YAML.")
    parser.add_argument("--policy", required=True, choices=sorted(POLICIES), help="Baseline policy name.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes. Defaults to config evaluation.num_episodes.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "csv"), help="Directory for CSV output.")
    args = parser.parse_args()

    scenario_name = Path(args.config).stem
    episodes_df, _ = run_policy(
        config=args.config,
        policy_name=args.policy,
        episodes=args.episodes,
        scenario_name=scenario_name,
        save_csv=True,
        output_dir=args.output_dir,
    )
    summary = episodes_df.drop(columns=["episode_id"], errors="ignore").groupby(["scenario_name", "policy_name"]).mean(numeric_only=True)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()

