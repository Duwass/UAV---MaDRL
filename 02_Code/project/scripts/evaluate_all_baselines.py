from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_baseline import POLICIES, run_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all baseline policies on all scenario configs.")
    parser.add_argument("--config-dir", default=str(ROOT / "configs"), help="Directory containing scenario YAML files.")
    parser.add_argument("--configs", nargs="*", default=None, help="Explicit scenario YAML paths. Overrides --config-dir.")
    parser.add_argument("--policies", nargs="*", default=None, choices=sorted(POLICIES), help="Policies to evaluate. Defaults to all.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes per policy/scenario.")
    parser.add_argument("--output", default=str(ROOT / "results" / "csv" / "all_baselines_summary.csv"))
    parser.add_argument("--output-prefix", default="", help="Prefix for combined CSV, e.g. phase3_1_calibrated.")
    args = parser.parse_args()

    config_paths = [Path(path) for path in args.configs] if args.configs else sorted(Path(args.config_dir).glob("scenario_*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No scenario_*.yaml files found in {args.config_dir}")
    policy_names = args.policies if args.policies else list(POLICIES)

    all_episode_frames: list[pd.DataFrame] = []
    for config_path in config_paths:
        scenario_name = config_path.stem
        for policy_name in policy_names:
            print(f"Running {policy_name} on {scenario_name}")
            episodes_df, _ = run_policy(
                config=config_path,
                policy_name=policy_name,
                episodes=args.episodes,
                scenario_name=scenario_name,
                save_csv=True,
            )
            all_episode_frames.append(episodes_df)

    combined = pd.concat(all_episode_frames, ignore_index=True)
    if args.output_prefix:
        output_path = ROOT / "results" / "csv" / f"{args.output_prefix}_all_baselines_summary.csv"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    grouped = combined.groupby(["scenario_name", "policy_name"]).mean(numeric_only=True)
    print(grouped[["total_throughput", "packet_drop_rate", "energy_efficiency", "jamming_failure_rate", "fairness_index"]].round(4))
    print(f"Saved combined CSV to {output_path}")


if __name__ == "__main__":
    main()
