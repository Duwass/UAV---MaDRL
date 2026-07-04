from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.ctde.baseline_evaluation import POLICIES, run_baseline_evaluation


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run a no-training 3D CTDE baseline evaluation.")
    parser.add_argument("--env-config", required=True, help="Path to the 3D environment YAML config.")
    parser.add_argument("--policy", required=True, choices=POLICIES, help="Baseline policy to evaluate.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--save-dir", default=None, help="Optional output directory for the baseline bundle.")
    parser.add_argument("--num-iterations", type=int, required=True, help="Number of evaluation iterations.")
    parser.add_argument("--steps-per-iteration", type=int, required=True, help="Steps per evaluation iteration.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing save directory.")
    args = parser.parse_args(argv)

    try:
        summary = run_baseline_evaluation(
            env_config_path=args.env_config,
            policy=args.policy,
            seed=args.seed,
            save_dir=args.save_dir,
            num_iterations=args.num_iterations,
            steps_per_iteration=args.steps_per_iteration,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
