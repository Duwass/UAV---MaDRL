from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.ctde.train_loop import DEFAULT_CONFIG_PATH, train_ctde_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal CTDE 3D smoke training loop.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to CTDE smoke YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Force tiny smoke settings.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--save-dir", default=None, help="Optional user-provided directory for summary JSON.")
    args = parser.parse_args()

    config = args.config
    if args.smoke or args.seed is not None:
        import yaml

        path = Path(config)
        if not path.is_absolute():
            path = ROOT / path
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if args.smoke:
            config.setdefault("smoke", {})
            config["smoke"].update(
                {
                    "num_iterations": 2,
                    "rollout_steps": 5,
                    "batch_size": 4,
                    "eval_episodes": 1,
                    "eval_max_steps": 5,
                }
            )
        if args.seed is not None:
            config["seed"] = int(args.seed)

    summary = train_ctde_smoke(config)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "ctde_smoke_summary.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Saved CTDE smoke summary to {output_path}")


if __name__ == "__main__":
    main()
