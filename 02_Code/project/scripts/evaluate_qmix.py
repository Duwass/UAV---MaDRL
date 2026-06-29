from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.qmix.qmix_trainer import QMIXTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained QMIX checkpoint.")
    parser.add_argument("--config", required=True, help="Path to QMIX YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--episodes", type=int, default=30, help="Number of deterministic evaluation episodes.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    trainer = QMIXTrainer(args.config)
    df = trainer.evaluate_episodes(args.episodes, checkpoint=args.checkpoint)
    output = Path(args.output) if args.output else ROOT / "results" / "csv" / f"{trainer.output_prefix}_final_eval.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    summary = df.mean(numeric_only=True)
    print(f"Saved final eval CSV to {output}")
    print(
        "summary "
        f"reward={summary.get('total_reward', 0.0):.4f} "
        f"throughput/frame={summary.get('avg_throughput_per_frame', 0.0):.4f} "
        f"drop={summary.get('packet_drop_rate', 0.0):.4f} "
        f"jam={summary.get('jamming_failure_rate', 0.0):.4f} "
        f"fairness={summary.get('fairness_index', 0.0):.4f} "
        f"fallback={summary.get('fallback_rate', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()

