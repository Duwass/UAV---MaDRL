from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.qmix.qmix_trainer import QMIXTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QMIX with hierarchical high-level actions.")
    parser.add_argument("--config", required=True, help="Path to QMIX YAML config.")
    parser.add_argument("--episodes", type=int, default=None, help="Override training.total_episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    args = parser.parse_args()

    trainer = QMIXTrainer(args.config, episodes_override=args.episodes, seed_override=args.seed)
    train_df, eval_df = trainer.train()
    print(f"Saved train log rows={len(train_df)} to results/csv/{trainer.output_prefix}_train_log.csv")
    print(f"Saved eval log rows={len(eval_df)} to results/csv/{trainer.output_prefix}_eval_log.csv")
    print(f"Saved checkpoints to results/checkpoints/{trainer.output_prefix}_latest.pt and *_best.pt")


if __name__ == "__main__":
    main()

