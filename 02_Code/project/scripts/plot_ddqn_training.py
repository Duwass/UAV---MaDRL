from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, output: Path) -> None:
    if x not in df.columns or y not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df[x], df[y], marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close(fig)


def plot_modes(train_df: pd.DataFrame, output: Path) -> None:
    mode_cols = [col for col in ["mode_usage_harvest", "mode_usage_backscatter", "mode_usage_active"] if col in train_df.columns]
    if not mode_cols:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col in mode_cols:
        ax.plot(train_df["episode"], train_df[col], label=col.replace("mode_usage_", ""))
    ax.set_title("DDQN Mode Usage Over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mode count per episode")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close(fig)


def maybe_plot_comparison(comparison_csv: str | None, output_dir: Path) -> None:
    if not comparison_csv:
        return
    df = pd.read_csv(comparison_csv)
    if "avg_throughput_per_frame" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(df["policy_name"], df["avg_throughput_per_frame"])
    ax.set_title("DDQN vs Baselines Throughput")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Avg throughput/frame")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{Path(comparison_csv).stem}_throughput.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DDQN training and evaluation curves.")
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--comparison-csv", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_log)
    eval_df = pd.read_csv(args.eval_log)
    output_dir = Path(args.output_dir)
    prefix = Path(args.train_log).stem.replace("_train_log", "")

    plot_line(train_df, "episode", "train_reward", "DDQN Training Reward", "Train reward", output_dir / f"{prefix}_training_reward.png")
    plot_line(
        eval_df,
        "episode",
        "eval_avg_throughput_per_frame",
        "DDQN Eval Throughput Per Frame",
        "Packets/frame",
        output_dir / f"{prefix}_eval_throughput.png",
    )
    plot_line(eval_df, "episode", "eval_packet_drop_rate", "DDQN Eval Drop Rate", "Drop rate", output_dir / f"{prefix}_eval_drop_rate.png")
    plot_line(train_df, "episode", "avg_loss", "DDQN Loss", "Average loss", output_dir / f"{prefix}_loss.png")
    plot_line(train_df, "episode", "epsilon", "DDQN Epsilon Schedule", "Epsilon", output_dir / f"{prefix}_epsilon.png")
    plot_modes(train_df, output_dir / f"{prefix}_mode_usage.png")
    maybe_plot_comparison(args.comparison_csv, output_dir)
    print(f"Saved DDQN figures to {output_dir}")


if __name__ == "__main__":
    main()

