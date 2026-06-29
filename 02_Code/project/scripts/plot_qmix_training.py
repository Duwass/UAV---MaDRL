from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_line(df: pd.DataFrame, x: str, ys: list[str], labels: list[str], title: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for y, label in zip(ys, labels):
        if y in df.columns:
            ax.plot(df[x], df[y], label=label)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if len(ys) > 1:
        ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def save_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str, output: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    ordered = df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered["policy_name"], ordered[metric])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QMIX training and Scenario 4 comparisons.")
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--comparison-csv", default=None)
    args = parser.parse_args()

    fig_dir = ROOT / "results" / "figures"
    train_df = pd.read_csv(args.train_log)
    eval_df = pd.read_csv(args.eval_log)
    save_line(train_df, "episode", ["train_reward"], ["train reward"], "QMIX Training Reward", "Reward", fig_dir / "qmix_training_reward.png")
    save_line(
        eval_df,
        "episode",
        ["eval_avg_throughput_per_frame"],
        ["eval throughput/frame"],
        "QMIX Eval Throughput",
        "Throughput/frame",
        fig_dir / "qmix_eval_throughput.png",
    )
    save_line(
        eval_df,
        "episode",
        ["eval_packet_drop_rate", "eval_jamming_failure_rate"],
        ["drop", "jam"],
        "QMIX Eval Drop and Jam",
        "Rate",
        fig_dir / "qmix_eval_drop_jam.png",
    )
    save_line(train_df, "episode", ["avg_loss"], ["loss"], "QMIX Loss", "MSE", fig_dir / "qmix_loss.png")
    save_line(train_df, "episode", ["epsilon"], ["epsilon"], "QMIX Epsilon", "Epsilon", fig_dir / "qmix_epsilon.png")
    save_line(
        eval_df,
        "episode",
        ["eval_mode_usage_harvest", "eval_mode_usage_backscatter", "eval_mode_usage_active"],
        ["harvest", "backscatter", "active"],
        "QMIX Eval Mode Usage",
        "Mean actions per episode",
        fig_dir / "qmix_mode_usage.png",
    )
    if args.comparison_csv:
        comp_df = pd.read_csv(args.comparison_csv)
        save_bar(
            comp_df,
            "avg_throughput_per_frame",
            "QMIX vs Baselines Throughput",
            "Throughput/frame",
            fig_dir / "qmix_vs_baselines_throughput.png",
        )
    print(f"Saved QMIX figures to {fig_dir}")


if __name__ == "__main__":
    main()

