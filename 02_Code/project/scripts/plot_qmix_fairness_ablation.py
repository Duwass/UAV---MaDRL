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


def scatter_tradeoff(mean_df: pd.DataFrame, x: str, y: str, output: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_df[x], mean_df[y], s=70)
    for _, row in mean_df.iterrows():
        ax.annotate(str(row["config_name"]), (row[x], row[y]), xytext=(5, 5), textcoords="offset points")
    if x == "final_throughput_per_frame":
        ax.axvline(0.9604, color="tab:gray", linestyle="--", label="QMIX base throughput")
    if y == "final_fairness":
        ax.axhline(0.5260, color="tab:orange", linestyle="--", label="QMIX base fairness")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def grouped_metric_bars(mean_df: pd.DataFrame, std_df: pd.DataFrame, output: Path) -> None:
    metrics = ["final_throughput_per_frame", "final_fairness", "final_jam", "final_drop"]
    labels = ["throughput", "fairness", "jam", "drop"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    std_lookup = std_df.set_index("config_name")
    for ax, metric, label in zip(axes.ravel(), metrics, labels):
        stds = [float(std_lookup.loc[name, metric]) if name in std_lookup.index else 0.0 for name in mean_df["config_name"]]
        ax.bar(mean_df["config_name"], mean_df[metric], yerr=stds, capsize=4)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def no_balance_comparison(mean_df: pd.DataFrame, output: Path) -> None:
    subset = mean_df[mean_df["config_name"].isin(["qmix_sc4_fair_w1", "qmix_sc4_no_balance_action"])].copy()
    if subset.empty:
        subset = mean_df.copy()
    metrics = ["final_throughput_per_frame", "final_fairness", "final_jam"]
    labels = ["throughput", "fairness", "jam"]
    x = range(len(subset))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (idx - 1) * width
        ax.bar([pos + offset for pos in x], subset[metric], width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(subset["config_name"], rotation=25)
    ax.set_title("QMIX Balance Action Ablation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QMIX fairness ablation.")
    parser.add_argument("--mean", default=str(ROOT / "results" / "csv" / "qmix_fairness_ablation_mean.csv"))
    parser.add_argument("--std", default=str(ROOT / "results" / "csv" / "qmix_fairness_ablation_std.csv"))
    args = parser.parse_args()

    mean_df = pd.read_csv(args.mean)
    std_df = pd.read_csv(args.std)
    fig_dir = ROOT / "results" / "figures"
    scatter_tradeoff(
        mean_df,
        "final_throughput_per_frame",
        "final_fairness",
        fig_dir / "qmix_fairness_tradeoff_throughput_fairness.png",
        "QMIX Fairness Tradeoff: Throughput vs Fairness",
        "Throughput/frame",
        "Fairness",
    )
    scatter_tradeoff(
        mean_df,
        "final_jam",
        "final_fairness",
        fig_dir / "qmix_fairness_tradeoff_jam_fairness.png",
        "QMIX Fairness Tradeoff: Jam vs Fairness",
        "Jamming failure",
        "Fairness",
    )
    grouped_metric_bars(mean_df, std_df, fig_dir / "qmix_fairness_ablation_bars.png")
    no_balance_comparison(mean_df, fig_dir / "qmix_fairness_no_balance_comparison.png")
    print(f"Saved QMIX fairness ablation figures to {fig_dir}")


if __name__ == "__main__":
    main()

