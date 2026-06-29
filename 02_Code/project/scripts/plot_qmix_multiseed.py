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


def bar_mean_std(mean_df: pd.DataFrame, std_df: pd.DataFrame, metric: str, title: str, ylabel: str, output: Path) -> None:
    if metric not in mean_df.columns:
        return
    labels = mean_df["config_name"].tolist()
    means = mean_df[metric].to_numpy()
    std_lookup = std_df.set_index("config_name")[metric].to_dict() if metric in std_df.columns else {}
    stds = [float(std_lookup.get(label, 0.0)) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def ranking_plot(ranking_df: pd.DataFrame, output: Path) -> None:
    if ranking_df.empty or "final_throughput_per_frame" not in ranking_df.columns:
        return
    df = ranking_df.sort_values("rank")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["config_name"], df["final_throughput_per_frame"])
    ax.axhline(0.9710, color="tab:orange", linestyle="--", label="hierarchical DDQN")
    ax.axhline(0.8522, color="tab:green", linestyle="--", label="backscatter_only")
    ax.set_title("QMIX Multiseed Config Ranking")
    ax.set_ylabel("Throughput/frame mean")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def loss_first_last(summary_df: pd.DataFrame, output: Path) -> None:
    if summary_df.empty:
        return
    grouped = summary_df[summary_df["status"] == "success"].groupby("config_name", as_index=False)[
        ["first50_loss_mean", "last50_loss_mean"]
    ].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(grouped))
    width = 0.35
    ax.bar([pos - width / 2 for pos in x], grouped["first50_loss_mean"], width=width, label="first50")
    ax.bar([pos + width / 2 for pos in x], grouped["last50_loss_mean"], width=width, label="last50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["config_name"], rotation=25)
    ax.set_title("QMIX Loss First vs Last 50 Updates")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QMIX multiseed summary.")
    parser.add_argument("--summary", default=str(ROOT / "results" / "csv" / "qmix_experiment_summary.csv"))
    parser.add_argument("--mean", default=str(ROOT / "results" / "csv" / "qmix_multiseed_mean.csv"))
    parser.add_argument("--std", default=str(ROOT / "results" / "csv" / "qmix_multiseed_std.csv"))
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary)
    mean_df = pd.read_csv(args.mean)
    std_df = pd.read_csv(args.std)
    ranking_path = ROOT / "results" / "csv" / "qmix_multiseed_ranking.csv"
    ranking_df = pd.read_csv(ranking_path) if ranking_path.exists() else mean_df
    fig_dir = ROOT / "results" / "figures"
    bar_mean_std(
        mean_df,
        std_df,
        "final_throughput_per_frame",
        "QMIX Multiseed Throughput",
        "Throughput/frame",
        fig_dir / "qmix_multiseed_throughput_mean_std.png",
    )
    bar_mean_std(mean_df, std_df, "final_jam", "QMIX Multiseed Jam", "Jamming failure", fig_dir / "qmix_multiseed_jam_mean_std.png")
    bar_mean_std(mean_df, std_df, "final_fairness", "QMIX Multiseed Fairness", "Fairness", fig_dir / "qmix_multiseed_fairness_mean_std.png")
    bar_mean_std(mean_df, std_df, "final_drop", "QMIX Multiseed Drop", "Drop rate", fig_dir / "qmix_multiseed_drop_mean_std.png")
    ranking_plot(ranking_df, fig_dir / "qmix_multiseed_config_ranking.png")
    loss_first_last(summary_df, fig_dir / "qmix_multiseed_loss_first_last.png")
    print(f"Saved QMIX multiseed figures to {fig_dir}")


if __name__ == "__main__":
    main()

