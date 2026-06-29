from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_results(input_csv: str | Path | None) -> pd.DataFrame:
    if input_csv is not None:
        return pd.read_csv(input_csv)
    combined_path = ROOT / "results" / "csv" / "all_baselines_summary.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)
    episode_files = [
        path
        for path in (ROOT / "results" / "csv").glob("*_episodes.csv")
        if path.name != "all_baselines_summary.csv"
    ]
    if not episode_files:
        raise FileNotFoundError("No episode CSV files found. Run scripts/evaluate_all_baselines.py first.")
    return pd.concat([pd.read_csv(path) for path in episode_files], ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path, filename: str, title: str, ylabel: str) -> None:
    grouped = df.groupby(["scenario_name", "policy_name"], as_index=False)[metric].mean(numeric_only=True)
    pivot = grouped.pivot(index="scenario_name", columns="policy_name", values=metric)
    ax = pivot.plot(kind="bar", figsize=(12, 5.5), width=0.82)
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.legend(title="Policy", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def maybe_plot_rewards(df: pd.DataFrame, output_dir: Path, output_prefix: str = "") -> None:
    if "episode_id" not in df.columns or "total_reward" not in df.columns:
        return
    grouped = df.groupby(["policy_name", "episode_id"], as_index=False)["total_reward"].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy_name, policy_df in grouped.groupby("policy_name"):
        ax.plot(policy_df["episode_id"], policy_df["total_reward"], marker="o", linewidth=1.5, label=policy_name)
    ax.set_title("Episode Reward Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{output_prefix}_" if output_prefix else ""
    plt.savefig(output_dir / f"{prefix}episode_reward_curves.png", dpi=160)
    plt.close(fig)


def maybe_plot_ranking_heatmap(output_dir: Path, output_prefix: str = "") -> None:
    prefix = f"{output_prefix}_" if output_prefix else ""
    ranking_path = ROOT / "results" / "csv" / f"{prefix}baseline_ranking_by_scenario.csv"
    if not ranking_path.exists():
        return
    ranking = pd.read_csv(ranking_path)
    pivot = ranking.pivot(index="scenario_name", columns="policy_name", values="rank")
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_title("Baseline Ranking Heatmap")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Scenario")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Rank (1 is best)")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{prefix}ranking_heatmap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline evaluation plots.")
    parser.add_argument("--input", "--input-csv", dest="input_csv", default=None, help="Episode-level CSV. Defaults to results/csv/all_baselines_summary.csv.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "figures"))
    parser.add_argument("--output-prefix", default="", help="Prefix for generated figure filenames.")
    args = parser.parse_args()

    df = load_results(args.input_csv)
    output_dir = Path(args.output_dir)
    metrics = {
        "avg_throughput_per_frame": (
            "throughput_by_policy_scenario.png",
            "Average Throughput Per Frame by Policy and Scenario",
            "Packets/frame",
        ),
        "packet_drop_rate": (
            "drop_rate_by_policy_scenario.png",
            "Packet Drop Rate by Policy and Scenario",
            "Drop rate",
        ),
        "energy_efficiency": (
            "energy_efficiency_by_policy_scenario.png",
            "Energy Efficiency by Policy and Scenario",
            "Packets per energy unit",
        ),
        "jamming_failure_rate": (
            "jamming_failure_by_policy_scenario.png",
            "Jamming Failure Rate by Policy and Scenario",
            "Jamming failure rate",
        ),
        "fairness_index": (
            "fairness_by_policy_scenario.png",
            "Jain Fairness Index by Policy and Scenario",
            "Fairness index",
        ),
        "total_reward": (
            "reward_by_policy_scenario.png",
            "Episode Reward by Policy and Scenario",
            "Total reward",
        ),
    }
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    for metric, (filename, title, ylabel) in metrics.items():
        if metric in df.columns:
            plot_metric(df, metric, output_dir, f"{prefix}{filename}", title, ylabel)
    maybe_plot_rewards(df, output_dir, args.output_prefix)
    maybe_plot_ranking_heatmap(output_dir, args.output_prefix)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
