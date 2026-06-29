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


SCENARIO = "scenario_4_backscatter_types_calibrated"
VARIANTS = [
    ("ddqn_original_tuned", "tuned_backscatter_types"),
    ("sc4_reward_scaled", "sc4_reward_scaled"),
    ("sc4_dueling_scaled", "sc4_dueling_scaled"),
    ("sc4_dueling_scaled_slow_epsilon", "sc4_dueling_scaled_slow_epsilon"),
    ("sc4_dueling_scaled_low_lr", "sc4_dueling_scaled_low_lr"),
]
BASELINE_POLICIES = ["random", "htt_only", "backscatter_only", "greedy_sinr", "greedy_nearest"]


def mean_metric(df: pd.DataFrame, metric: str) -> float:
    return float(df[metric].mean()) if metric in df.columns else 0.0


def loss_summary(prefix: str) -> tuple[float, float]:
    train_path = ROOT / "results" / "csv" / f"{prefix}_train_log.csv"
    if not train_path.exists():
        return 0.0, 0.0
    df = pd.read_csv(train_path)
    if "avg_loss" not in df.columns or df.empty:
        return 0.0, 0.0
    return float(df.head(50)["avg_loss"].mean()), float(df.tail(50)["avg_loss"].mean())


def ddqn_row(policy_name: str, prefix: str) -> dict[str, float | str] | None:
    final_path = ROOT / "results" / "csv" / f"{prefix}_final_eval.csv"
    if not final_path.exists():
        return None
    df = pd.read_csv(final_path)
    loss_first, loss_last = loss_summary(prefix)
    return {
        "scenario_name": SCENARIO,
        "policy_name": policy_name,
        "source_prefix": prefix,
        "total_reward": mean_metric(df, "total_reward"),
        "avg_throughput_per_frame": mean_metric(df, "avg_throughput_per_frame"),
        "packet_drop_rate": mean_metric(df, "packet_drop_rate"),
        "jamming_failure_rate": mean_metric(df, "jamming_failure_rate"),
        "fairness_index": mean_metric(df, "fairness_index"),
        "energy_efficiency": mean_metric(df, "energy_efficiency"),
        "backscatter_success_rate": mean_metric(df, "backscatter_success_rate"),
        "active_success_rate": mean_metric(df, "active_success_rate"),
        "mode_usage_backscatter": mean_metric(df, "mode_usage_backscatter"),
        "mode_usage_active": mean_metric(df, "mode_usage_active"),
        "loss_first_50_mean": loss_first,
        "loss_last_50_mean": loss_last,
    }


def baseline_rows(path: Path) -> list[dict[str, float | str]]:
    df = pd.read_csv(path)
    rows: list[dict[str, float | str]] = []
    scenario_df = df[(df["scenario_name"] == SCENARIO) & (df["policy_name"].isin(BASELINE_POLICIES))]
    for _, row in scenario_df.iterrows():
        rows.append(
            {
                "scenario_name": SCENARIO,
                "policy_name": str(row["policy_name"]),
                "source_prefix": "phase3_1_baseline",
                "total_reward": float(row.get("total_reward", 0.0)),
                "avg_throughput_per_frame": float(row.get("avg_throughput_per_frame", 0.0)),
                "packet_drop_rate": float(row.get("packet_drop_rate", 0.0)),
                "jamming_failure_rate": float(row.get("jamming_failure_rate", 0.0)),
                "fairness_index": float(row.get("fairness_index", 0.0)),
                "energy_efficiency": float(row.get("energy_efficiency", 0.0)),
                "backscatter_success_rate": float(row.get("backscatter_success_rate", 0.0)),
                "active_success_rate": float(row.get("active_success_rate", 0.0)),
                "mode_usage_backscatter": float(row.get("mode_usage_backscatter", 0.0)),
                "mode_usage_active": float(row.get("mode_usage_active", 0.0)),
                "loss_first_50_mean": 0.0,
                "loss_last_50_mean": 0.0,
            }
        )
    return rows


def save_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ordered = df.sort_values(metric, ascending=False)
    ax.bar(ordered["policy_name"], ordered[metric])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close(fig)


def plot_drop_jam(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(df))
    width = 0.38
    ax.bar([i - width / 2 for i in x], df["packet_drop_rate"], width=width, label="drop")
    ax.bar([i + width / 2 for i in x], df["jamming_failure_rate"], width=width, label="jam")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["policy_name"], rotation=35, ha="right")
    ax.set_title("Scenario 4 Drop and Jamming Failure")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def plot_modes(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(df))
    width = 0.38
    ax.bar([i - width / 2 for i in x], df["mode_usage_backscatter"], width=width, label="backscatter")
    ax.bar([i + width / 2 for i in x], df["mode_usage_active"], width=width, label="active")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["policy_name"], rotation=35, ha="right")
    ax.set_title("Scenario 4 Mode Usage")
    ax.set_ylabel("Mean actions per episode")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def plot_loss(df: pd.DataFrame, output: Path) -> None:
    ddqn = df[df["loss_first_50_mean"] + df["loss_last_50_mean"] > 0.0]
    if ddqn.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(ddqn))
    width = 0.38
    ax.bar([i - width / 2 for i in x], ddqn["loss_first_50_mean"], width=width, label="first 50")
    ax.bar([i + width / 2 for i in x], ddqn["loss_last_50_mean"], width=width, label="last 50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ddqn["policy_name"], rotation=35, ha="right")
    ax.set_title("Scenario 4 DDQN Loss Stability")
    ax.set_ylabel("Mean avg_loss")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Scenario 4 DDQN stabilization variants.")
    parser.add_argument(
        "--baseline-summary",
        default=str(ROOT / "results" / "csv" / "phase3_1_calibrated_baseline_summary_mean.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "csv" / "scenario4_stabilization_comparison.csv"),
    )
    args = parser.parse_args()

    rows = baseline_rows(Path(args.baseline_summary))
    for policy_name, prefix in VARIANTS:
        row = ddqn_row(policy_name, prefix)
        if row is not None:
            rows.append(row)
    result = pd.DataFrame(rows)
    result = result.sort_values("avg_throughput_per_frame", ascending=False).reset_index(drop=True)
    result["throughput_rank"] = result.index + 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    fig_dir = ROOT / "results" / "figures"
    save_bar(
        result,
        "avg_throughput_per_frame",
        "Scenario 4 Stabilization Throughput",
        "Throughput/frame",
        fig_dir / "scenario4_stabilization_throughput.png",
    )
    plot_drop_jam(result, fig_dir / "scenario4_stabilization_drop_jam.png")
    plot_modes(result, fig_dir / "scenario4_stabilization_mode_usage.png")
    plot_loss(result, fig_dir / "scenario4_stabilization_loss.png")
    print(f"Saved comparison CSV to {output}")
    print(f"Saved stabilization figures to {fig_dir}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
