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
BASELINES = ["random", "htt_only", "greedy_sinr", "backscatter_only", "greedy_nearest"]
DDQN_FINALS = [
    ("flat_ddqn_tuned", "tuned_backscatter_types"),
    ("flat_ddqn_stabilized_low_lr", "sc4_dueling_scaled_low_lr"),
    ("hier_sc4_basic", "hier_sc4_basic"),
    ("hier_sc4_slow_epsilon", "hier_sc4_slow_epsilon"),
]


def mean_metric(df: pd.DataFrame, metric: str) -> float:
    return float(df[metric].mean()) if metric in df.columns else 0.0


def final_eval_row(policy_name: str, prefix: str) -> dict[str, float | str] | None:
    path = ROOT / "results" / "csv" / f"{prefix}_final_eval.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
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
        "mode_usage_harvest": mean_metric(df, "mode_usage_harvest"),
        "mode_usage_backscatter": mean_metric(df, "mode_usage_backscatter"),
        "mode_usage_active": mean_metric(df, "mode_usage_active"),
        "mode_usage_idle": mean_metric(df, "mode_usage_idle"),
        "fallback_rate": mean_metric(df, "fallback_rate"),
    }


def baseline_rows(path: Path) -> list[dict[str, float | str]]:
    df = pd.read_csv(path)
    rows = []
    for _, row in df[(df["scenario_name"] == SCENARIO) & (df["policy_name"].isin(BASELINES))].iterrows():
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
                "mode_usage_harvest": float(row.get("mode_usage_harvest", 0.0)),
                "mode_usage_backscatter": float(row.get("mode_usage_backscatter", 0.0)),
                "mode_usage_active": float(row.get("mode_usage_active", 0.0)),
                "mode_usage_idle": float(row.get("mode_usage_idle", 0.0)),
                "fallback_rate": 0.0,
            }
        )
    return rows


def save_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str, output: Path, ascending: bool = False) -> None:
    ordered = df.sort_values(metric, ascending=ascending)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(ordered["policy_name"], ordered[metric])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close(fig)


def grouped_bars(df: pd.DataFrame, metrics: list[str], labels: list[str], title: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(df)))
    width = 0.8 / len(metrics)
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar([pos + offset for pos in x], df[metric], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(df["policy_name"], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hierarchical Scenario 4 DDQN with flat DDQN and baselines.")
    parser.add_argument(
        "--baseline-summary",
        default=str(ROOT / "results" / "csv" / "phase3_1_calibrated_baseline_summary_mean.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "csv" / "hierarchical_scenario4_comparison.csv"),
    )
    args = parser.parse_args()

    rows = baseline_rows(Path(args.baseline_summary))
    for policy_name, prefix in DDQN_FINALS:
        row = final_eval_row(policy_name, prefix)
        if row is not None:
            rows.append(row)
    result = pd.DataFrame(rows).sort_values("avg_throughput_per_frame", ascending=False).reset_index(drop=True)
    result["throughput_rank"] = result.index + 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    fig_dir = ROOT / "results" / "figures"
    save_bar(
        result,
        "avg_throughput_per_frame",
        "Hierarchical Scenario 4 Throughput",
        "Throughput/frame",
        fig_dir / "hierarchical_scenario4_throughput.png",
    )
    grouped_bars(
        result,
        ["packet_drop_rate", "jamming_failure_rate"],
        ["drop", "jam"],
        "Hierarchical Scenario 4 Drop and Jam",
        "Rate",
        fig_dir / "hierarchical_scenario4_drop_jam.png",
    )
    grouped_bars(
        result,
        ["mode_usage_backscatter", "mode_usage_active"],
        ["backscatter", "active"],
        "Hierarchical Scenario 4 Mode Usage",
        "Mean actions per episode",
        fig_dir / "hierarchical_scenario4_mode_usage.png",
    )
    save_bar(
        result,
        "fairness_index",
        "Hierarchical Scenario 4 Fairness",
        "Jain fairness",
        fig_dir / "hierarchical_scenario4_fairness.png",
    )
    print(f"Saved comparison CSV to {output}")
    print(f"Saved hierarchical figures to {fig_dir}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
