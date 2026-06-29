from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


NUMERIC_METRICS = [
    "final_throughput_per_frame",
    "final_reward",
    "final_drop",
    "final_jam",
    "final_fairness",
    "final_energy_efficiency",
    "final_backscatter_success",
    "final_active_success",
    "final_fallback_rate",
    "best_eval_throughput_per_frame",
    "last50_loss_mean",
]

BASELINE_THROUGHPUTS = {
    "hierarchical_ddqn": 0.9710,
    "greedy_nearest": 0.8977,
    "backscatter_only": 0.8522,
    "random": 0.1075,
}


def summarize_multiseed(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    successful = summary_df[summary_df["status"] == "success"].copy()
    available_metrics = [metric for metric in NUMERIC_METRICS if metric in successful.columns]
    mean_df = successful.groupby("config_name", as_index=False)[available_metrics].mean()
    std_df = successful.groupby("config_name", as_index=False)[available_metrics].std(ddof=0).fillna(0.0)
    count_df = successful.groupby("config_name", as_index=False).size().rename(columns={"size": "num_successful_runs"})
    mean_df = mean_df.merge(count_df, on="config_name", how="left")
    std_df = std_df.merge(count_df, on="config_name", how="left")

    ranking = mean_df.copy()
    if "final_throughput_per_frame" in ranking.columns:
        throughput_std = std_df[["config_name", "final_throughput_per_frame"]].rename(
            columns={"final_throughput_per_frame": "final_throughput_per_frame_std"}
        )
        ranking = ranking.merge(throughput_std, on="config_name", how="left")
        for name, value in BASELINE_THROUGHPUTS.items():
            ranking[f"throughput_vs_{name}"] = ranking["final_throughput_per_frame"] - value
        ranking = ranking.sort_values(
            ["final_throughput_per_frame", "final_jam", "final_fairness", "final_throughput_per_frame_std"],
            ascending=[False, True, False, True],
        ).reset_index(drop=True)
        ranking["rank"] = ranking.index + 1
    return mean_df, std_df, ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize QMIX multi-seed experiment results.")
    parser.add_argument("--input", default=str(ROOT / "results" / "csv" / "qmix_experiment_summary.csv"))
    args = parser.parse_args()

    summary_df = pd.read_csv(args.input)
    mean_df, std_df, ranking = summarize_multiseed(summary_df)
    csv_dir = ROOT / "results" / "csv"
    mean_path = csv_dir / "qmix_multiseed_mean.csv"
    std_path = csv_dir / "qmix_multiseed_std.csv"
    ranking_path = csv_dir / "qmix_multiseed_ranking.csv"
    mean_df.to_csv(mean_path, index=False)
    std_df.to_csv(std_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    print(f"Saved mean summary to {mean_path}")
    print(f"Saved std summary to {std_path}")
    print(f"Saved ranking to {ranking_path}")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()

