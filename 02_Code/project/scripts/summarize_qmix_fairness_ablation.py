from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


METRICS = [
    "final_throughput_per_frame",
    "final_reward",
    "final_drop",
    "final_jam",
    "final_fairness",
    "final_energy_efficiency",
    "final_backscatter_success",
    "final_active_success",
    "final_fallback_rate",
]

BASELINES = {
    "qmix_base_throughput": 0.9604,
    "qmix_base_fairness": 0.5260,
    "qmix_base_jam": 0.2056,
    "hier_ddqn_throughput": 0.9710,
    "hier_ddqn_fairness": 0.4754,
    "hier_ddqn_jam": 0.4403,
}


def _normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if abs(max_value - min_value) < 1.0e-12:
        return pd.Series([0.5] * len(series), index=series.index)
    scaled = (series - min_value) / (max_value - min_value)
    return scaled if higher_is_better else 1.0 - scaled


def summarize_fairness(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    successful = summary_df[summary_df["status"] == "success"].copy()
    metrics = [metric for metric in METRICS if metric in successful.columns]
    mean_df = successful.groupby("config_name", as_index=False)[metrics].mean()
    std_df = successful.groupby("config_name", as_index=False)[metrics].std(ddof=0).fillna(0.0)
    counts = successful.groupby("config_name", as_index=False).size().rename(columns={"size": "num_successful_runs"})
    mean_df = mean_df.merge(counts, on="config_name", how="left")
    std_df = std_df.merge(counts, on="config_name", how="left")

    ranking = mean_df.copy()
    if not ranking.empty:
        ranking["throughput_delta_vs_qmix_base"] = ranking["final_throughput_per_frame"] - BASELINES["qmix_base_throughput"]
        ranking["fairness_delta_vs_qmix_base"] = ranking["final_fairness"] - BASELINES["qmix_base_fairness"]
        ranking["jam_delta_vs_qmix_base"] = ranking["final_jam"] - BASELINES["qmix_base_jam"]
        ranking["drop_delta_vs_qmix_base"] = ranking["final_drop"] - 0.4744
        ranking["tradeoff_score"] = (
            _normalize(ranking["final_throughput_per_frame"], True)
            + _normalize(ranking["final_fairness"], True)
            + _normalize(ranking["final_jam"], False)
            + _normalize(ranking["final_drop"], False)
        )
        ranking = ranking.sort_values(
            ["tradeoff_score", "final_fairness", "final_throughput_per_frame", "final_jam"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        ranking["rank"] = ranking.index + 1
    return mean_df, std_df, ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize QMIX fairness ablation.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "results" / "csv" / "qmix_fairness_experiment_summary.csv"),
        help="Fairness experiment summary CSV.",
    )
    args = parser.parse_args()

    summary_df = pd.read_csv(args.input)
    mean_df, std_df, ranking = summarize_fairness(summary_df)
    csv_dir = ROOT / "results" / "csv"
    mean_path = csv_dir / "qmix_fairness_ablation_mean.csv"
    std_path = csv_dir / "qmix_fairness_ablation_std.csv"
    ranking_path = csv_dir / "qmix_fairness_tradeoff_ranking.csv"
    mean_df.to_csv(mean_path, index=False)
    std_df.to_csv(std_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    print(f"Saved fairness mean to {mean_path}")
    print(f"Saved fairness std to {std_path}")
    print(f"Saved fairness ranking to {ranking_path}")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()

