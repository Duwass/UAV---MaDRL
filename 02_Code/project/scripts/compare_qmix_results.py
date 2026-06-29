from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


SCENARIO = "scenario_4_backscatter_types_calibrated"
BASELINES = ["random", "htt_only", "greedy_sinr", "backscatter_only", "greedy_nearest"]
LEARNED_FINALS = [
    ("flat_ddqn_tuned", "tuned_backscatter_types"),
    ("hierarchical_ddqn", "hier_sc4_basic"),
    ("qmix", "qmix_hier_sc4_backscatter_types"),
]
METRICS = [
    "total_reward",
    "avg_throughput_per_frame",
    "packet_drop_rate",
    "jamming_failure_rate",
    "fairness_index",
    "energy_efficiency",
    "backscatter_success_rate",
    "active_success_rate",
    "mode_usage_harvest",
    "mode_usage_backscatter",
    "mode_usage_active",
    "fallback_rate",
]


def mean_metric(df: pd.DataFrame, metric: str) -> float:
    return float(df[metric].mean()) if metric in df.columns and not df.empty else 0.0


def final_eval_row(policy_name: str, prefix: str) -> dict[str, float | str] | None:
    path = ROOT / "results" / "csv" / f"{prefix}_final_eval.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row: dict[str, float | str] = {
        "scenario_name": SCENARIO,
        "policy_name": policy_name,
        "source_prefix": prefix,
    }
    for metric in METRICS:
        row[metric] = mean_metric(df, metric)
    return row


def baseline_rows(path: Path) -> list[dict[str, float | str]]:
    df = pd.read_csv(path)
    rows: list[dict[str, float | str]] = []
    scenario_df = df[(df["scenario_name"] == SCENARIO) & (df["policy_name"].isin(BASELINES))]
    for _, item in scenario_df.iterrows():
        row: dict[str, float | str] = {
            "scenario_name": SCENARIO,
            "policy_name": str(item["policy_name"]),
            "source_prefix": "phase3_1_baseline",
        }
        for metric in METRICS:
            row[metric] = float(item.get(metric, 0.0))
        rows.append(row)
    return rows


def main() -> None:
    baseline_path = ROOT / "results" / "csv" / "phase3_1_calibrated_baseline_summary_mean.csv"
    rows = baseline_rows(baseline_path)
    for policy, prefix in LEARNED_FINALS:
        row = final_eval_row(policy, prefix)
        if row is not None:
            rows.append(row)
    result = pd.DataFrame(rows).sort_values("avg_throughput_per_frame", ascending=False).reset_index(drop=True)
    result["throughput_rank"] = result.index + 1
    output = ROOT / "results" / "csv" / "qmix_scenario4_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved comparison CSV to {output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

