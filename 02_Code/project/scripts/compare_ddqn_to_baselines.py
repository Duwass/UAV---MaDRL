from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


METRICS = [
    "avg_throughput_per_frame",
    "packet_drop_rate",
    "jamming_failure_rate",
    "fairness_index",
    "energy_efficiency",
    "backscatter_success_rate",
    "active_success_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DDQN final evaluation against calibrated baselines.")
    parser.add_argument("--ddqn-final-eval", required=True)
    parser.add_argument("--baseline-summary", default=str(ROOT / "results" / "csv" / "phase3_1_calibrated_baseline_summary_mean.csv"))
    parser.add_argument("--scenario", default=None, help="Scenario name to compare. Defaults to DDQN eval scenario_name.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-prefix", default=None, help="Write results/csv/<prefix>_vs_baselines.csv when --output is not set.")
    args = parser.parse_args()

    ddqn = pd.read_csv(args.ddqn_final_eval)
    baseline = pd.read_csv(args.baseline_summary)
    scenario = args.scenario or str(ddqn["scenario_name"].iloc[0])
    baseline_scenario = baseline[baseline["scenario_name"] == scenario]
    if baseline_scenario.empty:
        raise ValueError(f"No baseline rows found for scenario={scenario}")

    rows: list[dict[str, float | str]] = []
    ddqn_mean = ddqn.mean(numeric_only=True)
    ddqn_row: dict[str, float | str] = {"scenario_name": scenario, "policy_name": "ddqn"}
    for metric in METRICS:
        ddqn_row[metric] = float(ddqn_mean.get(metric, 0.0))
    rows.append(ddqn_row)

    for _, row in baseline_scenario.iterrows():
        out: dict[str, float | str] = {"scenario_name": scenario, "policy_name": str(row["policy_name"])}
        for metric in METRICS:
            out[metric] = float(row.get(metric, 0.0))
        rows.append(out)

    result = pd.DataFrame(rows)
    random_t = float(result.loc[result["policy_name"] == "random", "avg_throughput_per_frame"].iloc[0])
    best_baseline_t = float(result[result["policy_name"] != "ddqn"]["avg_throughput_per_frame"].max())
    result["relative_to_random_throughput_pct"] = 100.0 * (result["avg_throughput_per_frame"] - random_t) / max(random_t, 1.0e-9)
    result["relative_to_best_baseline_throughput_pct"] = 100.0 * (
        result["avg_throughput_per_frame"] - best_baseline_t
    ) / max(best_baseline_t, 1.0e-9)

    if args.output:
        output = Path(args.output)
    elif args.output_prefix:
        output = ROOT / "results" / "csv" / f"{args.output_prefix}_vs_baselines.csv"
    else:
        output = ROOT / "results" / "csv" / f"ddqn_vs_baselines_{scenario}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved comparison CSV to {output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
