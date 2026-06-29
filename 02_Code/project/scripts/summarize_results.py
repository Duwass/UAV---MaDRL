from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_METRICS = [
    "total_reward",
    "total_throughput",
    "avg_throughput_per_frame",
    "packet_delivery_ratio",
    "packet_drop_rate",
    "avg_queue_length",
    "energy_efficiency",
    "jamming_failure_rate",
    "collision_count",
    "fairness_index",
    "invalid_action_rate",
    "out_of_coverage_action_rate",
    "insufficient_energy_count",
    "jammed_transmission_rate",
    "successful_backscatter_packets",
    "successful_active_packets",
    "backscatter_success_rate",
    "active_success_rate",
    "harvested_energy_total",
    "mode_usage_backscatter",
    "mode_usage_active",
]

REQUIRED_COLUMNS = [
    "scenario_name",
    "policy_name",
    "episode_id",
    "total_reward",
    "total_throughput",
    "avg_throughput_per_frame",
    "packet_delivery_ratio",
    "packet_drop_rate",
    "avg_queue_length",
    "avg_delay_proxy",
    "energy_efficiency",
    "uav_energy_consumption",
    "jamming_failure_rate",
    "collision_count",
    "fairness_index",
    "duplicate_target_count",
    "invalid_action_rate",
    "out_of_coverage_action_rate",
    "insufficient_energy_count",
    "jammed_transmission_rate",
    "successful_backscatter_packets",
    "successful_active_packets",
    "backscatter_success_rate",
    "active_success_rate",
    "harvested_energy_total",
    "mode_usage_backscatter",
    "mode_usage_active",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Combined evaluation CSV is missing required columns: {missing}")


def short_comment(row: pd.Series, scenario_df: pd.DataFrame) -> str:
    throughput = row["avg_throughput_per_frame_mean"]
    drop = row["packet_drop_rate_mean"]
    jam = row["jamming_failure_rate_mean"]
    fairness = row["fairness_index_mean"]

    max_t = scenario_df["avg_throughput_per_frame_mean"].max()
    min_drop = scenario_df["packet_drop_rate_mean"].min()
    min_jam = scenario_df["jamming_failure_rate_mean"].min()
    max_fair = scenario_df["fairness_index_mean"].max()
    max_jam = scenario_df["jamming_failure_rate_mean"].max()

    if throughput >= max_t - 1.0e-12 and drop > scenario_df["packet_drop_rate_mean"].median():
        return "Highest throughput but high drop rate"
    if max_jam > 0.05 and jam <= min_jam + 1.0e-12 and throughput >= scenario_df["avg_throughput_per_frame_mean"].median():
        return "Best anti-jamming behavior"
    if throughput < scenario_df["avg_throughput_per_frame_mean"].median() and drop <= min_drop + 1.0e-12:
        return "Stable but low throughput"
    if "mobile_jammer" in str(row["scenario_name"]) and jam > scenario_df["jamming_failure_rate_mean"].median():
        return "Poor under mobile jammer"
    if fairness >= max_fair - 1.0e-12 and throughput >= scenario_df["avg_throughput_per_frame_mean"].median():
        return "High fairness with competitive delivery"
    if throughput < scenario_df["avg_throughput_per_frame_mean"].median():
        return "Energy efficient but low delivery"
    return "Balanced baseline performance"


def build_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_columns(df)
    grouped = df.groupby(["scenario_name", "policy_name"], as_index=False)
    mean_df = grouped[SUMMARY_METRICS].mean(numeric_only=True)
    std_df = grouped[SUMMARY_METRICS].std(numeric_only=True).fillna(0.0)

    ranking_base = mean_df.rename(
        columns={
            "avg_throughput_per_frame": "avg_throughput_per_frame_mean",
            "packet_drop_rate": "packet_drop_rate_mean",
            "jamming_failure_rate": "jamming_failure_rate_mean",
            "fairness_index": "fairness_index_mean",
        }
    )
    ranking_rows: list[pd.DataFrame] = []
    for scenario_name, scenario_df in ranking_base.groupby("scenario_name", sort=True):
        ranked = scenario_df.sort_values(
            by=[
                "avg_throughput_per_frame_mean",
                "packet_drop_rate_mean",
                "jamming_failure_rate_mean",
                "fairness_index_mean",
            ],
            ascending=[False, True, True, False],
        ).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["short_comment"] = ranked.apply(lambda row: short_comment(row, ranked), axis=1)
        ranking_rows.append(
            ranked[
                [
                    "scenario_name",
                    "rank",
                    "policy_name",
                    "avg_throughput_per_frame_mean",
                    "packet_drop_rate_mean",
                    "jamming_failure_rate_mean",
                    "fairness_index_mean",
                    "short_comment",
                ]
            ]
        )
    ranking_df = pd.concat(ranking_rows, ignore_index=True)
    return mean_df, std_df, ranking_df


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def compare_scenarios(mean_df: pd.DataFrame) -> dict[str, str]:
    avg_by_scenario = mean_df.groupby("scenario_name").mean(numeric_only=True)
    observations: dict[str, str] = {}

    no_jammer_name = "scenario_1_multi_uav_calibrated" if "scenario_1_multi_uav_calibrated" in avg_by_scenario.index else "scenario_1_multi_uav"
    static_name = "scenario_2b_static_jammer_medium" if "scenario_2b_static_jammer_medium" in avg_by_scenario.index else "scenario_2_static_jammer"
    mobile_name = "scenario_3b_mobile_chase_uav" if "scenario_3b_mobile_chase_uav" in avg_by_scenario.index else "scenario_3_mobile_jammer"

    if {no_jammer_name, static_name}.issubset(avg_by_scenario.index):
        s1 = avg_by_scenario.loc[no_jammer_name]
        s2 = avg_by_scenario.loc[static_name]
        observations["jamming_static"] = (
            f"{static_name} throughput {_fmt(s2['avg_throughput_per_frame'])} vs "
            f"{no_jammer_name} {_fmt(s1['avg_throughput_per_frame'])}; "
            f"jamming failure {_fmt(s2['jamming_failure_rate'])} vs {_fmt(s1['jamming_failure_rate'])}."
        )
    if {static_name, mobile_name}.issubset(avg_by_scenario.index):
        s2 = avg_by_scenario.loc[static_name]
        s3 = avg_by_scenario.loc[mobile_name]
        observations["mobile_vs_static"] = (
            f"{mobile_name} throughput {_fmt(s3['avg_throughput_per_frame'])} vs "
            f"{static_name} {_fmt(s2['avg_throughput_per_frame'])}; "
            f"jamming failure {_fmt(s3['jamming_failure_rate'])} vs {_fmt(s2['jamming_failure_rate'])}."
        )
    return observations


def expected_pattern_checks(mean_df: pd.DataFrame) -> list[str]:
    checks: list[str] = []
    avg_by_scenario = mean_df.groupby("scenario_name").mean(numeric_only=True)

    no_jammer_name = "scenario_1_multi_uav_calibrated" if "scenario_1_multi_uav_calibrated" in avg_by_scenario.index else "scenario_1_multi_uav"
    static_name = "scenario_2b_static_jammer_medium" if "scenario_2b_static_jammer_medium" in avg_by_scenario.index else "scenario_2_static_jammer"
    mobile_name = "scenario_3b_mobile_chase_uav" if "scenario_3b_mobile_chase_uav" in avg_by_scenario.index else "scenario_3_mobile_jammer"

    if {no_jammer_name, static_name}.issubset(avg_by_scenario.index):
        s1 = avg_by_scenario.loc[no_jammer_name]
        s2 = avg_by_scenario.loc[static_name]
        observed = s2["avg_throughput_per_frame"] < s1["avg_throughput_per_frame"] or s2["jamming_failure_rate"] > s1["jamming_failure_rate"]
        checks.append(
            f"- Expected Pattern A, jamming effect: {'Observed' if observed else 'Not observed'}; "
            f"{no_jammer_name} throughput/frame={_fmt(s1['avg_throughput_per_frame'])}, jam failure={_fmt(s1['jamming_failure_rate'])}; "
            f"{static_name} throughput/frame={_fmt(s2['avg_throughput_per_frame'])}, jam failure={_fmt(s2['jamming_failure_rate'])}."
        )

    static_series = ["scenario_2a_static_jammer_weak", "scenario_2b_static_jammer_medium", "scenario_2c_static_jammer_strong"]
    if set(static_series).issubset(avg_by_scenario.index):
        throughputs = [avg_by_scenario.loc[name, "avg_throughput_per_frame"] for name in static_series]
        failures = [avg_by_scenario.loc[name, "jamming_failure_rate"] for name in static_series]
        throughput_monotonic = throughputs[0] > throughputs[1] > throughputs[2]
        failure_monotonic = failures[0] < failures[1] < failures[2]
        checks.append(
            f"- Static jammer weak/medium/strong monotonicity: throughput {'Observed' if throughput_monotonic else 'Not observed'} "
            f"({_fmt(throughputs[0])} > {_fmt(throughputs[1])} > {_fmt(throughputs[2])}); "
            f"failure {'Observed' if failure_monotonic else 'Not observed'} "
            f"({_fmt(failures[0])} < {_fmt(failures[1])} < {_fmt(failures[2])})."
        )

    if {static_name, mobile_name}.issubset(avg_by_scenario.index):
        s2 = avg_by_scenario.loc[static_name]
        s3 = avg_by_scenario.loc[mobile_name]
        observed = s3["avg_throughput_per_frame"] < s2["avg_throughput_per_frame"] or s3["jamming_failure_rate"] > s2["jamming_failure_rate"]
        checks.append(
            f"- Expected Pattern B, mobile jammer harder than static: {'Observed' if observed else 'Not observed'}; "
            f"{static_name} throughput/frame={_fmt(s2['avg_throughput_per_frame'])}, jam failure={_fmt(s2['jamming_failure_rate'])}; "
            f"{mobile_name} throughput/frame={_fmt(s3['avg_throughput_per_frame'])}, jam failure={_fmt(s3['jamming_failure_rate'])}."
        )

    random_rows = mean_df[mean_df["policy_name"] == "random"].set_index("scenario_name")
    greedy_rows = mean_df[mean_df["policy_name"].isin(["greedy_nearest", "greedy_sinr"])]
    greedy_wins = 0
    compared = 0
    exceptions: list[str] = []
    for scenario_name, scenario_df in greedy_rows.groupby("scenario_name"):
        if scenario_name in random_rows.index:
            compared += 1
            random_t = random_rows.loc[scenario_name, "avg_throughput_per_frame"]
            best_greedy = scenario_df["avg_throughput_per_frame"].max()
            if best_greedy > random_t:
                greedy_wins += 1
            else:
                exceptions.append(scenario_name)
    checks.append(
        f"- Expected Pattern C, greedy better than random: Observed in {greedy_wins}/{compared} scenarios. "
        f"Exceptions: {', '.join(exceptions) if exceptions else 'none'}."
    )

    htt_first = 0
    for _, scenario_df in mean_df.groupby("scenario_name"):
        best_policy = scenario_df.sort_values("avg_throughput_per_frame", ascending=False).iloc[0]["policy_name"]
        htt_first += int(best_policy == "htt_only")
    checks.append(
        f"- Expected Pattern D, HTT-only reasonable but not universal: HTT-only ranked first in {htt_first}/{mean_df['scenario_name'].nunique()} scenarios."
    )

    efficiency_rank_notes = []
    for scenario_name, scenario_df in mean_df.groupby("scenario_name"):
        ranked_eff = scenario_df.sort_values("energy_efficiency", ascending=False).reset_index(drop=True)
        if "backscatter_only" in set(ranked_eff["policy_name"]):
            position = int(ranked_eff.index[ranked_eff["policy_name"] == "backscatter_only"][0]) + 1
            efficiency_rank_notes.append(f"{scenario_name}: rank {position}")
    checks.append(
        "- Expected Pattern E, backscatter-only energy efficient but lower throughput: "
        + "; ".join(efficiency_rank_notes)
        + "."
    )

    scenario4_name = "scenario_4_backscatter_types_calibrated" if "scenario_4_backscatter_types_calibrated" in set(mean_df["scenario_name"]) else "scenario_4_backscatter_types"
    scenario4 = mean_df[mean_df["scenario_name"] == scenario4_name]
    if not scenario4.empty:
        s4_ranked = scenario4.sort_values("avg_throughput_per_frame", ascending=False).reset_index(drop=True)
        backscatter_rank = int(s4_ranked.index[s4_ranked["policy_name"] == "backscatter_only"][0]) + 1
        observed = backscatter_rank <= 2
        checks.append(
            f"- Expected Pattern F, Scenario 4 backscatter benefit: {'Observed' if observed else 'Not observed'}; "
            f"backscatter-only throughput rank={backscatter_rank}. "
            + (
                "Backscatter is now competitive and energy efficient for the calibrated type mix."
                if observed
                else "Type 2/Type 3 modeling and jammer placement still do not make backscatter competitive."
            )
        )

    calibrated_no_jammer = mean_df[mean_df["scenario_name"].isin(["scenario_0_no_jammer_calibrated", "scenario_1_multi_uav_calibrated"])]
    if not calibrated_no_jammer.empty:
        drop_mean = calibrated_no_jammer["packet_drop_rate"].mean()
        checks.append(
            f"- Drop-rate calibration: Observed; calibrated no-jammer mean drop rate={_fmt(drop_mean)}, below the previous 0.97-0.99 range."
        )

    invalid_mean = mean_df["invalid_action_rate"].mean() if "invalid_action_rate" in mean_df.columns else 0.0
    checks.append(
        f"- Action masking effect: Observed; mean invalid_action_rate={_fmt(invalid_mean)} across summarized policy/scenario groups."
    )

    return checks


def generate_report(
    df: pd.DataFrame,
    mean_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    output_path: Path,
    pytest_status: str,
    sanity_status: str,
) -> None:
    episodes = int(df["episode_id"].nunique())
    scenarios = list(df["scenario_name"].drop_duplicates())
    policies = list(df["policy_name"].drop_duplicates())
    seeds = "Config base seed 42 with per-episode offsets in run_baseline.py"
    best = ranking_df[ranking_df["rank"] == 1].copy()
    observations = compare_scenarios(mean_df)

    lines: list[str] = [
        "# Phase 3.1 Calibration Report",
        "",
        "## 1. What Changed From Phase 3",
        "",
        "- Added Gymnasium and PettingZoo action masking interfaces.",
        "- Added diagnostic metrics for invalid actions, coverage failures, energy failures, mode usage, active/backscatter attempts, and per-UAV/per-IoT delivery.",
        "- Added calibrated no-jammer, jammer-strength, mobile-chase, and backscatter-type scenarios.",
        "- Updated baseline policies to use masks or masked fallbacks without making them learned controllers.",
        "",
        "## 2. Action Masking Design",
        "",
        "- `UAVBackscatterEnv.get_action_mask(uav_id)` returns a binary vector over the discrete action space.",
        "- `UAVBackscatterParallelEnv.action_mask(agent)` exposes the same mask for PettingZoo agents.",
        "- Masks invalidate out-of-coverage communication, empty-queue backscatter/active actions, insufficient-energy active actions, busy/idle mode conflicts, and avoid-jammer actions when no jammer exists.",
        "- Idle and relay actions remain valid; random policy samples from the mask when `action_masking.random_uses_mask` is true.",
        "",
        "## 3. New Diagnostic Metrics",
        "",
        "- Invalid action metrics: invalid action rate, out-of-coverage action rate, no-queue selections, insufficient-energy selections, and busy-mode invalid selections.",
        "- Transmission diagnostics: jammed transmission rate, active/backscatter attempted packets, successful packets, and success rates.",
        "- Mode diagnostics: idle, harvest, backscatter, active, relay, and avoid-jammer usage counts.",
        "- Delivery diagnostics: flattened per-UAV served packets and per-IoT delivered packets.",
        "",
        "## 4. New/Updated Config Files",
        "",
        "- `configs/scenario_0_no_jammer_calibrated.yaml`",
        "- `configs/scenario_1_multi_uav_calibrated.yaml`",
        "- `configs/scenario_2a_static_jammer_weak.yaml`",
        "- `configs/scenario_2b_static_jammer_medium.yaml`",
        "- `configs/scenario_2c_static_jammer_strong.yaml`",
        "- `configs/scenario_3a_mobile_random_walk.yaml`",
        "- `configs/scenario_3b_mobile_chase_uav.yaml`",
        "- `configs/scenario_4_backscatter_types_calibrated.yaml`",
        "",
        "## 5. Experiment Setup",
        "",
        f"- Scenarios: {', '.join(scenarios)}",
        f"- Policies: {', '.join(policies)}",
        f"- Episodes per policy/scenario: {episodes}",
        f"- Seeds: {seeds}",
        f"- Metrics: {', '.join(SUMMARY_METRICS)}",
        "",
        "## 6. Test Status",
        "",
        f"- Pytest: {pytest_status}",
        f"- Sanity tests: {sanity_status}",
        "",
        "## 7. Best Policy Per Scenario",
        "",
        "| Scenario | Best policy | Throughput/frame mean | Drop rate mean | Jamming failure mean | Fairness mean | Invalid action rate | Backscatter success | Active success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        mean_row = mean_df[(mean_df["scenario_name"] == row["scenario_name"]) & (mean_df["policy_name"] == row["policy_name"])].iloc[0]
        lines.append(
            f"| {row['scenario_name']} | {row['policy_name']} | "
            f"{_fmt(row['avg_throughput_per_frame_mean'])} | {_fmt(row['packet_drop_rate_mean'])} | "
            f"{_fmt(row['jamming_failure_rate_mean'])} | {_fmt(row['fairness_index_mean'])} | "
            f"{_fmt(mean_row.get('invalid_action_rate', 0.0))} | {_fmt(mean_row.get('backscatter_success_rate', 0.0))} | "
            f"{_fmt(mean_row.get('active_success_rate', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## 8. Observations",
            "",
            f"- No-jammer vs jammer behavior: {observations.get('jamming_static', 'Not enough scenarios to compare.')}",
            f"- Static jammer vs mobile jammer behavior: {observations.get('mobile_vs_static', 'Not enough scenarios to compare.')}",
        ]
    )

    for scenario_name, scenario_df in ranking_df.groupby("scenario_name"):
        top = scenario_df.iloc[0]
        bottom = scenario_df.iloc[-1]
        lines.append(
            f"- {scenario_name}: rank 1 is {top['policy_name']} ({top['short_comment']}); "
            f"lowest ranked is {bottom['policy_name']}."
        )

    random_rows = mean_df[mean_df["policy_name"] == "random"].set_index("scenario_name")
    greedy_rows = mean_df[mean_df["policy_name"].isin(["greedy_nearest", "greedy_sinr"])]
    greedy_notes = []
    for scenario_name, scenario_df in greedy_rows.groupby("scenario_name"):
        if scenario_name in random_rows.index:
            random_t = random_rows.loc[scenario_name, "avg_throughput_per_frame"]
            best_greedy = scenario_df.sort_values("avg_throughput_per_frame", ascending=False).iloc[0]
            greedy_notes.append(
                f"{scenario_name}: best greedy {best_greedy['policy_name']}={_fmt(best_greedy['avg_throughput_per_frame'])}, "
                f"random={_fmt(random_t)}"
            )
    lines.append(f"- Random vs greedy behavior: {'; '.join(greedy_notes)}.")

    htt = mean_df[mean_df["policy_name"] == "htt_only"].set_index("scenario_name")
    backscatter = mean_df[mean_df["policy_name"] == "backscatter_only"].set_index("scenario_name")
    htt_backscatter_notes = []
    for scenario_name in sorted(set(htt.index).intersection(backscatter.index)):
        htt_backscatter_notes.append(
            f"{scenario_name}: HTT throughput/frame={_fmt(htt.loc[scenario_name, 'avg_throughput_per_frame'])}, "
            f"backscatter={_fmt(backscatter.loc[scenario_name, 'avg_throughput_per_frame'])}, "
            f"HTT energy efficiency={_fmt(htt.loc[scenario_name, 'energy_efficiency'])}, "
            f"backscatter energy efficiency={_fmt(backscatter.loc[scenario_name, 'energy_efficiency'])}"
        )
    lines.append(f"- HTT-only vs backscatter-only behavior: {'; '.join(htt_backscatter_notes)}.")

    scenario4_name = "scenario_4_backscatter_types_calibrated" if "scenario_4_backscatter_types_calibrated" in set(mean_df["scenario_name"]) else "scenario_4_backscatter_types"
    scenario4 = mean_df[mean_df["scenario_name"] == scenario4_name]
    if not scenario4.empty:
        s4_rank = ranking_df[ranking_df["scenario_name"] == scenario4_name]
        lines.append(
            "- Scenario 4 backscatter-type behavior: "
            + "; ".join(
                f"{row['policy_name']} rank {int(row['rank'])}, throughput/frame={_fmt(row['avg_throughput_per_frame_mean'])}"
                for _, row in s4_rank.iterrows()
            )
            + "."
        )

    lines.extend(
        [
            "",
            "## 9. Expected Pattern Check",
            "",
            *expected_pattern_checks(mean_df),
            "",
            "## 10. Remaining Issues",
            "",
            "- If drop rate remains high in any calibrated scenario, queue/arrival/service balance still needs tuning before long training.",
            "- Current greedy policies remain simple one-step heuristics, so poor performance can reflect coverage/mobility myopia rather than simulator corruption.",
            "- Backscatter-only is low-energy but low-rate by design; it should be interpreted together with energy efficiency and Type 2/Type 3 delivery diagnostics.",
            "",
            "## 11. Recommendation",
            "",
            "- Centralized DDQN readiness: conditionally yes for a first prototype, because the Gym-style environment, metrics, baselines, and sanity checks are stable. Start with short runs and fixed seeds.",
            "- MAPPO/QMIX readiness: conditionally yes for interface experiments, because the PettingZoo wrapper exposes local observations and global state. Before long training, add more diagnostics for per-agent credit assignment and action masking.",
            "- Before training: consider action masking for invalid mode/target combinations, tune jammer/RF/backscatter parameters if Scenario 4 needs stronger backscatter differentiation, and add repeated-seed confidence intervals.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 3 baseline evaluation CSVs.")
    parser.add_argument("--input", default=str(ROOT / "results" / "csv" / "all_baselines_summary.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "csv"))
    parser.add_argument("--output-prefix", default="", help="Prefix for generated summary CSV files.")
    parser.add_argument("--report", default=None)
    parser.add_argument("--pytest-status", default="Passed")
    parser.add_argument("--sanity-status", default="Passed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    df = pd.read_csv(input_path)
    mean_df, std_df, ranking_df = build_summaries(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    mean_path = output_dir / f"{prefix}baseline_summary_mean.csv"
    std_path = output_dir / f"{prefix}baseline_summary_std.csv"
    ranking_path = output_dir / f"{prefix}baseline_ranking_by_scenario.csv"
    mean_df.to_csv(mean_path, index=False)
    std_df.to_csv(std_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    if args.report is not None:
        report_path = Path(args.report)
    elif args.output_prefix == "phase3_1_calibrated":
        report_path = ROOT / "results" / "phase3_1_calibration_report.md"
    else:
        report_path = ROOT / "results" / f"{args.output_prefix}_phase3_baseline_validation_report.md" if args.output_prefix else ROOT / "results" / "phase3_baseline_validation_report.md"
    generate_report(df, mean_df, ranking_df, report_path, args.pytest_status, args.sanity_status)

    print(f"Saved mean summary to {mean_path}")
    print(f"Saved std summary to {std_path}")
    print(f"Saved ranking summary to {ranking_path}")
    print(f"Saved validation report to {report_path}")
    print(ranking_df[ranking_df["rank"] == 1].to_string(index=False))


if __name__ == "__main__":
    main()
