"""Generate publication-ready Phase 3 result tables, figures, and summaries.

This script is intentionally read-only with respect to experiment outputs: it
loads existing CSV files and writes derived artifacts under results/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIO4 = "scenario_4_backscatter_types_calibrated"

CSV_SOURCES = {
    "baseline_mean": "results/csv/phase3_1_calibrated_baseline_summary_mean.csv",
    "baseline_std": "results/csv/phase3_1_calibrated_baseline_summary_std.csv",
    "flat_ddqn": "results/csv/tuned_backscatter_types_final_eval.csv",
    "ddqn_s0": "results/csv/ddqn_scenario_0_no_jammer_final_eval.csv",
    "tuned_no_jammer": "results/csv/tuned_no_jammer_final_eval.csv",
    "tuned_static_weak": "results/csv/tuned_static_weak_final_eval.csv",
    "hier_ddqn": "results/csv/hier_sc4_basic_final_eval.csv",
    "hier_comparison": "results/csv/hierarchical_scenario4_comparison.csv",
    "qmix_single": "results/csv/qmix_hier_sc4_backscatter_types_final_eval.csv",
    "qmix_comparison": "results/csv/qmix_scenario4_comparison.csv",
    "qmix_experiments": "results/csv/qmix_experiment_summary.csv",
    "qmix_multiseed_mean": "results/csv/qmix_multiseed_mean.csv",
    "qmix_multiseed_std": "results/csv/qmix_multiseed_std.csv",
    "qmix_multiseed_ranking": "results/csv/qmix_multiseed_ranking.csv",
    "qmix_fairness_experiments": "results/csv/qmix_fairness_experiment_summary.csv",
    "qmix_fairness_mean": "results/csv/qmix_fairness_ablation_mean.csv",
    "qmix_fairness_std": "results/csv/qmix_fairness_ablation_std.csv",
    "qmix_fairness_ranking": "results/csv/qmix_fairness_tradeoff_ranking.csv",
    "flat_stabilized_low_lr": "results/csv/sc4_dueling_scaled_low_lr_final_eval.csv",
}

METRIC_COLUMNS = {
    "reward": "total_reward",
    "throughput": "avg_throughput_per_frame",
    "drop": "packet_drop_rate",
    "jam": "jamming_failure_rate",
    "fairness": "fairness_index",
    "energy": "energy_efficiency",
    "backscatter_success": "backscatter_success_rate",
    "active_success": "active_success_rate",
    "fallback_rate": "fallback_rate",
}

SUMMARY_COLUMNS = {
    "reward": "final_reward",
    "throughput": "final_throughput_per_frame",
    "drop": "final_drop",
    "jam": "final_jam",
    "fairness": "final_fairness",
    "energy": "final_energy_efficiency",
    "backscatter_success": "final_backscatter_success",
    "active_success": "final_active_success",
    "fallback_rate": "final_fallback_rate",
}


@dataclass
class ArtifactContext:
    output_root: Path
    tables_dir: Path
    figures_dir: Path
    reports_dir: Path
    frames: dict[str, pd.DataFrame]
    missing: list[str]
    generated_tables: list[Path]
    generated_figures: list[Path]
    generated_reports: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory relative to the project root.",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    tables_dir = output_root / "publication_tables"
    figures_dir = output_root / "publication_figures"
    reports_dir = output_root / "publication_reports"
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, reports_dir


def load_sources() -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for key, rel_path in CSV_SOURCES.items():
        path = ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        try:
            frames[key] = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive report path
            missing.append(f"{rel_path} (failed to read: {exc})")
    return frames, missing


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def fmt_pm(mean_value: Any, std_value: Any, digits: int = 4) -> str:
    if mean_value is None or pd.isna(mean_value):
        return "N/A"
    if std_value is None or pd.isna(std_value):
        return fmt(mean_value, digits)
    return f"{float(mean_value):.{digits}f} +/- {float(std_value):.{digits}f}"


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    out = [line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out) + "\n"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def simple_latex_table(df: pd.DataFrame) -> str:
    alignment = "l" * len(df.columns)
    rows = [rf"\begin{{tabular}}{{{alignment}}}", r"\hline"]
    rows.append(" & ".join(latex_escape(col) for col in df.columns) + r" \\")
    rows.append(r"\hline")
    for _, row in df.iterrows():
        rows.append(" & ".join(latex_escape(value) for value in row.tolist()) + r" \\")
    rows.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(rows)


def write_table(ctx: ArtifactContext, df: pd.DataFrame, stem: str) -> None:
    csv_path = ctx.tables_dir / f"{stem}.csv"
    md_path = ctx.tables_dir / f"{stem}.md"
    tex_path = ctx.tables_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(df), encoding="utf-8")
    tex_path.write_text(simple_latex_table(df), encoding="utf-8")
    ctx.generated_tables.extend([csv_path, md_path, tex_path])


def final_eval_metrics(df: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric, column in METRIC_COLUMNS.items():
        result[metric] = float(df[column].mean()) if column in df.columns else np.nan
    return result


def baseline_metrics(ctx: ArtifactContext, policy_name: str) -> dict[str, float]:
    df = ctx.frames.get("baseline_mean")
    if df is None:
        return {metric: np.nan for metric in METRIC_COLUMNS}
    subset = df
    if "scenario_name" in subset.columns:
        subset = subset[subset["scenario_name"].astype(str) == SCENARIO4]
    subset = subset[subset["policy_name"].astype(str) == policy_name]
    if subset.empty:
        return {metric: np.nan for metric in METRIC_COLUMNS}
    row = subset.iloc[0]
    return {metric: safe_float(row.get(column, np.nan)) for metric, column in METRIC_COLUMNS.items()}


def comparison_metrics(ctx: ArtifactContext, frame_key: str, policy_name: str) -> dict[str, float]:
    df = ctx.frames.get(frame_key)
    if df is None or "policy_name" not in df.columns:
        return {metric: np.nan for metric in METRIC_COLUMNS}
    subset = df[df["policy_name"].astype(str) == policy_name]
    if subset.empty:
        return {metric: np.nan for metric in METRIC_COLUMNS}
    row = subset.iloc[0]
    return {metric: safe_float(row.get(column, np.nan)) for metric, column in METRIC_COLUMNS.items()}


def qmix_base_stats(ctx: ArtifactContext) -> tuple[dict[str, float], dict[str, float]]:
    mean_df = ctx.frames.get("qmix_multiseed_mean", pd.DataFrame())
    std_df = ctx.frames.get("qmix_multiseed_std", pd.DataFrame())

    def row_for(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        subset = df[df.get("config_name", "").astype(str) == "qmix_sc4_base"]
        if subset.empty:
            return pd.Series(dtype=float)
        return subset.iloc[0]

    mean_row = row_for(mean_df)
    std_row = row_for(std_df)
    mean = {metric: safe_float(mean_row.get(column, np.nan)) for metric, column in SUMMARY_COLUMNS.items()}
    std = {metric: safe_float(std_row.get(column, np.nan)) for metric, column in SUMMARY_COLUMNS.items()}
    return mean, std


def qmix_fairness_stats(ctx: ArtifactContext, config_name: str) -> tuple[dict[str, float], dict[str, float]]:
    mean_df = ctx.frames.get("qmix_fairness_mean", pd.DataFrame())
    std_df = ctx.frames.get("qmix_fairness_std", pd.DataFrame())

    def row_for(df: pd.DataFrame) -> pd.Series:
        if df.empty or "config_name" not in df.columns:
            return pd.Series(dtype=float)
        subset = df[df["config_name"].astype(str) == config_name]
        if subset.empty:
            return pd.Series(dtype=float)
        return subset.iloc[0]

    mean_row = row_for(mean_df)
    std_row = row_for(std_df)
    mean = {metric: safe_float(mean_row.get(column, np.nan)) for metric, column in SUMMARY_COLUMNS.items()}
    std = {metric: safe_float(std_row.get(column, np.nan)) for metric, column in SUMMARY_COLUMNS.items()}
    return mean, std


def build_publication_values(ctx: ArtifactContext) -> dict[str, dict[str, float]]:
    values = {
        "Random": baseline_metrics(ctx, "random"),
        "HTT-only": baseline_metrics(ctx, "htt_only"),
        "Backscatter-only": baseline_metrics(ctx, "backscatter_only"),
        "Greedy SINR": baseline_metrics(ctx, "greedy_sinr"),
        "Greedy nearest": baseline_metrics(ctx, "greedy_nearest"),
    }
    if "flat_ddqn" in ctx.frames:
        values["Flat DDQN"] = final_eval_metrics(ctx.frames["flat_ddqn"])
    else:
        values["Flat DDQN"] = comparison_metrics(ctx, "qmix_comparison", "flat_ddqn")

    if "hier_ddqn" in ctx.frames:
        values["Hierarchical DDQN"] = final_eval_metrics(ctx.frames["hier_ddqn"])
    else:
        values["Hierarchical DDQN"] = comparison_metrics(ctx, "qmix_comparison", "hierarchical_ddqn")

    qmix_mean, _ = qmix_base_stats(ctx)
    values["QMIX base"] = qmix_mean
    return values


def create_table1(ctx: ArtifactContext, values: dict[str, dict[str, float]]) -> None:
    qmix_mean, qmix_std = qmix_base_stats(ctx)
    methods = [
        ("Random", "Stochastic baseline"),
        ("HTT-only", "Heuristic baseline"),
        ("Backscatter-only", "Heuristic baseline"),
        ("Greedy SINR", "Heuristic baseline"),
        ("Greedy nearest", "Heuristic baseline"),
        ("Flat DDQN", "Flat DRL"),
        ("Hierarchical DDQN", "Hierarchical DRL"),
        ("QMIX base", "MaDRL, hierarchical"),
    ]
    rows = []
    for method, category in methods:
        metrics = values[method]
        use_pm = method == "QMIX base"
        rows.append(
            {
                "Method": method,
                "Type / Category": category,
                "Throughput/frame": fmt_pm(qmix_mean["throughput"], qmix_std["throughput"]) if use_pm else fmt(metrics["throughput"]),
                "Drop rate": fmt_pm(qmix_mean["drop"], qmix_std["drop"]) if use_pm else fmt(metrics["drop"]),
                "Jamming failure": fmt_pm(qmix_mean["jam"], qmix_std["jam"]) if use_pm else fmt(metrics["jam"]),
                "Fairness": fmt_pm(qmix_mean["fairness"], qmix_std["fairness"]) if use_pm else fmt(metrics["fairness"]),
                "Energy efficiency": fmt_pm(qmix_mean["energy"], qmix_std["energy"]) if use_pm else fmt(metrics["energy"]),
                "Backscatter success": fmt_pm(qmix_mean["backscatter_success"], qmix_std["backscatter_success"]) if use_pm else fmt(metrics["backscatter_success"]),
                "Active success": fmt_pm(qmix_mean["active_success"], qmix_std["active_success"]) if use_pm else fmt(metrics["active_success"]),
            }
        )
    write_table(ctx, pd.DataFrame(rows), "table1_overall_scenario4_comparison")


def create_table2(ctx: ArtifactContext, values: dict[str, dict[str, float]]) -> None:
    qmix_mean, _ = qmix_base_stats(ctx)
    stabilized = (
        final_eval_metrics(ctx.frames["flat_stabilized_low_lr"])
        if "flat_stabilized_low_lr" in ctx.frames
        else {metric: np.nan for metric in METRIC_COLUMNS}
    )
    rows = [
        {
            "Method": "Flat DDQN",
            "Action interface": "Flat movement x IoT x mode",
            "Action dimension": 864,
            "Multi-agent coordination": "Centralized-factorized DDQN",
            "Throughput/frame": fmt(values["Flat DDQN"]["throughput"]),
            "Drop rate": fmt(values["Flat DDQN"]["drop"]),
            "Jamming failure": fmt(values["Flat DDQN"]["jam"]),
            "Fairness": fmt(values["Flat DDQN"]["fairness"]),
            "Interpretation": "Large flat action space limits learning in heterogeneous Scenario 4.",
        },
        {
            "Method": "Flat DDQN + tuning",
            "Action interface": "Flat movement x IoT x mode",
            "Action dimension": 864,
            "Multi-agent coordination": "Centralized-factorized DDQN",
            "Throughput/frame": fmt(stabilized["throughput"]),
            "Drop rate": fmt(stabilized["drop"]),
            "Jamming failure": fmt(stabilized["jam"]),
            "Fairness": fmt(stabilized["fairness"]),
            "Interpretation": "Reward scaling and dueling stabilized loss but did not solve the action-interface bottleneck.",
        },
        {
            "Method": "Hierarchical DDQN",
            "Action interface": "High-level executor",
            "Action dimension": 10,
            "Multi-agent coordination": "Centralized-factorized DDQN",
            "Throughput/frame": fmt(values["Hierarchical DDQN"]["throughput"]),
            "Drop rate": fmt(values["Hierarchical DDQN"]["drop"]),
            "Jamming failure": fmt(values["Hierarchical DDQN"]["jam"]),
            "Fairness": fmt(values["Hierarchical DDQN"]["fairness"]),
            "Interpretation": "Action abstraction is the dominant improvement over flat DDQN.",
        },
        {
            "Method": "QMIX-Hierarchical",
            "Action interface": "High-level executor",
            "Action dimension": 10,
            "Multi-agent coordination": "QMIX value decomposition",
            "Throughput/frame": fmt(qmix_mean["throughput"]),
            "Drop rate": fmt(qmix_mean["drop"]),
            "Jamming failure": fmt(qmix_mean["jam"]),
            "Fairness": fmt(qmix_mean["fairness"]),
            "Interpretation": "QMIX preserves high throughput and improves jamming/fairness trade-off over hierarchical DDQN.",
        },
    ]
    write_table(ctx, pd.DataFrame(rows), "table2_algorithm_progression")


def create_table3(ctx: ArtifactContext) -> None:
    exp_df = ctx.frames.get("qmix_experiments", pd.DataFrame())
    rows = []
    if not exp_df.empty:
        subset = exp_df[exp_df["config_name"].astype(str) == "qmix_sc4_base"].copy()
        subset = subset.sort_values("seed")
        for _, row in subset.iterrows():
            rows.append(
                {
                    "Run": f"seed {int(row['seed'])}",
                    "Throughput/frame": fmt(row.get("final_throughput_per_frame")),
                    "Reward": fmt(row.get("final_reward")),
                    "Drop rate": fmt(row.get("final_drop")),
                    "Jamming failure": fmt(row.get("final_jam")),
                    "Fairness": fmt(row.get("final_fairness")),
                    "Energy efficiency": fmt(row.get("final_energy_efficiency")),
                    "Backscatter success": fmt(row.get("final_backscatter_success")),
                    "Active success": fmt(row.get("final_active_success")),
                    "Fallback rate": fmt(row.get("final_fallback_rate")),
                }
            )

    mean, std = qmix_base_stats(ctx)
    rows.extend(
        [
            {
                "Run": "mean",
                "Throughput/frame": fmt(mean["throughput"]),
                "Reward": fmt(mean["reward"]),
                "Drop rate": fmt(mean["drop"]),
                "Jamming failure": fmt(mean["jam"]),
                "Fairness": fmt(mean["fairness"]),
                "Energy efficiency": fmt(mean["energy"]),
                "Backscatter success": fmt(mean["backscatter_success"]),
                "Active success": fmt(mean["active_success"]),
                "Fallback rate": fmt(mean["fallback_rate"]),
            },
            {
                "Run": "std",
                "Throughput/frame": fmt(std["throughput"]),
                "Reward": fmt(std["reward"]),
                "Drop rate": fmt(std["drop"]),
                "Jamming failure": fmt(std["jam"]),
                "Fairness": fmt(std["fairness"]),
                "Energy efficiency": fmt(std["energy"]),
                "Backscatter success": fmt(std["backscatter_success"]),
                "Active success": fmt(std["active_success"]),
                "Fallback rate": fmt(std["fallback_rate"]),
            },
        ]
    )
    write_table(ctx, pd.DataFrame(rows), "table3_qmix_multiseed")


def create_table4(ctx: ArtifactContext) -> None:
    base_mean, base_std = qmix_base_stats(ctx)
    configs = [
        ("QMIX base", "qmix_sc4_base", base_mean, base_std, "Best overall trade-off; final MaDRL setting."),
        (
            "fair_w2",
            "qmix_sc4_fair_w2",
            *qmix_fairness_stats(ctx, "qmix_sc4_fair_w2"),
            "Improves throughput and jamming but reduces mean fairness.",
        ),
        (
            "fair_w3",
            "qmix_sc4_fair_w3",
            *qmix_fairness_stats(ctx, "qmix_sc4_fair_w3"),
            "Excessive fairness weighting degrades performance.",
        ),
        (
            "no_balance_action",
            "qmix_sc4_no_balance_action",
            *qmix_fairness_stats(ctx, "qmix_sc4_no_balance_action"),
            "Disabling balance action hurts fairness, throughput, and jamming.",
        ),
    ]
    rows = []
    for display_name, _config_name, mean, std, interpretation in configs:
        rows.append(
            {
                "Variant": display_name,
                "Throughput/frame mean +/- std": fmt_pm(mean["throughput"], std["throughput"]),
                "Drop mean +/- std": fmt_pm(mean["drop"], std["drop"]),
                "Jamming failure mean +/- std": fmt_pm(mean["jam"], std["jam"]),
                "Fairness mean +/- std": fmt_pm(mean["fairness"], std["fairness"]),
                "Interpretation": interpretation,
            }
        )
    write_table(ctx, pd.DataFrame(rows), "table4_qmix_fairness_ablation")


def create_table5(ctx: ArtifactContext) -> None:
    rows = [
        {
            "Section": "Scenario",
            "Scenario / Metric": "No-jammer calibrated",
            "UAV count": "1",
            "IoT count": "5",
            "Jammer setting": "Disabled",
            "Backscatter/active relevance": "Basic backscatter/active scheduling without jamming.",
            "Purpose": "Learnability sanity check and no-jammer reference.",
        },
        {
            "Section": "Scenario",
            "Scenario / Metric": "Static weak jammer",
            "UAV count": "2",
            "IoT count": "10",
            "Jammer setting": "One weak static jammer",
            "Backscatter/active relevance": "Anti-jamming behavior with moderate coordination.",
            "Purpose": "Intermediate transfer and robustness validation.",
        },
        {
            "Section": "Scenario",
            "Scenario / Metric": "Scenario 4 backscatter types",
            "UAV count": "2",
            "IoT count": "15",
            "Jammer setting": "One calibrated mobile/chase jammer",
            "Backscatter/active relevance": "Heterogeneous Type 1/2/3 devices with active, backscatter, and harvest modes.",
            "Purpose": "Main heterogeneous RF-powered backscatter coordination benchmark.",
        },
    ]
    metric_descriptions = [
        ("Throughput/frame", "Delivered packets per frame; higher is better."),
        ("Drop rate", "Fraction of generated/queued packets dropped; lower is better."),
        ("Jamming failure", "Transmission failures attributed to jamming; lower is better."),
        ("Fairness", "Jain-style service fairness across IoT devices; higher is better."),
        ("Energy efficiency", "Delivered utility per energy cost; higher is better."),
        ("Backscatter success", "Successful backscatter transmission rate; higher is better."),
        ("Active success", "Successful active transmission rate; higher is better."),
    ]
    for metric, purpose in metric_descriptions:
        rows.append(
            {
                "Section": "Metric",
                "Scenario / Metric": metric,
                "UAV count": "",
                "IoT count": "",
                "Jammer setting": "",
                "Backscatter/active relevance": "",
                "Purpose": purpose,
            }
        )
    write_table(ctx, pd.DataFrame(rows), "table5_simulation_scenarios_metrics")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.2),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_figure(ctx: ArtifactContext, stem: str) -> None:
    path = ctx.figures_dir / f"{stem}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    ctx.generated_figures.append(path)


def add_bar_labels(ax: plt.Axes, bars: Any, digits: int = 3) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.{digits}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def create_figures(ctx: ArtifactContext, values: dict[str, dict[str, float]]) -> None:
    configure_plot_style()

    # Figure 1
    fig1_methods = [
        "Random",
        "HTT-only",
        "Flat DDQN",
        "Greedy SINR",
        "Backscatter-only",
        "Greedy nearest",
        "Hierarchical DDQN",
        "QMIX base",
    ]
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    x = np.arange(len(fig1_methods))
    y = [values[m]["throughput"] for m in fig1_methods]
    bars = ax.bar(x, y, color=["#8c8c8c", "#7aa6c2", "#c47f6f", "#b5a86a", "#6fa77a", "#5e8ec1", "#8e78bd", "#3d6f57"])
    ax.set_xticks(x)
    ax.set_xticklabels(fig1_methods, rotation=30, ha="right")
    ax.set_ylabel("Throughput per frame")
    ax.set_title("Overall Scenario 4 Throughput Comparison")
    add_bar_labels(ax, bars)
    ax.set_ylim(0, max(y) * 1.18)
    save_figure(ctx, "fig1_overall_throughput_comparison")

    # Figure 2
    fig2_methods = ["Flat DDQN", "Hierarchical DDQN", "QMIX base", "Greedy nearest", "Backscatter-only"]
    metrics = [("Drop", "drop"), ("Jamming", "jam"), ("Fairness", "fairness")]
    x = np.arange(len(fig2_methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for idx, (label, metric) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, [values[m][metric] for m in fig2_methods], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(fig2_methods, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Drop, Jamming, and Fairness Comparison")
    ax.legend()
    save_figure(ctx, "fig2_drop_jam_fairness_comparison")

    # Figure 3
    fig3_methods = ["Flat DDQN", "Hierarchical DDQN", "QMIX base"]
    metrics = [("Throughput", "throughput"), ("Jamming", "jam"), ("Fairness", "fairness")]
    x = np.arange(len(fig3_methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for idx, (label, metric) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, [values[m][metric] for m in fig3_methods], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(fig3_methods)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Metric value")
    ax.set_title("Algorithm Progression: Flat to Hierarchical to QMIX")
    ax.legend()
    save_figure(ctx, "fig3_algorithm_progression")

    # Figure 4
    qmix_mean, qmix_std = qmix_base_stats(ctx)
    labels = ["Throughput", "Jamming", "Fairness", "Drop"]
    keys = ["throughput", "jam", "fairness", "drop"]
    means = [qmix_mean[k] for k in keys]
    stds = [qmix_std[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color="#4f7d92")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean +/- std across seeds")
    ax.set_title("QMIX Base Multi-seed Mean/Std")
    add_bar_labels(ax, bars)
    save_figure(ctx, "fig4_qmix_multiseed_mean_std")

    # Figure 5
    fair_variants = fairness_variant_values(ctx)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for name, metrics_dict in fair_variants.items():
        ax.scatter(metrics_dict["fairness"], metrics_dict["throughput"], s=85)
        ax.annotate(
            f"{name}\njam={metrics_dict['jam']:.3f}",
            (metrics_dict["fairness"], metrics_dict["throughput"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Fairness mean")
    ax.set_ylabel("Throughput/frame mean")
    ax.set_title("Fairness Ablation Trade-off")
    save_figure(ctx, "fig5_fairness_ablation_tradeoff")

    # Figure 6
    variants = list(fair_variants.keys())
    x = np.arange(len(variants))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars_metrics = [("Throughput", "throughput"), ("Fairness", "fairness"), ("Jamming", "jam"), ("Drop", "drop")]
    for idx, (label, key) in enumerate(bars_metrics):
        ax.bar(x + (idx - 1.5) * width, [fair_variants[v][key] for v in variants], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean metric value")
    ax.set_title("QMIX Fairness Ablation Grouped Metrics")
    ax.legend(ncol=2)
    save_figure(ctx, "fig6_fairness_ablation_bars")


def fairness_variant_values(ctx: ArtifactContext) -> dict[str, dict[str, float]]:
    base_mean, _ = qmix_base_stats(ctx)
    values = {"QMIX base": base_mean}
    for display, config in [
        ("fair_w2", "qmix_sc4_fair_w2"),
        ("fair_w3", "qmix_sc4_fair_w3"),
        ("no_balance", "qmix_sc4_no_balance_action"),
    ]:
        mean, _ = qmix_fairness_stats(ctx, config)
        values[display] = mean
    return values


def create_summary_report(ctx: ArtifactContext, values: dict[str, dict[str, float]]) -> None:
    qmix_mean, qmix_std = qmix_base_stats(ctx)
    fair_values = fairness_variant_values(ctx)
    report_path = ctx.reports_dir / "experimental_results_summary.md"
    text = f"""# Experimental Results Summary

## 1. Overview

This package aggregates existing Phase 3 result CSV files into publication-ready tables and figures. It does not rerun training or modify simulator/training code.

## 2. Scenario 4 Main Comparison

Scenario 4 is the main heterogeneous RF-powered ambient backscatter benchmark. The key throughput/frame values are:

- Random: {fmt(values['Random']['throughput'])}
- HTT-only: {fmt(values['HTT-only']['throughput'])}
- Backscatter-only: {fmt(values['Backscatter-only']['throughput'])}
- Greedy SINR: {fmt(values['Greedy SINR']['throughput'])}
- Greedy nearest: {fmt(values['Greedy nearest']['throughput'])}
- Flat DDQN: {fmt(values['Flat DDQN']['throughput'])}
- Hierarchical DDQN: {fmt(values['Hierarchical DDQN']['throughput'])}
- QMIX base: {fmt_pm(qmix_mean['throughput'], qmix_std['throughput'])}

## 3. Flat DDQN Limitation

Flat DDQN uses the full movement x IoT target x communication-mode action interface with 864 actions in Scenario 4. It reaches throughput/frame {fmt(values['Flat DDQN']['throughput'])}, close to HTT-only and far below the strongest heuristic baselines. This supports the conclusion that the flat action interface is a bottleneck for heterogeneous backscatter control.

## 4. Hierarchical Action Interface Effect

The hierarchical executor reduces the Scenario 4 action dimension from 864 to 10 high-level actions. Hierarchical DDQN reaches throughput/frame {fmt(values['Hierarchical DDQN']['throughput'])}, exceeding the calibrated heuristic baselines in this result set. This is the main action-abstraction result.

## 5. QMIX Multi-agent Coordination Effect

QMIX reuses the 10-action hierarchical interface and applies cooperative value decomposition across UAV agents. Across seeds 42, 43, and 44, QMIX base obtains throughput/frame {fmt_pm(qmix_mean['throughput'], qmix_std['throughput'])}, jamming failure {fmt_pm(qmix_mean['jam'], qmix_std['jam'])}, fairness {fmt_pm(qmix_mean['fairness'], qmix_std['fairness'])}, and drop rate {fmt_pm(qmix_mean['drop'], qmix_std['drop'])}. Relative to hierarchical DDQN, QMIX improves the jamming/fairness trade-off while retaining high throughput.

## 6. Multi-seed Stability

QMIX base is stable across three seeds: throughput/frame std is {fmt(qmix_std['throughput'])}. The mean throughput remains above greedy nearest ({fmt(values['Greedy nearest']['throughput'])}) and backscatter-only ({fmt(values['Backscatter-only']['throughput'])}).

## 7. Fairness Ablation

The fairness ablation shows that naive executor fairness weighting does not improve mean fairness over QMIX base:

- QMIX base fairness: {fmt(fair_values['QMIX base']['fairness'])}, throughput/frame: {fmt(fair_values['QMIX base']['throughput'])}, jam: {fmt(fair_values['QMIX base']['jam'])}
- fair_w2 fairness: {fmt(fair_values['fair_w2']['fairness'])}, throughput/frame: {fmt(fair_values['fair_w2']['throughput'])}, jam: {fmt(fair_values['fair_w2']['jam'])}
- fair_w3 fairness: {fmt(fair_values['fair_w3']['fairness'])}, throughput/frame: {fmt(fair_values['fair_w3']['throughput'])}, jam: {fmt(fair_values['fair_w3']['jam'])}
- no_balance fairness: {fmt(fair_values['no_balance']['fairness'])}, throughput/frame: {fmt(fair_values['no_balance']['throughput'])}, jam: {fmt(fair_values['no_balance']['jam'])}

Disabling BALANCE_UNDERSERVED_IOT hurts throughput, fairness, and jamming relative to QMIX base, supporting the value of the high-level balance action. QMIX base remains the final recommended setting.

## 8. Recommended Figures and Tables for Final Report

- Table 1 and Figure 1 for the main Scenario 4 comparison.
- Table 2 and Figure 3 for the research progression from flat DDQN to hierarchical DDQN to QMIX.
- Table 3 and Figure 4 for multi-seed QMIX stability.
- Table 4, Figure 5, and Figure 6 for the fairness/coordination ablation.
- Table 5 for experimental setup and metric definitions.

## 9. Key Claims Supported by Data

- Hierarchical action abstraction reduces action dimension from 864 to 10.
- Flat DDQN is insufficient for heterogeneous backscatter Scenario 4.
- Hierarchical DDQN beats the heuristic baselines in Scenario 4.
- QMIX base provides stable multi-seed performance.
- QMIX improves the jamming/fairness trade-off relative to hierarchical DDQN.
- Disabling BALANCE_UNDERSERVED_IOT hurts fairness and throughput.
- Naive executor fairness weighting does not improve mean fairness, so QMIX base remains the final setting.

## 10. Cautions / Limitations

- QMIX base is reported as a three-seed aggregate, while hierarchical DDQN and flat DDQN are single-seed final evaluations unless otherwise noted.
- Scenario 4 results are tied to the calibrated simulator and the fixed hierarchical executor semantics.
- Fairness remains moderate; improving per-IoT service balance may require reward-level fairness shaping or more explicit coordination objectives.
"""
    report_path.write_text(text, encoding="utf-8")
    ctx.generated_reports.append(report_path)


def create_index(ctx: ArtifactContext) -> None:
    index_path = ctx.output_root / "publication_artifacts_index.md"
    used = [rel for rel in CSV_SOURCES.values() if (ROOT / rel).exists()]
    missing = ctx.missing
    reports_with_index = [*ctx.generated_reports, index_path]

    def rel_list(paths: list[Path]) -> str:
        return "\n".join(f"- {path.relative_to(ROOT).as_posix()}" for path in paths)

    text = f"""# Publication Artifacts Index

## Regeneration Command

```powershell
python scripts\\generate_publication_artifacts.py
```

## Source CSV Files Used

{chr(10).join(f"- {path}" for path in used)}

## Missing Optional Files

{chr(10).join(f"- {path}" for path in missing) if missing else "- None"}

## Generated Tables

{rel_list(ctx.generated_tables)}

## Generated Figures

{rel_list(ctx.generated_figures)}

## Generated Reports / Index Files

{rel_list(reports_with_index)}
"""
    index_path.write_text(text, encoding="utf-8")
    ctx.generated_reports.append(index_path)


def main() -> None:
    args = parse_args()
    output_root = (ROOT / args.output_dir).resolve()
    tables_dir, figures_dir, reports_dir = ensure_dirs(output_root)
    frames, missing = load_sources()
    ctx = ArtifactContext(
        output_root=output_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        frames=frames,
        missing=missing,
        generated_tables=[],
        generated_figures=[],
        generated_reports=[],
    )

    values = build_publication_values(ctx)
    create_table1(ctx, values)
    create_table2(ctx, values)
    create_table3(ctx)
    create_table4(ctx)
    create_table5(ctx)
    create_figures(ctx, values)
    create_summary_report(ctx, values)
    create_index(ctx)

    print("[publication] Generated tables:")
    for path in ctx.generated_tables:
        print(f"  - {path.relative_to(ROOT)}")
    print("[publication] Generated figures:")
    for path in ctx.generated_figures:
        print(f"  - {path.relative_to(ROOT)}")
    print("[publication] Generated reports:")
    for path in ctx.generated_reports:
        print(f"  - {path.relative_to(ROOT)}")
    if missing:
        print("[publication] Missing optional inputs:")
        for path in missing:
            print(f"  - {path}")
    else:
        print("[publication] Missing optional inputs: none")


if __name__ == "__main__":
    main()
