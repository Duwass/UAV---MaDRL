import pandas as pd

from scripts.summarize_qmix_fairness_ablation import summarize_fairness


def test_qmix_fairness_summary_tradeoff_ranking():
    df = pd.DataFrame(
        [
            {
                "config_name": "a",
                "status": "success",
                "final_throughput_per_frame": 0.95,
                "final_reward": -1.0,
                "final_drop": 0.4,
                "final_jam": 0.2,
                "final_fairness": 0.6,
                "final_energy_efficiency": 4.0,
                "final_backscatter_success": 0.7,
                "final_active_success": 0.8,
                "final_fallback_rate": 0.01,
            },
            {
                "config_name": "b",
                "status": "success",
                "final_throughput_per_frame": 0.90,
                "final_reward": -2.0,
                "final_drop": 0.5,
                "final_jam": 0.4,
                "final_fairness": 0.5,
                "final_energy_efficiency": 3.0,
                "final_backscatter_success": 0.6,
                "final_active_success": 0.7,
                "final_fallback_rate": 0.02,
            },
        ]
    )
    mean_df, std_df, ranking = summarize_fairness(df)
    assert set(mean_df["config_name"]) == {"a", "b"}
    assert set(std_df["config_name"]) == {"a", "b"}
    assert ranking.loc[0, "config_name"] == "a"
    assert "tradeoff_score" in ranking.columns

