import pandas as pd

from scripts.summarize_qmix_multiseed import summarize_multiseed


def test_qmix_multiseed_summary_grouping_and_ranking():
    df = pd.DataFrame(
        [
            {
                "config_name": "base",
                "seed": 1,
                "status": "success",
                "final_throughput_per_frame": 1.0,
                "final_reward": -1.0,
                "final_drop": 0.2,
                "final_jam": 0.3,
                "final_fairness": 0.6,
                "final_energy_efficiency": 3.0,
                "final_backscatter_success": 0.7,
                "final_active_success": 0.6,
                "final_fallback_rate": 0.01,
                "best_eval_throughput_per_frame": 1.1,
                "last50_loss_mean": 10.0,
            },
            {
                "config_name": "base",
                "seed": 2,
                "status": "success",
                "final_throughput_per_frame": 0.8,
                "final_reward": -2.0,
                "final_drop": 0.4,
                "final_jam": 0.5,
                "final_fairness": 0.4,
                "final_energy_efficiency": 2.0,
                "final_backscatter_success": 0.5,
                "final_active_success": 0.4,
                "final_fallback_rate": 0.03,
                "best_eval_throughput_per_frame": 0.9,
                "last50_loss_mean": 12.0,
            },
        ]
    )
    mean_df, std_df, ranking = summarize_multiseed(df)
    assert mean_df.loc[0, "final_throughput_per_frame"] == 0.9
    assert round(float(std_df.loc[0, "final_throughput_per_frame"]), 4) == 0.1
    assert ranking.loc[0, "config_name"] == "base"
    assert ranking.loc[0, "rank"] == 1

