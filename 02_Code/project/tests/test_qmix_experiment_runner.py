from scripts.run_qmix_experiments import build_run_config


def test_qmix_runner_seed_and_prefix_override():
    base_config = {
        "training": {"seed": 42, "total_episodes": 500},
        "logging": {"output_prefix": "qmix_sc4_base"},
    }
    run_config = build_run_config(base_config, "qmix_sc4_base", seed=43, episodes=300)
    assert run_config["training"]["seed"] == 43
    assert run_config["training"]["total_episodes"] == 300
    assert run_config["logging"]["output_prefix"] == "qmix_sc4_base_seed43"

