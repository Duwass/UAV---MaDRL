from marl.ddqn.reward_processing import RewardProcessor


def test_reward_processing_scale():
    processor = RewardProcessor.from_config({"enabled": True, "mode": "scale", "scale": 100.0})
    assert processor.process(-250.0) == -2.5


def test_reward_processing_clip_mode():
    processor = RewardProcessor.from_config(
        {"enabled": True, "mode": "clip", "scale": 10.0, "clip_min": -2.0, "clip_max": 2.0}
    )
    assert processor.process(-100.0) == -2.0
    assert processor.process(100.0) == 2.0


def test_reward_processing_none_when_disabled():
    processor = RewardProcessor.from_config({"enabled": False, "mode": "scale", "scale": 100.0})
    assert processor.process(-250.0) == -250.0
