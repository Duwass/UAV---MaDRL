from __future__ import annotations

import numpy as np
import torch

from marl.ddqn.action_masking import masked_argmax, masked_epsilon_greedy


def test_masked_argmax_never_selects_invalid_action():
    q_values = torch.tensor([10.0, 1.0, 5.0])
    mask = torch.tensor([0, 1, 1])
    assert masked_argmax(q_values, mask) == 2


def test_masked_epsilon_greedy_samples_only_valid_actions():
    q_values = np.array([10.0, 1.0, 5.0])
    mask = np.array([0, 1, 0])
    rng = np.random.default_rng(123)
    actions = [masked_epsilon_greedy(q_values, mask, epsilon=1.0, rng=rng) for _ in range(20)]
    assert set(actions) == {1}


def test_all_zero_mask_fallback_does_not_crash():
    q_values = np.array([1.0, 2.0, 3.0])
    mask = np.array([0, 0, 0])
    rng = np.random.default_rng(123)
    assert masked_epsilon_greedy(q_values, mask, epsilon=1.0, rng=rng) == 0

