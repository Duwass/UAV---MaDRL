import numpy as np
import torch

from marl.qmix.utils import masked_argmax, masked_epsilon_greedy


def test_masked_argmax_respects_invalid_actions():
    q_values = torch.tensor([[1.0, 9.0, 2.0], [5.0, 4.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    actions = masked_argmax(q_values, mask)
    assert actions.tolist() == [2, 0]


def test_masked_epsilon_greedy_never_selects_invalid_actions():
    rng = np.random.default_rng(123)
    q_values = np.array([[1.0, 99.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    mask = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    for _ in range(50):
        actions, fallback = masked_epsilon_greedy(q_values, mask, epsilon=1.0, rng=rng)
        assert actions[0] in {0, 2}
        assert actions[1] == 2
        assert fallback == 0

