import numpy as np
import torch

from marl.ddqn.ddqn_agent import CentralizedFactorizedDDQNAgent
from marl.ddqn.networks import DuelingQNetwork, QNetwork


def test_dueling_network_output_shape():
    network = DuelingQNetwork(input_dim=12, action_dim=7, hidden_sizes=[16])
    output = network(torch.zeros((4, 12), dtype=torch.float32))
    assert output.shape == (4, 7)


def test_standard_and_dueling_agents_select_actions():
    state = np.zeros(10, dtype=np.float32)
    mask = np.ones(5, dtype=np.int8)
    for network_type in ["standard", "dueling"]:
        agent = CentralizedFactorizedDDQNAgent(
            state_dim=10,
            action_dim=5,
            num_uav=1,
            hidden_sizes=[16],
            network_type=network_type,
            device="cpu",
            seed=123,
        )
        action = agent.select_action(state, uav_id=0, action_mask=mask, deterministic=True)
        assert 0 <= action < 5


def test_standard_network_output_shape():
    network = QNetwork(input_dim=12, action_dim=7, hidden_sizes=[16])
    output = network(torch.zeros((4, 12), dtype=torch.float32))
    assert output.shape == (4, 7)
