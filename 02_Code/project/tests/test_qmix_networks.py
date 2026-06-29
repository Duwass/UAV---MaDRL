import torch

from marl.qmix.networks import AgentQNetwork, QMixer


def test_agent_q_network_output_shape():
    net = AgentQNetwork(obs_dim=12, action_dim=10, n_agents=3, hidden_sizes=[16], use_agent_id=True)
    obs = torch.randn(3, 12)
    agent_ids = torch.tensor([0, 1, 2])
    q_values = net(obs, agent_ids)
    assert q_values.shape == (3, 10)


def test_qmixer_output_shape_and_nonnegative_weights():
    mixer = QMixer(n_agents=2, state_dim=20, mixing_embed_dim=8, hypernet_hidden_dim=16)
    agent_qs = torch.randn(4, 5, 2)
    states = torch.randn(4, 5, 20)
    q_tot = mixer(agent_qs, states)
    assert q_tot.shape == (4, 5)
    flat_states = states.reshape(-1, 20)
    assert torch.all(mixer.first_layer_weights(flat_states) >= 0)
    assert torch.all(mixer.final_layer_weights(flat_states) >= 0)

