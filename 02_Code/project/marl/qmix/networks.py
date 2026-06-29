from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_sizes: list[int] | tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden)))
        layers.append(nn.ReLU())
        last_dim = int(hidden)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class AgentQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 128),
        use_agent_id: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_agents = int(n_agents)
        self.use_agent_id = bool(use_agent_id)
        input_dim = self.obs_dim + (self.n_agents if self.use_agent_id else 0)
        self.net = _mlp(input_dim, hidden_sizes, self.action_dim)

    def forward(self, observations: torch.Tensor, agent_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = observations
        if self.use_agent_id:
            if agent_ids is None:
                raise ValueError("agent_ids are required when use_agent_id=True.")
            one_hot = torch.nn.functional.one_hot(agent_ids.long(), num_classes=self.n_agents).float()
            one_hot = one_hot.to(device=x.device)
            x = torch.cat([x, one_hot], dim=-1)
        return self.net(x)


class QMixer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.mixing_embed_dim = int(mixing_embed_dim)
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, int(hypernet_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hypernet_hidden_dim), self.n_agents * self.mixing_embed_dim),
        )
        self.hyper_b_1 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, int(hypernet_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hypernet_hidden_dim), self.mixing_embed_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, 1),
        )

    def first_layer_weights(self, states: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.hyper_w_1(states))

    def final_layer_weights(self, states: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.hyper_w_final(states))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = agent_qs.shape
        flat_states = states.reshape(batch_size * time_steps, self.state_dim)
        flat_qs = agent_qs.reshape(batch_size * time_steps, 1, self.n_agents)

        w1 = self.first_layer_weights(flat_states).view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b_1(flat_states).view(-1, 1, self.mixing_embed_dim)
        hidden = torch.nn.functional.elu(torch.bmm(flat_qs, w1) + b1)

        w_final = self.final_layer_weights(flat_states).view(-1, self.mixing_embed_dim, 1)
        v = self.v(flat_states).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        return y.view(batch_size, time_steps)

