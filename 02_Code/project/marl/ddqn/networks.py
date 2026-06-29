from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 256),
        layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden_size)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(hidden_size)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_size)
        layers.append(nn.Linear(last_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 256),
        layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden_size)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(hidden_size)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_size)
        self.features = nn.Sequential(*layers)
        self.value_stream = nn.Linear(last_dim, 1)
        self.advantage_stream = nn.Linear(last_dim, int(action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
