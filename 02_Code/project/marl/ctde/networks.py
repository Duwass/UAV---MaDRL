from __future__ import annotations

import torch
from torch import nn

from marl.ctde.utils import DEFAULT_NUM_MODES, DEFAULT_NUM_MOVEMENT_ACTIONS, DEFAULT_NUM_TARGETS


def _mlp(input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for _ in range(int(num_layers)):
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        last_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class FactorizedActor(nn.Module):
    """Decentralized actor: forward uses only one UAV's local observation, never global state."""

    def __init__(
        self,
        obs_dim: int = 114,
        hidden_dim: int = 128,
        num_layers: int = 2,
        movement_dim: int = DEFAULT_NUM_MOVEMENT_ACTIONS,
        target_dim: int = DEFAULT_NUM_TARGETS,
        mode_dim: int = DEFAULT_NUM_MODES,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.movement_dim = int(movement_dim)
        self.target_dim = int(target_dim)
        self.mode_dim = int(mode_dim)
        self.trunk = _mlp(self.obs_dim, hidden_dim, num_layers)
        self.movement_head = nn.Linear(int(hidden_dim), self.movement_dim)
        self.target_head = nn.Linear(int(hidden_dim), self.target_dim)
        self.mode_head = nn.Linear(int(hidden_dim), self.mode_dim)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        features = self.trunk(obs)
        return {
            "movement_logits": self.movement_head(features),
            "target_logits": self.target_head(features),
            "mode_logits": self.mode_head(features),
        }


class CentralizedVCritic(nn.Module):
    """Centralized-training value critic; not used during decentralized execution."""

    def __init__(
        self,
        state_dim: int = 89,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.trunk = _mlp(self.state_dim, hidden_dim, num_layers)
        self.value_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.value_head(self.trunk(state))
