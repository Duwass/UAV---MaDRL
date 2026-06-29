from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from marl.ddqn.action_masking import apply_action_mask, masked_epsilon_greedy
from marl.ddqn.networks import DuelingQNetwork, QNetwork


class CentralizedFactorizedDDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_uav: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 256),
        layer_norm: bool = False,
        learning_rate: float = 1.0e-3,
        gamma: float = 0.99,
        batch_size: int = 128,
        target_update_steps: int = 1000,
        gradient_clip_norm: float = 10.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        network_type: str = "standard",
        device: str | None = None,
        seed: int = 0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_uav = int(num_uav)
        self.uav_id_dim = self.num_uav if self.num_uav <= 10 else 1
        self.input_dim = self.state_dim + self.uav_id_dim
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_steps = int(target_update_steps)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.network_type = str(network_type).lower()
        self.train_steps = 0
        self.rng = np.random.default_rng(seed)
        device_name = str(device or "auto").lower()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DDQN config requested CUDA, but torch.cuda.is_available() is False.")
        self.device = torch.device(device_name)
        print(f"[DDQN] Using device: {self.device}")

        torch.manual_seed(seed)
        network_cls = self._network_class()
        self.online_net = network_cls(self.input_dim, self.action_dim, hidden_sizes, layer_norm).to(self.device)
        self.target_net = network_cls(self.input_dim, self.action_dim, hidden_sizes, layer_norm).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.SmoothL1Loss()

    def _network_class(self) -> type[nn.Module]:
        if self.network_type == "standard":
            return QNetwork
        if self.network_type == "dueling":
            return DuelingQNetwork
        raise ValueError(f"Unknown ddqn.network_type={self.network_type}")

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.train_steps / max(self.epsilon_decay_steps, 1))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def encode_uav_ids(self, uav_ids: torch.Tensor) -> torch.Tensor:
        uav_ids = uav_ids.to(self.device)
        if self.uav_id_dim == 1:
            denom = max(self.num_uav - 1, 1)
            return (uav_ids.float() / denom).unsqueeze(1)
        return torch.nn.functional.one_hot(uav_ids.long(), num_classes=self.uav_id_dim).float()

    def build_input(self, states: torch.Tensor, uav_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([states.to(self.device), self.encode_uav_ids(uav_ids)], dim=1)

    def select_action(self, state: np.ndarray, uav_id: int, action_mask: np.ndarray, deterministic: bool = False) -> int:
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            uav_id_t = torch.as_tensor([int(uav_id)], dtype=torch.long, device=self.device)
            action_mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
            q_values = self.online_net(self.build_input(state_t, uav_id_t)).squeeze(0)
        epsilon = 0.0 if deterministic else self.epsilon
        return masked_epsilon_greedy(q_values, action_mask_t, epsilon, self.rng)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.online_net.train()
        states = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        uav_ids = torch.as_tensor(batch["uav_id"], dtype=torch.long, device=self.device)
        actions = torch.as_tensor(batch["action"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        action_masks = torch.as_tensor(batch["action_mask"], dtype=torch.float32, device=self.device)
        next_action_masks = torch.as_tensor(batch["next_action_mask"], dtype=torch.float32, device=self.device)

        inputs = self.build_input(states, uav_ids)
        q_values_all = self.online_net(inputs)
        q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_inputs = self.build_input(next_states, uav_ids)
            next_online_q = self.online_net(next_inputs)
            next_online_masked = apply_action_mask(next_online_q, next_action_masks)
            next_actions = next_online_masked.argmax(dim=1)
            next_target_q = self.target_net(next_inputs)
            next_q = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip_norm).item())
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.update_target_network()

        masked_current = apply_action_mask(q_values_all.detach(), action_masks)
        return {
            "loss": float(loss.item()),
            "mean_q": float(q_values.detach().mean().item()),
            "max_q": float(masked_current.max(dim=1).values.mean().item()),
            "epsilon": float(self.epsilon),
            "gradient_norm": grad_norm,
        }

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_uav": self.num_uav,
            "uav_id_dim": self.uav_id_dim,
            "epsilon": self.epsilon,
            "network_type": self.network_type,
            "device": str(self.device),
            "extra": extra or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        return checkpoint.get("extra", {})
