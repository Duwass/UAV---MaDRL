from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from marl.qmix.networks import AgentQNetwork, QMixer
from marl.qmix.utils import apply_action_mask, masked_argmax, masked_epsilon_greedy


class QMIXAgent:
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        action_dim: int,
        use_agent_id: bool = True,
        agent_hidden_sizes: list[int] | tuple[int, ...] = (128, 128),
        mixing_embed_dim: int = 32,
        hypernet_hidden_dim: int = 64,
        learning_rate: float = 5.0e-4,
        gamma: float = 0.99,
        target_update_steps: int = 200,
        gradient_clip_norm: float = 10.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 25000,
        double_q: bool = True,
        device: str | None = "auto",
        seed: int = 0,
    ):
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.use_agent_id = bool(use_agent_id)
        self.gamma = float(gamma)
        self.target_update_steps = int(target_update_steps)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.double_q = bool(double_q)
        self.train_steps = 0
        self.env_steps = 0
        self.target_update_count = 0
        self.rng = np.random.default_rng(seed)

        device_name = str(device or "auto").lower()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("QMIX config requested CUDA, but torch.cuda.is_available() is False.")
        self.device = torch.device(device_name)
        print(f"[QMIX] Using device: {self.device}")

        torch.manual_seed(seed)
        self.agent_net = AgentQNetwork(self.obs_dim, self.action_dim, self.n_agents, agent_hidden_sizes, self.use_agent_id).to(
            self.device
        )
        self.target_agent_net = AgentQNetwork(
            self.obs_dim, self.action_dim, self.n_agents, agent_hidden_sizes, self.use_agent_id
        ).to(self.device)
        self.mixer = QMixer(self.n_agents, self.state_dim, mixing_embed_dim, hypernet_hidden_dim).to(self.device)
        self.target_mixer = QMixer(self.n_agents, self.state_dim, mixing_embed_dim, hypernet_hidden_dim).to(self.device)
        self.update_target_networks()
        self.target_agent_net.eval()
        self.target_mixer.eval()
        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=float(learning_rate),
        )
        self.loss_fn = nn.MSELoss(reduction="none")

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(self.epsilon_decay_steps, 1))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _agent_ids(self, *shape: int) -> torch.Tensor:
        base = torch.arange(self.n_agents, device=self.device, dtype=torch.long)
        if not shape:
            return base
        return base.view(*([1] * len(shape)), self.n_agents).expand(*shape, self.n_agents)

    def select_actions(self, observations: np.ndarray, action_masks: np.ndarray, deterministic: bool = False) -> tuple[list[int], int]:
        self.agent_net.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            ids_t = torch.arange(self.n_agents, dtype=torch.long, device=self.device)
            q_values = self.agent_net(obs_t, ids_t)
        epsilon = 0.0 if deterministic else self.epsilon
        actions, fallback_count = masked_epsilon_greedy(q_values, action_masks, epsilon, self.rng)
        if not deterministic:
            self.env_steps += 1
        return [int(action) for action in actions], int(fallback_count)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.agent_net.train()
        self.mixer.train()
        observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        global_states = torch.as_tensor(batch["global_states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        next_global_states = torch.as_tensor(batch["next_global_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        action_masks = torch.as_tensor(batch["action_masks"], dtype=torch.float32, device=self.device)
        next_action_masks = torch.as_tensor(batch["next_action_masks"], dtype=torch.float32, device=self.device)
        filled = torch.as_tensor(batch["filled"], dtype=torch.float32, device=self.device)

        batch_size, time_steps, _, _ = observations.shape
        flat_agent_ids = self._agent_ids(batch_size, time_steps).reshape(-1)

        q_all = self.agent_net(observations.reshape(-1, self.obs_dim), flat_agent_ids).view(
            batch_size, time_steps, self.n_agents, self.action_dim
        )
        chosen_agent_qs = q_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_agent_qs, global_states)

        with torch.no_grad():
            next_online_q = self.agent_net(next_observations.reshape(-1, self.obs_dim), flat_agent_ids).view(
                batch_size, time_steps, self.n_agents, self.action_dim
            )
            if self.double_q:
                next_actions = masked_argmax(next_online_q, next_action_masks)
            else:
                next_actions = masked_argmax(
                    self.target_agent_net(next_observations.reshape(-1, self.obs_dim), flat_agent_ids).view(
                        batch_size, time_steps, self.n_agents, self.action_dim
                    ),
                    next_action_masks,
                )
            target_next_q_all = self.target_agent_net(next_observations.reshape(-1, self.obs_dim), flat_agent_ids).view(
                batch_size, time_steps, self.n_agents, self.action_dim
            )
            target_next_q_all = apply_action_mask(target_next_q_all, next_action_masks)
            target_next_agent_qs = target_next_q_all.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)
            target_q_tot_next = self.target_mixer(target_next_agent_qs, next_global_states)
            targets = rewards + self.gamma * (1.0 - dones) * target_q_tot_next

        losses = self.loss_fn(q_tot, targets.detach()) * filled
        loss = losses.sum() / filled.sum().clamp_min(1.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            self.gradient_clip_norm,
        )
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.update_target_networks()
            self.target_update_count += 1

        masked_current = apply_action_mask(q_all.detach(), action_masks)
        return {
            "loss": float(loss.item()),
            "avg_q_tot": float((q_tot.detach() * filled).sum().item() / filled.sum().clamp_min(1.0).item()),
            "avg_agent_q": float(masked_current.max(dim=-1).values.mean().item()),
            "gradient_norm": float(grad_norm.item()),
            "epsilon": float(self.epsilon),
        }

    def update_target_networks(self) -> None:
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent_net": self.agent_net.state_dict(),
                "target_agent_net": self.target_agent_net.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "env_steps": self.env_steps,
                "target_update_count": self.target_update_count,
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "n_agents": self.n_agents,
                "action_dim": self.action_dim,
                "use_agent_id": self.use_agent_id,
                "double_q": self.double_q,
                "epsilon": self.epsilon,
                "device": str(self.device),
                "extra": extra or {},
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.agent_net.load_state_dict(checkpoint["agent_net"])
        self.target_agent_net.load_state_dict(checkpoint["target_agent_net"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.env_steps = int(checkpoint.get("env_steps", 0))
        self.target_update_count = int(checkpoint.get("target_update_count", 0))
        return dict(checkpoint.get("extra", {}))
