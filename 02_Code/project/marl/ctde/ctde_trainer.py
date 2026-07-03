from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class CTDETrainer:
    """Minimal centralized-training update component for CTDE smoke tests.

    The critic uses global state only during training. The actor loss uses local
    observations and a centralized advantage, but this trainer is not used for
    decentralized action selection; evaluation/deployment still does not use the
    critic.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        entropy_coef: float = 0.0,
        device: str | torch.device | None = None,
        grad_clip_norm: float | None = None,
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        if device is None:
            try:
                device = next(actor.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.critic.to(self.device)

    def compute_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        tensors = self._to_tensor_batch(batch)
        states = tensors["state"]
        next_states = tensors["next_state"]
        rewards = tensors["reward"]
        dones = tensors["done"].float()

        values = self.critic(states).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)
            target_v = rewards + self.gamma * (1.0 - dones) * next_values
        critic_loss = torch.nn.functional.mse_loss(values, target_v.detach())
        advantage = target_v.detach() - values.detach()

        selected_log_probs, entropy = self._compute_action_log_probs(
            tensors["obs"],
            tensors["movement_actions"],
            tensors["target_actions"],
            tensors["mode_actions"],
        )
        summed_log_probs = selected_log_probs.sum(dim=1)
        actor_loss = -(advantage * summed_log_probs).mean()
        if self.entropy_coef != 0.0:
            actor_loss = actor_loss - self.entropy_coef * entropy.mean()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "total_loss": critic_loss + actor_loss,
            "mean_value": values.detach().mean(),
            "mean_target": target_v.detach().mean(),
            "mean_advantage": advantage.detach().mean(),
            "mean_entropy": entropy.detach().mean(),
        }

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        losses = self.compute_losses(batch)

        self.critic_optimizer.zero_grad(set_to_none=True)
        losses["critic_loss"].backward()
        critic_grad_norm = self._clip_grad_norm(self.critic.parameters())
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        losses["actor_loss"].backward()
        actor_grad_norm = self._clip_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        return {
            "critic_loss": float(losses["critic_loss"].detach().item()),
            "actor_loss": float(losses["actor_loss"].detach().item()),
            "mean_value": float(losses["mean_value"].detach().item()),
            "mean_target": float(losses["mean_target"].detach().item()),
            "mean_advantage": float(losses["mean_advantage"].detach().item()),
            "mean_entropy": float(losses["mean_entropy"].detach().item()),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_grad_norm": float(actor_grad_norm),
        }

    def _to_tensor_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device),
            "state": torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device),
            "movement_actions": torch.as_tensor(batch["movement_actions"], dtype=torch.long, device=self.device),
            "target_actions": torch.as_tensor(batch["target_actions"], dtype=torch.long, device=self.device),
            "mode_actions": torch.as_tensor(batch["mode_actions"], dtype=torch.long, device=self.device),
            "reward": torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(batch["done"], dtype=torch.bool, device=self.device),
        }

    def _compute_action_log_probs(
        self,
        obs: torch.Tensor,
        movement_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mode_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_agents, obs_dim = obs.shape
        flat_obs = obs.reshape(batch_size * num_agents, obs_dim)
        outputs = self.actor(flat_obs)
        movement_logits = outputs["movement_logits"]
        target_logits = outputs["target_logits"]
        mode_logits = outputs["mode_logits"]

        movement_log_probs = torch.log_softmax(movement_logits, dim=-1)
        target_log_probs = torch.log_softmax(target_logits, dim=-1)
        mode_log_probs = torch.log_softmax(mode_logits, dim=-1)

        flat_movement = movement_actions.reshape(-1, 1)
        flat_target = target_actions.reshape(-1, 1)
        flat_mode = mode_actions.reshape(-1, 1)
        selected = (
            movement_log_probs.gather(1, flat_movement).squeeze(1)
            + target_log_probs.gather(1, flat_target).squeeze(1)
            + mode_log_probs.gather(1, flat_mode).squeeze(1)
        ).view(batch_size, num_agents)

        entropy = (
            torch.distributions.Categorical(logits=movement_logits).entropy()
            + torch.distributions.Categorical(logits=target_logits).entropy()
            + torch.distributions.Categorical(logits=mode_logits).entropy()
        ).view(batch_size, num_agents)
        return selected, entropy

    def _clip_grad_norm(self, parameters) -> float:
        params = [param for param in parameters if param.grad is not None]
        if not params:
            return 0.0
        if self.grad_clip_norm is None:
            total = torch.norm(torch.stack([param.grad.detach().norm(2) for param in params]), 2)
            return float(total.item())
        return float(torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm).item())
