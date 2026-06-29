from __future__ import annotations

from typing import Any

from pettingzoo import ParallelEnv

from envs.uav_backscatter_env import UAVBackscatterEnv


class UAVBackscatterParallelEnv(ParallelEnv):
    metadata = {"name": "uav_backscatter_parallel_v0", "render_modes": ["human"]}

    def __init__(self, config: str | dict[str, Any]):
        self.base_env = UAVBackscatterEnv(config)
        self.possible_agents = [f"uav_{idx}" for idx in range(self.base_env.num_uav)]
        self.agents = self.possible_agents[:]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, info = self.base_env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        obs_dict = {agent: observations[idx] for idx, agent in enumerate(self.agents)}
        info_dict = {agent: info for agent in self.agents}
        return obs_dict, info_dict

    def step(self, actions: dict[str, int]):
        observations, reward, terminated, truncated, info = self.base_env.step(actions)
        obs_dict = {agent: observations[idx] for idx, agent in enumerate(self.possible_agents)}
        rewards = {agent: reward for agent in self.possible_agents}
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.possible_agents}
        infos = {agent: info for agent in self.possible_agents}
        self.agents = [] if terminated or truncated else self.possible_agents[:]
        return obs_dict, rewards, terminations, truncations, infos

    def render(self) -> None:
        self.base_env.render()

    def close(self) -> None:
        self.base_env.close()

    def state(self):
        return self.base_env.get_global_state()

    def observation_space(self, agent: str):
        return self.base_env.single_observation_space

    def action_space(self, agent: str):
        return self.base_env.action_space

    def action_mask(self, agent: str):
        agent_idx = int(agent.split("_")[-1])
        return self.base_env.get_action_mask(agent_idx)
