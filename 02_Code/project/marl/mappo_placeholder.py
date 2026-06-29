from __future__ import annotations


class MAPPOPlaceholder:
    """Interface stub for centralized-critic MAPPO."""

    def __init__(self, config: dict):
        self.config = config

    def train(self, parallel_env) -> None:
        # TODO: Add rollout storage, centralized value function, GAE, and PPO clipping.
        raise NotImplementedError("MAPPO training is intentionally deferred until the testbed is stable.")

    def act(self, observations, state=None):
        # TODO: Replace with actor policy inference after implementation.
        raise NotImplementedError("MAPPO action selection is not implemented in Phase 3.")

