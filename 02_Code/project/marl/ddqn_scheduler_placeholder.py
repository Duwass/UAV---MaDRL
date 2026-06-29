from __future__ import annotations


class DDQNSchedulerPlaceholder:
    """Interface stub for future independent/shared DDQN scheduling experiments."""

    def __init__(self, config: dict):
        self.config = config

    def train(self, env) -> None:
        # TODO: Add replay buffer, target network, epsilon schedule, and parameter updates.
        raise NotImplementedError("DDQN training is intentionally deferred until the environment is validated.")

    def act(self, observations, env=None):
        # TODO: Replace with learned action selection after training is implemented.
        raise NotImplementedError("DDQN action selection is not implemented in Phase 3.")

