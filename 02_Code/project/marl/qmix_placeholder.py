from __future__ import annotations


class QMIXPlaceholder:
    """Interface stub for future value-decomposition training."""

    def __init__(self, config: dict):
        self.config = config

    def train(self, parallel_env) -> None:
        # TODO: Add agent Q-networks, mixer network, replay, and target updates.
        raise NotImplementedError("QMIX training is intentionally deferred until the simulator passes sanity checks.")

    def act(self, observations, state=None):
        # TODO: Replace with decentralized argmax action selection after implementation.
        raise NotImplementedError("QMIX action selection is not implemented in Phase 3.")

