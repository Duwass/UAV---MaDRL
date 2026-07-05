from __future__ import annotations

from typing import Any

import numpy as np

from marl.ctde.action_diagnostics import prefix_action_diagnostics, summarize_action_diagnostics
from marl.ctde.rollout import select_decentralized_actions
from marl.ctde.utils import DEFAULT_NUM_MODES, DEFAULT_NUM_MOVEMENT_ACTIONS, DEFAULT_NUM_TARGETS, FactorizedAction


def evaluate_decentralized_policy(
    env,
    actor,
    num_episodes: int = 1,
    max_steps: int | None = None,
    use_movement_mask: bool = True,
    deterministic: bool = True,
    epsilon: float | None = None,
    selection_mode: str = "epsilon_argmax",
    diagnostic_prefix: str = "eval_",
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run a decentralized execution smoke/evaluation path.

    Each action is selected from per-UAV local observations through the actor.
    This path does not use critic values, global state for action selection,
    env action masks, or any high-level action executor/wrapper. It is an
    evaluation utility, not a performance claim.
    """
    generator = rng if rng is not None else np.random.default_rng()
    episode_returns: list[float] = []
    episode_steps: list[int] = []
    episode_metrics: list[dict[str, Any]] = []
    frame_metrics: list[dict[str, Any]] = []
    terminated_count = 0
    truncated_count = 0
    selected_actions: list[FactorizedAction] = []
    raw_actions: list[FactorizedAction] = []
    head_diagnostics: list[dict[str, Any]] = []
    agent_ids: list[int] = []
    num_agents = 0
    policy_epsilon = 0.0 if deterministic else 1.0
    if epsilon is not None:
        policy_epsilon = float(epsilon)

    for _ in range(int(num_episodes)):
        observations, _ = env.reset()
        num_agents = max(num_agents, int(np.asarray(observations).shape[0]))
        step_limit = int(max_steps if max_steps is not None else env.max_steps)
        episode_return = 0.0
        steps = 0
        terminated = False
        truncated = False
        last_episode_metrics: dict[str, Any] = {}
        last_frame_metrics: dict[str, Any] = {}

        while steps < step_limit and not (terminated or truncated):
            selection = select_decentralized_actions(
                actor,
                observations,
                epsilon=policy_epsilon,
                rng=generator,
                use_movement_mask=use_movement_mask,
                selection_mode=selection_mode,
                temperature=temperature,
            )
            selected_actions.extend(selection.factorized_actions)
            raw_actions.extend(selection.raw_factorized_actions)
            head_diagnostics.extend(selection.head_diagnostics)
            agent_ids.extend(range(len(selection.factorized_actions)))
            observations, reward, terminated, truncated, info = env.step(selection.flat_actions)
            episode_return += float(reward)
            steps += 1
            last_episode_metrics = dict(info.get("episode_metrics", {}))
            last_frame_metrics = dict(info.get("frame_metrics", {}))

        episode_returns.append(float(episode_return))
        episode_steps.append(int(steps))
        episode_metrics.append(last_episode_metrics)
        frame_metrics.append(last_frame_metrics)
        terminated_count += int(bool(terminated))
        truncated_count += int(bool(truncated))

    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    action_diagnostics = summarize_action_diagnostics(
        selected_actions,
        raw_actions=raw_actions,
        head_diagnostics=head_diagnostics,
        agent_ids=agent_ids,
        num_agents=num_agents,
        deterministic=bool(deterministic and policy_epsilon <= 0.0 and str(selection_mode) != "stochastic"),
        epsilon=policy_epsilon,
        selection_mode=str(selection_mode),
        movement_count=DEFAULT_NUM_MOVEMENT_ACTIONS,
        target_count=DEFAULT_NUM_TARGETS,
        mode_count=DEFAULT_NUM_MODES,
    )
    prefixed_action_diagnostics = prefix_action_diagnostics(
        action_diagnostics,
        diagnostic_prefix,
        num_agents=max(num_agents, 1),
    )
    return {
        "episodes": int(num_episodes),
        "total_steps": int(sum(episode_steps)),
        "mean_return": mean_return,
        "episode_returns": episode_returns,
        "episode_steps": episode_steps,
        "terminated_count": int(terminated_count),
        "truncated_count": int(truncated_count),
        "episode_metrics": episode_metrics,
        "frame_metrics": frame_metrics,
        "last_episode_metrics": episode_metrics[-1] if episode_metrics else {},
        "last_frame_metrics": frame_metrics[-1] if frame_metrics else {},
        "action_diagnostics": action_diagnostics,
        f"{diagnostic_prefix}action_diagnostics": prefixed_action_diagnostics,
        "eval_action_diagnostics": prefixed_action_diagnostics if diagnostic_prefix == "eval_" else {},
    }
