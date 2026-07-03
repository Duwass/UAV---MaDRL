from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from marl.ctde.factorized_policy import select_factorized_action_from_logits
from marl.ctde.utils import (
    DEFAULT_NUM_MOVEMENT_ACTIONS,
    FactorizedAction,
    build_local_movement_mask_from_obs,
    encode_factorized_action,
)


@dataclass(frozen=True)
class DecentralizedActionSelection:
    factorized_actions: list[FactorizedAction]
    flat_actions: list[int]
    movement_masks: np.ndarray | None


def select_decentralized_actions(
    actor,
    observations: np.ndarray,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    use_movement_mask: bool = True,
) -> DecentralizedActionSelection:
    """DE-safe action selection from local observations only.

    This function does not use global state, critic values, env action masks, or
    the hierarchical executor. Each actor call receives one UAV local
    observation.
    """
    generator = rng if rng is not None else np.random.default_rng()
    obs_array = np.asarray(observations, dtype=np.float32)
    if obs_array.ndim != 2:
        raise ValueError("observations must have shape [num_agents, obs_dim].")

    factorized_actions: list[FactorizedAction] = []
    flat_actions: list[int] = []
    movement_masks: list[np.ndarray] = []

    for local_obs in obs_array:
        outputs = _actor_forward(actor, local_obs)
        movement_mask = (
            build_local_movement_mask_from_obs(local_obs, DEFAULT_NUM_MOVEMENT_ACTIONS)
            if use_movement_mask
            else None
        )
        action = select_factorized_action_from_logits(
            _to_1d_numpy(outputs["movement_logits"]),
            _to_1d_numpy(outputs["target_logits"]),
            _to_1d_numpy(outputs["mode_logits"]),
            movement_mask=movement_mask,
            epsilon=epsilon,
            rng=generator,
        )
        factorized_actions.append(action)
        flat_actions.append(encode_factorized_action(action))
        if movement_mask is not None:
            movement_masks.append(movement_mask)

    masks_array = np.stack(movement_masks) if use_movement_mask else None
    return DecentralizedActionSelection(factorized_actions, flat_actions, masks_array)


def collect_ctde_rollout(
    env,
    actor,
    buffer=None,
    max_steps: int | None = None,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    use_movement_mask: bool = True,
) -> dict[str, Any]:
    """Collect a short rollout while storing global state only for training data.

    Action selection is delegated to select_decentralized_actions, so the actor
    only receives local observations. Global state is read for replay-buffer
    storage intended for later centralized critic training.
    """
    observations, info = env.reset()
    state = _global_state_from_env_or_info(env, info)
    step_limit = int(max_steps if max_steps is not None else env.max_steps)
    episode_return = 0.0
    transitions_collected = 0
    terminated = False
    truncated = False
    last_info = dict(info)

    while transitions_collected < step_limit and not (terminated or truncated):
        selection = select_decentralized_actions(
            actor,
            observations,
            epsilon=epsilon,
            rng=rng,
            use_movement_mask=use_movement_mask,
        )
        next_observations, reward, terminated, truncated, next_info = env.step(selection.flat_actions)
        next_state = _global_state_from_env_or_info(env, next_info)
        done = bool(terminated or truncated)

        if buffer is not None:
            buffer.add(
                obs=observations,
                state=state,
                movement_actions=np.asarray([action.movement for action in selection.factorized_actions], dtype=np.int64),
                target_actions=np.asarray([action.target for action in selection.factorized_actions], dtype=np.int64),
                mode_actions=np.asarray([action.mode for action in selection.factorized_actions], dtype=np.int64),
                flat_actions=np.asarray(selection.flat_actions, dtype=np.int64),
                reward=float(reward),
                next_obs=next_observations,
                next_state=next_state,
                done=done,
                movement_masks=selection.movement_masks,
                info=dict(next_info),
            )

        episode_return += float(reward)
        transitions_collected += 1
        observations = next_observations
        state = next_state
        last_info = dict(next_info)

    return {
        "num_steps": int(transitions_collected),
        "episode_return": float(episode_return),
        "transitions_collected": int(transitions_collected),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "last_info": last_info,
        "episode_metrics": dict(last_info.get("episode_metrics", {})),
    }


def _actor_forward(actor, local_obs: np.ndarray) -> dict[str, Any]:
    obs_tensor = torch.as_tensor(local_obs, dtype=torch.float32)
    with torch.no_grad():
        outputs = actor(obs_tensor)
    if isinstance(outputs, dict):
        return outputs
    movement_logits, target_logits, mode_logits = outputs
    return {
        "movement_logits": movement_logits,
        "target_logits": target_logits,
        "mode_logits": mode_logits,
    }


def _to_1d_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError("actor logits must be 1D or [1, dim].")
    return array


def _global_state_from_env_or_info(env, info: dict[str, Any]) -> np.ndarray:
    if "global_state" in info:
        return np.asarray(info["global_state"], dtype=np.float32)
    return np.asarray(env.get_global_state(), dtype=np.float32)
