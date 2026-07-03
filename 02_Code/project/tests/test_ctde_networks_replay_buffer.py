from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde.factorized_policy import select_factorized_action_from_logits
from marl.ctde.networks import CentralizedVCritic, FactorizedActor
from marl.ctde.replay_buffer import CTDEReplayBuffer
from marl.ctde.utils import (
    DEFAULT_NUM_MODES,
    DEFAULT_NUM_MOVEMENT_ACTIONS,
    DEFAULT_NUM_TARGETS,
    encode_factorized_action,
)
from marl.ctde import networks as ctde_networks
from marl.ctde import replay_buffer as ctde_replay_buffer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
OBS_DIM = 114
STATE_DIM = 89
NUM_AGENTS = 2


def _transition_kwargs(seed: int = 0, with_masks: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    data = {
        "obs": rng.normal(size=(NUM_AGENTS, OBS_DIM)).astype(np.float32),
        "state": rng.normal(size=STATE_DIM).astype(np.float32),
        "movement_actions": np.asarray([0, 1], dtype=np.int64),
        "target_actions": np.asarray([0, 2], dtype=np.int64),
        "mode_actions": np.asarray([0, 3], dtype=np.int64),
        "flat_actions": np.asarray([0, 123], dtype=np.int64),
        "reward": float(seed),
        "next_obs": rng.normal(size=(NUM_AGENTS, OBS_DIM)).astype(np.float32),
        "next_state": rng.normal(size=STATE_DIM).astype(np.float32),
        "done": False,
    }
    if with_masks:
        data["movement_masks"] = np.ones((NUM_AGENTS, DEFAULT_NUM_MOVEMENT_ACTIONS), dtype=bool)
    return data


def test_factorized_actor_forward_shapes_single_obs():
    actor = FactorizedActor()
    outputs = actor(torch.zeros(OBS_DIM))
    assert outputs["movement_logits"].shape == (1, DEFAULT_NUM_MOVEMENT_ACTIONS)
    assert outputs["target_logits"].shape == (1, DEFAULT_NUM_TARGETS)
    assert outputs["mode_logits"].shape == (1, DEFAULT_NUM_MODES)


def test_factorized_actor_forward_shapes_batch():
    actor = FactorizedActor()
    batch_size = 4
    outputs = actor(torch.zeros(batch_size, OBS_DIM))
    assert outputs["movement_logits"].shape == (batch_size, DEFAULT_NUM_MOVEMENT_ACTIONS)
    assert outputs["target_logits"].shape == (batch_size, DEFAULT_NUM_TARGETS)
    assert outputs["mode_logits"].shape == (batch_size, DEFAULT_NUM_MODES)


def test_factorized_actor_does_not_accept_global_state():
    forbidden = {"state", "global_state", "env", "critic", "executor"}
    params = set(inspect.signature(FactorizedActor.forward).parameters)
    assert forbidden.isdisjoint(params)


def test_centralized_v_critic_forward_shape_single_state():
    critic = CentralizedVCritic()
    values = critic(torch.zeros(STATE_DIM))
    assert values.shape == (1, 1)


def test_centralized_v_critic_forward_shape_batch():
    critic = CentralizedVCritic()
    batch_size = 5
    values = critic(torch.zeros(batch_size, STATE_DIM))
    assert values.shape == (batch_size, 1)


def test_actor_and_critic_parameter_independence():
    actor = FactorizedActor()
    critic = CentralizedVCritic()
    actor_params = {id(param) for param in actor.parameters()}
    critic_params = {id(param) for param in critic.parameters()}
    assert actor_params.isdisjoint(critic_params)


def test_replay_buffer_add_and_len():
    buffer = CTDEReplayBuffer(capacity=4, seed=1)
    buffer.add(**_transition_kwargs())
    assert len(buffer) == 1


def test_replay_buffer_sample_shapes():
    buffer = CTDEReplayBuffer(capacity=8, seed=1)
    for idx in range(5):
        buffer.add(**_transition_kwargs(seed=idx))
    batch = buffer.sample(3)
    assert batch["obs"].shape == (3, NUM_AGENTS, OBS_DIM)
    assert batch["state"].shape == (3, STATE_DIM)
    assert batch["movement_actions"].shape == (3, NUM_AGENTS)
    assert batch["target_actions"].shape == (3, NUM_AGENTS)
    assert batch["mode_actions"].shape == (3, NUM_AGENTS)
    assert batch["flat_actions"].shape == (3, NUM_AGENTS)
    assert batch["reward"].shape == (3,)
    assert batch["next_obs"].shape == (3, NUM_AGENTS, OBS_DIM)
    assert batch["next_state"].shape == (3, STATE_DIM)
    assert batch["done"].shape == (3,)


def test_replay_buffer_optional_movement_masks_shapes():
    buffer = CTDEReplayBuffer(capacity=8, seed=1)
    for idx in range(4):
        buffer.add(**_transition_kwargs(seed=idx, with_masks=True))
    batch = buffer.sample(2)
    assert batch["movement_masks"].shape == (2, NUM_AGENTS, DEFAULT_NUM_MOVEMENT_ACTIONS)
    assert batch["movement_masks"].dtype == np.bool_


def test_replay_buffer_clear():
    buffer = CTDEReplayBuffer(capacity=4, seed=1)
    buffer.add(**_transition_kwargs())
    buffer.clear()
    assert len(buffer) == 0


def test_networks_do_not_import_hierarchical_executor():
    source = inspect.getsource(ctde_networks)
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for item in forbidden)


def test_replay_buffer_does_not_import_hierarchical_executor():
    source = inspect.getsource(ctde_replay_buffer)
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for item in forbidden)


def test_ctde_actor_outputs_compatible_with_selector():
    actor = FactorizedActor()
    outputs = actor(torch.zeros(OBS_DIM))
    action = select_factorized_action_from_logits(
        outputs["movement_logits"][0].detach().numpy(),
        outputs["target_logits"][0].detach().numpy(),
        outputs["mode_logits"][0].detach().numpy(),
    )
    flat_action = encode_factorized_action(action)
    assert 0 <= flat_action < 1056


def test_3d_env_dims_still_match_ctde_defaults():
    env = UAVBackscatterEnv(CONFIG_3D)
    assert env.get_obs_dim() == OBS_DIM
    assert env.get_state_dim() == STATE_DIM
    assert env.get_action_dim() == 1056
