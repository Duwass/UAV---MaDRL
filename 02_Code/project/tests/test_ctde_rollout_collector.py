from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde.networks import FactorizedActor
from marl.ctde.replay_buffer import CTDEReplayBuffer
from marl.ctde.rollout import collect_ctde_rollout, select_decentralized_actions
from marl.ctde.utils import DEFAULT_NUM_MOVEMENT_ACTIONS, MOVE_UP
from marl.ctde import rollout as ctde_rollout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"


class RecordingActor:
    def __init__(self, prefer_up: bool = False):
        self.inputs: list[np.ndarray] = []
        self.prefer_up = bool(prefer_up)

    def __call__(self, obs: torch.Tensor):
        obs_np = obs.detach().cpu().numpy()
        self.inputs.append(obs_np.copy())
        movement = torch.zeros(1, DEFAULT_NUM_MOVEMENT_ACTIONS)
        movement[0, MOVE_UP if self.prefer_up else 0] = 10.0
        target = torch.zeros(1, 16)
        target[0, 1] = 10.0
        mode = torch.zeros(1, 6)
        mode[0, 1] = 10.0
        return {
            "movement_logits": movement,
            "target_logits": target,
            "mode_logits": mode,
        }


def test_select_decentralized_actions_signature_has_no_env_state_critic():
    forbidden = {"env", "state", "global_state", "critic", "executor"}
    params = set(inspect.signature(select_decentralized_actions).parameters)
    assert forbidden.isdisjoint(params)


def test_select_decentralized_actions_shapes():
    actor = FactorizedActor()
    observations = np.zeros((2, 114), dtype=np.float32)
    selection = select_decentralized_actions(actor, observations, rng=np.random.default_rng(1))
    assert len(selection.flat_actions) == 2
    assert len(selection.factorized_actions) == 2
    assert selection.movement_masks.shape == (2, DEFAULT_NUM_MOVEMENT_ACTIONS)
    assert all(0 <= action < 1056 for action in selection.flat_actions)


def test_select_decentralized_actions_uses_local_obs_only():
    actor = RecordingActor()
    observations = np.zeros((2, 114), dtype=np.float32)
    observations[0, 0] = 0.25
    observations[1, 0] = 0.75
    select_decentralized_actions(actor, observations, use_movement_mask=False)
    assert len(actor.inputs) == 2
    assert actor.inputs[0].shape == (114,)
    assert actor.inputs[1].shape == (114,)
    assert actor.inputs[0].shape != (89,)
    assert actor.inputs[0][0] == 0.25
    assert actor.inputs[1][0] == 0.75


def test_select_decentralized_actions_applies_movement_mask():
    actor = RecordingActor(prefer_up=True)
    observations = np.zeros((1, 114), dtype=np.float32)
    observations[0, 2] = 1.0
    selection = select_decentralized_actions(actor, observations, use_movement_mask=True)
    assert selection.factorized_actions[0].movement != MOVE_UP


def test_collect_rollout_adds_transitions_to_buffer():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()
    buffer = CTDEReplayBuffer(capacity=8, seed=1)
    summary = collect_ctde_rollout(env, actor, buffer=buffer, max_steps=3, rng=np.random.default_rng(1))
    assert summary["transitions_collected"] == len(buffer)
    assert len(buffer) > 0
    batch = buffer.sample(1)
    assert batch["obs"].shape == (1, 2, 114)
    assert batch["state"].shape == (1, 89)
    assert batch["movement_actions"].shape == (1, 2)
    assert batch["next_obs"].shape == (1, 2, 114)
    assert batch["next_state"].shape == (1, 89)


def test_collect_rollout_summary_keys():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()
    summary = collect_ctde_rollout(env, actor, max_steps=2, rng=np.random.default_rng(1))
    assert {"num_steps", "episode_return", "transitions_collected"}.issubset(summary)


def test_collect_rollout_no_results_written():
    forbidden = {"save_path", "output_path", "results_dir", "checkpoint_path"}
    params = set(inspect.signature(collect_ctde_rollout).parameters)
    assert forbidden.isdisjoint(params)


def test_rollout_module_does_not_import_hierarchical():
    source = inspect.getsource(ctde_rollout)
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for item in forbidden)


def test_collect_rollout_does_not_use_env_action_mask_for_selection():
    env = UAVBackscatterEnv(CONFIG_3D)

    def fail_get_action_mask(_uav_id):
        raise AssertionError("env action mask must not be used for CTDE decentralized selection")

    env.get_action_mask = fail_get_action_mask
    actor = FactorizedActor()
    summary = collect_ctde_rollout(env, actor, max_steps=2, rng=np.random.default_rng(1))
    assert summary["num_steps"] == 2


def test_global_state_stored_but_not_used_for_actor_selection():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = RecordingActor()
    buffer = CTDEReplayBuffer(capacity=4, seed=1)
    collect_ctde_rollout(env, actor, buffer=buffer, max_steps=2, rng=np.random.default_rng(1))
    assert all(item.shape == (114,) for item in actor.inputs)
    batch = buffer.sample(1)
    assert batch["state"].shape == (1, 89)


def test_done_stored_as_bool():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()
    buffer = CTDEReplayBuffer(capacity=4, seed=1)
    collect_ctde_rollout(env, actor, buffer=buffer, max_steps=2, rng=np.random.default_rng(1))
    batch = buffer.sample(1)
    assert batch["done"].dtype == np.bool_


def test_existing_ctde_networks_and_actions_still_pass():
    actor = FactorizedActor()
    observations = np.zeros((2, 114), dtype=np.float32)
    selection = select_decentralized_actions(actor, observations)
    assert len(selection.flat_actions) == 2
