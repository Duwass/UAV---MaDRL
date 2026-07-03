from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from envs.uav_backscatter_env import UAVBackscatterEnv
from marl.ctde.evaluation import evaluate_decentralized_policy
from marl.ctde.networks import FactorizedActor
from marl.ctde.utils import MODE_HARVEST, encode_factorized_action
from marl.ctde import evaluation as ctde_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"


class FixedLogitActor:
    def __init__(self):
        self.inputs: list[np.ndarray] = []

    def __call__(self, obs: torch.Tensor):
        obs_np = obs.detach().cpu().numpy()
        self.inputs.append(obs_np.copy())
        movement = torch.zeros(1, 11)
        movement[0, 3] = 10.0
        target = torch.zeros(1, 16)
        target[0, 1] = 10.0
        mode = torch.zeros(1, 6)
        mode[0, MODE_HARVEST] = 10.0
        return {
            "movement_logits": movement,
            "target_logits": target,
            "mode_logits": mode,
        }


class NoGlobalStateProxy:
    def __init__(self, base_env):
        self.base_env = base_env
        self.max_steps = base_env.max_steps

    def reset(self):
        return self.base_env.reset()

    def step(self, actions):
        return self.base_env.step(actions)

    def get_global_state(self):
        raise AssertionError("evaluation must not call env.get_global_state")


def test_evaluate_decentralized_policy_signature_has_no_critic_or_global_state():
    forbidden = {"critic", "state", "global_state", "executor"}
    params = set(inspect.signature(evaluate_decentralized_policy).parameters)
    assert forbidden.isdisjoint(params)


def test_evaluate_decentralized_policy_runs_on_3d_env():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()
    summary = evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2)
    assert "mean_return" in summary
    assert summary["total_steps"] == 2


def test_evaluate_uses_greedy_epsilon_zero_by_default():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FixedLogitActor()
    original_step = env.step
    seen_actions: list[list[int]] = []

    def record_step(actions):
        seen_actions.append(list(actions))
        return original_step(actions)

    env.step = record_step
    evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=1)
    expected = encode_factorized_action(3, 1, MODE_HARVEST)
    assert seen_actions == [[expected, expected]]


def test_evaluate_does_not_call_env_get_global_state():
    env = NoGlobalStateProxy(UAVBackscatterEnv(CONFIG_3D))
    actor = FactorizedActor()
    summary = evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2)
    assert summary["total_steps"] == 2


def test_evaluate_does_not_call_env_get_action_mask():
    env = UAVBackscatterEnv(CONFIG_3D)

    def fail_get_action_mask(_uav_id):
        raise AssertionError("evaluation must not call env.get_action_mask")

    env.get_action_mask = fail_get_action_mask
    actor = FactorizedActor()
    summary = evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2)
    assert summary["total_steps"] == 2


def test_evaluate_does_not_use_critic():
    params = set(inspect.signature(evaluate_decentralized_policy).parameters)
    assert "critic" not in params


def test_evaluate_does_not_import_hierarchical():
    source = inspect.getsource(ctde_evaluation)
    forbidden = ["HierarchicalActionExecutor", "HierarchicalUAVBackscatterEnv", "envs.hierarchical"]
    assert all(item not in source for item in forbidden)


def test_actor_receives_local_obs_only_during_evaluation():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FixedLogitActor()
    evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2)
    assert len(actor.inputs) == 4
    assert all(item.shape == (114,) for item in actor.inputs)
    assert all(item.shape != (89,) for item in actor.inputs)


def test_evaluate_summary_includes_3d_metrics_when_present():
    env = UAVBackscatterEnv(CONFIG_3D)
    actor = FactorizedActor()
    summary = evaluate_decentralized_policy(env, actor, num_episodes=1, max_steps=2)
    assert isinstance(summary["last_episode_metrics"], dict)
    assert isinstance(summary["last_frame_metrics"], dict)
    assert "avg_uav_altitude" in summary["last_frame_metrics"]


def test_evaluate_no_results_written():
    forbidden = {"save_path", "output_path", "results_dir", "checkpoint_path"}
    params = set(inspect.signature(evaluate_decentralized_policy).parameters)
    assert forbidden.isdisjoint(params)
    source = inspect.getsource(ctde_evaluation)
    assert ".to_csv" not in source
    assert "torch.save" not in source
