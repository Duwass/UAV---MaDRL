from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from envs.channel_model import distance_3d
from envs.hierarchical_env import HierarchicalUAVBackscatterEnv
from envs.uav_backscatter_env import MODE_IDLE, UAVBackscatterEnv, encode_action
from marl.qmix.networks import AgentQNetwork, QMixer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_3D = PROJECT_ROOT / "configs" / "scenario_4_3d_backscatter_types_calibrated.yaml"
CONFIG_2D = PROJECT_ROOT / "configs" / "scenario_4_backscatter_types_calibrated.yaml"


def _idle_action(env: UAVBackscatterEnv) -> int:
    return encode_action(0, 0, MODE_IDLE, env.num_iot, env.num_movement_actions)


def test_3d_observation_contains_altitude_features():
    env = UAVBackscatterEnv(CONFIG_3D)
    obs, _ = env.reset(seed=123)
    single = env.get_local_observation(0)
    uav = env.uavs[0]
    jammer = env.jammers[0]
    iot = env.iot_devices[0]
    first_iot_offset = 9

    assert obs.shape == (env.num_uav, env.get_obs_dim())
    assert single.shape == (env.get_obs_dim(),)
    assert single[2] == pytest.approx(env._normalize_z(uav.z))
    assert single[6] == pytest.approx(env._normalize_dz(jammer.z - uav.z))
    assert single[7] == pytest.approx(distance_3d(uav, jammer) / env._max_3d_distance())
    assert single[first_iot_offset + 2] == pytest.approx(env._normalize_dz(iot.z - uav.z))
    assert single[first_iot_offset + 3] == pytest.approx(distance_3d(uav, iot) / env._max_3d_distance())
    assert single[first_iot_offset + 6] == float(env._in_coverage(uav, iot))


def test_3d_global_state_contains_3d_positions():
    env = UAVBackscatterEnv(CONFIG_3D)
    _, info = env.reset(seed=123)
    state = info["global_state"]
    first_iot_offset = env.num_uav * 4
    first_jammer_offset = first_iot_offset + env.num_iot * 5

    assert state.shape == (env.get_state_dim(),)
    assert state[2] == pytest.approx(env._normalize_z(env.uavs[0].z))
    assert state[first_iot_offset + 2] == pytest.approx(env._normalize_z(env.iot_devices[0].z))
    assert state[first_jammer_offset + 2] == pytest.approx(env._normalize_z(env.jammers[0].z))


def test_3d_obs_dim_expected_or_reported():
    env = UAVBackscatterEnv(CONFIG_3D)

    assert env.get_obs_dim() == 9 + 15 * 7
    assert env.observation_dim == 114


def test_3d_state_dim_expected_or_reported():
    env = UAVBackscatterEnv(CONFIG_3D)

    assert env.get_state_dim() == 2 * 4 + 15 * 5 + 1 * 4 + 2
    assert env.global_state_dim == 89


def test_legacy_obs_state_dims_unchanged():
    env = UAVBackscatterEnv(CONFIG_2D)

    assert env.get_obs_dim() == 97
    assert env.get_state_dim() == 71
    assert env.get_action_dim() == 864


def test_obs_changes_when_uav_altitude_changes():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)
    before = env.get_local_observation(0)

    env.uavs[0].z += env.uav_vertical_step
    after = env.get_local_observation(0)

    assert after[2] != pytest.approx(before[2])
    assert not np.allclose(before, after)


def test_state_changes_when_jammer_altitude_changes():
    env = UAVBackscatterEnv(CONFIG_3D)
    env.reset(seed=123)
    before = env.get_global_state()

    env.jammers[0].z += env.uav_vertical_step
    after = env.get_global_state()

    assert not np.allclose(before, after)


def test_3d_env_random_step_obs_state_shape_consistent():
    env = UAVBackscatterEnv(CONFIG_3D)
    rng = np.random.default_rng(7)
    obs, info = env.reset(seed=123)
    actions = [int(rng.integers(0, env.get_action_dim())) for _ in range(env.num_uav)]
    next_obs, reward, terminated, truncated, step_info = env.step(actions)

    assert obs.shape == (env.num_uav, env.get_obs_dim())
    assert next_obs.shape == (env.num_uav, env.get_obs_dim())
    assert info["global_state"].shape == (env.get_state_dim(),)
    assert step_info["global_state"].shape == (env.get_state_dim(),)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_hierarchical_env_3d_shape_no_crash():
    base_env = UAVBackscatterEnv(CONFIG_3D)
    env = HierarchicalUAVBackscatterEnv(base_env)
    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, step_info = env.step([0 for _ in range(env.num_uav)])

    assert env.action_size == 10
    assert obs.shape == (env.num_uav, env.get_obs_dim())
    assert next_obs.shape == (env.num_uav, env.get_obs_dim())
    assert info["global_state"].shape == (env.get_state_dim(),)
    assert step_info["global_state"].shape == (env.get_state_dim(),)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_qmix_network_init_with_3d_dims_no_crash():
    env = UAVBackscatterEnv(CONFIG_3D)
    obs, info = env.reset(seed=123)
    agent_net = AgentQNetwork(env.get_obs_dim(), env.get_action_dim(), env.num_uav, hidden_sizes=[16], use_agent_id=True)
    mixer = QMixer(env.num_uav, env.get_state_dim(), mixing_embed_dim=8, hypernet_hidden_dim=16)

    q_values = agent_net(torch.as_tensor(obs, dtype=torch.float32), torch.arange(env.num_uav))
    q_tot = mixer(
        torch.zeros(1, 1, env.num_uav, dtype=torch.float32),
        torch.as_tensor(info["global_state"], dtype=torch.float32).reshape(1, 1, env.get_state_dim()),
    )

    assert q_values.shape == (env.num_uav, env.get_action_dim())
    assert q_tot.shape == (1, 1)
