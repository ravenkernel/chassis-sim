"""Gymnasium 注册契约：三 id 可 make、check_env 通过、SyncVectorEnv 能跑 100 步。

动作一律从 env.action_space 采样：diff_drive 的动作空间正在被另一个执行者改成
归一化 [-1,1]^2，测试里写死 ±1.2 / ±omega_max 会在改动落地当天变成假红。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import registry
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import SyncVectorEnv

import chassis_sim  # noqa: F401 - import 副作用即注册
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.gym_registration import ENV_IDS, register_envs

ENV_ITEMS = sorted(ENV_IDS.items())
# 默认观测 = 最小集 odom+path+goal+act（act 段宽度随车型动作维）
GYM_OBS_DIMS = {
    "ChassisSim-Ackermann-v0": 128,
    "ChassisSim-DiffDrive-v0": 128,
    "ChassisSim-Mecanum-v0": 129,
}
N_VEC = 4
VEC_STEPS = 100


def test_registration_happens_on_import():
    for env_id in ENV_IDS:
        assert env_id in registry
        assert registry[env_id].entry_point == "chassis_sim.envs:ChassisSimEnv"


def test_register_envs_is_idempotent():
    # 模块 import 时已注册过 ⇒ 再调一次不应重复注册、也不该抛 AlreadyRegisteredError
    assert register_envs() == []
    assert register_envs() == []
    for env_id in ENV_IDS:
        assert env_id in registry


@pytest.mark.parametrize(("env_id", "chassis"), ENV_ITEMS)
def test_make_env_default_obs_dim(env_id, chassis):
    env = gym.make(env_id)
    try:
        assert env.observation_space.shape == (GYM_OBS_DIMS[env_id],)
        assert env.observation_space.dtype == np.float32
        obs, _ = env.reset(seed=0)
        assert obs.shape == (GYM_OBS_DIMS[env_id],)
    finally:
        env.close()


def test_make_env_forwards_obs_spec_kwarg():
    """gymnasium 把 make() 的多余关键字原样传给构造器 ⇒ 对象参数直接可用。"""
    from chassis_sim import (
        Act,
        Err,
        Goal,
        GoalSig,
        Lidar,
        LidarSegment,
        ObsSpec,
        Odom,
        PathProgress,
        PathSegment,
        PathValid,
    )

    spec = ObsSpec(
        (
            LidarSegment(
                name="lidar",
                sensors=("front", "rear"),
                ray_counts={"front": 180, "rear": 180},
            ),
            Odom(),
            PathSegment(points=30),
            Goal(),
            Act(action_dim=2),
            Err(),
            GoalSig(),
            PathValid(),
            PathProgress(),
        )
    )
    env = gym.make(
        "ChassisSim-Ackermann-v0",
        obs_spec=spec,
        sensors=[Lidar("front", n_rays=180), Lidar("rear", n_rays=180)],
    )
    try:
        assert env.observation_space.shape == (494,)
        assert env.unwrapped.obs_spec_describe()["lidar"] == slice(0, 360)
    finally:
        env.close()


@pytest.mark.parametrize(("env_id", "chassis"), ENV_ITEMS)
def test_make_env_passes_check_env_and_steps(env_id, chassis):
    env = gym.make(env_id)
    try:
        assert env.unwrapped.spec.id == env_id
        # 三 id → 三车型的映射：与外层 action_space 逐位一致（不写死物理界）
        ref = ChassisSimEnv(chassis)
        try:
            assert env.action_space.shape == ref.action_space.shape
            assert np.array_equal(env.action_space.low, ref.action_space.low)
            assert np.array_equal(env.action_space.high, ref.action_space.high)
            assert env.observation_space.shape == ref.observation_space.shape
            assert env.observation_space.dtype == ref.observation_space.dtype
        finally:
            ref.close()

        check_env(env.unwrapped, skip_render_check=True)

        obs, info = env.reset(seed=0)
        assert isinstance(info, dict)
        assert env.observation_space.contains(obs)

        low = env.action_space.low
        high = env.action_space.high
        for _ in range(100):
            action = env.action_space.sample()
            assert action.shape == env.action_space.shape
            assert np.all(action >= low) and np.all(action <= high)
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)
            assert "collision" in info and "goal_reached" in info
            if terminated or truncated:
                break
    finally:
        env.close()


def test_sync_vector_env_runs_100_steps():
    probe = gym.make("ChassisSim-Ackermann-v0")
    try:
        obs_dim = int(probe.observation_space.shape[0])
        act_dim = int(probe.action_space.shape[0])
    finally:
        probe.close()

    venv = SyncVectorEnv(
        [lambda: gym.make("ChassisSim-Ackermann-v0") for _ in range(N_VEC)]
    )
    try:
        obs, _ = venv.reset(seed=0)
        assert obs.shape == (N_VEC, obs_dim)
        for _ in range(VEC_STEPS):
            actions = venv.action_space.sample()
            assert actions.shape == (N_VEC, act_dim)
            obs, rewards, terminated, truncated, _ = venv.step(actions)
            assert obs.shape == (N_VEC, obs_dim)
            assert rewards.shape == (N_VEC,)
            assert terminated.shape == (N_VEC,)
            assert truncated.shape == (N_VEC,)
    finally:
        venv.close()
