"""env 的观测空间契约（3.0.0 W1）：有限 Box、越界计数、200 步不越界。

这里只断言**契约**（finite / contains / 计数语义），具体界的推导在
tests/test_obs_segment_bounds.py。
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

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
from chassis_sim.envs import ChassisSimEnv

KINDS = ("ackermann", "diff_drive", "mecanum")
STEPS = 200

FULL_SPEC = ObsSpec(
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


@pytest.mark.parametrize("kind", KINDS)
def test_observation_space_is_finite(kind):
    env = ChassisSimEnv(kind)
    try:
        low = env.observation_space.low
        high = env.observation_space.high
        assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))
        assert np.all(low < high)
    finally:
        env.close()


@pytest.mark.parametrize("kind", KINDS)
def test_check_env_reports_no_infinity_warning(kind, recwarn):
    env = ChassisSimEnv(kind)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    assert not [w for w in recwarn if "infinity" in str(w.message)]


@pytest.mark.parametrize("kind", KINDS)
def test_200_random_steps_stay_inside_the_space(kind):
    env = ChassisSimEnv(kind)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert "obs_clipped" not in info  # 合法回合不新增键
        rng = np.random.default_rng(0)
        for _ in range(STEPS):
            action = rng.uniform(env.action_space.low, env.action_space.high)
            obs, _r, terminated, truncated, info = env.step(action.astype(np.float32))
            assert env.observation_space.contains(obs), "obs 越界：界推导本身错了"
            assert "obs_clipped" not in info
            if terminated or truncated:
                obs, info = env.reset()
                assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_full_494_dim_spec_has_finite_bounds_and_stays_inside():
    env = ChassisSimEnv(
        "ackermann",
        obs_spec=FULL_SPEC,
        sensors=[Lidar("front", n_rays=180), Lidar("rear", n_rays=180)],
    )
    try:
        assert env.observation_space.shape == (494,)
        assert np.all(np.isfinite(env.observation_space.low))
        assert np.all(np.isfinite(env.observation_space.high))
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        rng = np.random.default_rng(0)
        for _ in range(STEPS):
            obs, _r, terminated, truncated, _info = env.step(
                rng.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
            )
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


def test_clip_is_a_noop_for_legal_obs():
    env = ChassisSimEnv("ackermann")
    try:
        obs, _ = env.reset(seed=0)
        assert env._obs_clipped == 0
        assert env._clip_obs(obs) is obs  # 零拷贝：合法观测不复制
        assert env._obs_clipped == 0
    finally:
        env.close()


def test_clip_counts_only_real_violations_and_clips_them():
    env = ChassisSimEnv("ackermann")
    try:
        obs, _ = env.reset(seed=0)
        bad = obs.copy()
        bad[0] = env.observation_space.high[0] * 2.0
        bad[1] = env.observation_space.low[1] - 1.0
        clipped = env._clip_obs(bad)
        assert env._obs_clipped == 2
        assert env.observation_space.contains(clipped)
        assert obs[0] != clipped[0]
    finally:
        env.close()


def test_clip_leaves_nan_alone():
    env = ChassisSimEnv("ackermann")
    try:
        obs, _ = env.reset(seed=0)
        bad = obs.copy()
        bad[2] = np.nan
        out = env._clip_obs(bad)
        assert np.isnan(out[2])  # NaN 是上游缺陷的信号，不夹也不计数
        assert env._obs_clipped == 0
    finally:
        env.close()


def test_info_reports_clipped_only_when_nonzero():
    env = ChassisSimEnv("ackermann")
    try:
        env.reset(seed=0)
        env._obs_clipped = 5
        _obs, _r, _term, _trunc, info = env.step(
            np.zeros(env.action_dim, dtype=np.float32)
        )
        assert info["obs_clipped"] >= 5
    finally:
        env.close()


def test_privileged_tail_is_inside_zero_one():
    from chassis_sim.config import EnvConfig

    cfg = EnvConfig(privileged_enabled=True, privileged_bins=8)
    env = ChassisSimEnv("ackermann", cfg=cfg)
    try:
        low = env.observation_space.low
        high = env.observation_space.high
        tail_low = low[-8:]
        tail_high = high[-8:]
        assert np.all(tail_low == 0.0) and np.all(tail_high == 1.0)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
    finally:
        env.close()
