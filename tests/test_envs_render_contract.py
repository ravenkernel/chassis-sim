"""render 契约（3.0.0 W1）：render_mode / metadata / rgb_array 帧。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import chassis_sim  # noqa: F401 - import 副作用即注册
from chassis_sim.envs import ChassisSimEnv


def test_metadata_declares_both_modes_and_fps():
    env = ChassisSimEnv("ackermann")
    try:
        assert set(env.metadata["render_modes"]) >= {"human", "rgb_array"}
        assert env.metadata["render_fps"] == max(1, int(round(1.0 / env.dt)))
        assert env.metadata["render_fps"] == 10
    finally:
        env.close()


def test_rgb_array_returns_uint8_hwc():
    env = ChassisSimEnv("ackermann", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.shape[0] > 0 and frame.shape[1] > 0
        assert frame.max() > 0  # 真的画了东西，不是全黑
    finally:
        env.close()


def test_render_without_mode_returns_none():
    env = ChassisSimEnv("ackermann")
    try:
        env.reset(seed=0)
        assert env.render() is None
    finally:
        env.close()


def test_human_mode_returns_none_under_agg():
    env = ChassisSimEnv("ackermann", render_mode="human")
    try:
        env.reset(seed=0)
        assert env.render() is None
        assert env._render_fig is not None  # 图仍然画出来（可 savefig）
    finally:
        env.close()


def test_unknown_render_mode_is_rejected():
    with pytest.raises(ValueError, match="render_mode"):
        ChassisSimEnv("ackermann", render_mode="ansi")


def test_gym_make_gives_rgb_array_frames():
    env = gym.make("ChassisSim-Ackermann-v1", render_mode="rgb_array")
    try:
        assert env.metadata["render_fps"] == 10
        env.reset(seed=0)
        frame = env.render()
        assert frame.dtype == np.uint8 and frame.shape[2] == 3
    finally:
        env.close()
