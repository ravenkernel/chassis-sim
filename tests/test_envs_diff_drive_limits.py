"""差速车型「限幅/闸门」活路径：物理量 → 归一化动作的往返一致性。

W15 之后 diff_drive 的 action_space 是 Box(2, [-1,-1], [1,1])，
command() 里 `v = a0·v_max`（a0<0 时 ·(−v_min)）、`ω = a1·omega_max`。
`ChassisSimEnv._encode_physical` 的差速分支因此必须做逆映射；旧实现直接
透传物理量，在 v_max != 1.0 时会把物理量当成归一化动作再乘一次上限。

该分支只经 `_apply_limits_and_gate` 可达（accel_max / steer_rate_max > 0，
或注入 safety_gate），默认全关 ⇒ 由本文件覆盖。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.chassis import make_chassis
from chassis_sim.envs.chassis_env import ChassisSimEnv
from chassis_sim.envs.config import EnvConfig

V_MAX = 1.2
OMEGA_MAX = 0.8
ACCEL_MAX = 0.001  # m/s² → 单步 Δv 上限 1e-4
STEER_RATE_MAX = 5.0
# 大加速度上限：只为打开 _apply_limits_and_gate，本身不产生限幅
# （单步上限 1000×0.1 = 100 m/s，远大于任何动作）。
ACCEL_UNLIMITED = 1000.0
CHASSIS_NAMES = ("ackermann", "diff_drive", "mecanum")


def _make_diff_drive(accel_max: float = ACCEL_UNLIMITED) -> ChassisSimEnv:
    """v_max=1.2 / omega_max=0.8 的差速 env，且强制走 _apply_limits_and_gate。

    注意：ChassisSimEnv 不收 v_max / omega_max 关键字 —— 车型名路径只把
    EnvConfig 里的 wheelbase / v_max 转发给底盘，而 omega_max 根本不在
    EnvConfig 里。要非默认的 omega_max 只能注入底盘实例（cfg 仍给同一个
    v_max，避免两处口径不一致）。
    """
    chassis = make_chassis("diff_drive", v_max=V_MAX, omega_max=OMEGA_MAX)
    return ChassisSimEnv(chassis, cfg=EnvConfig(v_max=V_MAX, accel_max=accel_max))


def test_diff_drive_full_speed_roundtrip():
    """a=[1,0] ⇒ 物理 1.2 m/s（不是被 clip 回 1.0），mapped_action 速度分量 = 1.2。

    反向对照（旧恒等实现，实测）：旧分支 act[0] = v = 1.2，随后
    :791 的 `np.clip(act, action_space.low, action_space.high)` 把它压成 1.0，
    送进 command() 又乘 v_max 得回 1.2 —— **这一档旧实现同样通过**，单靠它
    抓不到过时分支。真正有分辨力的是下面的 0.5 档。
    """
    env = _make_diff_drive()
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        v_actual = float(env.bridge.get_odometry()[3])
        assert v_actual == pytest.approx(V_MAX, abs=1e-6), f"实际线速度 {v_actual!r}"
        # mapped_action 口径 = [转向标度, v]；速度在第 1 维（diff_drive 无第 3 维）
        assert float(info["mapped_action"][1]) == pytest.approx(V_MAX, abs=1e-6)
    finally:
        env.close()


def test_diff_drive_mid_speed_roundtrip_rejects_identity_passthrough():
    """a=[0.5, 0.25] ⇒ 物理 (0.6 m/s, 0.2 rad/s)；旧恒等实现给 (0.72, 0.16)。

    反向对照（旧实现的实际结果，改前实测 odom_v=0.72 / odom_w=0.16）：旧分支
    把物理量 [0.6, 0.2] 当成归一化动作喂回 command() ⇒
    v = 0.6×1.2 = 0.72、ω = 0.2×0.8 = 0.16。两个值都在 ±1 之内，
    `_encode_physical` 末尾的 clip 救不回来 ⇒ 下面三条断言在旧实现下
    全部以 ~20% 的偏差失败。这就是本用例存在的理由。
    """
    env = _make_diff_drive()
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.5, 0.25], dtype=np.float32))
        odom = env.bridge.get_odometry()
        v_actual, omega_actual = float(odom[3]), float(odom[4])
        assert v_actual == pytest.approx(0.6, abs=1e-6), f"实际线速度 {v_actual!r}"
        assert omega_actual == pytest.approx(0.2, abs=1e-6), (
            f"实际角速度 {omega_actual!r}"
        )
        assert float(info["mapped_action"][1]) == pytest.approx(0.6, abs=1e-6)
    finally:
        env.close()


def test_diff_drive_accel_limit_roundtrip():
    """限幅后的物理量也要原样往返：首步 v = accel_max·dt = 1e-4 m/s。

    旧实现把 1e-4 当归一化动作 ⇒ 1.2e-4（偏 20%），本用例同时守住
    「限幅确实生效」与「编码不放大」两件事。
    """
    env = _make_diff_drive(accel_max=ACCEL_MAX)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        expected = ACCEL_MAX * env.dt
        v_actual = float(env.bridge.get_odometry()[3])
        assert v_actual == pytest.approx(expected, abs=1e-9), (
            f"实际线速度 {v_actual!r}，期望 {expected!r}"
        )
        assert float(info["mapped_action"][1]) == pytest.approx(expected, abs=1e-9)
    finally:
        env.close()


def test_default_config_diff_drive_skips_reencode():
    """默认 accel_max=0 / steer_rate_max=0 ⇒ 不进 _apply_limits_and_gate。

    动作原样送桥（与主仓直通逐位一致）：a=[0.5, 0] ⇒ v = 0.5×1.2 = 0.6。
    """
    env = ChassisSimEnv("diff_drive", cfg=EnvConfig())
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.5, 0.0], dtype=np.float32))
        v_actual = float(env.bridge.get_odometry()[3])
        assert v_actual == pytest.approx(0.6, abs=1e-6)
        assert float(info["mapped_action"][1]) == pytest.approx(0.6, abs=1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("chassis_name", CHASSIS_NAMES)
def test_limits_path_survives_on_all_chassis(chassis_name):
    """三个车型在 accel_max / steer_rate_max 打开时跑 20 步都不炸。"""
    env = ChassisSimEnv(
        chassis_name,
        cfg=EnvConfig(accel_max=ACCEL_UNLIMITED, steer_rate_max=STEER_RATE_MAX),
    )
    rng = np.random.default_rng(7)
    try:
        obs, _ = env.reset(seed=0)
        for step in range(20):
            action = rng.uniform(-0.5, 0.5, size=env.action_dim).astype(np.float32)
            obs, _, term, trunc, info = env.step(action)
            assert np.all(np.isfinite(obs)), f"{chassis_name} step {step} obs 非有限"
            assert np.all(np.isfinite(np.asarray(info["mapped_action"])))
            if term or trunc:
                obs, _ = env.reset(seed=0)
    finally:
        env.close()


@pytest.mark.parametrize("chassis_name", CHASSIS_NAMES)
def test_default_config_survives_on_all_chassis(chassis_name):
    """默认配置（accel_max=0 / steer_rate_max=0）下三车型各跑 20 步无异常。"""
    env = ChassisSimEnv(chassis_name, cfg=EnvConfig())
    rng = np.random.default_rng(11)
    try:
        obs, _ = env.reset(seed=0)
        for step in range(20):
            action = rng.uniform(-0.5, 0.5, size=env.action_dim).astype(np.float32)
            obs, _, term, trunc, _ = env.step(action)
            assert np.all(np.isfinite(obs)), f"{chassis_name} step {step} obs 非有限"
            if term or trunc:
                obs, _ = env.reset(seed=0)
    finally:
        env.close()
