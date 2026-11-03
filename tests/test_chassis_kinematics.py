"""闭式解运动学断言：阿克曼圆周半径、差速原地旋转、麦轮横移与 45° 螃蟹走位。"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.chassis import (
    AckermannChassis,
    Chassis,
    DiffDriveChassis,
    MecanumChassis,
)


def _ang_diff(a: float, b: float) -> float:
    """a − b 折到 [-π, π]。"""
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


# ══════════════════════════════════════════════════════════════════════
# 统一积分核只能有一份实现
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("cls", [AckermannChassis, DiffDriveChassis, MecanumChassis])
def test_integrate_is_not_overridden(cls):
    assert cls.integrate is Chassis.integrate


# ══════════════════════════════════════════════════════════════════════
# 阿克曼：固定 δ 走一圈，半径 = L / tan(δ)
# ══════════════════════════════════════════════════════════════════════

L, DELTA, V = 0.145, 0.3, 0.5
N_STEPS = 100_000


def test_ackermann_fixed_delta_traces_circle_of_radius_l_over_tan_delta():
    """固定 δ 走整圈：Euler 轨迹是正 N 边形，外接圆半径 → L/tan(δ)。"""
    ch = AckermannChassis(delta_max=0.526, curvature_action=False)
    # curvature_action=False ⇒ δ = a[0]·delta_max，可直接摆物理转向角
    cmd = ch.command(np.array([DELTA / ch.delta_max, V / ch.v_max]))
    assert cmd.v == pytest.approx(V, rel=1e-15)

    omega = V / L * np.tan(DELTA)
    # 取 ω·dt = 2π/N：整圈恰好 N 步，位移方向按 N 等分旋转 ⇒ 轨迹严格闭合
    dt = 2.0 * np.pi / (N_STEPS * omega)
    x = y = theta = 0.0
    pts = np.empty((N_STEPS, 2), dtype=np.float64)
    for i in range(N_STEPS):
        x, y, theta = ch.integrate(x, y, theta, cmd, dt)
        pts[i] = (x, y)

    centre = pts.mean(axis=0)
    radii = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
    step_angle = cmd.omega * dt
    # Euler 折线的每段弦长是 v·dt（不是圆弧长 R·θ），故正 N 边形外接圆半径
    #   r_poly = (v·dt/2) / sin(ω·dt/2) = R·(θ/2)/sin(θ/2)，θ = ω·dt
    r_poly = (cmd.v * dt / 2.0) / np.sin(step_angle / 2.0)
    r_target = L / np.tan(DELTA)  # 规格要求的连续圆闭式解
    r_exact = r_target * (step_angle / 2.0) / np.sin(step_angle / 2.0)

    # 1e5 步 float64 累加噪声实测：std/r_poly ≈ 1.8e-12、mean 差 ≈ 7e-13、闭合 ≈ 3.8e-12
    assert radii.std() / r_poly < 1e-11, f"多边形顶点半径不齐: std={radii.std():.3e}"
    assert abs(radii.mean() - r_poly) / r_poly < 1e-11
    # 与"弦长取成弦而非弧"的闭式解逐位吻合（差 0 才说明积分核没有额外误差）
    assert abs(r_poly - r_exact) / r_exact < 1e-12
    # 规格：半径 = L/tan(δ) 相对误差 < 1e-9（θ²/24 = 1.6e-10 已含在内）
    assert abs(radii.mean() - r_target) / r_target < 1e-9, (
        f"半径 {radii.mean():.15f} vs L/tan(δ)={r_target:.15f}"
    )
    # 一整圈后回到起点
    assert np.hypot(x, y) / r_target < 1e-10, f"闭合误差 {np.hypot(x, y):.3e}"


def test_ackermann_curvature_action_radius_is_inverse_curvature():
    """曲率参数化：|a|=1 ⇒ δ = arctan(κ_max·L) ⇒ R = L/tan δ = 1/κ_max。"""
    ch = AckermannChassis(curvature_action=True, kappa_max=4.0, v_max=1.0)
    cmd = ch.command(np.array([1.0, 1.0]))
    assert cmd.v == 1.0
    radius = cmd.v / abs(cmd.omega)
    assert abs(radius - 1.0 / 4.0) / (1.0 / 4.0) < 1e-12


# ══════════════════════════════════════════════════════════════════════
# 差速：原地旋转
# ══════════════════════════════════════════════════════════════════════


def test_diff_drive_spin_in_place_keeps_xy_and_accumulates_theta():
    ch = DiffDriveChassis(v_max=1.0, omega_max=1.0)
    cmd = ch.command(np.array([0.0, 1.0]))
    assert (cmd.v, cmd.omega, cmd.vy) == (0.0, 1.0, 0.0)

    dt, n = 0.1, 50
    x = y = theta = 0.0
    for _ in range(n):
        x, y, theta = ch.integrate(x, y, theta, cmd, dt)
        assert x == 0.0 and y == 0.0  # v=0, vy=0 ⇒ 逐位不变
        assert theta > 0.0  # θ 单调累加
    assert theta == pytest.approx(n * cmd.omega * dt, rel=1e-15)


# ══════════════════════════════════════════════════════════════════════
# 麦轮：横移与螃蟹走位
# ══════════════════════════════════════════════════════════════════════


def test_mecanum_pure_lateral_motion_moves_along_plus_y_only():
    """归一化动作 a = [0, 1, 0] ⇒ vy = vy_max = 0.5，每步沿 +y 走 0.5·dt。"""
    ch = MecanumChassis()
    cmd = ch.command(np.array([0.0, 1.0, 0.0]))
    assert (cmd.v, cmd.vy, cmd.omega) == (0.0, 0.5, 0.0)
    assert cmd.vy == ch.vy_max == 0.5

    dt, n = 0.1, 10
    x = y = theta = 0.0
    for _ in range(n):
        x, y, theta = ch.integrate(x, y, theta, cmd, dt)
    assert x == 0.0 and theta == 0.0
    assert y == pytest.approx(n * cmd.vy * dt, rel=1e-15)
    assert y > 0.0


def test_mecanum_crab_45deg_without_rotation():
    """vx = vy > 0、ω = 0 ⇒ 沿 +45° 直线斜行。"""
    ch = MecanumChassis()
    # 归一化动作：a[0]=0.5 ⇒ v=0.5·v_max=0.5；a[1]=1.0 ⇒ vy=vy_max=0.5
    cmd = ch.command(np.array([0.5, 1.0, 0.0]))
    assert (cmd.v, cmd.vy) == (0.5, 0.5)

    dt, n = 0.05, 40
    x = y = theta = 0.0
    for _ in range(n):
        x, y, theta = ch.integrate(x, y, theta, cmd, dt)

    assert theta == 0.0
    assert np.arctan2(y, x) == pytest.approx(np.pi / 4.0, abs=1e-15)
    assert np.hypot(x, y) == pytest.approx(n * np.hypot(cmd.v, cmd.vy) * dt, rel=1e-15)


def test_mecanum_crab_with_rotation_keeps_body_frame_velocity():
    """(vx, vy, ω) 恒定 ⇒ 每步位移 = 车体系速度矢量旋转 θ。"""
    ch = MecanumChassis()
    # 目标物理量 (v, vy, ω) = (0.6, 0.3, 1.0) ⇒ 归一化动作 [0.6/1.0, 0.3/0.5, 1.0]
    cmd = ch.command(np.array([0.6, 0.6, 1.0]))
    assert (cmd.v, cmd.vy, cmd.omega) == (0.6, 0.3, 1.0)
    body_angle = np.arctan2(cmd.vy, cmd.v)
    speed = np.hypot(cmd.v, cmd.vy)

    dt, n = 0.05, 200
    x = y = theta = 0.0
    for _ in range(n):
        nx, ny, ntheta = ch.integrate(x, y, theta, cmd, dt)
        dx, dy = nx - x, ny - y
        assert np.hypot(dx, dy) == pytest.approx(speed * dt, rel=1e-12)
        assert _ang_diff(np.arctan2(dy, dx), theta + body_angle) == pytest.approx(
            0.0, abs=1e-12
        )
        x, y, theta = nx, ny, ntheta

    assert theta == pytest.approx(n * cmd.omega * dt, rel=1e-15)
