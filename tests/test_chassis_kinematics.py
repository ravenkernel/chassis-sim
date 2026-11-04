"""运动学断言（**闭式**积分 + 轮级观察层）：阿克曼圆周半径、差速原地旋转、
麦轮横移与螃蟹走位，以及三种底盘的轮级运动学数值例。

2026-11-05 起 integrate() 是闭式解（恒定 v/vy/ω 下对任意 dt 精确到浮点噪声），
原先按 Euler 折线几何写的两条断言（正 N 边形外接圆半径 r_poly = R·(θ/2)/sin(θ/2)、
"每步位移 = 弦长 = speed·dt"）已随实现一起改写：现在断言的是**圆弧**几何。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chassis_sim.chassis import (
    AckermannChassis,
    Chassis,
    ChassisCommand,
    DiffDriveChassis,
    MecanumChassis,
    WheelKinematics,
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
N_STEPS = 20_000


def test_ackermann_fixed_delta_traces_circle_of_radius_l_over_tan_delta():
    """固定 δ 走整圈：闭式解的轨迹是**同心圆**，半径逐点等于 L/tan(δ)。

    Euler 时代这条轨迹是正 N 边形，外接圆半径 = R·(θ/2)/sin(θ/2)（θ = ω·dt），
    与连续圆的相对差 θ²/24；闭式解没有这一项，因此下面能直接用 1e-11 的相对容差
    对 R 本身断言，而不是对多边形半径断言。
    """
    ch = AckermannChassis(delta_max=0.526, curvature_action=False)
    # curvature_action=False ⇒ δ = a[0]·delta_max，可直接摆物理转向角
    cmd = ch.command(np.array([DELTA / ch.delta_max, V / ch.v_max]))
    assert cmd.v == pytest.approx(V, rel=1e-15)

    omega = V / L * np.tan(DELTA)
    assert cmd.omega == pytest.approx(omega, rel=1e-15)
    # 取 ω·dt = 2π/N：整圈恰好 N 步
    dt = 2.0 * np.pi / (N_STEPS * omega)
    x = y = theta = 0.0
    pts = np.empty((N_STEPS, 2), dtype=np.float64)
    for i in range(N_STEPS):
        x, y, theta = ch.integrate(x, y, theta, cmd, dt)
        pts[i] = (x, y)

    centre = pts.mean(axis=0)
    radii = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
    r_target = L / np.tan(DELTA)  # 连续圆半径 = L/tan(δ)

    assert radii.std() / r_target < 1e-11, f"顶点半径不齐: std={radii.std():.3e}"
    assert abs(radii.mean() - r_target) / r_target < 1e-11, (
        f"半径 {radii.mean():.15f} vs L/tan(δ)={r_target:.15f}"
    )
    # 圆心在起点法向 R 处（+y 方向，ω>0 左转）
    assert abs(centre[1] - r_target) / r_target < 1e-11
    # 一整圈后回到起点（Euler 时代是 r_poly 级的多边形闭合，现在只剩累加噪声）
    assert np.hypot(x, y) / r_target < 1e-11, f"闭合误差 {np.hypot(x, y):.3e}"


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


def test_mecanum_crab_with_rotation_follows_arc_not_chord():
    """(vx, vy, ω) 恒定 ⇒ 轨迹是半径 R = |u|/ω 的圆弧，每步位移是**弦**。

    闭式积分下每步位移 = 2R·sin(φ/2)（φ = ω·dt），方向 = θ + body_angle + φ/2。
    Euler 时代它是 u·dt（弦长恒等于速度×dt），这正是旧实现在圆周上偏 5e-2 m
    的同一个误差源，故这条断言必须跟着实现改写。
    """
    ch = MecanumChassis()
    # 目标物理量 (v, vy, ω) = (0.6, 0.3, 1.0) ⇒ 归一化动作 [0.6/1.0, 0.3/0.5, 1.0]
    cmd = ch.command(np.array([0.6, 0.6, 1.0]))
    assert (cmd.v, cmd.vy, cmd.omega) == (0.6, 0.3, 1.0)
    body_angle = np.arctan2(cmd.vy, cmd.v)
    speed = np.hypot(cmd.v, cmd.vy)
    dt = 0.05
    phi = cmd.omega * dt
    radius = speed / abs(cmd.omega)
    chord = 2.0 * radius * np.sin(abs(phi) / 2.0)

    n = 200
    x = y = theta = 0.0
    for _ in range(n):
        nx, ny, ntheta = ch.integrate(x, y, theta, cmd, dt)
        dx, dy = nx - x, ny - y
        assert np.hypot(dx, dy) == pytest.approx(chord, rel=1e-12)
        # 车体系速度在**区间中点**的朝向上（弦方向 = 中点切线方向）
        assert _ang_diff(np.arctan2(dy, dx), theta + body_angle + phi / 2.0) == (
            pytest.approx(0.0, abs=1e-12)
        )
        x, y, theta = nx, ny, ntheta

    assert theta == pytest.approx(n * phi, rel=1e-15)
    # 200 步 = 50 rad ≈ 7.96 圈：轨迹始终在半径 R 的圆上
    cx, cy = -radius * np.sin(body_angle), radius * np.cos(body_angle)
    assert abs(np.hypot(x - cx, y - cy) - radius) / radius < 1e-12


# ══════════════════════════════════════════════════════════════════════
# 轮级运动学层（**只读**：不参与位姿积分）
# ══════════════════════════════════════════════════════════════════════


def test_base_chassis_wheel_kinematics_is_none_not_empty_dict():
    """不支持轮级量的底盘必须**显式**返回 None（不做空 dict 的静默降级）。"""

    class _Bare(Chassis):
        name = "bare"
        action_dim = 1

        def action_space(self):
            raise NotImplementedError

        def command(self, action):
            return ChassisCommand(v=float(np.asarray(action)[0]), omega=0.0)

    bare = _Bare()
    assert bare.wheel_kinematics(ChassisCommand(v=1.0, omega=0.5)) is None


def test_wheel_kinematics_is_read_only():
    """调 wheel_kinematics 不得改动任何状态：执行器相位与之后的积分轨迹都不变。"""
    cmds = [
        ChassisCommand(v=0.5, omega=0.3),
        ChassisCommand(v=-0.4, omega=-0.2, vy=0.1),
        ChassisCommand(v=0.2, omega=0.9, vy=-0.3),
    ]
    for cls in (AckermannChassis, DiffDriveChassis, MecanumChassis):
        ch = cls()
        act = np.zeros((ch.action_dim,), dtype=np.float64)
        act[1] = 0.4
        ch.step(act)  # 让阿克曼建立执行器滞后状态
        phase = ch.steering_angle()
        before = [ch.wheel_kinematics(cmd) for cmd in cmds]
        assert ch.steering_angle() == phase
        after = [ch.wheel_kinematics(cmd) for cmd in cmds]
        assert before == after  # 纯函数：同输入逐位同输出
        x = y = theta = 0.0
        for cmd in cmds:
            x, y, theta = ch.integrate(x, y, theta, cmd, 0.1)
        assert math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)


# ── 阿克曼：左右前轮转角（R = L/tanδ，内轮角 > 外轮角）────────────────


def test_ackermann_front_wheel_angles_numbers():
    """三组数值例：δ=0 / 左满舵 / 右满舵；每组的解析值都写在注释里。

    L = 0.145 m，b = track_width = 0.20 m（默认取车宽），κ_max = 4.0
    ⇒ 满舵 δ = arctan(κ_max·L) = arctan(0.58) = 0.5255837...，R = L/tanδ = 0.25 m。
    左满舵（δ>0）：R_c = +0.25 ⇒ 左轮是内轮
        左 = atan2(L, R−b/2) = atan2(0.145, 0.15) = +0.7684506335910436
        右 = atan2(L, R+b/2) = atan2(0.145, 0.35) = +0.3927606656373215
    右满舵（δ<0）：镜像 ⇒ 左 = −0.3927606656373215、右 = −0.7684506335910436
    δ = 0：两轮都是 0（半径无穷，公式退化）。
    """
    L = 0.145
    R_turn = L / math.tan(math.atan(0.58))
    ch = AckermannChassis(curvature_action=True, kappa_max=4.0, delta_max=0.526)
    assert ch.track_width == pytest.approx(ch.width)

    cases = [
        (0.0, 0.5, 0.0, 0.0, 0.0),
        (1.0, 0.5, math.atan(0.58), 0.7684506335910436, 0.3927606656373215),
        (-1.0, 0.5, -math.atan(0.58), -0.3927606656373215, -0.7684506335910436),
    ]
    for a0, a1, exp_delta, exp_l, exp_r in cases:
        act = np.array([a0, a1])
        delta = ch.physical_action(act)[0]
        assert delta == pytest.approx(exp_delta, rel=1e-12, abs=1e-15)
        left, right = ch.front_wheel_angles(delta)
        assert left == pytest.approx(exp_l, rel=1e-12, abs=1e-15)
        assert right == pytest.approx(exp_r, rel=1e-12, abs=1e-15)
        if delta != 0.0:
            # R = L/tan(δ)：由**转角本身**反算半径，逐位回到 R_turn
            rad = L / math.tan(abs(delta))
            assert rad == pytest.approx(R_turn, rel=1e-15)
        if delta > 0.0:
            assert abs(left) > abs(right), "左转时左轮是内轮，转角必须更大"
        elif delta < 0.0:
            assert abs(right) > abs(left), "右转时右轮是内轮，转角必须更大"

    # 逐位可验的恒等式：cot(δ_out) − cot(δ_in) = b/L（bicycle 几何，与速度无关）
    left, right = ch.front_wheel_angles(math.atan(0.58))
    cot_gap = 1.0 / math.tan(right) - 1.0 / math.tan(left)
    assert cot_gap == pytest.approx(ch.track_width / ch.wheelbase, rel=1e-15)


def test_ackermann_wheel_kinematics_from_command():
    """wheel_kinematics 从 ω 反解 δ（δ = atan(ω·L/v)）再给左右轮角。"""
    ch = AckermannChassis(curvature_action=True, kappa_max=4.0)
    # a=(1.0, 0.5) ⇒ δ=0.525584, v=0.5 ⇒ ω = v/L·tanδ = 2.0
    cmd = ch.command(np.array([1.0, 0.5]))
    assert cmd.omega == pytest.approx(2.0, rel=1e-12)
    wk = ch.wheel_kinematics(cmd)
    assert isinstance(wk, WheelKinematics) and wk.kind == "ackermann"
    assert wk.wheel_left is None and wk.wheel_fl is None
    assert wk.steering_left == pytest.approx(0.7684506335910436, rel=1e-12)
    assert wk.steering_right == pytest.approx(0.3927606656373215, rel=1e-12)

    # 倒车同一条转向命令（v<0 ⇒ ω<0，但 δ 仍为正）：δ 由 ω·L/v 还原，符号抵消
    cmd_back = ch.command(np.array([1.0, -0.5]))
    assert cmd_back.omega == pytest.approx(-2.0, rel=1e-12)
    wk_back = ch.wheel_kinematics(cmd_back)
    assert wk_back.steering_left == pytest.approx(wk.steering_left, rel=1e-12)
    assert wk_back.steering_right == pytest.approx(wk.steering_right, rel=1e-12)

    # v=0 ⇒ ω=0（原地不动）⇒ 两轮 0
    wk_still = ch.wheel_kinematics(ChassisCommand(v=0.0, omega=0.0))
    assert (wk_still.steering_left, wk_still.steering_right) == (0.0, 0.0)


# ── 差速：v_L = v − ω·b/2、v_R = v + ω·b/2 ──────────────────────────


def test_diff_drive_wheel_speeds_numbers():
    """三组数值例（b = 0.3 ⇒ b/2 = 0.15），解析值见每行注释。

    (v, ω) = (0.5, 0.0)  ⇒ 左右都是 0.5（纯直行）
    (v, ω) = (0.5, 1.0)  ⇒ 左 = 0.5 − 1.0·0.15 = 0.35，右 = 0.5 + 0.15 = 0.65
    (v, ω) = (0.5, −1.0) ⇒ 左 = 0.65，右 = 0.35（右转，左轮在外侧更快）
    """
    b = 0.3
    ch = DiffDriveChassis(track_width=b, v_max=1.0, omega_max=1.0)
    cases = [
        (0.5, 0.0, 0.5, 0.5),
        (0.5, 1.0, 0.35, 0.65),
        (0.5, -1.0, 0.65, 0.35),
    ]
    for v, omega, exp_l, exp_r in cases:
        wk = ch.wheel_kinematics(ChassisCommand(v=v, omega=omega))
        assert wk.kind == "diff_drive"
        assert wk.wheel_left == pytest.approx(exp_l, rel=0.0, abs=1e-15)
        assert wk.wheel_right == pytest.approx(exp_r, rel=0.0, abs=1e-15)
        assert wk.steering_left is None
    # 归一化动作路径：[a0, a1] = [0.5, 1.0] ⇒ v = 0.5·v_max、ω = 1.0·omega_max
    wk = ch.wheel_kinematics(ch.command(np.array([0.5, 1.0])))
    assert (wk.wheel_left, wk.wheel_right) == pytest.approx((0.35, 0.65), abs=1e-15)


# ── 麦轮：逆 Jacobian（含 vy 与 ω 的耦合）────────────────────────────


def test_mecanum_wheel_speeds_numbers():
    """三组数值例 + 一组混合：r = 0.05、lx = 0.19、ly = 0.10（lx+ly = 0.29）。

    约定（x 前、y 左、ω 逆时针；FL=(+lx,+ly)、FR=(+lx,−ly)、RL=(−lx,+ly)、
    RR=(−lx,−ly)），逆 Jacobian：
        ω_FL = (v − vy − (lx+ly)·ω)/r，ω_FR = (v + vy + (lx+ly)·ω)/r
        ω_RL = (v + vy − (lx+ly)·ω)/r，ω_RR = (v − vy + (lx+ly)·ω)/r

    例 1 纯前进 (v,vy,ω) = (0.5, 0, 0)：四个轮都是 0.5/0.05 = +10.0
    例 2 纯横移 (0, 0.5, 0)：FL = −0.5/0.05 = −10、FR = +10、RL = +10、RR = −10
    例 3 纯自转 (0, 0, 1)：coupling = 0.29 ⇒ FL = RL = −0.29/0.05 = −5.8、
                          FR = RR = +5.8
    例 4 混合 (0.4, −0.15, 0.7)：coupling = 0.203
         FL = (0.4+0.15−0.203)/0.05 = 6.94   FR = (0.4−0.15+0.203)/0.05 = 9.06
         RL = (0.4−0.15−0.203)/0.05 = 0.94   RR = (0.4+0.15+0.203)/0.05 = 15.06
    """
    ch = MecanumChassis(wheel_radius=0.05, lx=0.19, ly=0.10)
    cases = [
        (0.5, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0),
        (0.0, 0.5, 0.0, -10.0, 10.0, 10.0, -10.0),
        (0.0, 0.0, 1.0, -5.8, 5.8, -5.8, 5.8),
        (0.4, -0.15, 0.7, 6.94, 9.06, 0.94, 15.06),
    ]
    for v, vy, omega, efl, efr, erl, err_ in cases:
        wk = ch.wheel_kinematics(ChassisCommand(v=v, vy=vy, omega=omega))
        assert wk.kind == "mecanum"
        got = (wk.wheel_fl, wk.wheel_fr, wk.wheel_rl, wk.wheel_rr)
        assert got == pytest.approx((efl, efr, erl, err_), rel=0.0, abs=1e-12)

    # 逆→正解自洽：用同一份约定反推回 (v, vy, ω)
    r, k = ch.wheel_radius, ch.lx + ch.ly
    for v, vy, omega, *_ in cases:
        wk = ch.wheel_kinematics(ChassisCommand(v=v, vy=vy, omega=omega))
        v_b = r * (wk.wheel_fl + wk.wheel_fr + wk.wheel_rl + wk.wheel_rr) / 4.0
        vy_b = r * (-wk.wheel_fl + wk.wheel_fr + wk.wheel_rl - wk.wheel_rr) / 4.0
        om_b = r * (-wk.wheel_fl + wk.wheel_fr - wk.wheel_rl + wk.wheel_rr) / (4.0 * k)
        assert (v_b, vy_b, om_b) == pytest.approx((v, vy, omega), rel=0.0, abs=1e-15)


def test_wheel_kinematics_as_dict_is_public_surface():
    """快照可被无 dataclasses 依赖的调用方取用（as_dict 给**全部**字段，含 None）。"""
    ch = DiffDriveChassis(track_width=0.3)
    d = ch.wheel_kinematics(ChassisCommand(v=0.5, omega=1.0)).as_dict()
    assert d["kind"] == "diff_drive"
    assert d["wheel_left"] == 0.35
    assert d["wheel_right"] == 0.65
    # 差速底盘没有转向轮/麦轮 ⇒ 这些键存在但显式为 None（不是缺键）
    assert d["steering_left"] is None and d["steering_right"] is None
    assert d["wheel_fl"] is None and d["wheel_rr"] is None
    assert list(d) == [
        "kind",
        "steering_left",
        "steering_right",
        "wheel_left",
        "wheel_right",
        "wheel_fl",
        "wheel_fr",
        "wheel_rl",
        "wheel_rr",
    ]
