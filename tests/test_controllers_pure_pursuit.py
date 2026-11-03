"""纯追踪控制器的收敛性：直线横向纠偏 + 圆弧跟踪。

闭环用本文件自带的 `_StubDiff`（差速，归一化 action 契约）与
`_integrate`（逐字等于 chassis.Chassis.integrate 的欧拉核）推演——
**不 import chassis 包**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from chassis_sim.controllers import PurePursuitController

DT = 0.05


@dataclass(frozen=True)
class _StubCommand:
    v: float
    omega: float
    vy: float = 0.0


class _StubDiff:
    """差速底盘桩：a = [v/v_max, ω/ω_max]，其余与 chassis 契约同签名。"""

    name = "diff_drive"
    action_dim = 2

    def __init__(self, v_max=1.0, v_min=-1.0, omega_max=1.0):
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)

    def action_space(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def command(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v = a[0] * self.v_max if a[0] >= 0.0 else a[0] * (-self.v_min)
        return _StubCommand(float(v), float(a[1] * self.omega_max), 0.0)

    def describe(self):
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "action_shape": (2,),
            "limits": {
                "v_min": self.v_min,
                "v_max": self.v_max,
                "omega_max": self.omega_max,
            },
            "params": {},
        }


def _integrate(state, cmd, dt):
    """等价于 Chassis.integrate（Euler，车体系速度旋到世界系）。"""
    x, y, th = state
    return (
        x + (cmd.v * np.cos(th) - cmd.vy * np.sin(th)) * dt,
        y + (cmd.v * np.sin(th) + cmd.vy * np.cos(th)) * dt,
        th + cmd.omega * dt,
    )


def _run(ctrl, chassis, path, x0, steps, dt=DT):
    ctrl.reset()
    ctrl.set_path(path)
    state = (float(x0[0]), float(x0[1]), float(x0[2]))
    rows = []
    for _ in range(steps):
        action = ctrl.act({"pose": np.array(state)})
        cmd = chassis.command(action)
        state = _integrate(state, cmd, dt)
        rows.append((state[0], state[1], state[2], cmd.v, cmd.omega))
    return np.asarray(rows)


def _straight_path(length=30.0, step=0.05):
    xs = np.arange(0.0, length + step, step)
    return np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])


def _circle_path(radius=1.0, laps=2, step=0.05):
    n = round(2.0 * np.pi * radius / step)
    ang = np.linspace(0.0, 2.0 * np.pi * laps, n * laps + 1)
    return np.column_stack(
        [radius * np.cos(ang), radius * np.sin(ang), ang + np.pi / 2.0]
    )


def test_pure_pursuit_converges_to_straight_line():
    """从 y 偏 0.3 m 出发，10 s 内横向误差收敛到 < 0.05 m。"""
    chassis = _StubDiff()
    ctrl = PurePursuitController(chassis=chassis, lookahead=0.6)
    traj = _run(ctrl, chassis, _straight_path(), (0.0, 0.3, 0.0), steps=200, dt=DT)
    err = np.abs(traj[:, 1])
    print(
        "直线横向误差 |y|：前 3 个采样点 =",
        np.round(err[:3], 6),
        "10 s 终值 =",
        round(float(err[-1]), 6),
        "末 2 s 最大 =",
        round(float(err[-40:].max()), 6),
    )
    assert err[-1] < 0.05, f"10 s 后横向误差 {err[-1]:.4f} m 未收敛到 0.05 m"
    assert err[-40:].max() < 0.05, f"末 2 s 横向误差峰值 {err[-40:].max():.4f} m"
    assert np.all(traj[:, 3] > 0.0), "纯追踪应始终前进（v > 0）"


def test_pure_pursuit_tracks_circle():
    """R = 1.0 m 圆弧，2 圈：暖机 2 s 后跟踪误差 < 0.1 m。"""
    chassis = _StubDiff()
    ctrl = PurePursuitController(chassis=chassis, lookahead=0.6)
    path = _circle_path(radius=1.0, laps=2)
    traj = _run(ctrl, chassis, path, (1.0, 0.0, np.pi / 2.0), steps=200, dt=DT)
    radial = np.abs(np.hypot(traj[:, 0], traj[:, 1]) - 1.0)
    warm = int(2.0 / DT)
    print(
        "圆弧径向误差：前 3 个采样点 =",
        np.round(radial[:3], 6),
        "10 s 终值 =",
        round(float(radial[-1]), 6),
        "暖机后最大 =",
        round(float(radial[warm:].max()), 6),
    )
    assert radial[warm:].max() < 0.1, (
        f"圆弧跟踪误差 {radial[warm:].max():.4f} m 超 0.1 m"
    )


def test_pure_pursuit_empty_path_is_zero_action():
    chassis = _StubDiff()
    ctrl = PurePursuitController(chassis=chassis)
    ctrl.set_path(np.empty((0, 3)))
    assert np.allclose(ctrl.act({"pose": np.zeros(3)}), 0.0)
