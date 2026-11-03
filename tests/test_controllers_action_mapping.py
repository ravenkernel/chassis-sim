"""`Controller.to_chassis_action` 与三种车型 `command()` 正映射的往返一致性。

**不 import chassis 包**：本文件自带 `_StubChassis`（差速 / 麦轮 / 阿克曼），
`command()` 里复刻的正映射就是契约本身：

* 阿克曼：逐字对应 /home/zhao/drl-sim/envs/ackermann_env.py:822-833
  （`mapped_action = action.copy()` → 曲率/线性转向 → 分段速度），随后
  `ω = v/L·tan(δ)`（同 envs 的 numpy_bridge.py:383 口径）。
* 差速 / 麦轮：`a[0] ≥ 0 → v = a[0]·v_max`、`a[0] < 0 → v = a[0]·(−v_min)`，
  `ω = a[·]·ω_max`、`vy = a[1]·vy_max`（归一化动作契约）。

`Command` 用本地 frozen dataclass 等价于 `chassis.ChassisCommand`（字段 v/omega/vy 同名），
这样测试可以在 chassis 包尚未装好时独立跑。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from gymnasium import spaces

from chassis_sim.controllers import Controller, DwaController, PurePursuitController


@dataclass(frozen=True)
class _StubCommand:
    """等价于 chassis_sim.chassis.base.ChassisCommand（v / omega / vy）。"""

    v: float
    omega: float
    vy: float = 0.0


class _StubDiff:
    """差速：a = [v/v_max, ω/ω_max]。"""

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
        low, high = self.action_space().low, self.action_space().high
        a = np.clip(np.asarray(action, dtype=np.float64), low, high)
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


class _StubMecanum:
    """麦轮：a = [vx/v_max, vy/vy_max, ω/ω_max]。"""

    name = "mecanum"
    action_dim = 3

    def __init__(self, v_max=1.0, v_min=-1.0, vy_max=0.5, omega_max=1.0):
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.vy_max = float(vy_max)
        self.omega_max = float(omega_max)

    def action_space(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def command(self, action):
        low, high = self.action_space().low, self.action_space().high
        a = np.clip(np.asarray(action, dtype=np.float64), low, high)
        vx = a[0] * self.v_max if a[0] >= 0.0 else a[0] * (-self.v_min)
        return _StubCommand(
            float(vx), float(a[2] * self.omega_max), float(a[1] * self.vy_max)
        )

    def describe(self):
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "action_shape": (3,),
            "limits": {
                "v_min": self.v_min,
                "v_max": self.v_max,
                "omega_max": self.omega_max,
                "vy_max": self.vy_max,
            },
            "params": {},
        }


class _StubAckermann:
    """阿克曼：正映射逐字复刻 ackermann_env.py:822-833 + ω = v/L·tan δ。"""

    name = "ackermann"
    action_dim = 2

    def __init__(
        self,
        wheelbase=0.145,
        delta_max=0.526,
        v_min=-1.0,
        v_max=1.0,
        curvature_action=True,
        kappa_max=4.0,
    ):
        self.wheelbase = float(wheelbase)
        self.delta_max = float(delta_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.curvature_action = bool(curvature_action)
        self.kappa_max = float(kappa_max)

    def action_space(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def command(self, action):
        low, high = self.action_space().low, self.action_space().high
        action = np.clip(np.asarray(action, dtype=np.float64), low, high)

        # ── 以下照抄 /home/zhao/drl-sim/envs/ackermann_env.py:822-833 ──
        # fmt: off
        mapped_action = action.copy()
        # 转向映射：曲率模式 (R ∝ 1/|a|) vs 线性模式 (δ ∝ a)
        if self.curvature_action:
            kappa = action[0] * self.kappa_max
            mapped_action[0] = np.arctan(np.clip(kappa, -self.kappa_max, self.kappa_max) * self.wheelbase)
        else:
            mapped_action[0] = action[0] * self.delta_max

        if action[1] >= 0:
            mapped_action[1] = action[1] * self.v_max        # [0, +1] → [0, v_max]
        else:
            mapped_action[1] = action[1] * (-self.v_min)     # [-1, 0] → [v_min, 0]
        # fmt: on
        # ── 照抄结束 ──

        delta = float(mapped_action[0])
        v = float(mapped_action[1])
        omega = 0.0 if v == 0.0 else v / self.wheelbase * np.tan(delta)
        return _StubCommand(v, omega, 0.0)

    def describe(self):
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "action_shape": (2,),
            "limits": {
                "v_min": self.v_min,
                "v_max": self.v_max,
                "omega_max": self.v_max * self.kappa_max,
            },
            "params": {
                "wheelbase": self.wheelbase,
                "delta_max": self.delta_max,
                "curvature_action": self.curvature_action,
                "kappa_max": self.kappa_max,
            },
        }


class _StubNoLimits(_StubDiff):
    """describe() 缺 'limits' 的坏底盘：用于验证缺键就地报错。"""

    def describe(self):
        desc = super().describe()
        del desc["limits"]
        return desc


def _roundtrip(ctrl, chassis, v, omega, vy=0.0):
    action = ctrl.to_chassis_action(v, omega, vy)
    assert action.shape == ctrl.action_space.shape
    assert np.all(np.abs(action) <= 1.0 + 1e-12), f"action 未归一化: {action}"
    cmd = chassis.command(action)
    return abs(float(cmd.v) - float(v)), abs(float(cmd.omega) - float(omega))


def _ackermann_feasible_omega(v, chassis):
    """阿克曼在给定 v 下可达的 |ω| 上限（曲率模式 κ_max、线性模式 tan(δ_max)/L）。"""
    if chassis.curvature_action:
        return abs(v) * chassis.kappa_max
    return abs(v) * np.tan(chassis.delta_max) / chassis.wheelbase


def test_action_shape_follows_chassis():
    for chassis in (_StubDiff(), _StubMecanum(), _StubAckermann()):
        ctrl = PurePursuitController(chassis=chassis)
        assert ctrl.action_space.shape == chassis.action_space().shape
        assert ctrl.act({"pose": np.zeros(3)}).shape == chassis.action_space().shape


def test_diff_drive_roundtrip_is_exact():
    chassis = _StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0)
    ctrl = PurePursuitController(chassis=chassis)
    rng = np.random.default_rng(0)
    cases = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, -1.0)]
    cases += [
        (float(v), float(o))
        for v, o in zip(rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200))
    ]
    worst_v = worst_o = 0.0
    for v, omega in cases:
        dv, do = _roundtrip(ctrl, chassis, v, omega)
        worst_v, worst_o = max(worst_v, dv), max(worst_o, do)
    assert worst_v <= 1e-9 and worst_o <= 1e-9, (worst_v, worst_o)


def test_mecanum_roundtrip_is_exact():
    chassis = _StubMecanum(v_max=1.0, v_min=-1.0, vy_max=0.5, omega_max=1.0)
    ctrl = PurePursuitController(chassis=chassis)
    rng = np.random.default_rng(1)
    cases = [(0.0, 0.0, 0.0), (1.0, 0.5, 1.0), (-1.0, -0.5, -1.0), (0.3, -0.2, 0.7)]
    cases += [
        (float(v), float(vy), float(o))
        for v, vy, o in zip(
            rng.uniform(-1, 1, 200),
            rng.uniform(-0.5, 0.5, 200),
            rng.uniform(-1, 1, 200),
        )
    ]
    worst_v = worst_o = worst_vy = 0.0
    for v, vy, omega in cases:
        action = ctrl.to_chassis_action(v, omega, vy)
        assert action.shape == (3,)
        cmd = chassis.command(action)
        worst_v = max(worst_v, abs(float(cmd.v) - v))
        worst_vy = max(worst_vy, abs(float(cmd.vy) - vy))
        worst_o = max(worst_o, abs(float(cmd.omega) - omega))
    assert max(worst_v, worst_vy, worst_o) <= 1e-9, (worst_v, worst_vy, worst_o)


@pytest.mark.parametrize("curvature_action", [True, False])
def test_ackermann_roundtrip_is_exact(curvature_action):
    chassis = _StubAckermann(curvature_action=curvature_action)
    ctrl = PurePursuitController(chassis=chassis)
    rng = np.random.default_rng(2)
    cases = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.05, 0.0)]
    vs = rng.uniform(-1.0, 1.0, 200)
    for v in vs:
        if abs(float(v)) < 1e-3:
            continue
        lim = _ackermann_feasible_omega(float(v), chassis)
        cases.append((float(v), float(rng.uniform(-1.0, 1.0)) * lim))
    for v in (-1.0, 1.0):  # 打满转向边界
        cases.append((v, _ackermann_feasible_omega(v, chassis)))
        cases.append((v, -_ackermann_feasible_omega(v, chassis)))
    worst_v = worst_o = 0.0
    for v, omega in cases:
        dv, do = _roundtrip(ctrl, chassis, v, omega)
        worst_v, worst_o = max(worst_v, dv), max(worst_o, do)
    assert worst_v <= 1e-9 and worst_o <= 1e-9, (worst_v, worst_o)


def test_ackermann_zero_speed_keeps_previous_delta():
    """v≈0 时 ω 不可反解：沿用上一拍 δ（首次为 0），且 ω 输出必然为 0。"""
    chassis = _StubAckermann()
    ctrl = PurePursuitController(chassis=chassis)
    ctrl.to_chassis_action(0.5, 0.5)  # 建立一拍 δ
    prev = ctrl._last_delta
    action = ctrl.to_chassis_action(0.0, 0.0)
    assert ctrl._last_delta == pytest.approx(prev)
    cmd = chassis.command(action)
    assert cmd.v == 0.0 and cmd.omega == 0.0
    # 零速请求非零 ω：物理不可达，被饱和成"有转向角但零速 ⇒ ω=0"
    cmd2 = chassis.command(ctrl.to_chassis_action(0.0, 1.0))
    assert cmd2.v == 0.0 and cmd2.omega == 0.0


def test_missing_limits_raises_keyerror_naming_the_key():
    with pytest.raises(KeyError, match="limits"):
        PurePursuitController(chassis=_StubNoLimits())
    with pytest.raises(KeyError, match="wheelbase"):
        DwaController(
            chassis=_StubAckermann(),
            params={"delta_max": 0.526, "curvature_action": True, "kappa_max": 4.0},
        )


def test_controller_is_abstract():
    with pytest.raises(TypeError):
        Controller(kind="diff_drive", limits={}, params={})
