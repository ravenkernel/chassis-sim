"""归一化动作契约：物理 (v, ω) → controller.to_chassis_action() → chassis.command()。

契约（本仓 A1 裁决）：**每个底盘的 action 一律是归一化 [-1,1]^action_dim**，
物理量只在 `Chassis.command()` 内产生。控制器出口与底盘入口因此必须逐位互逆：
`command(controller.to_chassis_action(v, ω))` 必须还原 (v, ω)，绝对容差 1e-9。

这条测试用 **available_chassis() 里的真实底盘类**（不是桩）覆盖三个车型 ——
任何底盘把物理量漏进 action_space / command()（例如 action_space 写成
`Box([-v_max, -omega_max], [v_max, omega_max])` 而 command() 原样透传），往返
就会被 1/v_max 的偏差打穿，这里立刻变红。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.chassis import available_chassis, make_chassis
from chassis_sim.controllers import PurePursuitController

TOL = 1e-9
# |v| 低于该值时阿克曼的 δ 无法由 (v, ω) 反解（controllers/base.py 的 V_EPS），
# 反解只能沿用上一拍 δ ⇒ 该分支不参与 ω 往返断言。
V_EPS = 1e-6
V_GRID = (0.3, 0.6, 1.2)
OMEGA_GRID = (0.0, 0.4, -0.4)
# v_max 档位：默认 1.0 只能覆盖 0.3 / 0.6；1.2 档让三个速度点全部落在界内。
V_MAX_CASES = (1.0, 1.2)


@pytest.mark.parametrize("name", available_chassis())
def test_every_real_chassis_exposes_normalized_action_space(name):
    space = make_chassis(name).action_space()
    assert space.dtype == np.float32
    np.testing.assert_array_equal(space.low, -np.ones(space.shape, dtype=np.float32))
    np.testing.assert_array_equal(space.high, np.ones(space.shape, dtype=np.float32))


@pytest.mark.parametrize("name", available_chassis())
@pytest.mark.parametrize("v_max", V_MAX_CASES)
def test_physical_command_roundtrips_through_controller(name, v_max):
    chassis = make_chassis(name, v_max=v_max)
    controller = PurePursuitController(chassis=chassis)
    space = chassis.action_space()
    omega_max = float(chassis.describe()["limits"]["omega_max"])

    covered = 0
    for v in V_GRID:
        if v > v_max + TOL:
            continue  # 超出该底盘 v_max 的请求会被 action_space 饱和，不在断言范围
        for omega in OMEGA_GRID:
            if abs(omega) > omega_max + TOL:
                continue
            action = controller.to_chassis_action(v, omega)
            ctx = f"{name} v_max={v_max} 请求 (v={v}, ω={omega}) → action={action}"
            assert action.shape == tuple(space.shape), ctx
            assert np.all(action >= space.low - TOL) and np.all(
                action <= space.high + TOL
            ), ctx
            cmd = chassis.command(action)
            assert cmd.v == pytest.approx(v, abs=TOL), ctx
            if name == "ackermann" and abs(v) < V_EPS:
                continue
            assert cmd.omega == pytest.approx(omega, abs=TOL), ctx
            covered += 1
    assert covered > 0, f"{name} v_max={v_max} 没有覆盖到任何 (v, ω) 组合"


def test_diff_drive_physical_velocity_survives_user_reported_case():
    """用户实测的原始场景：v_max=1.2 的差速底盘，v=0.6 必须原样回来。"""
    chassis = make_chassis("diff_drive", v_max=1.2)
    controller = PurePursuitController(chassis=chassis)
    action = controller.to_chassis_action(0.6, 0.0)
    np.testing.assert_allclose(action, np.array([0.5, 0.0]), atol=TOL)
    cmd = chassis.command(action)
    assert (cmd.v, cmd.omega, cmd.vy) == (0.6, 0.0, 0.0)
