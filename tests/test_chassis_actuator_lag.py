"""转向滞后一拍：`step()` 重放 golden 位姿，`command()` 作反向对照。

旧桥 /home/zhao/drl-sim/ros_bridge/numpy_bridge.py:376-390 的语义是
**速度立即生效、转向滞后一拍**：`self._v = v_cmd` 当步就用，而 ω/θ 取
`prev_steer = self._steer`（:383、:388），首帧 prev=0（:228），每步把 θ 折回
[-π,π]（:390）。本仓把这一拍做成显式执行器状态 `AckermannChassis.step()`，
`command()` 保持纯映射。

本文件的三件事：
1. 正向：golden 的 actions 走 `step()` + `integrate()`，逐位复现 golden 位姿；
2. 反向对照：同一串 action 改走 `command()`（无滞后），必须偏 > 1e-3，
   否则这条测试对滞后不敏感、等于没测；
3. `reset_state()` 后重放，结果与首次逐位一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.chassis import (
    AckermannChassis,
    DiffDriveChassis,
    MecanumChassis,
)

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "ackermann_golden.json"

TOL = 1e-12
# 反向对照的偏差下限：小于这个量级说明测试区分不出「滞后 / 不滞后」
LAG_TOL = 1e-3

# 与主仓 config/train.yaml 的 env 块 + config/env.yaml 的 robot.shape 同值
_PARAMS = {
    "wheelbase": 0.145,
    "delta_max": 0.526,
    "v_min": -1.0,
    "v_max": 1.0,
    "curvature_action": True,
    "kappa_max": 4.0,
}


def _chassis() -> AckermannChassis:
    return AckermannChassis(**_PARAMS)


def _golden() -> tuple[np.ndarray, np.ndarray, float]:
    """(golden actions, golden poses, dt)；只读 JSON，不依赖 /home/zhao/drl-sim。"""
    data = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    actions = np.array(
        [[float(s) for s in row] for row in data["actions"]], dtype=np.float64
    )
    poses = np.array(
        [[float(s) for s in row] for row in data["poses"]], dtype=np.float64
    )
    return actions, poses, float(data["meta"]["bridge_dt"])


def _ang_diff(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _wrap(theta: float) -> float:
    """与 numpy_bridge.py:390 同式的 θ 归一化。"""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _replay(ch, actions: np.ndarray, dt: float, mode: str) -> np.ndarray:
    """mode='step' 走执行器状态；mode='command' 走无滞后的纯映射。"""
    x = y = theta = 0.0
    poses = [(x, y, theta)]
    for a in actions:
        cmd = ch.step(a) if mode == "step" else ch.command(a)
        x, y, theta = ch.integrate(x, y, theta, cmd, dt)
        theta = _wrap(theta)
        poses.append((x, y, theta))
    return np.array(poses, dtype=np.float64)


def _max_err(got: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    err_xy = float(np.abs(got[:, :2] - expected[:, :2]).max())
    err_theta = float(np.abs(_ang_diff(got[:, 2], expected[:, 2])).max())
    return err_xy, err_theta


# ══════════════════════════════════════════════════════════════════════
# 正向：step() 复现旧桥的滞后一拍
# ══════════════════════════════════════════════════════════════════════


def test_step_replays_golden_poses_bit_exact():
    actions, expected, dt = _golden()
    ch = _chassis()
    got = _replay(ch, actions, dt, "step")

    assert got.shape == expected.shape
    err_xy, err_theta = _max_err(got, expected)
    print(
        "\n[step] steps="
        + str(len(actions))
        + " dt="
        + repr(dt)
        + " max|Δx,Δy|="
        + format(err_xy, ".3e")
        + " max|Δθ|="
        + format(err_theta, ".3e")
        + " tol="
        + format(TOL, ".0e")
    )
    assert err_xy <= TOL, "step 的 xy 最大误差 " + format(err_xy, ".3e") + " > 1e-12"
    assert err_theta <= TOL, (
        "step 的 θ 最大误差 " + format(err_theta, ".3e") + " > 1e-12"
    )


# ══════════════════════════════════════════════════════════════════════
# 反向对照：丢掉滞后这一拍，必须明显偏离
# ══════════════════════════════════════════════════════════════════════


def test_command_without_lag_diverges_from_golden():
    actions, expected, dt = _golden()
    ch = _chassis()
    got = _replay(ch, actions, dt, "command")

    err_xy, err_theta = _max_err(got, expected)
    print(
        "\n[no-lag] max|Δx,Δy|="
        + format(err_xy, ".3e")
        + " max|Δθ|="
        + format(err_theta, ".3e")
        + " 下限="
        + format(LAG_TOL, ".0e")
    )
    assert err_xy > LAG_TOL, (
        "无滞后重放只偏 "
        + format(err_xy, ".3e")
        + " ≤ 1e-3：这条测试区分不出滞后，失去意义"
    )


# ══════════════════════════════════════════════════════════════════════
# 执行器状态：reset_state() 回到首帧
# ══════════════════════════════════════════════════════════════════════


def test_reset_state_restores_initial_actuator_state():
    actions, expected, dt = _golden()
    ch = _chassis()

    first = _replay(ch, actions, dt, "step")
    # 同一实例重放第二次（未 reset）：首帧残存上一串末尾的转角 ⇒ 必然偏离
    stale = _replay(ch, actions, dt, "step")
    ch.reset_state()
    again = _replay(ch, actions, dt, "step")

    np.testing.assert_array_equal(again, first)
    err_xy, err_theta = _max_err(again, expected)
    print(
        "\n[reset] identical="
        + str(bool(np.array_equal(again, first)))
        + " max|Δx,Δy|="
        + format(err_xy, ".3e")
        + " 未 reset 重放偏差="
        + format(float(np.abs(stale[:, :2] - first[:, :2]).max()), ".3e")
    )
    assert err_xy <= TOL and err_theta <= TOL
    assert not np.array_equal(stale, first), "未 reset 的第二次重放不该与首次相同"


# ══════════════════════════════════════════════════════════════════════
# command() 仍是纯函数：不读也不写执行器状态
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("a0", [-0.9, -0.2, 0.0, 0.35, 0.8])
def test_command_is_pure_and_ignores_actuator_state(a0):
    a = np.array([a0, 0.6], dtype=np.float64)
    ch = _chassis()

    first = ch.command(a)
    second = ch.command(a)
    assert first == second, "同一 action 连调两次应返回相同命令"

    ch.step(np.array([-0.7, 0.3], dtype=np.float64))  # 弄脏执行器状态
    after_step = ch.command(a)
    assert after_step == first, "command() 不应受 step() 状态影响"


# ══════════════════════════════════════════════════════════════════════
# 基类默认：无滞后、无状态（差速 / 麦轮不覆盖也成立）
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "chassis, action",
    [
        (DiffDriveChassis(), np.array([0.5, -0.4], dtype=np.float64)),
        (MecanumChassis(), np.array([0.3, -0.2, 0.9], dtype=np.float64)),
    ],
)
def test_default_step_is_stateless(chassis, action):
    assert type(chassis).step is not AckermannChassis.step
    assert chassis.step(action) == chassis.command(action)
    chassis.reset_state()  # 默认无状态：调用是安全的空操作
    assert chassis.step(action) == chassis.command(action)
