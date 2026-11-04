"""三车型冒烟：同一个 SimpleBridge 换底盘就能跑，位姿符合运动学直觉。

这里**不**对拍旧桥（旧桥只有阿克曼），只验证"桥与底盘解耦"这件事本身：
同一份 `SimpleBridge` 代码分别配阿克曼 / 差速 / 麦轮，各跑 100 步，断言
位姿有限且落在直觉解上：

* 阿克曼：`δ=0` ⇒ 直线（θ 恒定，横向漂移 0）；
* 差速：定常 `(v, ω)` ⇒ 圆弧，半径 = v/ω，每步弦长 = v·dt，总转角 = ω·T；
* 麦轮：`(v, ω)` 全 0、只有 `vy` ⇒ 纯横移（x 不变，y 走 vy·T）。

外加一份三车型共用的**传感器契约**：

来自 get_sensor_reading 的 LiDAR 形状/量程/dtype、里程计长度、路径形状、布尔返回类型
—— 这些与车型无关，任何底盘都该满足。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis, DiffDriveChassis, MecanumChassis
from chassis_sim.sensors import Lidar

N_STEPS = 100
DT = 0.1

_PATH = np.column_stack(
    [
        np.linspace(2.0, 6.0, 20),
        np.full(20, 2.0),
        np.zeros(20),
    ]
)


def _bridge(chassis) -> SimpleBridge:
    # 雷达只有 get_sensor_reading 一个入口 ⇒ 前/后两支 Lidar 必须显式装配
    front, rear = chassis.lidar_mounts()
    bridge = SimpleBridge(
        chassis,
        dt=DT,
        sensors=[
            Lidar("front", mount=front, n_rays=180),
            Lidar("rear", mount=rear, n_rays=180),
        ],
    )
    bridge.reset(obstacles=[], path_waypoints=_PATH)
    return bridge


def _run(bridge: SimpleBridge, action: np.ndarray) -> np.ndarray:
    """跑 N_STEPS 步，返回逐步位姿 (N+1, 3)，第 0 行是初始位姿。"""
    poses = [[bridge._x, bridge._y, bridge._theta]]
    for _ in range(N_STEPS):
        bridge.step(action)
        poses.append([bridge._x, bridge._y, bridge._theta])
    poses = np.asarray(poses, dtype=np.float64)
    assert np.all(np.isfinite(poses)), "位姿出现 nan/inf"
    return poses


def test_ackermann_straight() -> None:
    """阿克曼零转向 ⇒ 严格直线（转向滞后也是 0）。"""
    bridge = _bridge(AckermannChassis())
    poses = _run(bridge, np.array([0.0, 0.6]))
    v = 0.6  # a1 · v_max = 0.6 · 1.0

    assert np.max(np.abs(poses[:, 2])) == 0.0
    assert np.max(np.abs(poses[:, 1] - 2.0)) == 0.0
    assert abs(poses[-1, 0] - (2.0 + v * DT * N_STEPS)) < 1e-12


def test_diff_drive_straight() -> None:
    bridge = _bridge(DiffDriveChassis())
    poses = _run(bridge, np.array([0.5, 0.0]))
    assert np.max(np.abs(poses[:, 2])) == 0.0
    assert np.max(np.abs(poses[:, 1] - 2.0)) == 0.0
    assert abs(poses[-1, 0] - (2.0 + 0.5 * DT * N_STEPS)) < 1e-12


def test_diff_drive_circle() -> None:
    """定常 (v, ω) ⇒ 精确半径 r = v/ω 的圆弧（2026-11-05 起闭式积分）。"""
    v, omega = 0.5, 0.5
    bridge = _bridge(DiffDriveChassis())
    poses = _run(bridge, np.array([v, omega]))

    radius = v / omega
    total_turn = omega * DT * N_STEPS  # 5.0 rad
    wrapped = (poses[-1, 2] + np.pi) % (2.0 * np.pi) - np.pi
    expected = (total_turn + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(wrapped - expected) < 1e-12

    # 闭式积分：每个采样点都落在**真正的**圆上（圆心在起点左侧 R 处），
    # 半径误差只剩浮点噪声。旧的 Euler 实现顶点落在 R_d = v·dt/(2·sin(dθ/2))
    # 的折线圆上（比 R 大半步，差 O(dθ²)），2026-11-05 换积分式前的断言
    # 正是按那个折线口径写的。
    center = np.array([2.0, 2.0 + radius])
    dist = np.hypot(poses[:, 0] - center[0], poses[:, 1] - center[1])
    dev = np.max(np.abs(dist - radius))
    assert dev < 1e-12, f"闭式圆周半径偏差 {dev}"

    # 弦长 = 2R·sin(ω·dt/2) 的精确式（Euler 下才会恒等于 v·dt）
    steps = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    chord = 2.0 * radius * np.sin(omega * DT / 2.0)
    assert np.max(np.abs(steps - chord)) < 1e-12


def test_mecanum_strafe() -> None:
    """麦轮纯横移：v=ω=0，只有 vy ⇒ x、θ 不变，y 走 vy·T。"""
    vy = 0.25  # a1 · vy_max = 0.5 · 0.5
    bridge = _bridge(MecanumChassis())
    poses = _run(bridge, np.array([0.0, 0.5, 0.0]))

    assert np.max(np.abs(poses[:, 0] - 2.0)) == 0.0
    assert np.max(np.abs(poses[:, 2])) == 0.0
    assert abs(poses[-1, 1] - (2.0 + vy * DT * N_STEPS)) < 1e-12


@pytest.mark.parametrize(
    "factory",
    [AckermannChassis, DiffDriveChassis, MecanumChassis],
)
def test_sensor_contract_all_chassis(factory) -> None:
    """三车型共用的传感器契约：形状 / dtype / 量程 / 返回类型。"""
    chassis = factory()
    bridge = _bridge(chassis)
    action = np.array([0.2, 0.3] + ([0.1] if chassis.action_dim == 3 else []))
    for _ in range(10):
        bridge.step(action)

    reading = bridge.get_sensor_reading() or {}
    assert sorted(reading) == ["front", "rear"]
    for name in ("front", "rear"):
        lidar = np.asarray(reading[name]["scan"])
        assert lidar.shape == (180,)
        assert lidar.dtype == np.float32
        assert np.all(lidar >= 0.05) and np.all(lidar <= 10.0)

    odom = bridge.get_odometry()
    assert odom.shape == (5,)
    assert odom.dtype == np.float32
    assert np.all(np.isfinite(odom))

    assert bridge.get_path().shape == (20, 3)
    assert bridge.get_path().dtype == np.float32
    assert bridge.get_path_global().shape == (20, 3)
    assert bridge.get_path_global().dtype == np.float32

    assert isinstance(bridge.check_collision(), bool)
    assert isinstance(bridge.check_goal_reached(), bool)
    assert isinstance(bridge.goal_heading, float)
    bridge.close()
