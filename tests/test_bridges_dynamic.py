"""SimpleBridge 的动态障碍：解析运动学 / 碰撞时序 / LiDAR / 确定性。

全部断言都基于"纯运动学、零随机"这一前提：位置是模拟时间 t = k·dt 的解析函数，
因此两次同样的 reset+step 序列必然逐位相同（max|Δ| = 0）。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.controllers import PurePursuitController

PATH3 = np.array([[1.0, 5.0, 0.0], [5.0, 5.0, 0.0]])

#: 与 scenarios/data/dynamic_crossing.yaml 同参数：t=2 s 时圆心正好压到 y=5
CROSSING = {
    "type": "linear",
    "start": [3.0, 4.4],
    "velocity": [0.0, 0.3],
    "radius": 0.25,
    "period": 20.0,
    "pingpong": True,
}


def _bridge(dynamics=None) -> SimpleBridge:
    return SimpleBridge(AckermannChassis(), dynamic_obstacles=dynamics)


def _reset(bridge: SimpleBridge) -> None:
    bridge.reset(obstacles=[], path_waypoints=PATH3, initial_heading=0.0)


def _straight(bridge: SimpleBridge, steps: int) -> None:
    """原地开环直行（a0=0 ⇒ δ=0，a1=1 ⇒ v=1 m/s），不依赖控制器。"""
    for _ in range(steps):
        bridge.step(np.array([0.0, 1.0]))


def _pursuit(
    bridge: SimpleBridge, steps: int
) -> tuple[list[np.ndarray], int | None, np.ndarray]:
    """pure_pursuit 跑 steps 步，返回 (逐步位姿, 首次碰撞的步号, 碰撞时刻的动态圆心)。"""
    ctrl = PurePursuitController(chassis=bridge.chassis)
    ctrl.set_path(PATH3)
    poses: list[np.ndarray] = []
    first_hit: int | None = None
    hit_pos = np.zeros(3)
    for k in range(1, steps + 1):
        odom = bridge.get_odometry()
        bridge.step(ctrl.act({"pose": np.array(odom[:3], dtype=np.float64)}))
        poses.append(np.array(bridge.get_odometry()[:3], dtype=np.float64))
        if first_hit is None and bridge.check_collision():
            first_hit = k
            hit_pos = bridge.dynamic_positions()[0].copy()
    return poses, first_hit, hit_pos


# ═════════════════════════ 运动学：t = 0 / 1 / 2 s ═════════════════════════
def test_linear_kinematics_matches_closed_form() -> None:
    """linear：p(t) = start + velocity·t，t = 0/1/2 s 与解析解**逐位**相等。"""
    start = np.array([-0.7, 1.3])
    velocity = np.array([0.25, -0.4])
    bridge = _bridge(
        [
            {
                "type": "linear",
                "start": start.tolist(),
                "velocity": velocity.tolist(),
                "radius": 0.2,
            }
        ]
    )
    _reset(bridge)
    got = {}
    for k in range(21):
        if k in (0, 10, 20):
            got[k] = bridge.dynamic_positions()[0, :2].copy()
        if k < 20:
            bridge.step(np.array([0.0, 0.0]))
    for k, t in ((0, 0.0), (10, 1.0), (20, 2.0)):
        expected = start + velocity * t
        assert np.array_equal(got[k], expected), (t, got[k], expected)


def test_circular_kinematics_matches_closed_form() -> None:
    """circular：p(t) = center + radius_orbit·(cos φ, sin φ)，φ = phase + omega·t。"""
    center = np.array([2.0, 3.0])
    radius_orbit, omega, phase = 0.5, 1.7, 0.4
    bridge = _bridge(
        [
            {
                "type": "circular",
                "center": center.tolist(),
                "radius_orbit": radius_orbit,
                "omega": omega,
                "phase": phase,
                "radius": 0.2,
            }
        ]
    )
    _reset(bridge)
    got = {}
    for k in range(21):
        if k in (0, 10, 20):
            got[k] = bridge.dynamic_positions()[0, :2].copy()
        if k < 20:
            bridge.step(np.array([0.0, 0.0]))
    for k, t in ((0, 0.0), (10, 1.0), (20, 2.0)):
        phi = phase + omega * t
        expected = center + radius_orbit * np.array([np.cos(phi), np.sin(phi)])
        assert np.array_equal(got[k], expected), (t, got[k], expected)


def test_linear_period_folds_sawtooth_and_pingpong() -> None:
    """period>0 的两种折叠：锯齿瞬移回起点；折返在中点半程处原路返回。"""
    base = {
        "type": "linear",
        "start": [0.0, 0.0],
        "velocity": [1.0, 0.0],
        "radius": 0.1,
    }
    saw = _bridge([dict(base, period=2.0, pingpong=False)])
    _reset(saw)
    _straight(saw, 25)  # t = 2.5 s ⇒ u = 0.5
    assert saw.dynamic_positions()[0, 0] == pytest.approx(0.5)
    ping = _bridge([dict(base, period=2.0, pingpong=True)])
    _reset(ping)
    _straight(ping, 15)  # t = 1.5 s ⇒ 折返：2.0/2 - |1.5-1.0| = 0.5
    assert ping.dynamic_positions()[0, 0] == pytest.approx(0.5)
    _straight(ping, 5)  # t = 2.0 s ⇒ 回到起点
    assert ping.dynamic_positions()[0, 0] == pytest.approx(0.0, abs=1e-12)


# ═════════════════════════ 碰撞：同一条时间轴上撞上 ═════════════════════════
def test_crossing_collides_while_static_run_does_not() -> None:
    """无动态障碍走满 100 步无碰撞；打开后**同一时间轴**（前 k 步位姿逐位相同）撞上。"""
    free = _bridge()
    _reset(free)
    free_poses, free_hit, _ = _pursuit(free, 100)
    assert free_hit is None, f"静态基线不该碰撞，却在第 {free_hit} 步碰撞"

    dyn = _bridge([CROSSING])
    _reset(dyn)
    dyn_poses, dyn_hit, hit_center = _pursuit(dyn, 100)
    assert dyn_hit is not None, "动态障碍横穿路径却没有触发碰撞"
    assert dyn_hit < 40, f"应在抵达终点（约第 40 步）前撞上，实际第 {dyn_hit} 步"

    # 位姿逐位相同 ⇒ 动态障碍只影响碰撞/雷达，不影响本车运动学
    for k in range(dyn_hit):
        assert np.array_equal(free_poses[k], dyn_poses[k]), k

    # 撞上那一刻：圆心到车心 < 本体半径 + 碰撞余量 + 车体半对角线（角点判定成立的必要条件）
    robot = dyn_poses[dyn_hit - 1]
    dist = float(np.hypot(hit_center[0] - robot[0], hit_center[1] - robot[1]))
    half_diag = float(np.hypot(0.38 / 2 + 0.03, 0.20 / 2 + 0.03))
    assert dist < hit_center[2] + 0.03 + half_diag, (dist, hit_center, robot)


# ═════════════════════════ LiDAR：看得见 ═════════════════════════
def test_lidar_sees_dynamic_obstacle() -> None:
    """同一步有/无动态障碍的 front 扫描：某条射线明显变近（给出索引与数值）。"""
    base = _bridge()
    _reset(base)
    _straight(base, 10)  # t = 1.0 s：车在 (2.0, 5.0)，圆心在 (3.0, 4.7)
    dyn = _bridge([CROSSING])
    _reset(dyn)
    _straight(dyn, 10)

    d_base = base.get_lidar_front()
    d_dyn = dyn.get_lidar_front()
    idx = int(np.argmin(d_dyn))
    assert d_dyn[idx] < d_base[idx] - 0.5, (idx, float(d_dyn[idx]), float(d_base[idx]))
    # 障碍在右前方 ⇒ 该射线方向约 -17°（车头为 0），不是正前方
    angle = float(-np.pi / 2.0 + idx * (np.pi / 179.0))
    assert -1.2 < angle < 0.0, angle
    assert float(d_dyn.min()) < float(d_base.min()) - 0.5


def test_lidar_unchanged_when_dynamic_list_is_empty() -> None:
    """默认（无动态障碍）构造与前向兼容：两条独立 bridge 的扫描逐位相同。"""
    a, b = _bridge(), _bridge([])
    _reset(a)
    _reset(b)
    _straight(a, 10)
    _straight(b, 10)
    assert np.array_equal(a.get_lidar_front(), b.get_lidar_front())


# ═════════════════════════ 确定性 ═════════════════════════
def test_reset_plus_fifty_steps_is_bitwise_reproducible() -> None:
    """同一场景两次完整 reset + 50 步：lidar 与位姿逐位相等（max|Δ| = 0）。"""

    def run() -> tuple[np.ndarray, np.ndarray]:
        bridge = _bridge([CROSSING])
        _reset(bridge)
        lidar = [bridge.get_lidar_front().copy(), bridge.get_lidar_rear().copy()]
        poses = []
        for _ in range(50):
            bridge.step(np.array([0.3, 1.0]))
            lidar[0] = np.minimum(lidar[0], bridge.get_lidar_front())
            lidar[1] = np.minimum(lidar[1], bridge.get_lidar_rear())
            poses.append(bridge.get_odometry()[:3].copy())
        return np.concatenate(lidar), np.concatenate(poses)

    lidar_a, pose_a = run()
    lidar_b, pose_b = run()
    assert np.array_equal(lidar_a, lidar_b)
    assert np.array_equal(pose_a, pose_b)
    assert float(np.max(np.abs(lidar_a - lidar_b))) == 0.0
    assert float(np.max(np.abs(pose_a - pose_b))) == 0.0


def test_set_dynamic_obstacles_takes_effect_without_reset() -> None:
    """运行中换一组动态障碍：不改变时间，下一步就按当前 t 求值；[] 清空。"""
    bridge = _bridge()
    _reset(bridge)
    _straight(bridge, 10)  # t = 1.0 s
    bridge.set_dynamic_obstacles([dict(CROSSING)])
    assert np.array_equal(
        bridge.dynamic_positions()[0, :2], np.array([3.0, 4.4 + 0.3 * 1.0])
    )
    bridge.set_dynamic_obstacles([])
    assert bridge.dynamic_positions().shape == (0, 3)


def test_unknown_dynamic_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="type 未知"):
        _bridge([{"type": "spiral", "radius": 0.2}])
    with pytest.raises(ValueError, match="radius 必须 > 0"):
        _bridge(
            [{"type": "linear", "start": [0, 0], "velocity": [0, 0], "radius": 0.0}]
        )
