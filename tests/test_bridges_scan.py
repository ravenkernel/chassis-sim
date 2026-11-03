"""get_scan() 的 ROS LaserScan 元数据自洽性 + scan_to_pointcloud 的解析几何。

测试用的位姿是**解析可算**的：机器人停在 10m x 10m 世界正中 (5, 5)、车头朝 +x，
无遮挡 ⇒ 每条射线都撞在某一面墙上，前端雷达安装位 (0.145, 0) 因此位于
(5.145, 5.0)，指向 +x 的射线落点 x 恒为 10.0。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.bridges.base import SCAN_FIELDS, scan_to_pointcloud
from chassis_sim.chassis import DiffDriveChassis

DT = 0.1
MAX_RANGE = 10.0
# 与 SimpleBridge 的扫描几何常量对拍（元数据必须由它们推导，不能另写一份）。
N_RAYS = SimpleBridge.N_LIDAR_RAYS
FOV = SimpleBridge.LIDAR_FOV
RANGE_MIN = SimpleBridge.RANGE_MIN
FRONT_OFFSET_X = 0.145  # Chassis.FRONT_LIDAR[0]：前雷达相对后轴中心的安装偏移
WORLD_MAX = 10.0


def _bridge() -> SimpleBridge:
    """世界正中 (5, 5)、车头 +x、无遮挡的桥。"""
    bridge = SimpleBridge(DiffDriveChassis(), dt=DT)
    path = np.array([[5.0, 5.0, 0.0], [6.0, 5.0, 0.0]])
    bridge.reset(obstacles=[], path_waypoints=path, initial_heading=0.0)
    return bridge


def _synthetic_scan(angle_min: float, angle_increment: float, ranges) -> dict:
    """手搓一个最小扫描字典（只有几何三个键被 scan_to_pointcloud 使用）。"""
    arr = np.asarray(ranges, dtype=np.float32)
    return {
        "angle_min": float(angle_min),
        "angle_max": float(angle_min + angle_increment * (arr.size - 1)),
        "angle_increment": float(angle_increment),
        "time_increment": 0.0,
        "scan_time": 0.0,
        "range_min": RANGE_MIN,
        "range_max": MAX_RANGE,
        "ranges": arr,
        "intensities": None,
        "frame_id": "laser_test",
    }


# ── 元数据：字段、顺序、自洽 ──────────────────────────────────────────
def test_scan_has_exactly_the_ten_ros_fields_in_order() -> None:
    scan = _bridge().get_scan("front")
    assert len(SCAN_FIELDS) == 10
    assert SCAN_FIELDS == (
        "angle_min",
        "angle_max",
        "angle_increment",
        "time_increment",
        "scan_time",
        "range_min",
        "range_max",
        "ranges",
        "intensities",
        "frame_id",
    )
    assert list(scan.keys()) == list(SCAN_FIELDS)


def test_front_scan_metadata_is_self_consistent() -> None:
    scan = _bridge().get_scan("front")
    ranges = scan["ranges"]
    n = len(ranges)

    assert isinstance(ranges, np.ndarray)
    assert ranges.dtype == np.float32
    assert ranges.shape == (N_RAYS,)
    assert np.all(np.isfinite(ranges))

    # 前视场 = [-pi/2, +pi/2]（单位弧度、相对车头）
    assert scan["angle_min"] == pytest.approx(-FOV / 2.0, abs=1e-15)
    assert scan["angle_max"] == pytest.approx(FOV / 2.0, abs=1e-15)

    # 约定：angle_increment == (angle_max - angle_min) / (n - 1)
    assert scan["angle_increment"] == pytest.approx(
        (scan["angle_max"] - scan["angle_min"]) / (n - 1), rel=0, abs=1e-12
    )
    assert scan["angle_increment"] == pytest.approx(
        FOV / (N_RAYS - 1), rel=0, abs=1e-12
    )
    # 端点 + 步长 + 长度三者互洽：angle_min + (n-1)*inc 必须回到 angle_max
    assert scan["angle_min"] + (n - 1) * scan["angle_increment"] == pytest.approx(
        scan["angle_max"], rel=0, abs=1e-12
    )

    assert scan["range_min"] == pytest.approx(RANGE_MIN)
    assert scan["range_max"] == pytest.approx(MAX_RANGE)
    # 瞬时快照（同一帧位姿发全部射线）⇒ 扫掠时间精确为 0
    assert scan["time_increment"] == 0.0
    assert scan["scan_time"] == 0.0
    assert scan["intensities"] is None
    assert scan["frame_id"] == "laser_front"


def test_rear_scan_metadata_is_unwrapped_and_self_consistent() -> None:
    scan = _bridge().get_scan("rear")
    n = len(scan["ranges"])

    # 后雷达安装朝向 = pi ⇒ 方位角区间 [pi/2, 3pi/2]，**不**折到 [-pi, pi]
    assert scan["angle_min"] == pytest.approx(np.pi / 2.0, abs=1e-15)
    assert scan["angle_max"] == pytest.approx(3.0 * np.pi / 2.0, abs=1e-15)
    assert scan["angle_increment"] == pytest.approx(
        (scan["angle_max"] - scan["angle_min"]) / (n - 1), rel=0, abs=1e-12
    )
    assert scan["frame_id"] == "laser_rear"
    assert scan["ranges"].shape == (N_RAYS,)


def test_scan_ranges_equal_the_matching_lidar_call() -> None:
    """同一侧、同一方位：有限读数与 get_lidar_*() 逐元素（bit 级）相等。"""
    bridge = _bridge()
    front = bridge.get_scan("front")
    rear = bridge.get_scan("rear")
    raw_front = bridge.get_lidar_front()
    raw_rear = bridge.get_lidar_rear()

    # 本场景无遮挡、机器人居中 ⇒ 无饱和射线，两个数组直接相等
    assert np.all(np.isfinite(front["ranges"]))
    assert np.all(np.isfinite(rear["ranges"]))
    np.testing.assert_array_equal(front["ranges"], raw_front)
    np.testing.assert_array_equal(rear["ranges"], raw_rear)


def test_no_return_is_inf_not_range_max() -> None:
    """量程内无返回报 +inf——不拿 range_max 冒充有效测距；原始 lidar 行为不变。"""
    bridge = SimpleBridge(DiffDriveChassis(), dt=DT, max_range=5.0)
    path = np.array([[5.0, 5.0, 0.0], [6.0, 5.0, 0.0]])
    bridge.reset(obstacles=[], path_waypoints=path, initial_heading=0.0)

    scan = bridge.get_scan("front")
    raw = bridge.get_lidar_front()
    ranges = scan["ranges"]
    n = len(ranges)

    # 解析解：前雷达在 (5.145, 5)，指向 +x 的射线撞右墙 x = 10
    angles = scan["angle_min"] + np.arange(n) * scan["angle_increment"]
    analytic = (WORLD_MAX - (5.0 + FRONT_OFFSET_X)) / np.cos(angles)
    finite_expected = analytic < 5.0

    assert 0 < int(finite_expected.sum()) < n  # 本场景必须同时存在两种射线
    assert np.array_equal(np.isfinite(ranges), finite_expected)
    assert np.all(np.isinf(ranges[~finite_expected]))
    # 饱和射线的原始读数是 range_max（get_lidar_front() 行为未变）
    assert np.all(raw[~finite_expected] == pytest.approx(5.0))
    # 有限射线逐元素相等，且量程取扫描字典里的 range_max
    np.testing.assert_array_equal(ranges[finite_expected], raw[finite_expected])
    assert scan["range_max"] == pytest.approx(5.0)

    # scan_to_pointcloud 只保留有限值的射线
    pts = scan_to_pointcloud(scan, origin=(5.145, 5.0), heading=0.0)
    assert pts.shape == (int(finite_expected.sum()), 2)
    assert pts.dtype == np.float64


def test_scan_angles_describe_the_actual_rays() -> None:
    """用墙的解析解反查：元数据里的角度确实就是射线角度，不是自说自话。"""
    bridge = _bridge()
    scan = bridge.get_scan("front")
    ranges = scan["ranges"].astype(np.float64)
    angles = scan["angle_min"] + np.arange(ranges.size) * scan["angle_increment"]

    # |angle| < 0.7 rad 时最近的面是右墙 x = 10（夹角更大时才轮到 y 墙）
    mask = np.abs(angles) < 0.7
    assert int(mask.sum()) == 80  # [-0.7, 0.7] 内正好 80 根（i = 50..129）
    expected = (WORLD_MAX - (5.0 + FRONT_OFFSET_X)) / np.cos(angles[mask])
    np.testing.assert_allclose(ranges[mask], expected, rtol=0, atol=1e-4)


def test_invalid_side_raises_value_error_listing_legal_values() -> None:
    with pytest.raises(ValueError) as excinfo:
        _bridge().get_scan("left")
    msg = str(excinfo.value)
    assert "'left'" in msg
    assert "'front'" in msg
    assert "'rear'" in msg


# ── scan -> 点云 ─────────────────────────────────────────────────────
def test_scan_to_pointcloud_analytic_rays() -> None:
    """一条已知射线：3 根有效 + 1 根无返回，坐标全解析可算。"""
    scan = _synthetic_scan(0.0, np.pi / 2.0, [1.0, 2.0, np.inf, 3.0])
    pts = scan_to_pointcloud(scan, origin=(1.0, 2.0), heading=0.0)

    assert pts.shape == (3, 2)
    assert pts.dtype == np.float64
    # i=0: 0 rad -> (1+1, 2+0)；i=1: pi/2 -> (1, 2+2)；i=3: 3pi/2 -> (1, 2-3)
    np.testing.assert_allclose(
        pts, [[2.0, 2.0], [1.0, 4.0], [1.0, -1.0]], rtol=0, atol=1e-12
    )


def test_scan_to_pointcloud_applies_heading() -> None:
    scan = _synthetic_scan(0.0, 0.0, [1.0])
    pts = scan_to_pointcloud(scan, origin=(0.0, 0.0), heading=np.pi / 2.0)
    np.testing.assert_allclose(pts, [[0.0, 1.0]], rtol=0, atol=1e-12)


def test_scan_to_pointcloud_drops_nonfinite_and_returns_empty_shape() -> None:
    scan = _synthetic_scan(-np.pi / 2.0, np.pi / 2.0, [np.inf, np.nan, np.inf])
    pts = scan_to_pointcloud(scan)
    assert pts.shape == (0, 2)
    assert pts.dtype == np.float64


def test_real_scan_pointcloud_lands_on_the_four_walls() -> None:
    """整条链路：真实 scan -> 点云，所有落点都该在世界边界上（无遮挡）。"""
    bridge = _bridge()
    scan = bridge.get_scan("front")
    origin = (5.0 + FRONT_OFFSET_X, 5.0)
    pts = scan_to_pointcloud(scan, origin=origin, heading=0.0)

    assert pts.shape == (N_RAYS, 2)
    on_x = np.isclose(pts[:, 0], 0.0, atol=1e-4) | np.isclose(
        pts[:, 0], WORLD_MAX, atol=1e-4
    )
    on_y = np.isclose(pts[:, 1], 0.0, atol=1e-4) | np.isclose(
        pts[:, 1], WORLD_MAX, atol=1e-4
    )
    assert np.all(on_x | on_y), pts[~(on_x | on_y)][:5]
