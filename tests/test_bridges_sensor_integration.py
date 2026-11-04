"""SimpleBridge 传感器接线测试：列表式 spec / 几何注入 / 随机流隔离 / 退化模型。

四条硬约束各有一组断言：

1. 传感器**不动**其它读数：加不加传感器，get_odometry 的返回值逐位相同；
   雷达帧只经 get_sensor_reading 的 lidar 条目交付。
2. 传感器**不消耗**全局 np.random：domain_rand=True 时开关传感器，LiDAR 与里程计
   序列 max|Δ| = 0。
3. 几何由桥注入：两支不同 mount 的 LiDAR 在同一场景下测距不同，且 scan_meta 的
   angle_min 反映 mount 的 yaw；默认 mount + 默认几何与旧接口逐位对拍。
4. 退化模型不变：量化（0.1 的整数倍）、延迟（第 k 步交付第 k-2 步）、丢帧（1.0 全丢
   / 0.0 不丢）。

env 侧（info 组装、sensors= 与 bridge= 互斥）属于 S3，本文件不 import
chassis_sim.envs。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.sensors import Imu, Lidar, WheelOdom

_STEPS = 20
_DT = 0.1

_PATH = np.column_stack([np.linspace(1.0, 6.0, 40), np.full(40, 2.0), np.zeros(40)])
_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.30}, "state": [3.0, 2.5, 0.0]},
    {
        "shape": {"name": "rectangle", "length": 0.60, "width": 0.40},
        "state": [4.2, 1.4, 0.4],
    },
]
_ACTIONS = np.tile(np.array([0.05, 0.5]), (_STEPS, 1))


def _make_bridge(
    sensors=None, *, domain_rand: bool = False, sensor_seed: int = 0
) -> SimpleBridge:
    return SimpleBridge(
        AckermannChassis(),
        domain_rand=domain_rand,
        sensors=sensors,
        sensor_seed=sensor_seed,
    )


def _reset(bridge: SimpleBridge) -> None:
    bridge.reset(
        obstacles=[dict(o) for o in _OBSTACLES],
        path_waypoints=_PATH.copy(),
        initial_heading=0.0,
    )


def _run(bridge: SimpleBridge, steps: int = _STEPS) -> list[dict]:
    """推进 steps 步，逐步取读数（未启用传感器时为空字典）。"""
    readings: list[dict] = []
    for action in _ACTIONS[:steps]:
        bridge.step(action)
        readings.append(bridge.get_sensor_reading() or {})
    return readings


def _stock_specs() -> list:
    """一套"厂配"传感器：前雷达用底盘的真实安装位，外加 IMU 与轮速计。"""
    front, rear = AckermannChassis().lidar_mounts()
    return [
        Lidar("front", mount=front),
        Lidar("rear", mount=rear),
        Imu("imu"),
        WheelOdom("wheel"),
    ]


# ═════════════════════════════════════════════════════════════════════
# 1. 关闭：读数为 None（env 的 info 键属于 S3）
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("sensors", [None, []], ids=["none", "empty-list"])
def test_disabled_sensors_yield_no_reading(sensors) -> None:
    bridge = _make_bridge(sensors)
    _reset(bridge)
    for _ in range(3):
        bridge.step(_ACTIONS[0])
        assert bridge.get_sensor_reading() is None


def test_reading_keys_are_sensor_names_in_spec_order() -> None:
    specs = _stock_specs()
    bridge = _make_bridge(specs)
    _reset(bridge)
    bridge.step(_ACTIONS[0])
    reading = bridge.get_sensor_reading()
    assert list(reading) == ["front", "rear", "imu", "wheel"]
    assert set(reading["front"]) == {"scan", "scan_meta"}
    assert set(reading["imu"]) == {"ax", "ay", "yaw_rate", "yaw", "t"}
    assert set(reading["wheel"]) == {"v_left", "v_right", "v", "t"}


# ═════════════════════════════════════════════════════════════════════
# 2. 旧接口逐位不变 + 全局随机流不被消耗
# ═════════════════════════════════════════════════════════════════════


def test_sensors_do_not_change_other_interfaces() -> None:
    """装配传感器不改变其它读数：odometry 逐位相同；无传感器时读数是 None。"""
    plain = _make_bridge(None)
    loaded = _make_bridge(_stock_specs())
    _reset(plain)
    _reset(loaded)
    for action in _ACTIONS:
        plain.step(action)
        loaded.step(action)
        assert np.array_equal(plain.get_odometry(), loaded.get_odometry())
    assert plain.get_sensor_reading() is None
    assert sorted(loaded.get_sensor_reading()) == ["front", "imu", "rear", "wheel"]


def test_sensors_do_not_consume_global_np_random() -> None:
    """domain_rand=True 时开关传感器，里程计漂移序列一个数都不许平移。"""

    def run(sensors, domain_rand: bool = True) -> np.ndarray:
        np.random.seed(12345)
        bridge = _make_bridge(sensors, domain_rand=domain_rand)
        _reset(bridge)
        odoms = []
        for action in _ACTIONS:
            bridge.step(action)
            odoms.append(bridge.get_odometry())
        return np.array(odoms)

    odom_a = run(None)
    odom_b = run(_stock_specs())
    assert np.array_equal(odom_a, odom_b), (
        f"odometry max|Δ| = {np.max(np.abs(odom_a - odom_b))}"
    )
    # 反向对照：domain_rand 本身确实带来了漂移（否则上面的相等可能是平凡成立）
    assert not np.array_equal(odom_a, run(_stock_specs(), domain_rand=False))


# ═════════════════════════════════════════════════════════════════════
# 3. 几何：mount 决定测距与 angle_min；与旧接口对拍
# ═════════════════════════════════════════════════════════════════════


def test_two_mounts_measure_differently_and_report_their_yaw() -> None:
    front, rear = AckermannChassis().lidar_mounts()
    bridge = _make_bridge([Lidar("front", mount=front), Lidar("rear", mount=rear)])
    _reset(bridge)
    bridge.step(_ACTIONS[0])
    reading = bridge.get_sensor_reading()

    front_scan = reading["front"]["scan"]
    rear_scan = reading["rear"]["scan"]
    assert not np.array_equal(front_scan, rear_scan)
    assert reading["front"]["scan_meta"]["angle_min"] == pytest.approx(-np.pi / 2.0)
    assert reading["rear"]["scan_meta"]["angle_min"] == pytest.approx(np.pi / 2.0)
    assert reading["front"]["scan_meta"]["mount"] == pytest.approx(front)
    assert reading["rear"]["scan_meta"]["mount"] == pytest.approx(rear)
    assert reading["front"]["scan_meta"]["name"] == "front"
    assert reading["front"]["scan_meta"]["angle_max"] == pytest.approx(np.pi / 2.0)


def test_scan_meta_keeps_legacy_keys_and_adds_name_mount() -> None:
    bridge = _make_bridge([Lidar("scan")])
    _reset(bridge)
    bridge.step(_ACTIONS[0])
    meta = bridge.get_sensor_reading()["scan"]["scan_meta"]
    legacy_keys = {
        "n_rays",
        "fov",
        "angle_min",
        "angle_max",
        "angle_increment",
        "range_min",
        "range_max",
        "latency_steps",
        "t",
        "frame_t",
        "valid",
        "dropped",
    }
    assert legacy_keys <= set(meta)
    assert {"name", "mount"} <= set(meta)
    # 旧 schema 的 side 由 name + mount 取代
    assert "side" not in meta


# ═════════════════════════════════════════════════════════════════════
# 4. 随机流隔离：加第三支不改变前两支（含噪声）
# ═════════════════════════════════════════════════════════════════════


def test_adding_a_sensor_does_not_change_the_first_two() -> None:
    front, rear = AckermannChassis().lidar_mounts()
    base = [
        Lidar("front", mount=front, noise_std=0.05),
        Lidar("rear", mount=rear, noise_std=0.05),
    ]
    extra = [
        Lidar("extra", mount=(0.0, 0.4, -0.3), noise_std=0.5, n_rays=32),
        Imu("extra_imu", noise_std=0.2),
    ]
    two = _make_bridge(base, sensor_seed=99)
    four = _make_bridge(base + extra, sensor_seed=99)
    _reset(two)
    _reset(four)

    for step in range(_STEPS):
        action = _ACTIONS[step]
        two.step(action)
        four.step(action)
        a = two.get_sensor_reading()
        b = four.get_sensor_reading()
        for name in ("front", "rear"):
            assert np.array_equal(a[name]["scan"], b[name]["scan"])
            assert a[name]["scan_meta"] == b[name]["scan_meta"]

    # 反向对照：同一几何、noise_std=0 的"静默"配置读数必须不同，
    # 否则上面的逐位相等可能只是因为根本没加噪声。
    quiet_specs = [
        Lidar("front", mount=front),
        Lidar("rear", mount=rear),
    ]
    quiet = _make_bridge(quiet_specs, sensor_seed=99)
    _reset(quiet)
    quiet.step(_ACTIONS[-1])
    quiet_scan = quiet.get_sensor_reading()["front"]["scan"]
    assert not np.array_equal(quiet_scan, a["front"]["scan"])


def test_imu_and_wheel_follow_the_bridge_state() -> None:
    bridge = _make_bridge([Imu("imu"), WheelOdom("wheel")])
    _reset(bridge)
    readings = _run(bridge, steps=4)
    # 逐步重放桥的内部状态：odom = [x, y, theta, v, omega]（domain_rand=False ⇒ 无漂移）
    bridge2 = _make_bridge([Imu("imu"), WheelOdom("wheel")])
    _reset(bridge2)
    # S3b：reset() 末尾已产出 t=0、v=0 的首帧，因此第一步的 ax 是相对它求差分
    # （不再走"无历史 ⇒ ax=0"的首拍分支）。
    prev_v = 0.0
    for step, reading in enumerate(readings):
        bridge2.step(_ACTIONS[step])
        odom = bridge2.get_odometry()
        theta, v, omega = float(odom[2]), float(odom[3]), float(odom[4])
        assert reading["imu"]["t"] == pytest.approx(bridge2.sim_time)
        assert reading["imu"]["yaw"] == pytest.approx(theta, abs=1e-6)
        assert reading["imu"]["yaw_rate"] == pytest.approx(omega, abs=1e-6)
        expected_ax = 0.0 if prev_v is None else (v - prev_v) / _DT
        assert reading["imu"]["ax"] == pytest.approx(expected_ax, abs=1e-6)
        # 轮速计：轮距缺省取底盘 footprint()[1]
        half = 0.5 * float(AckermannChassis().footprint()[1])
        assert reading["wheel"]["v_left"] == pytest.approx(v - omega * half, abs=1e-6)
        assert reading["wheel"]["v_right"] == pytest.approx(v + omega * half, abs=1e-6)
        prev_v = v


def test_wheel_track_width_defaults_to_chassis_footprint() -> None:
    bridge = _make_bridge([WheelOdom("wheel")])
    assert bridge._sensors is not None
    assert bridge._sensors.track_width == pytest.approx(
        float(AckermannChassis().footprint()[1])
    )
    explicit = _make_bridge([WheelOdom("wheel", track_width=0.75)])
    assert explicit._sensors is not None
    assert explicit._sensors.track_width == pytest.approx(0.75)


# ═════════════════════════════════════════════════════════════════════
# 5. 退化模型：延迟 / 丢帧 / 量化
# ═════════════════════════════════════════════════════════════════════


def test_latency_two_delivers_none_then_a_lagged_frame() -> None:
    """S3b：reset() 末尾产出的 t=0 首帧同样进延迟缓冲，预热两拍后才交付。"""
    bridge = _make_bridge([Lidar("scan", latency_steps=2)])
    _reset(bridge)
    # 预热期未结束：reset 后的读数里 scan 仍是 None（交付方据 valid=False 判定）
    assert bridge.get_sensor_reading()["scan"]["scan"] is None
    assert bridge.get_sensor_reading()["scan"]["scan_meta"]["valid"] is False
    readings = _run(bridge, steps=5)
    # 缓冲长度 latency+1：reset 帧 + 其后一拍仍在预热，第二拍起交付
    assert readings[0]["scan"]["scan"] is None
    for step in range(1, 5):
        meta = readings[step]["scan"]["scan_meta"]
        assert readings[step]["scan"]["scan"] is not None
        assert meta["t"] == pytest.approx((step + 1) * _DT)
        assert meta["frame_t"] == pytest.approx((step - 1) * _DT)
        assert meta["valid"] is True


def test_dropout_one_drops_every_step_and_zero_drops_none() -> None:
    def n_dropped(prob: float) -> int:
        bridge = _make_bridge([Lidar("scan", dropout_prob=prob)], sensor_seed=17)
        _reset(bridge)
        readings = _run(bridge)
        return sum(1 for r in readings if r["scan"]["scan"] is None)

    assert n_dropped(1.0) == _STEPS
    assert n_dropped(0.0) == 0


def test_quantized_ranges_are_multiples_of_the_step() -> None:
    bridge = _make_bridge([Lidar("scan", quant_step=0.1)])
    _reset(bridge)
    readings = _run(bridge, steps=5)
    for reading in readings:
        scan = reading["scan"]["scan"].astype(np.float64)
        assert np.allclose(scan, np.round(scan / 0.1) * 0.1, atol=1e-5)
        # 读数跨多个台阶（空世界下 180 条射线的距离本来就各不相同）
        assert len(np.unique(np.round(scan / 0.1))) > 10
