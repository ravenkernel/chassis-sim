"""SimpleBridge 传感器接线测试：默认关闭逐位不变 / 随机源隔离 / 延迟 / 丢帧 / 几何。

三条硬约束各有一组断言：
1. 关闭时 `get_sensor_reading()` 返回 None 且 env 的 info 不新增任何键；
   默认 `LidarParams()` 的扫描与历史 `get_lidar_front()` / `get_lidar_rear()` **逐位相同**。
2. 打开时**绝不**改动 obs 维度，也**绝不**消耗全局 `np.random`（domain_rand 的序列
   一个数都不许平移）；同一 seed 下开/关传感器，lidar / odom / 位姿 max|Δ| = 0。
3. 延迟 = 环形缓冲（第 k 步交付第 k-latency 步采集的帧）、丢帧 = 定义良好的 None。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.bridges.sensors import LidarParams, SensorConfig
from chassis_sim.chassis import AckermannChassis, make_chassis
from chassis_sim.envs.chassis_env import ChassisSimEnv

_STEPS = 20
_DT = 0.1

_PATH = np.column_stack(
    [np.linspace(1.0, 6.0, 40), np.full(40, 2.0), np.zeros(40)]
)
_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.30}, "state": [3.0, 2.5, 0.0]},
    {
        "shape": {"name": "rectangle", "length": 0.60, "width": 0.40},
        "state": [4.2, 1.4, 0.4],
    },
]
_ACTIONS = np.tile(np.array([0.05, 0.5]), (_STEPS, 1))


def _make_bridge(sensors: SensorConfig | None = None, *, domain_rand: bool = False):
    return SimpleBridge(AckermannChassis(), domain_rand=domain_rand, sensors=sensors)


def _reset(bridge: SimpleBridge) -> None:
    bridge.reset(
        obstacles=[dict(o) for o in _OBSTACLES],
        path_waypoints=_PATH.copy(),
        initial_heading=0.0,
    )


def _sensor_keys(reading: dict) -> set:
    return set(reading)


# ═════════════════════════════════════════════════════════════════════
# 1. 默认关闭：逐位不变 + info 零新键
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("side", "angle_min"), [("front", -np.pi / 2.0), ("rear", np.pi / 2.0)]
)
def test_default_lidar_params_reproduce_legacy_scan_bitwise(
    side: str, angle_min: float
) -> None:
    """默认 LidarParams()（180 线 / π / 10 m / 0.05 m）必须与历史扫描逐位相同。"""
    legacy = _make_bridge()
    sensor = _make_bridge(SensorConfig(lidar=LidarParams(side=side), sensor_seed=3))
    _reset(legacy)
    _reset(sensor)
    for k in range(_STEPS):
        legacy.step(_ACTIONS[k])
        sensor.step(_ACTIONS[k])
        expected = (
            legacy.get_lidar_front() if side == "front" else legacy.get_lidar_rear()
        )
        reading = sensor.get_sensor_reading()
        assert reading is not None and "scan" in reading
        scan = reading["scan"]
        assert scan is not None
        assert scan.shape == (180,)
        assert scan.dtype == np.float32
        assert np.array_equal(scan, expected), f"第 {k + 1} 步 {side} 不再逐位一致"
        meta = reading["scan_meta"]
        assert meta["angle_increment"] == float(np.pi) / 179
        assert meta["angle_min"] == pytest.approx(angle_min)
        assert meta["range_max"] == 10.0 and meta["range_min"] == 0.05


def test_no_sensors_no_reading_and_no_info_keys() -> None:
    bridge = _make_bridge()
    _reset(bridge)
    for k in range(_STEPS):
        bridge.step(_ACTIONS[k])
        assert bridge.get_sensor_reading() is None

    env = ChassisSimEnv("ackermann", max_steps=5)
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == (494,)
        _obs, _r, _term, _trunc, info = env.step(
            np.array([0.0, 0.5], dtype=np.float32)
        )
        assert not ({"imu", "wheel", "scan", "scan_meta"} & set(info))
    finally:
        env.close()


# ═════════════════════════════════════════════════════════════════════
# 2. 随机源隔离：开/关传感器，domain_rand 序列逐位相同
# ═════════════════════════════════════════════════════════════════════


def _record(bridge: SimpleBridge) -> dict:
    lidar, odom, pose = [], [], []
    for k in range(_STEPS):
        bridge.step(_ACTIONS[k])
        lidar.append(bridge.get_lidar_front().copy())
        odom.append(bridge.get_odometry().copy())
        pose.append(np.array([bridge._x, bridge._y, bridge._theta]))
    return {"lidar": lidar, "odom": odom, "pose": pose}


def test_sensors_do_not_shift_domain_rand_sequences() -> None:
    cfg = SensorConfig(
        lidar=LidarParams(noise_std=0.05),
        imu=True,
        wheel=True,
        imu_noise_std=0.05,
        wheel_noise_std=0.05,
        sensor_seed=42,
    )
    np.random.seed(2026)
    off_bridge = _make_bridge(None, domain_rand=True)
    _reset(off_bridge)
    off = _record(off_bridge)

    np.random.seed(2026)
    on_bridge = _make_bridge(cfg, domain_rand=True)
    _reset(on_bridge)
    on = _record(on_bridge)

    for key in ("lidar", "odom", "pose"):
        delta = max(
            float(np.max(np.abs(a - b))) for a, b in zip(off[key], on[key])
        )
        assert delta == 0.0, f"{key} 序列被传感器平移了，max|Δ|={delta}"


def test_sensor_noise_is_actually_nonzero() -> None:
    """反向对照：隔离不能靠"传感器根本不加噪声"来假装通过。"""
    cfg = SensorConfig(
        lidar=LidarParams(noise_std=0.05),
        imu=True,
        wheel=True,
        imu_noise_std=0.05,
        wheel_noise_std=0.05,
        sensor_seed=7,
    )
    clean = _make_bridge()
    noisy = _make_bridge(cfg)
    _reset(clean)
    _reset(noisy)
    lidar_diffs, imu_ax, wheel_diffs = [], [], []
    for k in range(_STEPS):
        clean.step(_ACTIONS[k])
        noisy.step(_ACTIONS[k])
        reading = noisy.get_sensor_reading()
        assert reading is not None
        lidar_diffs.append(reading["scan"] - clean.get_lidar_front())
        imu_ax.append(reading["imu"]["ax"])
        v = float(clean._v)
        omega = float(clean._omega)
        wheel_diffs.append(reading["wheel"]["v_left"] - (v - omega * 0.20 / 2.0))
    assert float(np.std(np.concatenate(lidar_diffs))) > 0.0
    assert float(np.std(imu_ax)) > 0.0
    assert float(np.std(wheel_diffs)) > 0.0


# ═════════════════════════════════════════════════════════════════════
# 3. 延迟与丢帧
# ═════════════════════════════════════════════════════════════════════


def test_latency_three_delivers_scan_from_three_steps_earlier() -> None:
    legacy = _make_bridge()
    sensor = _make_bridge(
        SensorConfig(lidar=LidarParams(latency_steps=3), sensor_seed=0)
    )
    _reset(legacy)
    _reset(sensor)
    truth = []
    for k in range(1, 9):
        legacy.step(_ACTIONS[k - 1])
        sensor.step(_ACTIONS[k - 1])
        truth.append(legacy.get_lidar_front().copy())
        reading = sensor.get_sensor_reading()
        assert reading is not None
        meta = reading["scan_meta"]
        assert meta["t"] == pytest.approx(k * _DT)
        if k <= 3:
            assert reading["scan"] is None, f"预热期第 {k} 步必须还没有可交付帧"
            assert meta["valid"] is False
        else:
            delivered = reading["scan"]
            assert delivered is not None
            assert np.array_equal(delivered, truth[k - 3 - 1]), (
                f"第 {k} 步交付的不是第 {k - 3} 步的扫描"
            )
            assert meta["frame_t"] == pytest.approx((k - 3) * _DT)


def test_full_dropout_returns_none_and_env_survives() -> None:
    cfg = SensorConfig(
        lidar=LidarParams(dropout_prob=1.0, noise_std=0.02), sensor_seed=1
    )
    chassis = make_chassis("ackermann")
    env = ChassisSimEnv(chassis, bridge=SimpleBridge(chassis, sensors=cfg), max_steps=30)
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == (494,)
        for k in range(10):
            obs, _r, terminated, truncated, info = env.step(
                np.array([0.0, 0.4], dtype=np.float32)
            )
            assert info["scan"] is None
            assert info["scan_meta"]["dropped"] is True
            assert info["scan_meta"]["valid"] is False
            assert obs.shape == (494,)
            assert np.isfinite(obs).all()
            if terminated or truncated:
                break
    finally:
        env.close()


# ═════════════════════════════════════════════════════════════════════
# 4. 参数化几何：90 线 / π/2 视场
# ═════════════════════════════════════════════════════════════════════


def test_custom_geometry_90_rays_and_half_pi_fov() -> None:
    """空 10×10 世界里从 (5, 5, 0) 出发：射线距离可解析算出，逐条对拍。"""
    cfg = SensorConfig(
        lidar=LidarParams(n_rays=90, fov=float(np.pi) / 2.0), sensor_seed=0
    )
    bridge = _make_bridge(cfg)
    bridge.reset(
        obstacles=[],
        path_waypoints=np.column_stack(
            [np.linspace(5.0, 9.0, 20), np.full(20, 5.0), np.zeros(20)]
        ),
        initial_heading=0.0,
    )
    bridge.step(np.array([0.0, 0.0]))  # v=0 ⇒ 位姿仍为 (5, 5, 0)
    reading = bridge.get_sensor_reading()
    assert reading is not None
    scan, meta = reading["scan"], reading["scan_meta"]
    assert scan.shape == (90,)
    inc = meta["angle_increment"]
    assert inc == pytest.approx((np.pi / 2.0) / 89, abs=1e-15)
    assert meta["angle_min"] == pytest.approx(-np.pi / 4.0)
    assert meta["angle_max"] == pytest.approx(np.pi / 4.0)

    bearings = meta["angle_min"] + np.arange(90) * inc
    ox = 5.0 + 0.145  # 前雷达安装 x 偏移
    oy = 5.0
    t_x = (10.0 - ox) / np.cos(bearings)
    t_y = np.where(
        np.sin(bearings) >= 0.0,
        (10.0 - oy) / np.sin(bearings),
        (0.0 - oy) / np.sin(bearings),
    )
    expected = np.minimum(t_x, t_y)
    assert np.allclose(scan, expected, atol=1e-5)
    assert np.all(scan > 0.0)


# ═════════════════════════════════════════════════════════════════════
# 5. obs 维度契约：传感器只进 info
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("name", "obs_dim"),
    [("ackermann", 494), ("diff_drive", 494), ("mecanum", 495)],
)
def test_obs_dim_unchanged_with_sensors_enabled(name: str, obs_dim: int) -> None:
    chassis = make_chassis(name)
    cfg = SensorConfig(
        lidar=LidarParams(n_rays=90, fov=float(np.pi) / 2.0, noise_std=0.01),
        imu=True,
        wheel=True,
        sensor_seed=2,
    )
    env = ChassisSimEnv(chassis, bridge=SimpleBridge(chassis, sensors=cfg), max_steps=40)
    rng = np.random.default_rng(0)
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == (obs_dim,)
        for _ in range(20):
            action = rng.uniform(-0.1, 0.1, size=env.action_dim).astype(np.float32)
            obs, _r, terminated, truncated, info = env.step(action)
            assert obs.shape == (obs_dim,), f"{name} 的 obs 维度被传感器改变了"
            assert {"imu", "wheel", "scan", "scan_meta"} <= set(info)
            assert info["scan"].shape == (90,)
            assert info["imu"]["t"] == pytest.approx(info["scan_meta"]["t"])
            if terminated or truncated:
                break
    finally:
        env.close()
