"""传感器层单元测试（纯 numpy）：IMU / 轮速计 / 参数化 LiDAR / 延迟 / 丢帧。

两个主题：
1. **数值公式**：IMU 的 ax/ay、轮速计的左右轮差速、LiDAR 的角度几何 / 量化 /
   延迟环形缓冲 / 丢帧语义，全部对**手算值**断言，不是拿实现对自己。
2. **随机源隔离**（硬约束）：传感器的随机数只来自从 `sensor_seed` 派生的独立
   Generator，模块级 `np.random` 序列一个数都不许动；且各路传感器互不移位。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges.sensors import (
    Imu,
    LidarParams,
    ParametricLidar,
    SensorConfig,
    SensorSuite,
    WheelOdometry,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ═════════════════════════════════════════════════════════════════════
# 默认值 / 参数校验
# ═════════════════════════════════════════════════════════════════════


def test_default_lidar_params_match_legacy_geometry() -> None:
    """默认参数必须逐位等于历史 180 线雷达的常量（否则 obs 对拍会碎）。"""
    p = LidarParams()
    assert (p.n_rays, p.max_range, p.range_min) == (180, 10.0, 0.05)
    assert p.fov == float(np.pi)
    assert (p.noise_std, p.quant_step, p.latency_steps, p.dropout_prob) == (
        0.0,
        0.0,
        0,
        0.0,
    )
    assert p.side == "front"
    # 端点包含：179 个间隔铺满 pi（与 get_scan 的 angle_increment 同式）
    assert p.angle_increment == float(np.pi) / 179
    assert p.angle_span == pytest.approx(float(np.pi), abs=0.0)


def test_sensor_config_defaults_are_all_off() -> None:
    cfg = SensorConfig()
    assert cfg.lidar is None
    assert cfg.imu is False and cfg.wheel is False
    assert cfg.enabled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_rays": 1},
        {"fov": 0.0},
        {"fov": float("nan")},
        {"max_range": 0.0},
        {"range_min": 0.0},
        {"range_min": 11.0},
        {"noise_std": -0.1},
        {"quant_step": -0.1},
        {"latency_steps": -1},
        {"dropout_prob": -0.1},
        {"dropout_prob": 1.1},
        {"side": "left"},
    ],
)
def test_lidar_params_reject_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LidarParams(**kwargs)


def test_sensor_config_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        SensorConfig(imu_noise_std=-1.0)
    with pytest.raises(ValueError):
        SensorConfig(wheel_noise_std=-1.0)
    with pytest.raises(ValueError):
        SensorConfig(track_width=0.0)
    with pytest.raises(ValueError):
        SensorConfig(sensor_seed=-1)


# ═════════════════════════════════════════════════════════════════════
# 参数化 LiDAR：延迟 / 丢帧 / 量化 / 噪声
# ═════════════════════════════════════════════════════════════════════


def test_latency_ring_buffer_delivers_k_minus_latency() -> None:
    """latency=3：第 k 步交付第 k-3 步采集的帧，前 3 步是预热期（None）。"""
    lidar = ParametricLidar(LidarParams(n_rays=4, latency_steps=3), _rng(0))
    delivered: list = []
    frame_ts: list = []
    for k in range(1, 8):
        raw = np.full(4, float(k))
        scan, meta = lidar.update(raw, t=0.1 * k)
        delivered.append(None if scan is None else float(scan[0]))
        frame_ts.append(meta["frame_t"])
        assert meta["t"] == pytest.approx(0.1 * k)
        assert meta["valid"] is (scan is not None)
    assert delivered == [None, None, None, 1.0, 2.0, 3.0, 4.0]
    expected_ts = [None, None, None, 0.1, 0.2, 0.3, 0.4]
    for got, want in zip(frame_ts, expected_ts):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want)


def test_zero_latency_delivers_current_frame() -> None:
    lidar = ParametricLidar(LidarParams(n_rays=3), _rng(0))
    scan, meta = lidar.update(np.array([1.0, 2.0, 3.0]), t=0.5)
    assert scan is not None
    assert np.array_equal(scan, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert meta["frame_t"] == pytest.approx(0.5)


def test_dropout_one_drops_every_frame_without_crashing() -> None:
    lidar = ParametricLidar(LidarParams(n_rays=8, dropout_prob=1.0), _rng(1))
    for k in range(10):
        scan, meta = lidar.update(np.linspace(0.5, 8.0, 8), t=0.1 * k)
        assert scan is None
        assert meta["valid"] is False
        assert meta["dropped"] is True
        assert meta["frame_t"] is None


def test_quantization_snaps_to_step_multiples() -> None:
    lidar = ParametricLidar(LidarParams(n_rays=3, quant_step=0.1), _rng(0))
    scan, _ = lidar.update(np.array([1.24, 1.26, 2.04]), t=0.0)
    assert scan is not None
    assert np.allclose(scan, np.array([1.2, 1.3, 2.0]), atol=1e-6)
    # 每个读数都是步长的整数倍（float32 下用 ULP 级容差）
    q = scan.astype(np.float64) / 0.1
    assert np.allclose(q, np.round(q), atol=1e-6)


def test_noise_is_nonzero_and_seed_reproducible() -> None:
    raw = np.full(180, 2.0)
    a = ParametricLidar(LidarParams(noise_std=0.05), _rng(7))
    b = ParametricLidar(LidarParams(noise_std=0.05), _rng(7))
    c = ParametricLidar(LidarParams(noise_std=0.05), _rng(8))
    sa, _ = a.update(raw, t=0.0)
    sb, _ = b.update(raw, t=0.0)
    sc, _ = c.update(raw, t=0.0)
    assert sa is not None and sb is not None and sc is not None
    assert float(np.std(sa.astype(np.float64) - 2.0)) > 0.0
    assert np.array_equal(sa, sb), "同 seed 必须逐位复现"
    assert not np.array_equal(sa, sc), "换 seed 必须变"


def test_scan_meta_geometry_matches_params() -> None:
    p = LidarParams(n_rays=90, fov=float(np.pi) / 2.0)
    lidar = ParametricLidar(p, _rng(0), mount_yaw=float(np.pi))
    _, meta = lidar.update(np.full(90, 1.0), t=0.0)
    assert meta["angle_increment"] == pytest.approx((np.pi / 2.0) / 89)
    assert meta["angle_min"] == pytest.approx(np.pi - np.pi / 4.0)
    assert meta["angle_max"] == pytest.approx(np.pi + np.pi / 4.0)
    assert meta["n_rays"] == 90 and meta["side"] == "front"
    assert np.isclose(
        meta["angle_max"] - meta["angle_min"], 89 * meta["angle_increment"]
    )


def test_update_rejects_wrong_ray_count() -> None:
    lidar = ParametricLidar(LidarParams(n_rays=8), _rng(0))
    with pytest.raises(ValueError):
        lidar.update(np.zeros(7), t=0.0)


# ═════════════════════════════════════════════════════════════════════
# IMU / 轮速计：手算值
# ═════════════════════════════════════════════════════════════════════


def test_imu_formulas_and_first_step() -> None:
    imu = Imu(dt=0.1, rng=_rng(0))
    first = imu.update(v=0.5, vy=0.0, omega=0.2, theta=0.1, t=0.1)
    assert first["ax"] == 0.0, "首步没有速度历史 ⇒ 纵向加速度为 0"
    assert first["ay"] == pytest.approx(0.5 * 0.2)
    assert first["yaw_rate"] == pytest.approx(0.2)
    assert first["yaw"] == pytest.approx(0.1)
    assert first["t"] == pytest.approx(0.1)

    second = imu.update(v=0.8, vy=0.0, omega=-0.1, theta=0.3, t=0.2)
    assert second["ax"] == pytest.approx((0.8 - 0.5) / 0.1)
    assert second["ay"] == pytest.approx(0.8 * -0.1)
    assert second["yaw_rate"] == pytest.approx(-0.1)


def test_imu_lateral_term_uses_vy_difference_for_mecanum() -> None:
    imu = Imu(dt=0.1, rng=_rng(0))
    imu.update(v=0.5, vy=0.0, omega=0.0, theta=0.0, t=0.1)
    out = imu.update(v=0.5, vy=0.25, omega=0.0, theta=0.0, t=0.2)
    assert out["ay"] == pytest.approx((0.25 - 0.0) / 0.1)


def test_imu_reset_clears_velocity_history() -> None:
    imu = Imu(dt=0.1, rng=_rng(0))
    imu.update(v=1.0, vy=0.0, omega=0.0, theta=0.0, t=0.1)
    imu.reset()
    out = imu.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.1)
    assert out["ax"] == 0.0, "reset 后首步不能再拿上一回合末速度做差分"


def test_wheel_odometry_differential_formula() -> None:
    wheel = WheelOdometry(rng=_rng(0), track_width=0.2)
    out = wheel.update(v=1.0, omega=0.5, t=0.3)
    assert out["v_left"] == pytest.approx(1.0 - 0.5 * 0.2 / 2.0)
    assert out["v_right"] == pytest.approx(1.0 + 0.5 * 0.2 / 2.0)
    assert out["v"] == pytest.approx(1.0)
    assert out["t"] == pytest.approx(0.3)


def test_wheel_noise_nonzero_and_midpoint_consistency() -> None:
    wheel = WheelOdometry(rng=_rng(3), track_width=0.2, noise_std=0.01)
    outs = [wheel.update(v=0.8, omega=0.0, t=0.1 * k) for k in range(1, 21)]
    diffs = [o["v_left"] - 0.8 for o in outs]
    assert float(np.std(diffs)) > 0.0
    assert all(o["v"] == pytest.approx(0.5 * (o["v_left"] + o["v_right"])) for o in outs)


# ═════════════════════════════════════════════════════════════════════
# 随机源隔离
# ═════════════════════════════════════════════════════════════════════


def test_sensors_never_touch_global_np_random() -> None:
    """正反证据：跑一遍传感器后，全局 np.random 的下一个数与不跑时完全相同。"""
    np.random.seed(12345)
    expected = np.random.random(5)

    np.random.seed(12345)
    cfg = SensorConfig(
        lidar=LidarParams(n_rays=64, noise_std=0.05, quant_step=0.002, dropout_prob=0.3),
        imu=True,
        wheel=True,
        imu_noise_std=0.02,
        wheel_noise_std=0.02,
        sensor_seed=99,
    )
    suite = SensorSuite(
        cfg,
        dt=0.1,
        fallback_track_width=0.2,
        lidar_mounts=((0.145, 0.0, 0.0), (-0.155, 0.0, float(np.pi))),
    )
    for k in range(20):
        suite.update(
            v=0.5,
            vy=0.0,
            omega=0.1,
            theta=0.0,
            t=0.1 * k,
            raw_scan=np.linspace(0.5, 9.0, 64),
        )
    actual = np.random.random(5)
    assert np.array_equal(expected, actual), "传感器消耗了全局 np.random 序列"


def test_lidar_stream_independent_of_other_sensors() -> None:
    """开/关 IMU 与轮速计不得平移 LiDAR 的随机流（SeedSequence.spawn 分路）。"""
    mounts = ((0.145, 0.0, 0.0), (-0.155, 0.0, float(np.pi)))
    only_lidar = SensorSuite(
        SensorConfig(lidar=LidarParams(n_rays=32, noise_std=0.05), sensor_seed=5),
        dt=0.1,
        fallback_track_width=0.2,
        lidar_mounts=mounts,
    )
    all_on = SensorSuite(
        SensorConfig(
            lidar=LidarParams(n_rays=32, noise_std=0.05),
            imu=True,
            wheel=True,
            imu_noise_std=0.05,
            wheel_noise_std=0.05,
            sensor_seed=5,
        ),
        dt=0.1,
        fallback_track_width=0.2,
        lidar_mounts=mounts,
    )
    raw = np.full(32, 3.0)
    a, b = [], []
    for k in range(5):
        a.append(only_lidar.update(v=0.5, vy=0.0, omega=0.0, theta=0.0, t=0.1 * k, raw_scan=raw)["scan"])
        b.append(all_on.update(v=0.5, vy=0.0, omega=0.0, theta=0.0, t=0.1 * k, raw_scan=raw)["scan"])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert set(only_lidar.update(v=0.5, vy=0.0, omega=0.0, theta=0.0, t=1.0, raw_scan=raw)) == {
        "scan",
        "scan_meta",
    }


def test_suite_only_emits_enabled_keys() -> None:
    mounts = ((0.145, 0.0, 0.0), (-0.155, 0.0, float(np.pi)))
    suite = SensorSuite(
        SensorConfig(imu=True, sensor_seed=1),
        dt=0.1,
        fallback_track_width=0.2,
        lidar_mounts=mounts,
    )
    reading = suite.update(v=0.5, vy=0.0, omega=0.0, theta=0.0, t=0.1)
    assert set(reading) == {"imu"}
    assert suite.lidar_params is None
    assert suite.track_width == 0.2
