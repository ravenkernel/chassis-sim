"""`chassis_sim.sensors` 单元测试：spec 校验 / 几何 / 随机流隔离 / 退化模型。

三条硬约束各有独立断言组：

1. **随机流按 `(seed, name)` 派生**：跨进程稳定（不依赖 `hash()` 的加盐），
   且增删任何一支传感器都不平移其它支的序列（这是 `spawn` 做不到的）。
2. **几何外置**：套件只把车体系 (原点, 方位角, 量程) 交给注入的 `raycast`，
   自己不碰世界坐标、不 import 桥。
3. **处理顺序固定**：clip -> 噪声 -> clip -> 量化 -> float32 -> 延迟缓冲 -> 丢帧。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from chassis_sim.sensors import Imu, Lidar, SensorSuite, WheelOdom, derive_rng

_DT = 0.1


def _constant(value: float):
    """返回一个恒定距离的 raycast（与角度/原点无关）。"""

    def raycast(
        origin_xy: tuple[float, float], angle_rad: float, max_range: float
    ) -> float:
        return float(value)

    return raycast


def _by_angle(offset: float = 1.0, gain: float = 0.37):
    """返回一个距离随**车体系方位角**线性变化的 raycast。"""

    def raycast(
        origin_xy: tuple[float, float], angle_rad: float, max_range: float
    ) -> float:
        return float(offset + gain * angle_rad)

    return raycast


def _recording(log: list):
    """返回一个把每次调用参数记进 `log` 的 raycast。"""

    def raycast(
        origin_xy: tuple[float, float], angle_rad: float, max_range: float
    ) -> float:
        log.append((tuple(origin_xy), float(angle_rad), float(max_range)))
        return 1.0

    return raycast


def _run_steps(
    suite: SensorSuite,
    n: int,
    *,
    v: float = 1.0,
    vy: float = 0.0,
    omega: float = 0.3,
    theta: float = 0.1,
) -> list[dict]:
    return [
        suite.update(v=v, vy=vy, omega=omega, theta=theta, t=i * _DT) for i in range(n)
    ]


# ═════════════════════════════════════════════════════════════════════
# 1. spec：默认值 / 不可变 / 参数校验
# ═════════════════════════════════════════════════════════════════════


def test_lidar_defaults_match_documentation() -> None:
    spec = Lidar("scan")
    assert spec.mount == (0.0, 0.0, 0.0)
    assert spec.fov == pytest.approx(np.pi)
    assert spec.n_rays == 180
    assert spec.max_range == pytest.approx(10.0)
    assert spec.range_min == pytest.approx(0.05)
    assert spec.noise_std == 0.0
    assert spec.quant_step == 0.0
    assert spec.latency_steps == 0
    assert spec.dropout_prob == 0.0


def test_imu_and_wheel_defaults() -> None:
    assert Imu("imu").noise_std == 0.0
    wheel = WheelOdom("wheel")
    assert wheel.track_width is None
    assert wheel.noise_std == 0.0


def test_specs_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    spec = Lidar("scan")
    with pytest.raises(FrozenInstanceError):
        spec.n_rays = 90  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: Lidar(""), id="empty-name"),
        pytest.param(lambda: Lidar("x", n_rays=1), id="n_rays<2"),
        pytest.param(lambda: Lidar("x", fov=0.0), id="fov=0"),
        pytest.param(lambda: Lidar("x", fov=float("nan")), id="fov=nan"),
        pytest.param(lambda: Lidar("x", max_range=-1.0), id="max_range<0"),
        pytest.param(lambda: Lidar("x", range_min=0.0), id="range_min=0"),
        pytest.param(
            lambda: Lidar("x", max_range=1.0, range_min=2.0), id="range_min>max"
        ),
        pytest.param(lambda: Lidar("x", noise_std=-0.1), id="noise<0"),
        pytest.param(lambda: Lidar("x", quant_step=-0.1), id="quant<0"),
        pytest.param(lambda: Lidar("x", latency_steps=-1), id="latency<0"),
        pytest.param(lambda: Lidar("x", dropout_prob=1.5), id="dropout>1"),
        pytest.param(lambda: Lidar("x", dropout_prob=-0.1), id="dropout<0"),
        pytest.param(
            lambda: Lidar("x", mount=(0.0, float("inf"), 0.0)), id="mount-inf"
        ),
        pytest.param(lambda: Lidar("x", mount=(1.0, 2.0)), id="mount-arity"),
        pytest.param(lambda: Imu("x", noise_std=-1.0), id="imu-noise<0"),
        pytest.param(lambda: WheelOdom("x", track_width=0.0), id="track_width=0"),
        pytest.param(lambda: WheelOdom("x", track_width=-0.5), id="track_width<0"),
    ],
)
def test_invalid_specs_raise_value_error(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_lidar_geometry_properties() -> None:
    spec = Lidar("scan", mount=(0.1, -0.2, 0.4), fov=np.pi / 2.0, n_rays=90)
    assert spec.angle_increment == pytest.approx((np.pi / 2.0) / 89)
    # angle_min 直接反映 mount 的 yaw：yaw - fov/2
    assert spec.angle_min == pytest.approx(0.4 - np.pi / 4.0)
    assert spec.angle_span == pytest.approx((89) * spec.angle_increment)
    assert spec.angle_span == pytest.approx(np.pi / 2.0)


# ═════════════════════════════════════════════════════════════════════
# 2. 装配：空列表 / 重名 / 类型 / 必需依赖
# ═════════════════════════════════════════════════════════════════════


def test_empty_spec_list_yields_no_readings() -> None:
    suite = SensorSuite([], seed=0)
    assert suite.names == ()
    assert len(suite) == 0
    assert suite.track_width is None
    assert suite.update(v=1.0, vy=0.0, omega=0.0, theta=0.0, t=0.0) == {}


def test_reading_keys_follow_spec_order() -> None:
    specs = [
        WheelOdom("wheel", track_width=0.3),
        Lidar("scan"),
        Imu("imu"),
    ]
    suite = SensorSuite(specs, raycast=_constant(1.0))
    reading = suite.update(v=1.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)
    assert list(reading) == ["wheel", "scan", "imu"]
    assert suite.names == ("wheel", "scan", "imu")
    assert suite.specs == tuple(specs)
    assert set(reading["scan"]) == {"scan", "scan_meta"}
    assert set(reading["imu"]) == {"ax", "ay", "yaw_rate", "yaw", "t"}
    assert set(reading["wheel"]) == {"v_left", "v_right", "v", "t"}


def test_duplicate_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="重复"):
        SensorSuite([Imu("same"), Imu("same")])


def test_lidar_requires_raycast() -> None:
    with pytest.raises(ValueError, match="raycast"):
        SensorSuite([Lidar("scan")])


def test_wheel_requires_track_width() -> None:
    with pytest.raises(ValueError, match="track_width"):
        SensorSuite([WheelOdom("wheel")])
    # 给缺省值即可
    suite = SensorSuite([WheelOdom("wheel")], default_track_width=0.42)
    assert suite.track_width == pytest.approx(0.42)


def test_unknown_spec_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        SensorSuite([object()])  # type: ignore[list-item]


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError):
        SensorSuite([], seed=-1)


# ═════════════════════════════════════════════════════════════════════
# 3. 几何外置：raycast 收到的就是车体系 (原点, 方位角, 量程)
# ═════════════════════════════════════════════════════════════════════


def test_raycast_receives_body_frame_origin_and_bearings() -> None:
    log: list = []
    spec = Lidar(
        "scan", mount=(0.3, -0.2, 0.4), fov=np.pi / 2.0, n_rays=3, max_range=5.0
    )
    suite = SensorSuite([spec], raycast=_recording(log))
    suite.update(v=1.0, vy=0.0, omega=0.0, theta=0.7, t=0.0)

    assert len(log) == 3
    for origin_xy, _angle, max_range in log:
        # 原点恒为 mount 的 (x, y)：套件不知道、也不关心车的世界位姿
        assert origin_xy == pytest.approx((0.3, -0.2))
        assert max_range == pytest.approx(5.0)
    bearings = np.array([entry[1] for entry in log])
    # 方位角 = mount.yaw - fov/2 + i * fov/(n_rays-1)，与底盘停在哪里无关
    assert bearings == pytest.approx([0.4 - np.pi / 4.0, 0.4, 0.4 + np.pi / 4.0])


def test_sensors_package_does_not_import_bridges() -> None:
    code = (
        "import sys, chassis_sim.sensors\n"
        "bad = [m for m in sys.modules if m.startswith('chassis_sim.bridges')]\n"
        "print(bad)\n"
        "assert not bad, bad\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ═════════════════════════════════════════════════════════════════════
# 4. LiDAR：读数结构与退化模型
# ═════════════════════════════════════════════════════════════════════


def test_scan_is_float32_with_consistent_meta() -> None:
    spec = Lidar(
        "scan", mount=(0.1, 0.2, 0.3), fov=np.pi / 3.0, n_rays=45, max_range=8.0
    )
    suite = SensorSuite([spec], raycast=_constant(2.0))
    reading = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)

    scan = reading["scan"]
    assert scan["scan"].dtype == np.float32
    assert scan["scan"].shape == (45,)
    meta = scan["scan_meta"]
    assert meta["name"] == "scan"
    assert meta["mount"] == (0.1, 0.2, 0.3)
    assert meta["n_rays"] == 45
    assert meta["fov"] == pytest.approx(np.pi / 3.0)
    assert meta["angle_min"] == pytest.approx(0.3 - np.pi / 6.0)
    assert meta["angle_max"] == pytest.approx(0.3 + np.pi / 6.0)
    assert meta["angle_increment"] == pytest.approx((np.pi / 3.0) / 44.0)
    assert meta["range_min"] == pytest.approx(0.05)
    assert meta["range_max"] == pytest.approx(8.0)
    assert meta["t"] == pytest.approx(0.0)
    assert meta["frame_t"] == pytest.approx(0.0)
    assert meta["valid"] is True
    assert meta["dropped"] is False


def test_two_mounts_produce_different_ranges() -> None:
    # 距离随方位角变化 ⇒ 两个 mount 的读数必然不同
    suite = SensorSuite(
        [
            Lidar("left", mount=(0.0, 0.3, 0.5), n_rays=8, fov=np.pi / 2.0),
            Lidar("right", mount=(0.0, -0.3, -0.5), n_rays=8, fov=np.pi / 2.0),
        ],
        raycast=_by_angle(),
    )
    reading = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)
    assert not np.array_equal(reading["left"]["scan"], reading["right"]["scan"])
    assert reading["left"]["scan_meta"]["angle_min"] == pytest.approx(0.5 - np.pi / 4.0)
    assert reading["right"]["scan_meta"]["angle_min"] == pytest.approx(
        -0.5 - np.pi / 4.0
    )


def test_latency_two_steps() -> None:
    spec = Lidar("scan", n_rays=4, latency_steps=2)
    suite = SensorSuite([spec], raycast=_constant(1.0))
    readings = _run_steps(suite, 5)

    # 前两步：缓冲未填满 ⇒ 无有效帧
    for step in (0, 1):
        assert readings[step]["scan"]["scan"] is None
        assert readings[step]["scan"]["scan_meta"]["valid"] is False
        assert readings[step]["scan"]["scan_meta"]["frame_t"] is None
        assert readings[step]["scan"]["scan_meta"]["dropped"] is False
    # 第三步起：交付第 k-2 步采集的帧，时戳同步回退两拍
    for step in (2, 3, 4):
        meta = readings[step]["scan"]["scan_meta"]
        assert readings[step]["scan"]["scan"] is not None
        assert meta["t"] == pytest.approx(step * _DT)
        assert meta["frame_t"] == pytest.approx((step - 2) * _DT)
        assert meta["valid"] is True


def test_zero_latency_delivers_current_frame() -> None:
    suite = SensorSuite([Lidar("scan", n_rays=4)], raycast=_constant(1.0))
    for step, reading in enumerate(_run_steps(suite, 4)):
        meta = reading["scan"]["scan_meta"]
        assert meta["t"] == pytest.approx(step * _DT)
        assert meta["frame_t"] == pytest.approx(step * _DT)


def test_dropout_one_drops_every_frame() -> None:
    suite = SensorSuite(
        [Lidar("scan", n_rays=4, dropout_prob=1.0)], raycast=_constant(1.0), seed=5
    )
    readings = _run_steps(suite, 6)
    assert [r["scan"]["scan"] for r in readings] == [None] * 6
    for reading in readings:
        assert reading["scan"]["scan_meta"]["dropped"] is True
        assert reading["scan"]["scan_meta"]["valid"] is False


def test_dropout_zero_drops_nothing() -> None:
    suite = SensorSuite(
        [Lidar("scan", n_rays=4, dropout_prob=0.0)], raycast=_constant(1.0), seed=5
    )
    readings = _run_steps(suite, 6)
    for reading in readings:
        assert reading["scan"]["scan"] is not None
        assert reading["scan"]["scan_meta"]["dropped"] is False


def test_partial_dropout_is_reproducible_for_a_fixed_seed() -> None:
    def dropped(seed: int) -> list[bool]:
        suite = SensorSuite(
            [Lidar("scan", n_rays=4, dropout_prob=0.4)],
            raycast=_constant(1.0),
            seed=seed,
        )
        return [r["scan"]["scan"] is None for r in _run_steps(suite, 20)]

    first = dropped(11)
    assert first == dropped(11)
    assert any(first) and not all(first)


# ═════════════════════════════════════════════════════════════════════
# 5. 处理顺序：clip -> 噪声 -> clip -> 量化 -> float32
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("raw", "expected"), [(1.24, 1.2), (1.26, 1.3), (2.04, 2.0)])
def test_quantization_rounds_to_step(raw: float, expected: float) -> None:
    suite = SensorSuite(
        [Lidar("scan", n_rays=4, quant_step=0.1)], raycast=_constant(raw)
    )
    scan = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)["scan"]["scan"]
    assert scan == pytest.approx(expected, abs=1e-6)


def test_quantized_readings_are_multiples_of_the_step() -> None:
    step = 0.1
    suite = SensorSuite(
        [Lidar("scan", n_rays=64, quant_step=step)], raycast=_by_angle(offset=3.0)
    )
    scan = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)["scan"]["scan"]
    quantized = np.round(scan.astype(np.float64) / step) * step
    assert np.allclose(scan, quantized, atol=1e-5)
    # 读数确实跨了多个台阶（否则"是步长整数倍"可以平凡成立）
    assert len(np.unique(np.round(scan / step))) > 5


def test_noise_is_reproducible_and_clipped() -> None:
    def run(seed: int) -> np.ndarray:
        suite = SensorSuite(
            [Lidar("scan", n_rays=32, noise_std=0.2, max_range=4.0)],
            raycast=_constant(4.0),
            seed=seed,
        )
        return suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)["scan"]["scan"]

    scan = run(3)
    assert np.array_equal(scan, run(3))
    assert not np.array_equal(scan, run(4))
    # 上限侧在加噪后被重新 clip 回 max_range
    assert scan.max() <= 4.0 + 1e-6
    assert scan.min() >= 0.05 - 1e-6


def test_noise_and_quantization_keep_the_processing_order() -> None:
    """量化必须在加噪之后：两步都开时读数仍是步长整数倍。"""
    suite = SensorSuite(
        [Lidar("scan", n_rays=32, noise_std=0.05, quant_step=0.1)],
        raycast=_by_angle(2.0),
    )
    scan = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)["scan"]["scan"]
    quantized = np.round(scan.astype(np.float64) / 0.1) * 0.1
    assert np.allclose(scan, quantized, atol=1e-5)


# ═════════════════════════════════════════════════════════════════════
# 6. 随机流：按 (seed, name) 派生、跨进程稳定、增删无关
# ═════════════════════════════════════════════════════════════════════


def test_derive_rng_depends_on_seed_and_name_only() -> None:
    assert np.array_equal(
        derive_rng(7, "scan").random(4), derive_rng(7, "scan").random(4)
    )
    assert not np.array_equal(
        derive_rng(7, "scan").random(4), derive_rng(7, "imu").random(4)
    )
    assert not np.array_equal(
        derive_rng(7, "scan").random(4), derive_rng(8, "scan").random(4)
    )


def test_derive_rng_is_stable_across_processes() -> None:
    code = (
        "import json\n"
        "from chassis_sim.sensors import derive_rng\n"
        "print(json.dumps(derive_rng(7, 'scan').normal(size=3).tolist()))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    remote = np.array(json.loads(out.stdout.strip()))
    assert np.array_equal(remote, derive_rng(7, "scan").normal(size=3))


def test_adding_a_sensor_does_not_shift_other_streams() -> None:
    """硬要求：加第三支（even 带噪声 + 丢帧）后，前两支的读数逐位不变。"""

    def base() -> list:
        return [
            Lidar("scan", n_rays=16, noise_std=0.05, dropout_prob=0.3),
            Imu("imu", noise_std=0.01),
            WheelOdom("wheel", track_width=0.4, noise_std=0.02),
        ]

    extra = [
        Lidar("extra", n_rays=8, noise_std=0.7, dropout_prob=0.5),
        WheelOdom("extra_wheel", track_width=0.9, noise_std=0.3),
    ]
    short = SensorSuite(base(), seed=21, raycast=_by_angle(1.5))
    long = SensorSuite(base() + extra, seed=21, raycast=_by_angle(1.5))

    for step in range(10):
        kwargs = {"v": 1.0, "vy": 0.1, "omega": 0.3, "theta": 0.2, "t": step * _DT}
        a = short.update(**kwargs)
        b = long.update(**kwargs)
        assert (a["scan"]["scan"] is None) == (b["scan"]["scan"] is None)
        assert a["scan"]["scan_meta"] == b["scan"]["scan_meta"]
        if a["scan"]["scan"] is not None:
            assert np.array_equal(a["scan"]["scan"], b["scan"]["scan"])
        assert a["imu"] == b["imu"]
        assert a["wheel"] == b["wheel"]


def test_sensors_never_touch_global_np_random() -> None:
    np.random.seed(12345)
    before = np.random.get_state()
    suite = SensorSuite(
        [
            Lidar("scan", n_rays=8, noise_std=0.1, dropout_prob=0.5),
            Imu("imu", noise_std=0.1),
            WheelOdom("wheel", track_width=0.3, noise_std=0.1),
        ],
        seed=5,
        raycast=_by_angle(),
    )
    _run_steps(suite, 5)
    after = np.random.get_state()

    def same_state(a, b) -> bool:
        return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]

    assert same_state(before, after)
    # 反向对照：证明上面的比较不是恒真
    np.random.normal()
    assert not same_state(before, np.random.get_state())


# ═════════════════════════════════════════════════════════════════════
# 7. IMU / 轮速计模型
# ═════════════════════════════════════════════════════════════════════


def test_imu_first_step_has_zero_acceleration() -> None:
    suite = SensorSuite([Imu("imu")])
    reading = suite.update(v=2.0, vy=0.3, omega=0.5, theta=0.4, t=0.0)["imu"]
    assert reading["ax"] == pytest.approx(0.0)
    assert reading["ay"] == pytest.approx(2.0 * 0.5)  # 向心项；无历史 ⇒ 无差分项
    assert reading["yaw_rate"] == pytest.approx(0.5)
    assert reading["yaw"] == pytest.approx(0.4)
    assert reading["t"] == pytest.approx(0.0)


def test_imu_differences_from_previous_sample() -> None:
    suite = SensorSuite([Imu("imu")])
    suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)
    reading = suite.update(v=1.0, vy=0.2, omega=0.5, theta=0.05, t=0.1)["imu"]
    assert reading["ax"] == pytest.approx(1.0 / 0.1)
    assert reading["ay"] == pytest.approx(1.0 * 0.5 + 0.2 / 0.1)
    assert reading["yaw"] == pytest.approx(0.05)


def test_imu_rejects_non_increasing_time() -> None:
    suite = SensorSuite([Imu("imu")])
    suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.1)
    with pytest.raises(ValueError, match="严格递增"):
        suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.1)


def test_imu_noise_is_reproducible() -> None:
    def run(seed: int) -> dict:
        suite = SensorSuite([Imu("imu", noise_std=0.1)], seed=seed)
        return suite.update(v=1.0, vy=0.0, omega=0.2, theta=0.0, t=0.0)["imu"]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_wheel_odometry_differential_formula() -> None:
    suite = SensorSuite([WheelOdom("wheel", track_width=0.5)])
    reading = suite.update(v=1.0, vy=0.0, omega=0.4, theta=0.0, t=0.0)["wheel"]
    assert reading["v_left"] == pytest.approx(1.0 - 0.4 * 0.25)
    assert reading["v_right"] == pytest.approx(1.0 + 0.4 * 0.25)
    assert reading["v"] == pytest.approx(1.0)
    assert reading["t"] == pytest.approx(0.0)


def test_wheel_explicit_track_width_wins_over_default() -> None:
    suite = SensorSuite([WheelOdom("wheel", track_width=0.8)], default_track_width=0.5)
    assert suite.track_width == pytest.approx(0.8)
    reading = suite.update(v=0.0, vy=0.0, omega=1.0, theta=0.0, t=0.0)["wheel"]
    assert reading["v_left"] == pytest.approx(-0.4)
    assert reading["v_right"] == pytest.approx(0.4)


def test_wheel_noise_is_reproducible() -> None:
    def run(seed: int) -> dict:
        suite = SensorSuite(
            [WheelOdom("wheel", track_width=0.3, noise_std=0.05)], seed=seed
        )
        return suite.update(v=1.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)["wheel"]

    assert run(9) == run(9)
    assert run(9) != run(10)


# ═════════════════════════════════════════════════════════════════════
# 8. reset：清空延迟缓冲与差分历史，但不重播种
# ═════════════════════════════════════════════════════════════════════


def test_reset_clears_latency_buffer_and_imu_history() -> None:
    suite = SensorSuite(
        [Lidar("scan", n_rays=4, latency_steps=1), Imu("imu")], raycast=_constant(1.0)
    )
    _run_steps(suite, 3)
    suite.reset()
    first = suite.update(v=1.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)
    # 延迟缓冲重新预热：第一步交付 None
    assert first["scan"]["scan"] is None
    # IMU 差分历史清零：又是"首步 ax=0"
    assert first["imu"]["ax"] == pytest.approx(0.0)


# ═════════════════════════════════════════════════════════════════════
# 9. import 纯净性（子进程断言）
# ═════════════════════════════════════════════════════════════════════


def test_sensors_import_does_not_pull_heavy_modules() -> None:
    code = (
        "import sys\n"
        "import chassis_sim\n"
        "import chassis_sim.sensors\n"
        "heavy = ('matplotlib', 'irsim', 'numba', 'torch', 'shapely', 'scipy')\n"
        "bad = [m for m in heavy if m in sys.modules]\n"
        "print(bad)\n"
        "assert not bad, bad\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip() == "[]"
