"""批量射线钩子 ``raycast_many``：逐位等价 / 调用次数 / 兼容性。

四组断言：

1. **逐位相同**（真桥，不是假钩子）：``SimpleBridge`` 的新路径产出与"逐条调用
   ``_raycast_body_ray`` 再拼起来"在多个 pose、``N in {60, 360}`` 下
   ``np.array_equal`` 为 True —— 不是 allclose。
2. **调用次数**：同时注入两个钩子时，一帧内 ``raycast_many`` 只调 1 次、
   标量 ``raycast`` 调 0 次（批量钩子优先）。
3. **兼容**：只给标量钩子仍能出帧，且每帧按射线数逐条调用。
4. **构造校验**：两者都不给且有 Lidar 时抛 ValueError，消息含 ``raycast_many``。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.sensors import Lidar, SensorSuite

_MOUNT = (0.12, 0.0, 0.0)
_PATH = np.column_stack([np.linspace(1.0, 6.0, 40), np.full(40, 2.0), np.zeros(40)])
_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.30}, "state": [3.0, 2.5, 0.0]},
    {
        "shape": {"name": "rectangle", "length": 0.60, "width": 0.40},
        "state": [4.2, 1.4, 0.4],
    },
]


def _spec(n_rays: int) -> Lidar:
    return Lidar("front", mount=_MOUNT, fov=np.pi, n_rays=n_rays)


def _bearings(spec: Lidar) -> np.ndarray:
    """与套件内部一致的一整帧方位角（float64，端点包含）。"""
    return spec.angle_min + np.arange(spec.n_rays, dtype=np.float64) * (
        spec.angle_increment
    )


def _make_bridge(n_rays: int) -> SimpleBridge:
    return SimpleBridge(AckermannChassis(), sensors=[_spec(n_rays)], domain_rand=False)


@pytest.mark.parametrize("n_rays", [60, 360])
@pytest.mark.parametrize("pose_steps", [0, 7, 23])
def test_batched_scan_is_bitwise_identical_to_scalar_stitching(
    n_rays: int, pose_steps: int
) -> None:
    bridge = _make_bridge(n_rays)
    bridge.reset(
        obstacles=[dict(o) for o in _OBSTACLES],
        path_waypoints=_PATH.copy(),
        initial_heading=0.0,
    )
    rng = np.random.default_rng(1_000 + pose_steps)
    for _ in range(pose_steps):
        bridge.step(rng.uniform(-0.6, 0.6, size=2))

    reading = bridge.get_sensor_reading()
    assert reading is not None
    scan = reading["front"]["scan"]

    spec = _spec(n_rays)
    origin = (float(spec.mount[0]), float(spec.mount[1]))
    per_ray = np.array(
        [
            bridge._raycast_body_ray(origin, float(a), float(spec.max_range))
            for a in _bearings(spec)
        ],
        dtype=np.float32,
    )
    assert scan.dtype == np.float32
    assert np.array_equal(scan, per_ray)


def test_batched_hook_runs_once_per_frame_and_scalar_never() -> None:
    calls = {"scalar": 0, "many": 0}
    spec = Lidar("scan", n_rays=64)

    def scalar(
        origin_xy: tuple[float, float], angle_rad: float, max_range: float
    ) -> float:
        calls["scalar"] += 1
        return 1.0

    def many(
        origin_xy: tuple[float, float], angles: np.ndarray, max_range: float
    ) -> np.ndarray:
        calls["many"] += 1
        return np.full(angles.shape[0], 1.0, dtype=np.float32)

    suite = SensorSuite([spec], raycast=scalar, raycast_many=many)
    reading = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)
    assert calls == {"scalar": 0, "many": 1}
    assert np.array_equal(reading["scan"]["scan"], np.full(64, 1.0, dtype=np.float32))


def test_batched_hook_alone_is_enough() -> None:
    spec = Lidar("scan", n_rays=5)

    def many(
        origin_xy: tuple[float, float], angles: np.ndarray, max_range: float
    ) -> np.ndarray:
        return np.linspace(0.1, 0.2, angles.shape[0], dtype=np.float32)

    suite = SensorSuite([spec], raycast_many=many)
    reading = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)
    assert np.array_equal(
        reading["scan"]["scan"], np.linspace(0.1, 0.2, 5, dtype=np.float32)
    )


def test_scalar_hook_only_still_produces_frames() -> None:
    angles_seen: list[float] = []

    def scalar(
        origin_xy: tuple[float, float], angle_rad: float, max_range: float
    ) -> float:
        angles_seen.append(float(angle_rad))
        return 2.0

    suite = SensorSuite([Lidar("scan", n_rays=8)], raycast=scalar)
    reading = suite.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.0)
    assert len(angles_seen) == 8
    assert np.array_equal(reading["scan"]["scan"], np.full(8, 2.0, dtype=np.float32))
    assert angles_seen == _bearings(Lidar("scan", n_rays=8)).tolist()


def test_no_hook_is_rejected_and_message_names_raycast_many() -> None:
    with pytest.raises(ValueError, match="raycast_many"):
        SensorSuite([Lidar("scan")])


def test_batched_path_matches_scalar_path_through_full_pipeline() -> None:
    """噪声 / 量化 / 延迟 / 丢帧全开时，两条路径的帧与随机消耗仍然逐位相同。"""
    spec = Lidar(
        "scan",
        mount=_MOUNT,
        n_rays=17,
        noise_std=0.05,
        quant_step=0.01,
        latency_steps=2,
        dropout_prob=0.3,
    )

    def scalar(
        origin_xy: tuple[float, float], angle_rad: float, max_range: float
    ) -> float:
        return float(np.float32(1.0 + 0.3 * np.sin(float(angle_rad))))

    def many(
        origin_xy: tuple[float, float], angles: np.ndarray, max_range: float
    ) -> np.ndarray:
        return (1.0 + 0.3 * np.sin(angles)).astype(np.float32)

    suite_scalar = SensorSuite([spec], raycast=scalar, seed=7)
    suite_many = SensorSuite([spec], raycast_many=many, seed=7)
    for k in range(6):
        a = suite_scalar.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.1 * k)
        b = suite_many.update(v=0.0, vy=0.0, omega=0.0, theta=0.0, t=0.1 * k)
        assert a["scan"]["scan_meta"] == b["scan"]["scan_meta"]
        if a["scan"]["scan"] is None:
            assert b["scan"]["scan"] is None
        else:
            assert np.array_equal(a["scan"]["scan"], b["scan"]["scan"])
