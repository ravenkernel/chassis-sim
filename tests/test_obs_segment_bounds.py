"""观测段的物理界（3.0.0 W1）：Segment.bounds 与 ResolvedObsSpec.bounds。

规则：有推导依据的段给有限界；没有依据的段（Custom、未声明量程的 lidar 传感器）
保持 ±inf —— 不猜。界必须能**证明**，不能靠观测到的最大值拍出来。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.obs.spec import (
    Act,
    BoundContext,
    Custom,
    Err,
    Goal,
    GoalSig,
    LidarSegment,
    Odom,
    PathProgress,
    PathSegment,
    PathValid,
    ResolvedObsSpec,
    _bounds_inf,
)

CTX = BoundContext(
    dt=0.1,
    v_max=1.2,
    v_min=-0.6,
    vy_max=0.5,
    omega_max=4.8,
    lookahead=20.0,
    world_diag=float(np.hypot(10.0, 10.0)),
    lidar={"front": (0.05, 10.0), "rear": (0.1, 6.0)},
)


def test_default_bounds_are_unbounded():
    low, high = _bounds_inf(3)
    assert np.all(low == -np.inf) and np.all(high == np.inf)


def test_custom_segment_keeps_unbounded_on_purpose():
    seg = Custom(name="x", dim=2, fn=lambda ctx: np.zeros(2, dtype=np.float32))
    low, high = seg.bounds(CTX)
    assert low.shape == high.shape == (2,)
    assert np.all(np.isinf(low)) and np.all(np.isinf(high))


def test_odom_bounds_follow_the_spec_formula():
    low, high = Odom().bounds(CTX)
    span = max(abs(CTX.v_max), abs(CTX.v_min)) + CTX.vy_max
    disp = 10.0 * 1.5 * CTX.dt * span
    assert np.allclose(low, [-disp, -disp, CTX.v_min, -CTX.omega_max])
    assert np.allclose(high, [disp, disp, CTX.v_max, CTX.omega_max])
    assert np.all(np.isfinite([low, high]))


def test_path_rho_is_bounded_by_diag_plus_lookahead():
    seg = PathSegment(points=30)
    low, high = seg.bounds(CTX)
    assert low.shape == high.shape == (120,)
    assert np.allclose(high[0::4], CTX.world_diag + CTX.lookahead + 0.1)
    assert np.all(low[0::4] == 0.0)  # ρ 是距离，非负
    assert np.allclose(low[1::4], -1.0) and np.allclose(high[1::4], 1.0)
    assert np.allclose(low[2::4], -1.0) and np.allclose(high[2::4], 1.0)
    assert np.allclose(low[3::4], -np.pi) and np.allclose(high[3::4], np.pi)


def test_simple_segment_bounds():
    d = CTX.world_diag
    assert np.allclose(Goal().bounds(CTX)[0], [-d, -d])
    assert np.allclose(Act(3).bounds(CTX)[1], [1.0, 1.0, 1.0])
    assert np.allclose(Err().bounds(CTX)[0], [-1.0, -np.pi])
    assert np.allclose(GoalSig().bounds(CTX)[1], [1.0, 1.0])
    assert np.allclose(PathValid().bounds(CTX)[0], [0.0])
    assert np.allclose(PathProgress().bounds(CTX)[1], [1.0])


def test_lidar_bounds_are_per_ray_sensor_ranges():
    seg = LidarSegment(
        name="lidar", sensors=("front", "rear"), ray_counts={"front": 180, "rear": 90}
    )
    low, high = seg.bounds(CTX)
    assert low.size == high.size == 270
    assert np.allclose(low[:180], 0.05) and np.allclose(high[:180], 10.0)
    assert np.allclose(low[180:], 0.1) and np.allclose(high[180:], 6.0)


def test_pooled_lidar_uses_the_envelope_of_all_sensors():
    for pool in ("max", "mean"):
        # pool="max"/"mean" 要求各传感器射线数相等（宽度必须定义良好）
        seg = LidarSegment(
            name="lidar",
            sensors=("front", "rear"),
            pool=pool,
            ray_counts={"front": 180, "rear": 180},
        )
        low, high = seg.bounds(CTX)
        assert low.size == high.size == 180
        assert np.allclose(low, 0.05) and np.allclose(high, 10.0)


def test_lidar_with_unknown_sensor_is_unbounded():
    seg = LidarSegment(name="lidar", sensors=("nosuch",), ray_counts={"nosuch": 8})
    low, high = seg.bounds(CTX)
    assert np.all(np.isinf(low)) and np.all(np.isinf(high))


def test_resolved_bounds_match_describe_widths():
    spec = ResolvedObsSpec(
        (Odom(), PathSegment(points=10), Goal(), Act(2), PathProgress())
    )
    assert spec.total_dim == 49
    low, high = spec.bounds(CTX)
    assert low.shape == high.shape == (49,)
    assert low.dtype == high.dtype == np.float32
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))
    assert low[4] == 0.0  # 0.0 不扩 ulp：转 float32 精确，不会向内舍入


def test_resolved_bounds_reject_wrong_width():
    class Bad(Odom):
        def bounds(self, ctx):
            return np.zeros(1), np.ones(1)

    spec = ResolvedObsSpec((Bad(),))
    with pytest.raises(ValueError, match="期望 4 维界"):
        spec.bounds(CTX)


def test_bounds_are_padded_by_one_ulp_outwards():
    low, high = Odom().bounds(CTX)
    spec = ResolvedObsSpec((Odom(),))
    lo32, hi32 = spec.bounds(CTX)
    # float64 → float32 的向内舍入会被 nextafter 抵消：界不会比 float64 值更紧
    assert np.all(lo32.astype(np.float64) <= low)
    assert np.all(hi32.astype(np.float64) >= high)
