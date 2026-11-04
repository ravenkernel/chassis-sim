"""S1：可组合观测段（chassis_sim.obs）的契约测试。

对应 dispatch 的 7 条验收：

1. default 总宽 128（action_dim=2）/ 129（action_dim=3）；
2. 与旧 ObsLayout（src/chassis_sim/envs/observation.py:35）逐位对拍：9 段同序
   同宽 = 494 维，np.array_equal == True 且 max|Δ| == 0.0；
3. Custom 宽度不符 / 返回 2-D 抛 ValueError；
4. describe() 切片连续、首尾相接、总和 == total_dim、键序 == 段序；
5. 同名段构造期 ValueError；lidar 引用未知传感器时 resolved() ValueError；
6. pool="max" 射线数不等 → ValueError；相等 → 逐元素 max（含数值例）；
7. dtype 一律 float32、形状一维。

输入全部是写死的常数（不用 RNG）：对拍要能逐位复现。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.envs.observation import ObsLayout
from chassis_sim.obs import (
    Act,
    Custom,
    Err,
    Goal,
    GoalSig,
    LidarSegment,
    ObsContext,
    ObsSpec,
    Odom,
    Path,
    PathProgress,
    PathValid,
    Segment,
)

# 旧 ObsLayout 的 lidar 分量 = concat(lidar_front, lidar_rear)，这里保证逐位可切
LIDAR_360 = np.linspace(-3.0, 3.0, 360, dtype=np.float32)
PATH_FLAT = (np.arange(120, dtype=np.float32) - 60.0) / 8.0


def _ctx(**over) -> ObsContext:
    base = {
        "lidar": {
            "lidar": LIDAR_360,
            "front": LIDAR_360[:180].copy(),
            "rear": LIDAR_360[180:].copy(),
        },
        "odom_rel": np.array([1.5, -2.25, 0.75, -0.125], dtype=np.float32),
        "path_flat": PATH_FLAT,
        "goal": np.array([3.5, -1.25], dtype=np.float32),
        "act": np.array([-0.5, 0.25], dtype=np.float32),
        "err": np.array([0.125, -0.375], dtype=np.float32),
        "goal_sig": np.array([0.5, -0.8660254], dtype=np.float32),
        "path_valid": 0.9,
        "path_progress": 0.375,
    }
    base.update(over)
    return ObsContext(**base)


def _nine_segments() -> tuple:
    """与旧 ObsLayout 同序同宽的 9 段（lidar 用 360 根射线）== 494 维。"""
    return (
        LidarSegment(name="lidar", sensors=("lidar",), ray_counts={"lidar": 360}),
        Odom(),
        Path(points=30),
        Goal(),
        Act(action_dim=2),
        Err(),
        GoalSig(),
        PathValid(),
        PathProgress(),
    )


def _legacy_build(ctx: ObsContext) -> np.ndarray:
    layout = ObsLayout(action_dim=2, max_path_points=30)
    return layout.build(
        ctx.lidar["lidar"],
        ctx.odom_rel,
        ctx.path_flat,
        ctx.goal,
        ctx.act,
        ctx.err,
        ctx.goal_sig,
        np.array([ctx.path_valid], dtype=np.float32),
        path_progress=ctx.path_progress,
    )


# ── 验收 1：default 总宽 ──────────────────────────────────
def test_default_total_dim_is_128():
    assert ObsSpec.default().resolved().total_dim == 128
    assert 4 + 30 * 4 + 2 + 2 == 128


def test_default_action_dim_3_total_dim_is_129():
    assert ObsSpec.default(action_dim=3).resolved().total_dim == 129


def test_default_segments_are_odom_path_goal_act():
    res = ObsSpec.default().resolved()
    assert [s.name for s in res.segments] == ["odom", "path", "goal", "act"]
    assert res.describe() == {
        "odom": slice(0, 4),
        "path": slice(4, 124),
        "goal": slice(124, 126),
        "act": slice(126, 128),
    }


def test_default_build_places_each_segment():
    ctx = _ctx()
    res = ObsSpec.default().resolved()
    obs = res.build(ctx)
    d = res.describe()
    assert obs.shape == (128,)
    assert np.array_equal(obs[d["odom"]], ctx.odom_rel)
    assert np.array_equal(obs[d["path"]], ctx.path_flat)
    assert np.array_equal(obs[d["goal"]], ctx.goal)
    assert np.array_equal(obs[d["act"]], ctx.act)


# ── 验收 2：494 逐位对拍 ─────────────────────────────────
def test_494_bit_exact_against_legacy_layout():
    ctx = _ctx()
    res = ObsSpec(_nine_segments()).resolved()
    old = _legacy_build(ctx)
    new = res.build(ctx)
    layout = ObsLayout(action_dim=2, max_path_points=30)

    assert old.shape == new.shape == (494,)
    assert res.total_dim == layout.total_dim == 494
    assert np.array_equal(new, old) is True
    assert np.array_equal(new.view(np.uint32), old.view(np.uint32)) is True
    assert float(np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))) == 0.0


def test_494_slices_match_legacy_order_and_bits():
    ctx = _ctx()
    res = ObsSpec(_nine_segments()).resolved()
    layout = ObsLayout(action_dim=2, max_path_points=30)
    old = _legacy_build(ctx)
    new = res.build(ctx)

    assert list(res.describe()) == list(layout.ORDER)
    for name, sl in res.describe().items():
        assert (sl.start, sl.stop) == layout._slices[name]
        assert np.array_equal(new[sl], old[sl.start : sl.stop])
        assert np.array_equal(new[sl], layout.slice(old)[name].reshape(-1))


def test_front_rear_via_sensor_meta_also_matches_legacy():
    ctx = _ctx()
    segs = list(_nine_segments())
    segs[0] = LidarSegment(
        sensors=("front", "rear")
    )  # 不给 ray_counts，靠 sensor_meta 补
    res = ObsSpec(tuple(segs)).resolved(
        {"front": {"n_rays": 180}, "rear": {"n_rays": 180}}
    )
    assert res.describe()["lidar"] == slice(0, 360)
    assert res.total_dim == 494
    assert np.array_equal(res.build(ctx), _legacy_build(ctx)) is True


def test_path_progress_is_clipped_like_legacy():
    layout = ObsLayout(action_dim=2, max_path_points=30)
    for raw in (-0.5, 0.0, 0.375, 1.0, 2.0, float("nan")):
        ctx = _ctx(path_progress=raw)
        seg = PathProgress()
        assert np.array_equal(
            seg.build(ctx),
            layout.build(
                ctx.lidar["lidar"],
                ctx.odom_rel,
                ctx.path_flat,
                ctx.goal,
                ctx.act,
                ctx.err,
                ctx.goal_sig,
                np.array([ctx.path_valid], np.float32),
                path_progress=raw,
            )[-1:],
        )


# ── 验收 3：Custom 契约 ─────────────────────────────────
def test_custom_width_mismatch_raises():
    seg = Custom("bad_width", 3, lambda ctx: np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError) as ei:
        ObsSpec((seg,)).resolved().build(_ctx())
    assert "期望宽度 3，实际 4" in str(ei.value)
    print(str(ei.value))


def test_custom_two_dim_output_raises():
    seg = Custom("bad_ndim", 4, lambda ctx: np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError) as ei:
        ObsSpec((seg,)).resolved().build(_ctx())
    assert "ndim=2" in str(ei.value)
    print(str(ei.value))


def test_custom_converts_float64_to_float32():
    seg = Custom("double", 2, lambda ctx: np.array([1.5, -2.5], dtype=np.float64))
    res = ObsSpec((seg,)).resolved()
    out = res.build(_ctx())
    assert out.dtype == np.float32
    assert np.array_equal(out, np.array([1.5, -2.5], dtype=np.float32))


# ── 验收 4：describe() 切片几何 ──────────────────────────
def test_describe_is_contiguous_and_ordered():
    res = ObsSpec(_nine_segments()).resolved()
    d = res.describe()
    assert list(d) == [
        "lidar",
        "odom",
        "path",
        "goal",
        "act",
        "err",
        "goal_sig",
        "path_valid",
        "path_progress",
    ]
    assert d["lidar"] == slice(0, 360)
    assert d["odom"] == slice(360, 364)
    assert d["path"] == slice(364, 484)
    assert d["path_progress"] == slice(493, 494)

    prev = 0
    for sl in d.values():
        assert sl.start == prev
        assert sl.stop > sl.start
        prev = sl.stop
    assert prev == 494
    assert sum(sl.stop - sl.start for sl in d.values()) == res.total_dim


def test_describe_returns_a_copy():
    res = ObsSpec.default().resolved()
    d = res.describe()
    d["odom"] = slice(0, 999)
    assert res.describe()["odom"] == slice(0, 4)


# ── 验收 5：重名 / 未知传感器 ───────────────────────────
def test_duplicate_segment_names_rejected_at_construction():
    with pytest.raises(ValueError) as ei:
        ObsSpec((Odom(), Odom()))
    assert "段名重复" in str(ei.value)
    assert "odom" in str(ei.value)


def test_lidar_unknown_sensor_reported_by_resolved():
    spec = ObsSpec((LidarSegment(sensors=("front", "ghost"), pool="none"),))
    with pytest.raises(ValueError) as ei:
        spec.resolved({"front": {"n_rays": 4}})
    msg = str(ei.value)
    assert "缺少传感器射线数 ['ghost']" in msg
    print(msg)


def test_lidar_dim_without_metadata_raises():
    with pytest.raises(ValueError) as ei:
        _ = LidarSegment().dim
    assert "缺少传感器射线数" in str(ei.value)


def test_resolved_lidar_dim_uses_meta():
    seg = LidarSegment(sensors=("front", "rear"), pool="none")
    assert seg._resolve({"front": {"n_rays": 180}, "rear": {"n_rays": 120}}).dim == 300


# ── 验收 6：pool ────────────────────────────────────────
def test_pool_max_rejects_unequal_ray_counts():
    seg = LidarSegment(
        sensors=("front", "rear"), pool="max", ray_counts={"front": 4, "rear": 3}
    )
    with pytest.raises(ValueError) as ei:
        ObsSpec((seg,)).resolved()
    assert "要求所有传感器射线数相等" in str(ei.value)
    print(str(ei.value))


def test_pool_max_elementwise():
    ctx = _ctx(
        lidar={
            "front": np.array([1.0, 5.0, 3.0], dtype=np.float32),
            "rear": np.array([4.0, 2.0, 3.0], dtype=np.float32),
        }
    )
    res = ObsSpec(
        (
            LidarSegment(
                sensors=("front", "rear"),
                pool="max",
                ray_counts={"front": 3, "rear": 3},
            ),
        )
    ).resolved()
    assert res.total_dim == 3
    out = res.build(ctx)
    assert np.array_equal(out, np.array([4.0, 5.0, 3.0], dtype=np.float32))
    print("pool=max ->", out.tolist())


def test_pool_mean_elementwise():
    ctx = _ctx(
        lidar={
            "front": np.array([1.0, 5.0, 3.0], dtype=np.float32),
            "rear": np.array([4.0, 2.0, 3.0], dtype=np.float32),
        }
    )
    res = ObsSpec(
        (
            LidarSegment(
                sensors=("front", "rear"),
                pool="mean",
                ray_counts={"front": 3, "rear": 3},
            ),
        )
    ).resolved()
    out = res.build(ctx)
    assert np.array_equal(out, np.array([2.5, 3.5, 3.0], dtype=np.float32))
    print("pool=mean ->", out.tolist())


def test_pool_none_concatenates_in_sensor_order():
    ctx = _ctx(
        lidar={
            "front": np.array([1.0, 2.0], dtype=np.float32),
            "rear": np.array([3.0], dtype=np.float32),
        }
    )
    res = ObsSpec(
        (LidarSegment(sensors=("front", "rear"), ray_counts={"front": 2, "rear": 1}),)
    ).resolved()
    assert res.total_dim == 3
    assert np.array_equal(res.build(ctx), np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_pool_mode_validated():
    with pytest.raises(ValueError):
        LidarSegment(pool="sum")


# ── 验收 7：dtype / 一维 ────────────────────────────────
def test_every_segment_returns_float32_1d():
    ctx = _ctx()
    res = ObsSpec(_nine_segments()).resolved()
    for seg in res.segments:
        part = seg.build(ctx)
        assert isinstance(part, np.ndarray)
        assert part.dtype == np.float32, seg.name
        assert part.ndim == 1, seg.name
        assert part.shape == (seg.dim,), seg.name
    fused = res.build(ctx)
    assert fused.dtype == np.float32
    assert fused.ndim == 1
    assert fused.flags["C_CONTIGUOUS"]


def test_segment_width_mismatch_raises_instead_of_padding():
    ctx = _ctx(odom_rel=np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError) as ei:
        Odom().build(ctx)
    assert "期望 4 维，实际 5 维" in str(ei.value)


class _RogueSegment(Segment):
    """自报 dim=3 却只吐 2 维的坏段（绕过段级校验，专门验 spec 的总宽断言）。"""

    name = "rogue"
    dim = 3

    def build(self, ctx: ObsContext) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


def test_build_rejects_total_width_mismatch():
    res = ObsSpec((_RogueSegment(),)).resolved()
    with pytest.raises(ValueError) as ei:
        res.build(_ctx())
    assert "期望 3 维，实际拼接出 2 维" in str(ei.value)
