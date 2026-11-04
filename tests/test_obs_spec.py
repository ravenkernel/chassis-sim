"""S1：可组合观测段（chassis_sim.obs）的契约测试。

对应 dispatch 的 7 条验收：

1. default 总宽 128（action_dim=2）/ 129（action_dim=3）；
2. 与**内联金标准**逐位对拍：9 段同序同宽 = 494 维（段序/宽度取自主仓实测），
   np.array_equal == True 且 max|Δ| == 0.0；金标准是手写拼接，不 import 任何段实现；
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
    PathProgress,
    PathSegment,
    PathValid,
    Segment,
)

# 金标准的 lidar 分量 = concat(front, rear)，这里保证逐位可切
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
    """与金标准同序同宽的 9 段（lidar 用 360 根射线）== 494 维。"""
    return (
        LidarSegment(name="lidar", sensors=("lidar",), ray_counts={"lidar": 360}),
        Odom(),
        PathSegment(points=30),
        Goal(),
        Act(action_dim=2),
        Err(),
        GoalSig(),
        PathValid(),
        PathProgress(),
    )


#: 主仓实测的段序与切片（1.0.0 的金标准；ObsLayout 已删，这组数是手抄的）
GOLDEN_ORDER = (
    "lidar",
    "odom",
    "path",
    "goal",
    "act",
    "err",
    "goal_sig",
    "path_valid",
    "path_progress",
)
GOLDEN_SLICES = {
    "lidar": (0, 360),
    "odom": (360, 364),
    "path": (364, 484),
    "goal": (484, 486),
    "act": (486, 488),
    "err": (488, 490),
    "goal_sig": (490, 492),
    "path_valid": (492, 493),
    "path_progress": (493, 494),
}
GOLDEN_TOTAL = 494


def _golden_494(ctx: ObsContext) -> np.ndarray:
    """494 维的参考实现：按 GOLDEN_SLICES 手工拼接（与段类实现无关）。

    段实现是否逐位正确因此是一个真问题，而不是自证。path_progress 的夹紧
    （nan→0 / inf→1 / 越界裁到 [0,1]）也是从旧实现搬来的契约，这里照抄。
    """
    prog = np.asarray(ctx.path_progress, dtype=np.float32).ravel()
    prog = np.clip(np.nan_to_num(prog, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return np.concatenate(
        [
            np.asarray(ctx.lidar["lidar"], dtype=np.float32).ravel(),
            np.asarray(ctx.odom_rel, dtype=np.float32).ravel(),
            np.asarray(ctx.path_flat, dtype=np.float32).ravel(),
            np.asarray(ctx.goal, dtype=np.float32).ravel(),
            np.asarray(ctx.act, dtype=np.float32).ravel(),
            np.asarray(ctx.err, dtype=np.float32).ravel(),
            np.asarray(ctx.goal_sig, dtype=np.float32).ravel(),
            np.asarray(ctx.path_valid, dtype=np.float32).ravel(),
            prog,
        ]
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
def test_494_bit_exact_against_golden():
    ctx = _ctx()
    res = ObsSpec(_nine_segments()).resolved()
    old = _golden_494(ctx)
    new = res.build(ctx)

    assert old.shape == new.shape == (GOLDEN_TOTAL,)
    assert res.total_dim == GOLDEN_TOTAL
    assert np.array_equal(new, old) is True
    assert np.array_equal(new.view(np.uint32), old.view(np.uint32)) is True
    assert float(np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))) == 0.0


def test_494_slices_match_golden_order_and_bits():
    ctx = _ctx()
    res = ObsSpec(_nine_segments()).resolved()
    old = _golden_494(ctx)
    new = res.build(ctx)

    assert list(res.describe()) == list(GOLDEN_ORDER)
    assert res.describe() == {k: slice(*v) for k, v in GOLDEN_SLICES.items()}
    for name, sl in res.describe().items():
        assert (sl.start, sl.stop) == GOLDEN_SLICES[name]
        assert np.array_equal(new[sl], old[sl.start : sl.stop])


def test_front_rear_via_sensor_meta_also_matches_golden():
    ctx = _ctx()
    segs = list(_nine_segments())
    segs[0] = LidarSegment(
        sensors=("front", "rear")
    )  # 不给 ray_counts，靠 sensor_meta 补
    res = ObsSpec(tuple(segs)).resolved(
        {"front": {"n_rays": 180}, "rear": {"n_rays": 180}}
    )
    assert res.describe()["lidar"] == slice(0, 360)
    assert res.total_dim == GOLDEN_TOTAL
    assert np.array_equal(res.build(ctx), _golden_494(ctx)) is True


def test_path_progress_is_clipped_like_golden():
    for raw in (-0.5, 0.0, 0.375, 1.0, 2.0, float("nan")):
        ctx = _ctx(path_progress=raw)
        assert np.array_equal(PathProgress().build(ctx), _golden_494(ctx)[-1:])


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
