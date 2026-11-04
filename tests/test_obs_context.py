"""ContextBuilder：raw 传感器字典 -> ObsContext 的算式与时序契约。

本文件承接原 `tests/test_envs_observation.py` 里**仍然存在**的那层语义（1.0.0 起
类名 ContextBuilder、模块 `chassis_sim.obs.context`）：里程计差分、路径提取、
CTE/航向误差、path_valid 两种填充模式、reset 清时序、以及雷达帧按名字原样透传。

原文件里针对已删 API（ObsLayout 的 `build()` / `slice()` / `compute_obs_dims` /
`BASE_RAW_OBS_DIM` / ObservationBuilder 的 494 维）的用例不再存在：1.0.0 没有定长
观测布局，494 只能由显式 9 段 ObsSpec 表达（见 `tests/test_obs_spec.py` 与
`tests/test_envs_obs_spec.py` 的金标准切片表）。

输入全部写死、不用 RNG：契约要能逐位复现。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.obs.context import ContextBuilder


def _straight_path(length: float = 5.0, n: int = 60) -> np.ndarray:
    """y=0 直线，第 3 列是鼻向（0）。"""
    x = np.linspace(0.0, length, n)
    return np.column_stack([x, np.zeros(n), np.zeros(n)])


def _raw(
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    v: float = 0.0,
    omega: float = 0.0,
    path_world: np.ndarray | None = None,
) -> dict:
    """桥的 raw 读数（只带 build_context 真正用到的两个键）。"""
    return {
        "odometry": np.array([x, y, theta, v, omega], dtype=np.float32),
        "path_world": _straight_path() if path_world is None else path_world,
    }


class TestOdomDifferencing:
    def test_first_call_is_zero_displacement(self):
        ctx = ContextBuilder().build_context(_raw(v=0.4, omega=-0.25))
        assert np.array_equal(
            ctx.odom_rel, np.array([0.0, 0.0, 0.4, -0.25], np.float32)
        )

    def test_second_call_uses_previous_theta_and_scales_by_ten(self):
        b = ContextBuilder()
        b.build_context(_raw(x=0.0, y=0.0, theta=np.pi / 2.0))
        # 世界系 +y 位移 0.1 m，上一拍朝向 +y ⇒ 车体系 +x 位移 0.1 m
        ctx = b.build_context(_raw(x=0.0, y=0.1, theta=np.pi / 2.0))
        assert ctx.odom_rel[0] == pytest.approx(1.0, abs=1e-6)
        assert ctx.odom_rel[1] == pytest.approx(0.0, abs=1e-6)

    def test_reset_clears_temporal_state(self):
        b = ContextBuilder()
        b.build_context(_raw(x=0.0))
        b.reset()
        ctx = b.build_context(_raw(x=5.0))
        assert ctx.odom_rel[0] == 0.0 and ctx.odom_rel[1] == 0.0


class TestPathAndGoal:
    def test_path_flat_is_four_channels_per_point(self):
        ctx = ContextBuilder(max_path_points=30).build_context(_raw())
        assert ctx.path_flat.shape == (30 * 4,)
        assert ctx.path_flat.dtype == np.float32
        # 直线朝前：车正上方那个点 rho=0（它就是最近点），远处的点 cos_phi ≈ 1
        pts = ctx.path_flat.reshape(-1, 4)
        far = pts[int(np.argmax(pts[:, 0]))]
        assert far[0] > 0.0
        assert far[2] == pytest.approx(1.0, abs=1e-3)
        assert abs(far[1]) < 1e-3

    def test_goal_is_the_last_path_point_in_body_frame(self):
        ctx = ContextBuilder().build_context(_raw(x=1.0, y=0.0, theta=0.0))
        assert ctx.goal[0] == pytest.approx(4.0, abs=1e-5)
        assert ctx.goal[1] == pytest.approx(0.0, abs=1e-5)

    def test_err_is_zero_on_the_path_and_scaled_by_path_d_max(self):
        on_path = ContextBuilder(path_d_max=4.0).build_context(_raw())
        assert on_path.err[0] == pytest.approx(0.0, abs=1e-3)
        assert on_path.err[1] == pytest.approx(0.0, abs=1e-3)

        offset_path = _straight_path()
        offset_path[:, 1] = 0.1  # 路径整体左移 0.1 m ⇒ cte ≈ -0.1
        wide = ContextBuilder(path_d_max=4.0).build_context(
            _raw(path_world=offset_path)
        )
        narrow = ContextBuilder(path_d_max=0.5).build_context(
            _raw(path_world=offset_path)
        )
        assert narrow.err[0] == pytest.approx(8.0 * wide.err[0], rel=1e-3)

    def test_missing_path_zeroes_path_and_err(self):
        empty = np.zeros((0, 3))
        ctx = ContextBuilder(max_path_points=30).build_context(_raw(path_world=empty))
        assert np.array_equal(ctx.path_flat, np.zeros(120, np.float32))
        assert np.array_equal(ctx.err, np.zeros(2, np.float32))
        assert np.array_equal(ctx.goal, np.zeros(2, np.float32))
        assert ctx.path_valid[0] == 1.0  # extrapolate 模式：填充的点也算有效

    def test_sentinel_fill_mode_reports_real_fraction(self):
        b = ContextBuilder(max_path_points=30, path_fill_mode="sentinel")
        ctx = b.build_context(_raw())
        n_valid = b.path_extractor.last_n_valid
        assert 0 < n_valid <= 30
        assert ctx.path_valid[0] == pytest.approx(n_valid / 30.0)


class TestPassThrough:
    def test_lidar_frames_are_passed_by_name_verbatim(self):
        frames = {
            "front": np.linspace(1.0, 2.0, 8, dtype=np.float32),
            "rear": np.linspace(2.0, 1.0, 8, dtype=np.float32),
        }
        ctx = ContextBuilder().build_context(_raw(), lidar=frames)
        # 1.0.0 起没有 legacy 两键：上下文里只有调用方给的名字
        assert set(ctx.lidar) == {"front", "rear"}
        for name, scan in frames.items():
            assert np.array_equal(ctx.lidar[name], scan)

    def test_no_lidar_means_empty_mapping(self):
        ctx = ContextBuilder().build_context(_raw())
        assert dict(ctx.lidar) == {}

    def test_goal_signals_default_to_zero_and_path_progress_is_passed_through(self):
        ctx = ContextBuilder().build_context(_raw(), path_progress=0.375)
        assert np.array_equal(ctx.goal_sig, np.zeros(2, np.float32))
        assert float(np.asarray(ctx.path_progress)) == pytest.approx(0.375)
