"""ObsLayout 的维度契约：与主仓逐位对拍 + action_dim 泛化。

金标准来源：主仓 /home/zhao/drl-sim/envs/observation.py
    · 第 77-79 行  ORDER = ("lidar", "odom", "path", "goal", "act", "err",
                          "goal_sig", "path_valid", "path_progress")
    · 第 81-91 行  LIDAR=360 / ODOM=4 / GOAL=2 / ACT=2 / ERR=2 / GOAL_SIG=2 /
                   PATH_VALID=1 / PATH_PROGRESS=1（CTX=KIN=0，不在 ORDER 中）
    · 第 94-109 行 __init__ 按 max_path_points*4 累加出各段 (start, stop)
实测值（在主仓跑 `ObsLayout()._slices` / `.total_dim` 打印得到）：

    {"lidar": (0, 360), "odom": (360, 364), "path": (364, 484),
     "goal": (484, 486), "act": (486, 488), "err": (488, 490),
     "goal_sig": (490, 492), "path_valid": (492, 493),
     "path_progress": (493, 494)},  total = 494,  BASE_RAW_OBS_DIM = 494

本文件的 GOLDEN_SLICES 就是上面这组数：**不许**为了让测试通过而改它。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.envs.observation import (
    BASE_RAW_OBS_DIM,
    ObservationBuilder,
    ObsLayout,
    compute_obs_dims,
)

ORDER = (
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

# 主仓实测（见模块 docstring 的来源行号）——不是从本包实现反推出来的。
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


def _components(layout: ObsLayout, act_dim: int, **over):
    """按 layout 的分量宽度造一份合法输入（用于 build 往返测试）。"""
    kw = {
        "lidar": np.arange(360, dtype=np.float32) / 100.0,
        "odom": np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32),
        "path_flat": np.linspace(-1.0, 1.0, layout.path_dim, dtype=np.float32),
        "rel_goal": np.array([0.5, -0.5], dtype=np.float32),
        "prev_act": np.arange(act_dim, dtype=np.float32) / 10.0,
        "err": np.array([-0.3, 0.4], dtype=np.float32),
        "goal_sig": np.array([0.6, 0.8], dtype=np.float32),
        "path_valid": np.array([0.9], dtype=np.float32),
        "path_progress": 0.25,
    }
    kw.update(over)
    return kw


class TestObsLayoutGolden:
    """ObsLayout(2) 必须与主仓逐位相同。"""

    def test_default_is_action_dim_2(self):
        assert ObsLayout().action_dim == 2
        assert ObsLayout().max_path_points == 30
        assert ObsLayout().path_dim == 120

    def test_order_is_unchanged(self):
        assert ObsLayout.ORDER == ORDER

    def test_nine_slices_match_golden(self):
        layout = ObsLayout(2)
        assert layout._slices == GOLDEN_SLICES

    def test_total_dim_is_494(self):
        assert ObsLayout(2).total_dim == GOLDEN_TOTAL
        assert ObsLayout().total_dim == GOLDEN_TOTAL
        assert ObsLayout(max_path_points=30).total_dim == GOLDEN_TOTAL

    def test_base_raw_obs_dim_constant(self):
        assert BASE_RAW_OBS_DIM == GOLDEN_TOTAL
        assert BASE_RAW_OBS_DIM == ObsLayout().total_dim

    def test_offsets_are_contiguous_and_derived_from_widths(self):
        layout = ObsLayout(2)
        widths = [360, 4, 120, 2, 2, 2, 2, 1, 1]
        start = 0
        for name, width in zip(ORDER, widths):
            assert layout._slices[name] == (start, start + width)
            start += width
        assert start == layout.total_dim

    def test_slice_returns_expected_widths(self):
        layout = ObsLayout(2)
        parts = layout.slice(np.zeros(GOLDEN_TOTAL, dtype=np.float32))
        assert set(parts) == set(ORDER)
        assert [parts[name].size for name in ORDER] == [360, 4, 120, 2, 2, 2, 2, 1, 1]

    def test_slice_works_on_batched_arrays(self):
        layout = ObsLayout(2)
        obs = np.zeros((3, GOLDEN_TOTAL), dtype=np.float32)
        parts = layout.slice(obs)
        assert parts["lidar"].shape == (3, 360)
        assert parts["path_progress"].shape == (3, 1)

    def test_build_roundtrip(self):
        layout = ObsLayout(2)
        comps = _components(layout, act_dim=2)
        out = layout.build(**comps)
        assert out.shape == (GOLDEN_TOTAL,)
        assert out.dtype == np.float32
        parts = layout.slice(out)
        assert np.allclose(parts["lidar"], comps["lidar"])
        assert np.allclose(parts["odom"], comps["odom"])
        assert np.allclose(parts["path"], comps["path_flat"])
        assert np.allclose(parts["goal"], comps["rel_goal"])
        assert np.allclose(parts["act"], comps["prev_act"])
        assert np.allclose(parts["err"], comps["err"])
        assert np.allclose(parts["goal_sig"], comps["goal_sig"])
        assert np.allclose(parts["path_valid"], comps["path_valid"])
        assert parts["path_progress"][0] == pytest.approx(0.25)

    def test_build_rejects_wrong_width(self):
        layout = ObsLayout(2)
        comps = _components(layout, act_dim=3)  # act 多一维 ⇒ 495 ≠ 494
        with pytest.raises(ValueError, match="期望 494 维"):
            layout.build(**comps)

    def test_path_progress_is_clipped_and_nan_safe(self):
        layout = ObsLayout(2)
        for raw, want in (
            (5.0, 1.0),
            (-2.0, 0.0),
            (float("nan"), 0.0),
            (float("inf"), 1.0),
            (float("-inf"), 0.0),
        ):
            out = layout.build(**_components(layout, act_dim=2, path_progress=raw))
            assert out[-1] == pytest.approx(want), f"path_progress={raw!r}"

    def test_path_progress_wrong_size_raises(self):
        layout = ObsLayout(2)
        with pytest.raises(ValueError, match="path_progress"):
            layout.build(
                **_components(layout, act_dim=2, path_progress=np.array([0.1, 0.2]))
            )


class TestObsLayoutActionDim:
    """ObsLayout(action_dim) 泛化：顺序不变，偏移由宽度累加，新增分量仍在末尾。"""

    def test_action_dim_3_total_is_495(self):
        assert ObsLayout(3).total_dim == GOLDEN_TOTAL + 1

    def test_action_dim_3_slices(self):
        layout = ObsLayout(3)
        assert layout._slices == {
            "lidar": (0, 360),
            "odom": (360, 364),
            "path": (364, 484),
            "goal": (484, 486),
            "act": (486, 489),
            "err": (489, 491),
            "goal_sig": (491, 493),
            "path_valid": (493, 494),
            "path_progress": (494, 495),
        }

    def test_action_dim_3_keeps_prefix_identical(self):
        small, big = ObsLayout(2), ObsLayout(3)
        for name in ("lidar", "odom", "path", "goal"):
            assert small._slices[name] == big._slices[name]

    def test_action_dim_3_build_roundtrip(self):
        layout = ObsLayout(3)
        comps = _components(layout, act_dim=3)
        out = layout.build(**comps)
        assert out.shape == (495,)
        assert np.allclose(layout.slice(out)["act"], comps["prev_act"])
        assert layout.slice(out)["path_progress"][0] == pytest.approx(0.25)

    def test_action_dim_generic_widths(self):
        for act_dim in (1, 4, 7):
            layout = ObsLayout(act_dim)
            assert layout.total_dim == GOLDEN_TOTAL - 2 + act_dim
            assert layout._slices["path_progress"][1] == layout.total_dim

    def test_non_positive_action_dim_rejected(self):
        with pytest.raises(ValueError, match="action_dim"):
            ObsLayout(0)
        with pytest.raises(ValueError, match="action_dim"):
            ObsLayout(-1)

    def test_action_dim_is_keyword_friendly(self):
        """主仓所有调用点都用 keyword（max_path_points=），不能被位置参数破坏。"""
        assert ObsLayout(max_path_points=10).total_dim == 494 - 80
        assert ObsLayout(action_dim=2, max_path_points=10).total_dim == 414


class TestComputeObsDims:
    """compute_obs_dims：包里没有 train.yaml ⇒ 回落默认，不再 FileNotFoundError。"""

    def test_defaults_without_config_dir(self):
        assert compute_obs_dims() == (494, 494)
        assert compute_obs_dims(None) == (494, 494)

    def test_missing_train_yaml_falls_back(self, tmp_path):
        assert compute_obs_dims(str(tmp_path)) == (494, 494)

    def test_missing_directory_falls_back(self, tmp_path):
        assert compute_obs_dims(str(tmp_path / "nope")) == (494, 494)

    def test_reads_train_yaml_when_present(self, tmp_path):
        (tmp_path / "train.yaml").write_text(
            "network:\n  max_path_points: 30\n"
            "privileged:\n  enabled: true\n  n_bins: 36\n",
            encoding="utf-8",
        )
        assert compute_obs_dims(str(tmp_path)) == (494, 530)

    def test_privileged_disabled_adds_nothing(self, tmp_path):
        (tmp_path / "train.yaml").write_text(
            "network:\n  max_path_points: 20\n"
            "privileged:\n  enabled: false\n  n_bins: 36\n",
            encoding="utf-8",
        )
        assert compute_obs_dims(str(tmp_path)) == (454, 454)

    def test_action_dim_propagates(self, tmp_path):
        (tmp_path / "train.yaml").write_text(
            "network:\n  max_path_points: 30\n", encoding="utf-8"
        )
        assert compute_obs_dims(str(tmp_path), action_dim=3) == (495, 495)


class TestObservationBuilderDim:
    """ObservationBuilder 仍是 ackermann 专用（prev_act 恒 2 维），维度必须 494。"""

    def test_obs_dim_is_494(self):
        builder = ObservationBuilder()
        assert builder.obs_dim == 494
        assert builder.layout.total_dim == 494
        assert builder.layout._slices == GOLDEN_SLICES

    def test_max_path_points_scales_path_block(self):
        builder = ObservationBuilder(max_path_points=10)
        assert builder.obs_dim == 414
        assert builder.layout._slices["path"] == (364, 404)
