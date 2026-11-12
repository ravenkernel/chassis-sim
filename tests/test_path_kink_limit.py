"""单点折角上限（`max_single_ratio`）的回归测试。"""

import numpy as np

from chassis_sim.planning.map_generator import MapGenerator

SPACING = 0.05
R_MIN = 0.25


def _straight(n, step=SPACING, theta=0.0):
    x = np.arange(n) * step
    return np.stack([x, np.zeros(n), np.full(n, theta)], axis=1)


def test_clean_arc_passes():
    r = 0.30  # 比最小转弯半径宽 ⇒ 每步 ~9.5°
    ang = np.arange(40) * (SPACING / r)
    path = np.stack([r * np.sin(ang), r * (1 - np.cos(ang)), ang], axis=1)
    assert bool(MapGenerator._validate_path_curvature(path, SPACING, R_MIN)) is True


def test_single_spike_is_rejected():
    path = _straight(40)
    # 模拟「平滑把绕行揉成尖刺」：第 20/21 点几乎重合（0.03 m 尖峰），
    # 运动方向从 +31° 一步翻到 −80°（实测 seed1004 平滑后是 65°/R=0.02 m）
    path[20, :2] = [1.00, 0.03]
    path[21, :2] = [1.005, 0.00]
    # 鼻向跟着各自的运动方向走（否则会被判成换档点、被 mask 跳过）
    path[20, 2] = np.deg2rad(-80.0)
    path[21, 2] = 0.0
    assert MapGenerator._max_geometric_kink(path) > np.deg2rad(46.0)
    assert bool(MapGenerator._validate_path_curvature(path, SPACING, R_MIN)) is False


def test_clean_cusp_passes():
    # 前进 1 m 后原路倒车 0.5 m：运动方向翻转 π，但那是换档、不是折角
    fwd = _straight(20)
    rev = _straight(11)
    rev[:, 0] = fwd[-1, 0] - rev[:, 0]
    rev[:, 2] = np.pi + np.pi  # 倒车腿鼻向 = 运动方向(π) + π ⇒ 0
    path = np.vstack([fwd, rev[1:]])
    assert MapGenerator._max_geometric_kink(path) < np.deg2rad(5.0)
    assert bool(MapGenerator._validate_path_curvature(path, SPACING, R_MIN)) is True


def test_planner_discretization_kinks_still_pass():
    # 规划器离散化留下的 20–34° 折角属于已知放宽，不能被新上限误杀
    path = _straight(40)
    ang = np.deg2rad(30.0)
    path[20:, 0] += np.arange(20) * SPACING * (np.cos(ang) - 1.0)
    path[20:, 1] += np.arange(20) * SPACING * np.sin(ang)
    assert MapGenerator._max_geometric_kink(path) == pytest_approx(np.deg2rad(30.0))
    assert bool(MapGenerator._validate_path_curvature(path, SPACING, R_MIN)) is True


def pytest_approx(x, tol=1e-6):
    import pytest

    return pytest.approx(x, abs=tol)
