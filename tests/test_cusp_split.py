"""cusp_split 平滑：假翻转抹掉、真折返保留且折返顶点不被揉走（2026-11-12）。"""

import numpy as np

from chassis_sim.planning.map_generator import MapGenerator


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _dirs(path):
    mv = np.diff(path[:, :2], axis=0)
    return np.arctan2(mv[:, 1], mv[:, 0])


def _rev_mask(path):
    d = _dirs(path)
    return np.abs(wrap(path[:-1, 2] - d)) > np.pi / 2


def _bogus_flip_path():
    """几何一路 +x，鼻向列在中间突然翻 π（车不可能原地掉头 ⇒ 假翻转）。"""
    xs = np.linspace(0.0, 3.0, 61)
    nose = np.zeros_like(xs)
    nose[30:] = np.pi
    return np.column_stack([xs, np.zeros_like(xs), nose])


def _real_cusp_path():
    """真折返：先 +x 走 1.5 m，再沿同一走廊退回（运动方向翻 180°），鼻向保持 +x。

    注意：真折返的两条腿**本来就重合**（车从原路退回来），所以"两腿分开"不是判据；
    要守的是**折返顶点不被平滑拉走**、以及倒车段标记保留。
    """
    xs = np.concatenate([np.linspace(0.0, 1.5, 31), np.linspace(1.5, 0.0, 31)[1:]])
    ys = np.zeros_like(xs)
    nose = np.zeros_like(xs)
    return np.column_stack([xs, ys, nose])


def test_bogus_flip_is_dropped():
    path = _bogus_flip_path()
    out = MapGenerator._smooth_path(path, [], cusp_split=True)
    assert out.shape == path.shape
    assert _rev_mask(out).sum() == 0, (
        f"几何没折返的 π 翻转应被抹掉，实测仍有 {_rev_mask(out).sum()} 个倒车段"
    )


def test_bogus_flip_kept_when_option_off():
    path = _bogus_flip_path()
    out = MapGenerator._smooth_path(path, [])
    assert _rev_mask(out).mean() > 0.3, "默认关时行为必须保持历史样子（假翻转仍存在）"


def test_real_cusp_is_kept_and_vertex_frozen():
    path = _real_cusp_path()
    out = MapGenerator._smooth_path(path, [], cusp_split=True)
    rev = _rev_mask(out)
    assert rev[:5].sum() == 0, "第一条腿是前进段，不该被判成倒车"
    assert rev[-5:].all(), "真折返的第二条腿必须保留倒车标记"
    # 折返顶点位于 index 30：分腿平滑把两条腿的端点都当固定点 ⇒ 顶点不动
    assert np.allclose(out[30, :2], path[30, :2], atol=1e-9), (
        f"折返顶点被平滑拉走了：{out[30, :2]} vs {path[30, :2]}"
    )


def test_option_off_is_bit_identical_to_default():
    path = _real_cusp_path()
    a = MapGenerator._smooth_path(path, [], cusp_split=False)
    b = MapGenerator._smooth_path(path, [])
    assert np.array_equal(a, b)
