"""折线与几何工具：从主仓 benchmarks/controllers.py 逐字移植，外加纯追踪所需的弧长工具。

逐字来源（/home/zhao/drl-sim/benchmarks/controllers.py，行号为该文件 1-based 行号）：

    wrap_angle          L54-56
    path_segments       L80-87
    _segment_distances  L90-111
    polyline_project    L114-124
    polyline_query      L126-129
    obstacle_clearance  L227-266

新增（主仓无对应实现）：as_path、arc_lengths、point_at_arc、windowed_project。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "arc_lengths",
    "as_path",
    "obstacle_clearance",
    "path_segments",
    "point_at_arc",
    "polyline_project",
    "polyline_query",
    "windowed_project",
    "wrap_angle",
]

_TWO_PI = 2.0 * np.pi


def wrap_angle(a):
    """把角度归一化到 [-pi, pi)；标量入标量出，数组入数组出。（主仓 L54-56）"""
    return (np.asarray(a, dtype=np.float64) + np.pi) % _TWO_PI - np.pi


def as_path(path):
    """把入参统一成 (N,3) float64；空路径 / None 返回 None。"""
    if path is None:
        return None
    arr = np.asarray(path, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"path 需为 (N,3) 的 [x, y, θ]，实得 shape {arr.shape}")
    return arr[:, :3]


def path_segments(path):
    """把 (N,3) 路径拆成折线段：(起点 (M,2), 终点 (M,2), 起点鼻向 (M,))。（主仓 L80-87）"""
    if path.shape[0] == 1:
        return path[:, :2].copy(), path[:, :2].copy(), path[:, 2].copy()
    return path[:-1, :2], path[1:, :2], path[:-1, 2]


def _segment_distances(px, py, segs):
    """点集到**每一条**折线段的距离平方 + 段内投影比例：返回 (d2 (P,M), t (P,M))。（主仓 L90-111）"""
    ax, ay = segs[0][:, 0], segs[0][:, 1]
    bx, by = segs[1][:, 0], segs[1][:, 1]
    ex = bx - ax
    ey = by - ay
    l2 = ex * ex + ey * ey
    l2 = np.where(l2 < 1e-12, 1e-12, l2)
    rx = np.asarray(px, dtype=np.float64).reshape(-1, 1) - ax.reshape(1, -1)
    ry = np.asarray(py, dtype=np.float64).reshape(-1, 1) - ay.reshape(1, -1)
    t = np.clip(
        (rx * ex.reshape(1, -1) + ry * ey.reshape(1, -1)) / l2.reshape(1, -1), 0.0, 1.0
    )
    dx = rx - t * ex.reshape(1, -1)
    dy = ry - t * ey.reshape(1, -1)
    return dx * dx + dy * dy, t


def polyline_project(px, py, segs):
    """点集到折线的最近距离 + 最近段索引 + 段内位置 t。返回 (dist (P,), idx (P,), t (P,))。

    （主仓 L114-124：对 d2 取 argmin、只对选中项开方。）
    """
    d2, t = _segment_distances(px, py, segs)
    idx = np.argmin(d2, axis=1)
    ar = np.arange(d2.shape[0])
    return np.sqrt(d2[ar, idx]), idx, t[ar, idx]


def polyline_query(px, py, segs):
    """点集到折线的最近距离 + 最近段起点的鼻向：返回 (dist (P,), theta_ref (P,))。（主仓 L126-129）"""
    dist, idx, _ = polyline_project(px, py, segs)
    return dist, segs[2][idx]


def windowed_project(px, py, segs, anchor, back=10, fwd=40):
    """只在 [anchor-back, anchor+fwd] 的**段窗口**内取最近点，返回 (dist, idx, t)。

    anchor < 0（尚无历史）⇒ 全路径搜索。窗口解决两件事：
    1. 自交 / 首尾重合路径（例如跑第二圈的圆）上，全路径 argmin 会锚回时间序最早的那条
       支路，弧长从此不再前进；窗口把它粘在当前段附近。
    2. 一处误投影不会把"进度"永久拽走。
    back 只留少量余量（噪声/倒车抖动），fwd 决定一帧最多前进多远。
    """
    n_seg = int(segs[0].shape[0])
    if anchor is None or int(anchor) < 0:
        return polyline_project(px, py, segs)
    lo = max(0, int(anchor) - int(back))
    hi = min(n_seg - 1, int(anchor) + int(fwd))
    sub = (segs[0][lo : hi + 1], segs[1][lo : hi + 1], segs[2][lo : hi + 1])
    d2, t = _segment_distances(px, py, sub)
    k = np.argmin(d2, axis=1)
    ar = np.arange(d2.shape[0])
    gidx = (k + lo).astype(np.int64)
    return np.sqrt(d2[ar, k]), gidx, t[ar, k]


def obstacle_clearance(px, py, obstacles):
    """点集到最近障碍物**边界**的最近距离（无障碍时全 +inf）。（主仓 L227-266）

    支持 {"shape": {"name": "circle", "radius": r}, "state": [x, y, theta]} 与
    {"shape": {"name": "rectangle", "length": l, "width": w}, ...}。
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    out = np.full(px.shape, np.inf)
    if not obstacles:
        return out
    for obs in obstacles:
        try:
            shape = obs["shape"]
            state = obs["state"]
            cx, cy = float(state[0]), float(state[1])
            ct = float(state[2]) if len(state) > 2 else 0.0
            name = shape.get("name", "")
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if name == "circle":
            d = np.hypot(px - cx, py - cy) - float(shape.get("radius", 0.0))
        elif name == "rectangle":
            hl = 0.5 * float(shape.get("length", 0.0))
            hw = 0.5 * float(shape.get("width", 0.0))
            dx = px - cx
            dy = py - cy
            c, s = np.cos(-ct), np.sin(-ct)
            lx = dx * c - dy * s  # 旋到矩形本体系
            ly = dx * s + dy * c
            qx = np.clip(lx, -hl, hl)
            qy = np.clip(ly, -hw, hw)
            d = np.hypot(lx - qx, ly - qy)
        else:
            continue
        np.minimum(out, d, out=out)
    return out


# ── 弧长工具（纯追踪用；主仓无对应实现）─────────────────────────────


def arc_lengths(path):
    """沿折线用**点间欧氏间距累加**出来的弧长，返回 (s_cum (N,), seg_len (M,))。"""
    xy = np.asarray(path, dtype=np.float64)[:, :2]
    if xy.shape[0] < 2:
        return np.zeros(1), np.zeros(1)
    seg_len = np.hypot(*(xy[1:] - xy[:-1]).T)
    s_cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    return s_cum, seg_len


def point_at_arc(path, s_cum, s):
    """折线上弧长 s 处的点 (x, y)；s 超出两端时夹到端点。"""
    xy = np.asarray(path, dtype=np.float64)
    if xy.shape[0] == 1 or s_cum[-1] <= 0.0:
        return float(xy[0, 0]), float(xy[0, 1])
    sc = float(np.clip(s, s_cum[0], s_cum[-1]))
    k = int(np.searchsorted(s_cum, sc, side="right") - 1)
    k = min(max(k, 0), s_cum.shape[0] - 2)
    span = float(s_cum[k + 1] - s_cum[k])
    u = 0.0 if span <= 0.0 else (sc - float(s_cum[k])) / span
    x = xy[k, 0] + u * (xy[k + 1, 0] - xy[k, 0])
    y = xy[k, 1] + u * (xy[k + 1, 1] - xy[k, 1])
    return float(x), float(y)
