"""路径降采样工具 —— 仿真训练与 ROS2 部署共享。

将任意密度的输入路径通过非均匀采样生成固定尺寸的路径表示：
近处点密集，远处点稀疏。满足局部路径跟踪控制器
"局部精确、全局方向"的需求。
"""

import math

import numpy as np


def angle_wrap(a: float) -> float:
    """角度归一化到 [-pi, pi]（标量，基于 math）。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def np_angle_wrap(a: np.ndarray | float) -> np.ndarray | float:
    """角度归一化到 [-pi, pi]（矢量化，基于 numpy）。"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def clip_path_to_bounds(
    path: np.ndarray,
    bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
    margin: float = 0.0,
) -> np.ndarray:
    """将路径点 (x, y) 位置裁剪到 *bounds* 范围内。

    仅裁剪位置列（列 0 和列 1）；朝向（列 2）保持不变。
    可选的 *margin* 从内侧收缩有效边界
    （例如 ``margin=0.2`` 使点距离边缘 ≥ 0.2 m）。

    Args:
        path:   (N, 2|3) 路径点数组。
        bounds: (x_min, x_max, y_min, y_max) 世界坐标。
        margin: 每条边界向内收缩的距离（≥ 0）。

    Returns:
        *path* 的**副本**，位置已裁剪。若 *path* 仅 2 列则返回 (x, y)；
        3+ 列时保留朝向。
    """
    x_min, x_max, y_min, y_max = bounds
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin

    clipped = path.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], x_min, x_max)
    clipped[:, 1] = np.clip(clipped[:, 1], y_min, y_max)
    return clipped


def subsample_path(
    path: np.ndarray,
    target_count: int,
    power: float = 1.0,
) -> np.ndarray:
    n_in = len(path)
    if n_in <= target_count:
        return path.copy()

    # ── 1. 预处理：剔除距离极小的冗余点 (Numerical Stability) ────
    # 这一步能防止后面 searchsorted 和除零风险
    dist_sq = np.sum(np.diff(path[:, :2], axis=0) ** 2, axis=1)
    mask = np.concatenate([[True], dist_sq > 1e-6])
    path = path[mask]
    n_in = len(path)
    if n_in < 2:
        return np.tile(path[0:1], (target_count, 1))

    # ── 2. 累积弧长 (保持原逻辑) ──────────────────────────────────
    xy = path[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]

    # ── 3. 采样计算 ──────────────────────────────────────────────
    fractions = (np.arange(target_count) / max(target_count - 1, 1)) ** power
    sample_dists = fractions * total

    has_theta = path.shape[1] >= 3
    out_dim = 3 if has_theta else 2
    result = np.empty((target_count, out_dim), dtype=path.dtype)

    # 向量化寻找线段索引 (替代 for 循环内的 searchsorted)
    # searchsorted 也可以向量化，效率更高
    indices = np.searchsorted(cumlen, sample_dists) - 1
    indices = np.clip(indices, 0, n_in - 2)

    for i in range(target_count):
        idx = indices[i]
        d = sample_dists[i]

        # 避免除零，虽然预处理了冗余点，但为了稳健性保留 epsilon
        seg_dist = max(seg_lens[idx], 1e-9)
        t = (d - cumlen[idx]) / seg_dist
        t = np.clip(t, 0.0, 1.0)

        # 位置插值
        result[i, :2] = path[idx, :2] + t * diffs[idx]

        # 朝向插值
        if has_theta:
            t0 = path[idx, 2]
            t1 = path[idx + 1, 2]
            dtheta = np_angle_wrap(t1 - t0)
            result[i, 2] = np_angle_wrap(t0 + t * dtheta)

    # 确保首尾锚点完全对齐
    result[0] = path[0, :out_dim]
    result[-1] = path[-1, :out_dim]

    return result


def cartesian_to_polar(body_path: "np.ndarray") -> "np.ndarray":
    """Convert body-frame [dx, dy, dtheta] to polar [rho, sin_phi, cos_phi, dtheta].

    Args:
        body_path: (N, 3) float32 array with columns [dx, dy, dtheta] in body frame.
                   dx, dy in metres; dtheta in radians, wrapped to [-pi, pi].

    Returns:
        (N, 4) float32 array with columns [rho, sin_phi, cos_phi, dtheta].
        rho in metres; sin_phi, cos_phi dimensionless in [-1, 1];
        dtheta in radians (raw, no normalisation applied).

    2026-10-28 删除第 5 列 ddtheta：它是 dtheta 的**一阶差分**，零独立信息
    （Conv1d 在相邻 dθ 上本来就能学到同一个量），却占了整条观测的 30 维。
    dtheta 自身已由调用方 wrap 到 [-π, π]（PathExtractor._world_to_body 用
    np_angle_wrap），所以这里不再需要任何 wrap 逻辑。
    """
    import numpy as np

    dx = body_path[:, 0]
    dy = body_path[:, 1]
    rho = np.hypot(dx, dy)
    eps = 1e-8
    sin_phi = dy / np.maximum(rho, eps)
    cos_phi = dx / np.maximum(rho, eps)
    dtheta = body_path[:, 2]
    return np.column_stack([rho, sin_phi, cos_phi, dtheta])


def _arc_length(path: np.ndarray) -> float:
    """Compute cumulative arc length along a path of [x, y, ...] waypoints.

    Args:
        path: (N, 2+) array of waypoints.

    Returns:
        Total arc length in metres. Returns 0.0 for paths with < 2 points.
    """
    if len(path) < 2:
        return 0.0
    diffs = np.diff(path[:, :2], axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
