"""模块级 ray-casting 辅助函数 —— 从 numpy_bridge.py:30-132 原样移植。

两个函数都是"射线 × 障碍物"维度上的全向量化实现（没有单条射线的 Python 循环），
公式一字未改：圆的解析二次方程、矩形的"世界→局部"变换 + slab test。

⚠ 保留的两条旧坑（注释即代码的一部分，别删）：

* `_ray_rects_intersection` 里的世界→局部变换用 **R(−θ)**
  （`l = (c·dx + s·dy, −s·dx + c·dy)`）。旧实现曾写成 cos(−θ)/sin(−θ) 搭配
  转置形式，两个取负相互抵消等于 R(+θ)，把矩形镜像成 −θ 朝向 —— 非轴对齐时
  约 90% 射线量错。
* slab test 的退化分支必须保留 `parallel_x/parallel_y` 与 `out_x/out_y`：
  方向分量为 0 时 `t = ±inf` 会污染 `t_min`/`t_max` 的比较。
"""

from __future__ import annotations

import numpy as np


def _ray_circles_intersection(
    origins: np.ndarray,  # (R, 2)
    directions: np.ndarray,  # (R, 2)  单位向量
    centers: np.ndarray,  # (C, 2)
    radii: np.ndarray,  # (C,)
    max_range: float,
) -> np.ndarray:  # (R,)
    """每条射线到最近圆形交点的距离。

    对每一对 (射线, 圆) 使用 broadcasting 求解 |origin + t·direction − center|² = radius²，
    然后取每条射线的最小 *t*。
    """
    # to_center: (R, 1, 2) − (1, C, 2) → (R, C, 2)
    to_center = origins[:, None, :] - centers[None, :, :]

    # b = 2·(S−C)·D   → (R, C)
    b = 2.0 * np.sum(to_center * directions[:, None, :], axis=-1)

    # c = |S−C|² − r²   → (R, C)
    c = np.sum(to_center * to_center, axis=-1) - radii[None, :] ** 2

    disc = b * b - 4.0 * c  # (R, C)
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5

    # 每对 (射线, 圆) 的最小正 t
    t = np.where((t1 > 0.0) & (disc >= 0.0), t1, np.inf)
    t = np.where((t2 > 0.0) & (disc >= 0.0) & (t2 < t), t2, t)

    t_min = np.min(t, axis=1)  # (R,)
    t_min = np.where(np.isfinite(t_min), t_min, max_range)
    return np.clip(t_min, 0.0, max_range)


def _ray_rects_intersection(
    origins: np.ndarray,  # (R, 2)
    directions: np.ndarray,  # (R, 2)  单位向量
    centers: np.ndarray,  # (M, 2)
    half_dims: np.ndarray,  # (M, 2)  [半长, 半宽]
    thetas: np.ndarray,  # (M,)    矩形在世界坐标系中的朝向
    max_range: float,
) -> np.ndarray:  # (R,)
    """每条射线到最近矩形交点的距离。

    将每条射线转换到每个矩形的局部坐标系（AABB），然后应用标准的 slab test。
    在射线 × 矩形维度上完全向量化。
    """
    # ── 将射线起点转换到每个矩形的局部坐标系 ──────────────
    # 世界→局部 需要 R(−θ)：l = (c·dx + s·dy, −s·dx + c·dy)，c=cosθ, s=sinθ。
    # ⚠ 修复 (2026-08-17): 旧代码写成 cos(−θ)/sin(−θ) 搭配 R 的转置形式，
    # 两个取负相互抵消等于 R(+θ) —— 即把矩形镜像成 −θ 朝向，
    # 导致 LiDAR 看到的矩形与 env SAT 碰撞/真实几何不一致（非轴对齐时 ~90% 射线错误）。
    cos_t = np.cos(thetas)  # (M,)
    sin_t = np.sin(thetas)

    local_o = origins[:, None, :] - centers[None, :, :]  # (R, M, 2)
    local_ox = cos_t[None, :] * local_o[:, :, 0] + sin_t[None, :] * local_o[:, :, 1]
    local_oy = -sin_t[None, :] * local_o[:, :, 0] + cos_t[None, :] * local_o[:, :, 1]

    # ── 转换射线方向 ──────────────────────────────────────
    local_dx = (
        cos_t[None, :] * directions[:, None, 0]
        + sin_t[None, :] * directions[:, None, 1]
    )
    local_dy = (
        -sin_t[None, :] * directions[:, None, 0]
        + cos_t[None, :] * directions[:, None, 1]
    )

    hl = half_dims[None, :, 0]  # (1, M)
    hw = half_dims[None, :, 1]  # (1, M)

    with np.errstate(divide="ignore", invalid="ignore"):
        # ── slab test: x 轴 ─────────────────────────────────────
        tx1 = (-hl - local_ox) / local_dx
        tx2 = (hl - local_ox) / local_dx
        tx_min = np.minimum(tx1, tx2)
        tx_max = np.maximum(tx1, tx2)

        # ── slab test: y 轴 ─────────────────────────────────────
        ty1 = (-hw - local_oy) / local_dy
        ty2 = (hw - local_oy) / local_dy
        ty_min = np.minimum(ty1, ty2)
        ty_max = np.maximum(ty1, ty2)

    t_min = np.maximum(tx_min, ty_min)
    t_max = np.minimum(tx_max, ty_max)

    # ── 退化情况 ──────────────────────────────────────────
    eps = 1e-10
    parallel_x = np.abs(local_dx) < eps
    parallel_y = np.abs(local_dy) < eps
    out_x = parallel_x & ((local_ox < -hl) | (local_ox > hl))
    out_y = parallel_y & ((local_oy < -hw) | (local_oy > hw))

    t = np.where(t_min > 0.0, t_min, t_max)
    valid = (t_min <= t_max) & (t_max > 0.0) & ~out_x & ~out_y
    t = np.where(valid, t, np.inf)

    t_min_ray = np.min(t, axis=1)
    t_min_ray = np.where(np.isfinite(t_min_ray), t_min_ray, max_range)
    return np.clip(t_min_ray, 0.0, max_range)
