"""非对称 Actor-Critic 的特权障碍物编码器。

训练期间 Critic 通过基于真实障碍物几何形状的角度分箱射线投射
接收完美的障碍物距离信息（含世界边界）。Actor
只能看到带噪声的 LiDAR。这种非对称性让 Critic 能够学习准确的
Q 值，同时迫使 Actor 从真实传感器数据中提取相同的信息
—— 一种众所周知的 sim-to-real 技术。

2026-07-17 重写：向量化射线投射（消除 O(n_bins × n_obs) 标量循环）
+ 世界边界纳入特权信息（修复 Wall Ghost Bug）。
"""

import numpy as np


class PrivilegedEncoder:
    """在机器人周围的角分箱中对真实障碍物距离进行编码。

    输出是一个固定大小的归一化向量，可以拼接到
    Actor 的观测上，形成 Critic 的特权输入。
    """

    def __init__(
        self,
        n_bins: int = 36,
        max_range: float = 5.0,
        world_bounds: tuple[float, float, float, float] | None = None,
    ):
        self.n_bins = n_bins
        self.max_range = max_range
        self._world_bounds = world_bounds  # (xmin, xmax, ymin, ymax) or None
        # 分箱中心角，对称分布在 [-π, π)，0° 位于 bin 中心
        self._angles = np.linspace(
            -np.pi + np.pi / n_bins, np.pi - np.pi / n_bins, n_bins
        )
        # 预计算射线方向（单位向量），避免每步重复计算 trig
        self._ray_dx = np.cos(self._angles).astype(np.float64)
        self._ray_dy = np.sin(self._angles).astype(np.float64)
        self._eps = 1e-12

    @property
    def output_dim(self) -> int:
        return self.n_bins

    def encode(
        self, robot_pose: tuple[float, float, float], obstacles: list[dict]
    ) -> np.ndarray:
        """在 *n_bins* 个方向上投射射线并返回归一化距离。

        向量化实现：在外层仅遍历障碍物（通常 0-10 个），每个障碍物
        一次性计算全部 n_bins 条射线的交点，取代原 O(n_bins × n_obs)
        的标量级 Python 双重循环。

        Args:
            robot_pose: (x, y, theta)，世界坐标系。
            obstacles:  障碍物字典列表（MapGenerator 格式）。

        Returns:
            Float32 数组，形状为 (n_bins,)，每个元素在 [0, 1] 范围内，
            其中 0 = 障碍物在机器人原点处，1 = 范围内无障碍物。
        """
        rx, ry, rtheta = (
            float(robot_pose[0]),
            float(robot_pose[1]),
            float(robot_pose[2]),
        )

        distances = np.full(self.n_bins, self.max_range, dtype=np.float64)

        # 预计算世界坐标系下的射线方向 [n_bins]
        cos_theta = np.cos(rtheta)
        sin_theta = np.sin(rtheta)
        world_dx = cos_theta * self._ray_dx - sin_theta * self._ray_dy
        world_dy = sin_theta * self._ray_dx + cos_theta * self._ray_dy

        # ── 障碍物射线投射（向量化：外层循障碍物，内层 36 条射线一次算完）──
        for obs in obstacles:
            shape = obs["shape"]
            sx, sy = float(obs["state"][0]), float(obs["state"][1])
            stheta = float(obs["state"][2])

            if shape["name"] == "circle":
                t = self._ray_circle_batch(
                    rx, ry, world_dx, world_dy, sx, sy, shape["radius"]
                )
            elif shape["name"] == "rectangle":
                t = self._ray_rect_batch(
                    rx,
                    ry,
                    world_dx,
                    world_dy,
                    sx,
                    sy,
                    stheta,
                    shape["length"],
                    shape["width"],
                )
            else:
                continue

            distances = np.minimum(distances, t)

        # ── 世界边界射线投射（修复 Wall Ghost Bug）──────────────
        if self._world_bounds is not None:
            t_wall = self._ray_aabb_batch(
                rx, ry, world_dx, world_dy, self._world_bounds
            )
            distances = np.minimum(distances, t_wall)

        return (np.minimum(distances, self.max_range) / self.max_range).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # 向量化射线投射 — 每个障碍物一次处理全部 n_bins 条射线
    # ------------------------------------------------------------------

    def _ray_circle_batch(
        self,
        ox: float,
        oy: float,
        dx: np.ndarray,
        dy: np.ndarray,
        cx: float,
        cy: float,
        r: float,
    ) -> np.ndarray:
        """向量化射线–圆相交。返回每条射线的最小正 t [n_bins]，无命中为 inf。"""
        fx = ox - cx
        fy = oy - cy
        # dx, dy 是单位向量，故 a = 1（省去乘法）
        b = 2.0 * (fx * dx + fy * dy)  # [n_bins]
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4.0 * c  # [n_bins]
        hit = disc >= 0.0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = np.where(hit, (-b - sqrt_disc) * 0.5, np.inf)
        t2 = np.where(hit, (-b + sqrt_disc) * 0.5, np.inf)
        # 取最小正 t（射线起点可能在圆内也可能在圆外）
        t_pos1 = np.where(t1 > self._eps, t1, np.inf)
        t_pos2 = np.where(t2 > self._eps, t2, np.inf)
        return np.minimum(t_pos1, t_pos2)

    def _ray_rect_batch(
        self,
        ox: float,
        oy: float,
        dx: np.ndarray,
        dy: np.ndarray,
        cx: float,
        cy: float,
        theta: float,
        length: float,
        width: float,
    ) -> np.ndarray:
        """向量化射线–OBB 相交。返回每条射线的最小正 t [n_bins]，无命中为 inf。"""
        half_l = length * 0.5
        half_w = width * 0.5

        # 变换到矩形局部坐标系
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        lx = (ox - cx) * cos_t - (oy - cy) * sin_t  # scalar
        ly = (ox - cx) * sin_t + (oy - cy) * cos_t  # scalar
        rdx = dx * cos_t - dy * sin_t  # [n_bins]
        rdy = dx * sin_t + dy * cos_t  # [n_bins]

        # X 轴切片 [n_bins]
        parallel_x = np.abs(rdx) <= self._eps
        outside_x = parallel_x & ((lx < -half_l) | (lx > half_l))
        t1x = np.where(parallel_x, np.inf, (-half_l - lx) / rdx)
        t2x = np.where(parallel_x, np.inf, (half_l - lx) / rdx)
        tmin_x = np.where(parallel_x, -np.inf, np.minimum(t1x, t2x))
        tmax_x = np.where(parallel_x, np.inf, np.maximum(t1x, t2x))

        # Y 轴切片 [n_bins]
        parallel_y = np.abs(rdy) <= self._eps
        outside_y = parallel_y & ((ly < -half_w) | (ly > half_w))
        t1y = np.where(parallel_y, np.inf, (-half_w - ly) / rdy)
        t2y = np.where(parallel_y, np.inf, (half_w - ly) / rdy)
        tmin_y = np.where(parallel_y, -np.inf, np.minimum(t1y, t2y))
        tmax_y = np.where(parallel_y, np.inf, np.maximum(t1y, t2y))

        t_enter = np.maximum(tmin_x, tmin_y)
        t_exit = np.minimum(tmax_x, tmax_y)

        valid = (t_enter <= t_exit) & (t_exit > self._eps) & ~outside_x & ~outside_y
        return np.where(valid, np.where(t_enter > self._eps, t_enter, t_exit), np.inf)

    def _ray_aabb_batch(
        self, ox: float, oy: float, dx: np.ndarray, dy: np.ndarray, world_bounds: tuple
    ) -> np.ndarray:
        """向量化射线–AABB 相交（世界边界）。返回每条射线到最近墙壁的距离 [n_bins]。"""
        xmin, xmax, ymin, ymax = world_bounds

        # X 平面
        parallel_x = np.abs(dx) <= self._eps
        t1x = np.where(parallel_x, np.inf, (xmin - ox) / dx)
        t2x = np.where(parallel_x, np.inf, (xmax - ox) / dx)
        tmin_x = np.where(parallel_x, -np.inf, np.minimum(t1x, t2x))
        tmax_x = np.where(parallel_x, np.inf, np.maximum(t1x, t2x))

        # Y 平面
        parallel_y = np.abs(dy) <= self._eps
        t1y = np.where(parallel_y, np.inf, (ymin - oy) / dy)
        t2y = np.where(parallel_y, np.inf, (ymax - oy) / dy)
        tmin_y = np.where(parallel_y, -np.inf, np.minimum(t1y, t2y))
        tmax_y = np.where(parallel_y, np.inf, np.maximum(t1y, t2y))

        t_enter = np.maximum(tmin_x, tmin_y)
        t_exit = np.minimum(tmax_x, tmax_y)

        # 机器人恒在世界内 (t_enter <= 0)，取 t_exit（前向墙壁）
        valid = (t_enter <= t_exit) & (t_exit > self._eps)
        return np.where(valid, t_exit, np.inf)
