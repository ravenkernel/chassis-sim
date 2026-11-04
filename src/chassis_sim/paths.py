"""路径工具、几何判定与路径提取 —— 1.0 公共路径 `chassis_sim.paths`。

本模块收拢 1.0 之前分散在 `chassis_sim.envs` 下的三处私有深路径：

* `envs.path_utils`：`angle_wrap` / `np_angle_wrap` / `clip_path_to_bounds` /
  `subsample_path` / `cartesian_to_polar` / `_arc_length`；
* `envs.path_extractor.PathExtractor`；
* `envs.chassis_env` 的档位几何判定：`segment_is_reverse` / `GEAR_EPS` /
  `gear_is_compliant`。

训练（env 观测）与部署（Nav2 控制器插件）都从这里导入，路径不再有包内私有深路径。

角度工具
--------
`angle_wrap` 是标量版（math），`np_angle_wrap` 是矢量化版（numpy），二者都归一到
[-pi, pi]。
"""

from __future__ import annotations

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


def cartesian_to_polar(body_path: np.ndarray) -> np.ndarray:
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


def segment_is_reverse(
    path_body: np.ndarray,
    nearest_idx: int,
    heading_error_local: float,
    prev_dir: float = 1.0,
) -> float:
    """判定最近路径点所在段是"正着开"还是"倒着开" (2026-10-05)。

    返回 +1（前进段）/ −1（倒车段）；路径点重合无法判定时返回 *prev_dir*。

    **为什么必须是几何判定**：`path_body[:, 2]` 存的是**鼻向**（RS 积分，
    nose direction），而路径点之间的位置差给出的是**运动方向**。两者点积 < 0
    ⇔ 这一段要求倒车。

    旧实现 `1.0 if abs(heading_error_local) < π/2 else -1.0` 是错的：
    `heading_error_local` 正是"路径鼻向 − 车头朝向"，**正确跟踪时前进段与
    倒车段都 ≈ 0** ⇒ path_dir 恒为 +1，导致倒车段的 progress 符号翻转永不生效、
    传给 reward 的 path_dir 也失去意义。

    用**中心差分**取运动方向：比单边差抗噪，且天然处理路径两端。
    """
    n = len(path_body)
    if n < 2:
        return prev_dir
    i0 = max(0, nearest_idx - 1)
    i1 = min(n - 1, nearest_idx + 1)
    mdx = float(path_body[i1, 0] - path_body[i0, 0])
    mdy = float(path_body[i1, 1] - path_body[i0, 1])
    if float(np.hypot(mdx, mdy)) <= 1e-6:
        return prev_dir  # 路径点重合，沿用上次判定
    nx, ny = float(np.cos(heading_error_local)), float(np.sin(heading_error_local))
    return 1.0 if (mdx * nx + mdy * ny) >= 0.0 else -1.0


# 判定"车是否在动"的速度阈值（m/s）。低于它视为原地转向/静止，
# 不参与档位合规判定（否则原地掉头的瞬间会被误判成"反向前进"）。
GEAR_EPS = 0.01


def gear_is_compliant(path_dir: float, v: float, eps: float = GEAR_EPS) -> bool:
    """路径段规定的档位与车辆实际档位是否一致 (2026-10-05)。

    路径第 2 列是**鼻向**，所以倒车段（`path_dir < 0`）要求车头背对运动方向
    ⇒ 车辆必须以 **v < 0** 前进才能同时满足"沿路径推进"与"车头对齐路径鼻向"。
    前进段则要求 v ≥ 0。

    为什么需要它：`arc_progress` 是**位置势能**（沿路径的弧长推进），
    **方向无关** —— 反向前进与正确倒车拿到完全相同的进度分。
    实测 run025 在倒车段上 82~90% 的步是**反向前进**（stage4 82.0%、
    stage5 89.5%），即"掉头正着开"骗过了位置势能。此函数用于把那部分补贴掐掉。
    """
    if path_dir < 0:
        return v < -eps  # 要求倒车 ⇒ 必须真的在倒
    return v >= -eps  # 要求前进 ⇒ 不得倒车（v≈0 的原地转向不受罚）


class PathExtractor:
    """Extract a fixed-size local path segment from a variable-length global path.

    Algorithm:
        1. Find the waypoint nearest to the robot (Euclidean distance).
        2. Walk forward along the path from that point until *lookahead*
           metres of cumulative arc-length have been covered.
        3. Transform the segment to the robot's body frame.
        4. Subsample to exactly *target_count* points using the same
           ``power``-law spacing as training (near-dense, far-sparse).

    Zero ROS dependencies — operates on raw numpy arrays so it can be
    shared between training (SimBridge), deployment (ROS2Bridge), and
    a Nav2 Controller Plugin.

    Args:
        lookahead:    metres to extract forward along the path.
        target_count: number of output waypoints.
        power:        subsample spacing curve (2.0 = near-dense-far-sparse).
    """

    def __init__(
        self,
        lookahead: float = 8.0,
        target_count: int = 30,
        power: float = 2.0,
        extrapolate: bool = True,
        fill_mode: str = "extrapolate",  # "extrapolate" | "sentinel"
        lookahead_floor: float = 1.0,
    ):
        self._lookahead = float(lookahead)
        self._target_count = int(target_count)
        self._power = float(power)
        self._extrapolate = bool(extrapolate)
        self._fill_mode = str(fill_mode)
        self._lookahead_floor = float(lookahead_floor)
        self.last_n_valid: int = target_count  # 上次 extract 的真点数

    # -- Public API --

    def extract(
        self,
        path_world: np.ndarray,
        robot_pose: np.ndarray,
        velocity: float = 0.0,
    ) -> np.ndarray:
        """Return ``(target_count, 3)`` local waypoints in body frame.

        Args:
            path_world:  ``(N, 2|3)`` global waypoints [x, y, (theta)]。
                         **时间序**：索引 0=起点，N-1=终点；第 2 列为鼻向。
            robot_pose:  ``(3,)`` robot pose ``[x, y, theta]`` in world frame.
            velocity:   当前车速 (m/s)。**仅为兼容既有调用方签名而保留，
                         不参与任何方向/里程决策**（B3 修复，2026-10-05）：
                         路径为时间序，倒车段同样沿索引增大方向推进，所以
                         "前方" 恒为 nearest→goal，剩余弧长也恒为
                         ``_arc_length(path[nearest:])``——二者都与 v 符号无关。
                         调用方仍传该参数（``envs/observation.py``、
                         ``ros_bridge/controller_node.py``、
                         ``ros_bridge/nav2_controller_plugin.py``），保留签名
                         可避免改动全部调用点；符号敏感性由
                         ``tests/test_direction_semantics.py`` 锁死为"零影响"。

        Returns:
            ``(target_count, 3)`` float32 [dx, dy, dtheta] in body frame.
            Zero-filled when *path_world* has fewer than 2 points.
        """
        if len(path_world) < 2 or path_world.shape[1] < 2:
            return np.zeros((self._target_count, 3), dtype=np.float32)

        # 1. 找最近路径点（欧几里得距离，与朝向/速度无关）
        robot_xy = np.asarray(robot_pose[:2], dtype=np.float64)
        dists = np.hypot(path_world[:, 0] - robot_xy[0], path_world[:, 1] - robot_xy[1])
        nearest_idx = int(np.argmin(dists))

        # 2. 计算从最近点到终点的剩余弧长——近终点时缩小 lookahead，
        #     避免直线外推污染 PathEncoder 输入。
        #     恒取 nearest→goal（不是 start→nearest）：路径为时间序，倒车段
        #     同样沿索引增大方向推进，剩余里程与 velocity 符号无关。
        remaining = _arc_length(path_world[nearest_idx:])
        effective_lookahead = min(
            self._lookahead, max(remaining * 1.2, self._lookahead_floor)
        )

        # 3. 恒定按索引增大方向提取（nearest→goal）。
        #    _walk_forward 保证 end_idx >= nearest_idx + 1 ⇒ segment 非空，
        #    且最近点天然在首位（近端密集采样方向不变，subsample_path 依赖此约定）。
        start_idx, end_idx = self._walk_along(
            path_world,
            nearest_idx,
            effective_lookahead,
        )
        segment = path_world[start_idx:end_idx]

        # 4. 世界坐标系 → 车身坐标系
        body = self._world_to_body(segment, robot_pose)

        # 5. 记录真点数（外推/填充前）
        n_real = len(body)
        self.last_n_valid = min(n_real, self._target_count)

        # 6. 短段外推（仅 extrapolate 模式）
        if self._fill_mode == "extrapolate" and self._extrapolate:
            body = self._fill_lookahead(body)

        # 7. 降采样到固定点数
        result = subsample_path(body, self._target_count, power=self._power)

        # 8. 填充不足点数
        if len(result) < self._target_count:
            if self._fill_mode == "sentinel":
                # 填零哨兵——PathEncoder Conv1d 局部感受野可区分零与真点，
                # 配合 path_valid_frac 让网络知晓有效区域边界。
                pad = np.zeros(
                    (self._target_count - len(result), result.shape[1]),
                    dtype=result.dtype,
                )
            else:
                pad = np.tile(result[-1:], (self._target_count - len(result), 1))
            result = np.concatenate([result, pad], axis=0)
        return result.astype(np.float32)

    # -- Internal helpers --

    def _fill_lookahead(self, body: np.ndarray) -> np.ndarray:
        """Extrapolate body-frame segment forward to cover at least *lookahead* metres."""
        if len(body) == 0:
            return body

        # 如果只有 1 个点，无法通过两点求差得到方向，
        # 则直接使用该点自带的 theta 角度作为外推的单位方向向量
        if len(body) == 1:
            total_len = 0.0
            last_pt = body[-1].astype(np.float64)
            # body_frame 下的 theta 是相对于车头的目标朝向
            target_theta = last_pt[2]
            direction = np.array(
                [np.cos(target_theta), np.sin(target_theta)], dtype=np.float64
            )
        else:
            xy = body[:, :2]
            diffs = np.diff(xy, axis=0)
            seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
            total_len = float(np.sum(seg_lens))

            if total_len >= self._lookahead:
                return body

            # 从最后两个不同点取外推方向
            direction = np.array([1.0, 0.0], dtype=np.float64)
            for i in range(len(body) - 1, 0, -1):
                d = np.hypot(body[i, 0] - body[i - 1, 0], body[i, 1] - body[i - 1, 1])
                if d > 1e-6:
                    direction = (
                        np.array(
                            [body[i, 0] - body[i - 1, 0], body[i, 1] - body[i - 1, 1]],
                            dtype=np.float64,
                        )
                        / d
                    )
                    break

            last_pt = body[-1].astype(np.float64)

        # 10 cm 步长外推，覆盖剩余 lookahead 距离
        remaining = self._lookahead - total_len
        step = 0.1
        n_extra = int(np.ceil(max(remaining, 0.1) / step))  # 保证至少外推一点

        extra = np.empty((n_extra, body.shape[1]), dtype=np.float64)
        for i in range(n_extra):
            dist = (i + 1) * step
            extra[i, 0] = last_pt[0] + direction[0] * dist
            extra[i, 1] = last_pt[1] + direction[1] * dist
            extra[i, 2] = last_pt[2]  # 继承末点航向

        return np.concatenate([body, extra], axis=0)

    @staticmethod
    def _walk_along(
        path_world: np.ndarray,
        start_idx: int,
        lookahead: float,
    ) -> tuple[int, int]:
        """Return ``(start, end)`` indices covering *lookahead* metres
        forward along the path topology.

        恒定沿**索引增大方向**（nearest→goal）行走。B3 修复（2026-10-05）:
        删除了原 ``reverse=True`` 分支（nearest→start）——路径是时间序，
        倒车段也沿索引增大方向推进（鼻向与运动方向相反只是"车头朝向"这一
        几何事实，不改变行进顺序），按速度符号反向提取会输出整段反向的
        "已走过的路"。
        """
        end_idx = PathExtractor._walk_forward(path_world, start_idx, lookahead)
        return start_idx, end_idx

    @staticmethod
    def _walk_forward(path_world: np.ndarray, start_idx: int, lookahead: float) -> int:
        """Return exclusive end index after walking *lookahead* metres
        forward along the path from *start_idx*."""
        if start_idx >= len(path_world) - 1:
            return len(path_world)
        tail = path_world[start_idx:]
        diffs = np.diff(tail[:, :2], axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cumdist = np.concatenate([[0.0], np.cumsum(seg_lens)])
        offset = int(np.searchsorted(cumdist, lookahead)) + 1
        return min(start_idx + offset, len(path_world))

    @staticmethod
    def _world_to_body(points: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """Transform [x, y, (theta)] from world to body frame -> [dx, dy, dtheta]."""
        x, y, theta = (
            float(robot_pose[0]),
            float(robot_pose[1]),
            float(robot_pose[2]),
        )
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t

        if points.shape[1] >= 3:
            dtheta = np_angle_wrap(points[:, 2] - theta)
            return np.column_stack([rx, ry, dtheta])
        return np.column_stack([rx, ry, np.zeros(len(rx))])
