"""时间序（索引增大方向）局部路径提取 — 训练与 Nav2 部署共用。

取代旧的"前方过滤"（``path[path[:,0] > -0.5]``）——后者会丢弃机器人身后的
有效倒车航点。本模块从最近航点出发、**恒定沿索引增大方向**（nearest→goal）
走，与机器人朝向、车速符号都无关，然后变换到车身系并降采样。

为什么恒定沿索引增大方向（B3 修复，2026-10-05）:
    参考路径的索引是**时间序**——索引 0 是任务起点，N-1 是终点，车辆沿索引
    增大方向完成任务（``envs/ackermann_env.py`` 的 ``target_idx`` 单向推进，
    注释明确写"无论前进还是倒车，车辆总是沿着索引增大方向"）。
    因此"接下来要走的路"永远是 nearest→goal。
    路径第 2 列存的是**车头朝向（nose，RS 积分所得）而非运动方向**
    （见 ``envs/ackermann_env.py`` 的 heading_error_local 注释、
    ``envs/map_generator.py`` 里 "Keep as-is — downstream consumers expect
    nose direction"）：倒车段上 ``path[i+1,:2] - path[i,:2]``（运动方向）与
    ``path[i,2]``（鼻向）点积为负。按速度符号切换提取方向会把这一几何事实
    误当成"路径要反着走"，导致：
      1. 中段倒车时输出整体翻 180°（把"已走过的路"当成"前方的路"，
         实测 dx<0 比例 0.10→1.00）；
      2. 起点附近倒车时 nearest→start 无点可取，提取段塌成 1–3 个真点，
         ``last_n_valid`` 从 30 掉到 1–3（path_valid 1.00→0.03），
         策略执行正确倒车的第一步就把主引导输入打崩。
"""

from __future__ import annotations

import numpy as np

from chassis_sim.envs.path_utils import np_angle_wrap, subsample_path


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
        from .path_utils import _arc_length

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
