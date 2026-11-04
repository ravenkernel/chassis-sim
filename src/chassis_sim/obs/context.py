"""观测上下文构建（1.0）：raw 传感器字典 → `ObsContext`。

本模块是原 `envs/observation.py` 里**仍然在用**的那一层：`ContextBuilder.build_context`
算式与时序状态逐位保持 0.2.0 的行为（里程计差分、路径
提取器、CTE/航向误差、path_valid），只把产物交给 `ObsSpec` 决定怎么拼。

1.0.0 删掉的东西（都在原 `envs/observation.py`）：

* 9 段定长布局与它的 494 维定长 `build()` —— 494 只能由显式 `ObsSpec` 表达；
* 旧类名、按 train.yaml 推算维度的辅助函数、以及定长观测总维常量；
* legacy 两键 `lidar_front` / `lidar_rear`——雷达帧只来自
  `bridge.get_sensor_reading()`，由 `lidar=` 以「传感器名 → ranges」传入。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from chassis_sim.obs.spec import ObsContext
from chassis_sim.paths import PathExtractor, cartesian_to_polar

#: `goal_sig` 段宽度：sin/cos(终点航向误差)。与 `obs.GoalSig.dim` 同值。
GOAL_SIG_DIM = 2


class ContextBuilder:
    """把 raw 传感器字典算成 :class:`ObsContext`（只计算，不拼接）。

    时序状态（里程计差分 `_prev_odom_*`、`PathExtractor`）在 `reset()` 里清空；
    跨回合不残留上一回合的路径上下文。

    Args:
        max_path_points: 路径段点数（每点 4 通道，见 `obs.PathSegment`）。
        path_power: 降采样间距曲线（2.0 = 近密远疏）。
        lookahead: 路径提取的前向弧长（米）。
        v_max / v_min: 速度归一化区间（act 段第 2 通道）。
        delta_max: 前轮转角归一化除数（act 段第 1 通道）；None = 由
            `arctan(0.145/0.25)` 反解（与底盘默认几何一致）。
        path_d_max: CTE 归一化除数（err 段第 0 维 = clip(cte / path_d_max, ±1)）。
            由 env 从 `RewardConfig.path_d_max` 转发，小于 0.01 时按 0.01 取。
        extrapolate_path / path_fill_mode / path_lookahead_floor:
            原样转发给 `PathExtractor`。
    """

    def __init__(
        self,
        *,
        max_path_points: int = 30,
        path_power: float = 2.0,
        lookahead: float = 20.0,
        v_max: float = 1.3,
        v_min: float = -0.6,
        delta_max: float | None = None,
        path_d_max: float = 4.0,
        extrapolate_path: bool = True,
        path_fill_mode: str = "extrapolate",
        path_lookahead_floor: float = 1.0,
    ) -> None:
        self.max_path_points = int(max_path_points)
        self.path_power = float(path_power)
        # CTE 归一化除数：err 分量第 0 维 = clip(cte / path_d_max, ±1)。
        # 【P2-19】它必须由 **RewardConfig.path_d_max**（config/reward.yaml 的
        # path_d_max）转发进来 —— 此前 AckermannEnv 构造本类时没传，于是这里恒取
        # 签名默认 4.0，reward.yaml 的同名键等于死键（改 YAML 不改变观测尺度）。
        self.path_d_max = max(float(path_d_max), 0.01)
        self._path_fill_mode = str(path_fill_mode)

        # PathExtractor: world→body transform + directional subsample -> (30, 3)
        self._path_extractor = PathExtractor(
            lookahead=lookahead,
            target_count=self.max_path_points,
            power=path_power,
            extrapolate=extrapolate_path,
            fill_mode=path_fill_mode,
            lookahead_floor=path_lookahead_floor,
        )

        # 运动学限幅（act 段归一化用；值来自 cfg，不写死）
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.delta_max = (
            float(delta_max)
            if delta_max is not None
            else float(np.arctan(0.145 / 0.25))
        )

        # 里程计历史值（用于计算相对位移）
        self._prev_odom_x: float = 0.0
        self._prev_odom_y: float = 0.0
        self._prev_odom_theta: float = 0.0
        self._odom_initialized: bool = False

    @property
    def path_extractor(self) -> PathExtractor:
        """暴露 PathExtractor（预览渲染画车体系路径时用）。"""
        return self._path_extractor

    def reset(self) -> None:
        """清空时序状态（回合起点调用）。"""
        self._odom_initialized = False

    def build_context(
        self,
        raw: dict,
        prev_delta: float = 0.0,
        prev_v: float = 0.0,
        err_signals: np.ndarray | None = None,
        goal_signals: np.ndarray | None = None,
        path_progress: float | np.ndarray = 0.0,
        lidar: Mapping[str, np.ndarray] | None = None,
    ) -> ObsContext:
        """把 raw 传感器字典算成 :class:`ObsContext`（只计算，不拼接）。

        Args:
            raw: 桥的原始读数，用到 `raw["odometry"]`（[x, y, θ, v, ω]）与
                `raw["path_world"]`（世界系 (N, 3) 航点）。
            prev_delta / prev_v: 上一拍的物理量（act 段归一化输入）。
            err_signals: 显式 CTE/航向误差；None = 由本方法按当步路径算。
            goal_signals: 显式 goal_sig；None = 全零（与原实现相同）。
            path_progress: 沿参考路径的**已完成比例**（env 侧真值
                `arc_completion = _target_idx/(n_pts-1)`）。必须传入**当步**的值
                （与奖励侧 timeout 缩放用的是同一个量），reset 时传 0.0。
            lidar: 传感器名 → 当帧 ranges（`LidarSegment` 段按名字引用）。
                缺帧的保持/兜底由 env 负责，本方法只按原样放进 `ObsContext.lidar`。
        """
        # ── LiDAR & Odometry ────────────────────────────────
        lidar_frames: dict[str, np.ndarray] = {
            str(name): np.asarray(scan, dtype=np.float32)
            for name, scan in (lidar or {}).items()
        }

        odom_raw = raw["odometry"]
        x, y, theta = float(odom_raw[0]), float(odom_raw[1]), float(odom_raw[2])
        v, omega = float(odom_raw[3]), float(odom_raw[4])

        if self._odom_initialized:
            dx_world = x - self._prev_odom_x
            dy_world = y - self._prev_odom_y
            cos_prev = np.cos(-self._prev_odom_theta)
            sin_prev = np.sin(-self._prev_odom_theta)
            dx_body = dx_world * cos_prev - dy_world * sin_prev
            dy_body = dx_world * sin_prev + dy_world * cos_prev
        else:
            dx_body, dy_body = 0.0, 0.0
            self._odom_initialized = True

        self._prev_odom_x, self._prev_odom_y = x, y
        self._prev_odom_theta = theta
        # 通道顺序 [dx_body*10, dy_body*10, v, ω]（2026-10-28 删除 cosθ/sinθ：
        # 世界系绝对航向破坏任务等变性，且与 path/goal/err 的车体系表达冗余）。
        odom_rel = np.array(
            [dx_body * 10.0, dy_body * 10.0, v, omega], dtype=np.float32
        )

        # ── Path & Goal (修复状态泄漏) ──────────────────────
        path_world = raw["path_world"]
        robot_pose = raw["odometry"][:3]

        # 显式跟踪当前步的 valid 数量，切断对提取器内部状态的盲目依赖
        current_n_valid = 0

        if len(path_world) >= 2:
            body_path_cartesian = self._path_extractor.extract(
                path_world,
                robot_pose,
                velocity=v,
            )
            current_n_valid = self._path_extractor.last_n_valid
        else:
            body_path_cartesian = np.zeros(
                (self.max_path_points, 3), dtype=np.float32
            )  # [N,3]: dx, dy, dθ
            current_n_valid = 0  # 彻底阻断上回合数据的泄露

        # Cartesian body_path 必须先于极坐标转换保留：CTE/heading_err 用的是
        # 原始 (dx, dy)，转换后就丢了。
        # Convert to polar: (dx, dy, dθ) → (ρ, sinφ, cosφ, dθ)
        # Shared function with controller_node for deployment consistency.
        body_path = cartesian_to_polar(body_path_cartesian)  # [N, 4]

        path_flat = body_path.flatten().astype(np.float32)
        rel_goal = self._compute_relative_goal(path_world, robot_pose)

        # ── Prev Action（kin 分量已删：它是 prev_action 的确定性函数）──
        prev_delta_norm = prev_delta / max(self.delta_max, 0.01)
        v_range = max(self.v_max - self.v_min, 0.01)
        prev_v_norm = (prev_v - self.v_min) / v_range * 2.0 - 1.0
        prev_act = np.array([prev_delta_norm, prev_v_norm], dtype=np.float32)

        # ── Error signals ──────────────────────────────────
        if err_signals is None:
            # 依赖 current_n_valid 进行真实的有效性判断，而不是虚假的 len(body_path)
            if current_n_valid > 0:
                # 仅在有效点内寻找最近点（使用 Cartesian 副本——polar 已丢失 dx,dy）
                valid_path = body_path_cartesian[:current_n_valid]
                nearest_idx = int(
                    np.argmin(np.hypot(valid_path[:, 0], valid_path[:, 1]))
                )
                nx, ny = valid_path[nearest_idx, 0], valid_path[nearest_idx, 1]
                path_tangent = float(valid_path[nearest_idx, 2])
                cte_val = -nx * np.sin(path_tangent) + ny * np.cos(path_tangent)
                heading_err = path_tangent
            else:
                cte_val = 0.0
                heading_err = 0.0

            cte_norm = np.clip(cte_val / self.path_d_max, -1.0, 1.0)
            # 加上标准化，防止角度超出 [-pi, pi]
            heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
            err_signals = np.array([cte_norm, heading_err], dtype=np.float32)

        # ── Path valid fraction ─────────────────────────────
        if self._path_fill_mode == "sentinel":
            path_valid_frac = np.array(
                [current_n_valid / max(self.max_path_points, 1)],
                dtype=np.float32,
            )
        else:
            path_valid_frac = np.array([1.0], dtype=np.float32)

        # ── Goal signals (终点方向预判) ──────────────────────
        if goal_signals is None:
            goal_signals = np.zeros(GOAL_SIG_DIM, dtype=np.float32)

        return ObsContext(
            lidar=lidar_frames,
            odom_rel=odom_rel,
            path_flat=path_flat,
            goal=rel_goal,
            act=prev_act,
            err=err_signals,
            goal_sig=goal_signals,
            path_valid=path_valid_frac,
            path_progress=path_progress,
        )

    @staticmethod
    def _compute_relative_goal(
        path_world: np.ndarray, robot_pose: np.ndarray | None = None
    ) -> np.ndarray:
        """返回路径终点（goal）在车身坐标系中的位置。

        终点方向是独立的 destination signal，与 path_body（30 点路径形状）
        互补。模型不需要从 path_body 的最后一个点中猜终点位置。
        """
        if len(path_world) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 终点 = 路径最后一个点
        gx, gy = float(path_world[-1, 0]), float(path_world[-1, 1])

        if robot_pose is not None:
            x, y, theta = (
                float(robot_pose[0]),
                float(robot_pose[1]),
                float(robot_pose[2]),
            )
            dx = gx - x
            dy = gy - y
            cos_t = np.cos(-theta)
            sin_t = np.sin(-theta)
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
        else:
            rx, ry = gx, gy

        return np.array([rx, ry], dtype=np.float32)
