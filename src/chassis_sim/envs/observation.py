"""Observation builder — 组装原始传感器数据供 Actor 端到端编码。

将 encoder 移入 Actor 后，ObservationBuilder 不再持有或调用任何 encoder。
它只负责：
  1. 从 raw sensor dict 提取原始数据
  2. 将里程计转换为车身坐标系相对表示
  3. 用 PathExtractor 将世界坐标系路径转换为车身坐标系
  4. 拼接为固定长度的原始观测向量（numpy array）

Actor 在 forward 时自行调用 encoder 子模块，实现端到端梯度传递。

============================================================================
ObsLayout — 原始观测布局的单一真相源
============================================================================
所有切片索引（build / Actor.encode / controller_node）都从此派生。
修改 max_path_points 只需更新 ObsLayout(max_path_points=N) 一处。
============================================================================
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch
from chassis_sim.envs.path_extractor import PathExtractor
from chassis_sim.envs.path_utils import cartesian_to_polar as _cartesian_to_polar


class ObsLayout:
    """原始观测的维度布局 —— 单一真相源。

    所有构建（ObservationBuilder / controller_node）和切片（Actor.encode）
    都通过此类进行，消除硬编码索引造成的静默错位风险。

    组件顺序（固定）:
        lidar(360) → odom(4) → path(N×4) → goal(2) → act(2) → err(2)
        → goal_sig(2) → path_valid(1) → path_progress(1)

    【2026-10-28 删除 34 维零信息分量】审计证明原始观测里有 34 维不含独立信息，
    已全部删除（529 → 494，max_path_points=30）。被删的四类：
      · kin(2)          max_accel/max_steer_rate 是 prev_action 的**确定性函数**
                        （prev_delta、prev_v 都在 act 分量里）⇒ 零新信息；
      · path 的 ddθ(30) 是 dθ 的**一阶差分** ⇒ Conv1d 可自行学到；
      · goal_sig[2]     1/(goal_dist+0.5) 是**终点距离的函数**，而距离本身可由
                        goal(2) 与 path 末点算出；
      · odom 的 cosθ/sinθ(2) 是**世界系绝对航向**，破坏任务等变性（同一张地图
                        平移/旋转后任务不变，但这两维会变）。
    **不用"填零/保留空通道"保形**——形状变化必须对任何错位的调用点立刻报错。

    goal(2) = 终点在车体系中的 (x, y)，提供独立的 destination signal。
    path_progress(1) = **env 侧真值 arc_completion**（沿参考路径的路点推进比例，
        [0,1]，单调不减，与奖励侧 timeout 缩放用的是**同一个量**）。2026-10-05
        新增：此前观测里没有任何"我沿路径走到哪了"的标量，actor 只能从 30 点
        路径形状隐式推断进度，折返/迷宫平行走廊上直接失效。

    【顺序约定】新增分量一律**追加在末尾**，前面所有分量的切片偏移保持不变。

    用法::

        layout = ObsLayout(max_path_points=30)          # 训练: 30 pts

        # 构建
        raw = layout.build(lidar_360, odom_rel, path_flat,
                          rel_goal, prev_act, err, goal_sig, path_valid,
                          path_progress=arc_completion)

        # 切片 (numpy 或 torch)
        parts = layout.slice(obs)   # → {"lidar": ..., "odom": ..., ...}

    Args:
        action_dim: prev_action 分量宽度。默认 2（ackermann: [delta_norm, v_norm]）。
            非 2 的底盘（如差速/麦轮若要塞更多量）由调用方显式传值，切片偏移
            随之整体后移；顺序与各分量含义不变。
        max_path_points: 路径点数量（与 PathEncoder 输入一致）。
    """

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

    # 固定维度分量
    LIDAR = 360
    ODOM = 4  # [dx_body*10, dy_body*10, v, ω]（世界系 cosθ/sinθ 已删）
    CTX = 0  # 已移除 prev_path_ctx — MLP 无记忆，时间对齐错位风险 (2026-07-15)
    GOAL = 2
    ACT = 2
    KIN = 0  # 已删除（2026-10-28）：max_accel/max_steer_rate 是 prev_action 的
    # 确定性函数，零独立信息。常量保留为显式 0 只为可读性，**不在
    # ORDER 中** ⇒ 多传一个 kin 会直接撞维度校验，不会静默错位。
    ERR = 2  # CTE normalized + heading error
    GOAL_SIG = 2  # sin(goal_heading_err), cos(goal_heading_err)
    PATH_VALID = 1  # fraction of real path points [0,1]
    PATH_PROGRESS = 1  # env 真值 arc_completion = target_idx/(n_pts-1)，∈[0,1]

    def __init__(self, action_dim: int = 2, max_path_points: int = 30):
        if action_dim <= 0:
            raise ValueError(f"action_dim 必须为正整数，得到 {action_dim!r}")
        self.action_dim = int(action_dim)
        self.max_path_points = max_path_points
        self.path_dim = max_path_points * 4  # rho, sin_phi, cos_phi, dtheta

        # 维度表按 ORDER 顺序排列，偏移由**宽度累加**得出（不写死数字）：
        # 新增分量只许追加在末尾，前面分量的 (start, stop) 因此保持不变。
        dims = [
            self.LIDAR,
            self.ODOM,
            self.path_dim,
            self.GOAL,
            self.action_dim,
            self.ERR,
            self.GOAL_SIG,
            self.PATH_VALID,
            self.PATH_PROGRESS,
        ]
        self._slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, dim in zip(self.ORDER, dims):
            self._slices[name] = (offset, offset + dim)
            offset += dim
        self._total = offset

    @property
    def total_dim(self) -> int:
        """原始观测总维度 (e.g. 494 for 30 path points, no ctx)。"""
        return self._total

    def slice(self, obs) -> dict[str, np.ndarray | torch.Tensor]:
        """将原始观测拆分为具名分量 dict。

        Args:
            obs: 1D/2D numpy array 或 torch Tensor。

        Returns:
            dict with keys "lidar", "odom", "path", "goal", "act",
            "err", "goal_sig", "path_valid", "path_progress"。
        """
        return {name: obs[..., s:e] for name, (s, e) in self._slices.items()}

    def build(
        self,
        lidar,
        odom,
        path_flat,
        rel_goal,
        prev_act,
        err,
        goal_sig,
        path_valid,
        path_progress=0.0,
    ) -> np.ndarray:
        """拼接各分量为总观测向量，附带维度校验。

        Args:
            path_progress: 沿路径已走比例（env 真值 arc_completion），标量或
                shape (1,) 数组；默认 0.0 = 回合起点/无路径（旧调用点行为）。

        Raises:
            ValueError: 如果拼接后维度不等于 total_dim。
        """
        # 夹紧到 [0,1]：真值本身已在 [0,1]，这里只把 NaN/越界挡在观测之外
        # （观测一写进 buffer 就会污染整个训练，宁可在此静默夹紧）。
        _prog = np.asarray(path_progress, dtype=np.float32).ravel()
        if _prog.size != self.PATH_PROGRESS:
            raise ValueError(
                f"ObsLayout.build: path_progress 期望 {self.PATH_PROGRESS} 个值，"
                f"实际 {_prog.size} 个"
            )
        _prog = np.nan_to_num(_prog, nan=0.0, posinf=1.0, neginf=0.0)
        _prog = np.clip(_prog, 0.0, 1.0)

        fused = np.concatenate(
            [
                np.asarray(lidar, dtype=np.float32).ravel(),
                np.asarray(odom, dtype=np.float32).ravel(),
                np.asarray(path_flat, dtype=np.float32).ravel(),
                np.asarray(rel_goal, dtype=np.float32).ravel(),
                np.asarray(prev_act, dtype=np.float32).ravel(),
                np.asarray(err, dtype=np.float32).ravel(),
                np.asarray(goal_sig, dtype=np.float32).ravel(),
                np.asarray(path_valid, dtype=np.float32).ravel(),
                _prog,
            ]
        )
        if len(fused) != self._total:
            raise ValueError(
                f"ObsLayout.build: 期望 {self._total} 维，实际拼接出 {len(fused)} 维。"
                f" 请检查各分量维度: "
                f"lidar={np.asarray(lidar).size}, "
                f"odom={np.asarray(odom).size}, "
                f"path={np.asarray(path_flat).size}, "
                f"goal={np.asarray(rel_goal).size}, "
                f"act={np.asarray(prev_act).size}, "
                f"err={np.asarray(err).size}, "
                f"goal_sig={np.asarray(goal_sig).size}, "
                f"path_valid={np.asarray(path_valid).size}, "
                f"path_progress={_prog.size}"
            )
        return fused


# 向后兼容常量
BASE_RAW_OBS_DIM = ObsLayout().total_dim  # 494（max_path_points=30, no ctx）


@functools.lru_cache(maxsize=1)
def compute_obs_dims(
    config_dir: str | None = None, action_dim: int = 2
) -> tuple[int, int]:
    """Compute base and full observation dimensions from train.yaml config.

    Returns (base_obs_dim, full_obs_dim) where full = base + privileged bins.

    抽包后的语义变化：train.yaml **不在** wheel 里（它是训练配置，不属于 env
    层）。因此 config_dir 为 None、或目录里没有 train.yaml 时，回落到
    ObsLayout 的默认值（max_path_points=30、privileged 关闭），而不是像主仓
    那样直接 FileNotFoundError。存在该文件时逐位沿用主仓的读法。
    """
    train_cfg: dict = {}
    if config_dir is not None:
        train_path = Path(config_dir) / "train.yaml"
        if train_path.is_file():
            with open(train_path) as f:
                train_cfg = yaml.safe_load(f) or {}

    network = train_cfg.get("network", {})
    max_path_points = network.get("max_path_points", 30)
    privileged_enabled = train_cfg.get("privileged", {}).get("enabled", False)
    priv_bins = train_cfg.get("privileged", {}).get("n_bins", 36)

    layout = ObsLayout(action_dim=action_dim, max_path_points=max_path_points)
    base = layout.total_dim
    full = base + (priv_bins if privileged_enabled else 0)
    return base, full


class ObservationBuilder:
    """Builds the raw fused observation from raw sensor data.

    Pipeline (encoder-free):
        raw_lidar [360]      -> direct concat -> [360]
        raw_odom  [4]        -> direct concat -> [4]
        raw_path  [30,4]     -> flatten       -> [120]  (4ch polar)
        goal_dest [2]        -> direct concat -> [2]   ← 终点方向（车体系）
        prev_action   [2]    -> direct concat -> [2]
        err_signals   [2]    -> direct concat -> [2]   ← CTE + heading_error
        goal_sig      [2]    -> direct concat -> [2]   ← sin/cos(ghe)
        path_valid    [1]    -> direct concat -> [1]   ← 真实路点占比
        path_progress [1]    -> direct concat -> [1]   ← env 真值 arc_completion
        Total: 360+4+120+2+2+2+2+1+1 = 494

    Encoder modules (LidarEncoder, OdomEncoder, PathEncoder) are now
    owned by AckermannActor and applied during Actor.forward().
    """

    def __init__(
        self,
        device: str = "cpu",
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
    ):
        self.device = device
        self.max_path_points = max_path_points
        self.path_power = path_power
        # CTE 归一化除数：err 分量第 0 维 = clip(cte / path_d_max, ±1)。
        # 【P2-19】它必须由 **RewardConfig.path_d_max**（config/reward.yaml 的
        # path_d_max）转发进来 —— 此前 AckermannEnv 构造本类时没传，于是这里恒取
        # 签名默认 4.0，reward.yaml 的同名键等于死键（改 YAML 不改变观测尺度）。
        # 现在 wire 在 AckermannEnv.__init__ 的 ObservationBuilder(...) 里。
        self.path_d_max = max(path_d_max, 0.01)
        self._path_fill_mode = path_fill_mode

        # ObsLayout: 单一真相源 — 所有维度索引从此派生
        self._layout = ObsLayout(max_path_points=max_path_points)

        # PathExtractor: world→body transform + directional subsample -> (30, 3)
        self._path_extractor = PathExtractor(
            lookahead=lookahead,
            target_count=max_path_points,
            power=path_power,
            extrapolate=extrapolate_path,
            fill_mode=path_fill_mode,
            lookahead_floor=path_lookahead_floor,
        )

        # Kinematic limits for flag computation (from config, not hardcoded)
        self.v_max = v_max
        self.v_min = v_min
        self.delta_max = delta_max if delta_max is not None else np.arctan(0.145 / 0.25)

        # 里程计历史值（用于计算相对位移）
        self._prev_odom_x: float = 0.0
        self._prev_odom_y: float = 0.0
        self._prev_odom_theta: float = 0.0
        self._odom_initialized: bool = False

    @property
    def obs_dim(self) -> int:
        """Total raw observation dimension (derived from ObsLayout)."""
        return self._layout.total_dim

    @property
    def layout(self) -> ObsLayout:
        """Expose layout for external consumers (Actor, controller)."""
        return self._layout

    @property
    def path_extractor(self) -> PathExtractor:
        """Expose PathExtractor for preview rendering (body-frame path viz)."""
        return self._path_extractor

    def reset(self) -> None:
        """Clear temporal state at episode start."""
        self._odom_initialized = False

    def build(
        self,
        raw: dict,
        prev_delta: float = 0.0,
        prev_v: float = 0.0,
        err_signals: np.ndarray | None = None,
        goal_signals: np.ndarray | None = None,
        path_progress: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """组装原始观测。

        Args:
            path_progress: 沿参考路径的**已完成比例**（env 侧真值
                ``arc_completion = _target_idx/(n_pts-1)``）。必须传入**当步**
                的值（与奖励侧 timeout 缩放用的是同一个量）；reset 时传 0.0。
                默认 0.0 仅用于无路径 / 旧调用点（如 ROS 部署节点）。
        """
        # ── LiDAR & Odometry (保持原样) ──────────────────────
        lf = np.asarray(raw["lidar_front"], dtype=np.float32)
        lr = np.asarray(raw["lidar_rear"], dtype=np.float32)
        lidar_360 = np.concatenate([lf, lr])

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
        body_path = _cartesian_to_polar(body_path_cartesian)  # [N, 4]

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
            goal_signals = np.zeros(self._layout.GOAL_SIG, dtype=np.float32)

        return self._layout.build(
            lidar_360,
            odom_rel,
            path_flat,
            rel_goal,
            prev_act,
            err_signals,
            goal_signals,
            path_valid_frac,
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
