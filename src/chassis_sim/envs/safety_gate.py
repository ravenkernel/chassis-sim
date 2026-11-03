"""Safety Gate：用于推理的硬运动学和碰撞约束。"""

from __future__ import annotations

import numpy as np


class SafetyGate:
    """部署和评估的 Post-Actor 安全层。

    依次应用三项硬约束：
    1. 运动学限制（转向速率、加速度）
    2. 紧急刹车（基于 LiDAR 最小距离）
    3. 转弯半径限制（基于轴距和最小转弯半径）

    训练流程不使用此模块 —— 仅在评估和部署时使用。
    """

    def __init__(
        self,
        wheelbase: float = 0.145,
        min_turning_radius: float = 0.25,
        max_steer_rate: float = np.pi / 2,  # rad/s
        max_accel: float = 2.0,  # m/s²
        max_decel: float = -4.0,  # m/s²
        brake_distance: float = 0.3,  # m — 急刹车
        slow_distance: float = 1.0,  # m — 线性减速
        v_max: float = 2.0,
        v_min: float = -1.0,
        dt: float = 0.1,
    ):
        self.wheelbase = wheelbase
        self.min_turning_radius = min_turning_radius
        self.max_steer_rate = max_steer_rate
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.brake_distance = brake_distance
        self.slow_distance = slow_distance
        self.v_max = v_max
        self.v_min = v_min
        self.dt = dt

        # 缓存的上限值
        self._max_delta = np.arctan(wheelbase / min_turning_radius)

        # 状态
        self._prev_delta = 0.0
        self._prev_v = 0.0
        self._brake_triggered = False
        self._slowdown_triggered = False

    def reset(self):
        """为新 episode 重置内部状态。"""
        self._prev_delta = 0.0
        self._prev_v = 0.0
        self._brake_triggered = False
        self._slowdown_triggered = False

    def __call__(
        self, action: np.ndarray, lidar_360: np.ndarray | None = None
    ) -> np.ndarray:
        """应用所有安全约束。

        Args:
            action: 来自 Actor 的 [delta, v]
            lidar_360: [360] LiDAR 距离，若为 None 则跳过刹车检查

        Returns:
            安全的 [delta, v] 动作
        """
        delta = float(action[0])
        v = float(action[1])
        self._brake_triggered = False
        self._slowdown_triggered = False

        # 1. 运动学限制（使用 prev_delta/prev_v 进行速率限制）
        delta, v = self._apply_kinematic_limits(delta, v)

        # 2. 紧急刹车（可能将 v 覆盖为 0）
        if lidar_360 is not None and len(lidar_360) > 0:
            delta, v = self._apply_emergency_brake(delta, v, lidar_360)

        # 3. 转弯半径限制
        delta, v = self._apply_turning_radius_clamp(delta, v)

        # 保存 APPLIED 值（刹车/限制后），供下一步速率限制使用。
        # 正确反映机器人实际状态，避免刹车释放后瞬时跳回高速度。
        self._prev_delta = delta
        self._prev_v = v
        return np.array([delta, v], dtype=np.float32)

    def _apply_kinematic_limits(self, delta: float, v: float) -> tuple[float, float]:
        # 转向速率限制
        delta_max_step = self.max_steer_rate * self.dt
        delta = np.clip(
            delta, self._prev_delta - delta_max_step, self._prev_delta + delta_max_step
        )

        # 加速度限制
        v_target = np.clip(v, self.v_min, self.v_max)
        if v_target >= self._prev_v:
            v_max_step = self.max_accel * self.dt
            v = min(v_target, self._prev_v + v_max_step)
        else:
            v_min_step = self.max_decel * self.dt
            v = max(v_target, self._prev_v + v_min_step)

        return delta, v

    def _apply_emergency_brake(
        self, delta: float, v: float, lidar_360: np.ndarray
    ) -> tuple[float, float]:
        min_dist = float(np.min(lidar_360))

        if min_dist < self.brake_distance:
            self._brake_triggered = True
            return 0.0, 0.0

        if min_dist < self.slow_distance:
            self._slowdown_triggered = True
            v_max_allowed = (min_dist / self.slow_distance) * self.v_max
            v = np.clip(v, self.v_min, v_max_allowed)

        return delta, v

    def _apply_turning_radius_clamp(
        self, delta: float, v: float
    ) -> tuple[float, float]:
        delta = np.clip(delta, -self._max_delta, self._max_delta)
        return delta, v

    @property
    def brake_triggered(self) -> bool:
        return self._brake_triggered

    @property
    def slowdown_triggered(self) -> bool:
        return self._slowdown_triggered

    @property
    def max_delta(self) -> float:
        return self._max_delta
