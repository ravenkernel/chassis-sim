"""差速底盘：动作 = [v, ω] 的归一化量（[-1,1]^2），物理速度 = 动作 × 各自上限。

与阿克曼、麦轮同一条契约（本仓 A1 裁决）：**原始动作一律是归一化量**，物理量
只在 `command()` 内产生。控制器出口 `to_chassis_action()` 的逆映射
（`a = v/v_max`）因此与本文的正映射逐位互逆 —— 动作空间若暴露物理界
（`[-v_max, v_max]`），控制器与底盘之间的往返会被打上 1/v_max 的偏差。
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .base import Chassis, ChassisCommand, WheelKinematics


class DiffDriveChassis(Chassis):
    name = "diff_drive"
    action_dim = 2

    def __init__(
        self,
        wheelbase: float = 0.145,
        v_max: float = 1.0,
        omega_max: float = 1.0,
        track_width: float | None = None,
    ) -> None:
        self.wheelbase = float(wheelbase)
        self.v_max = float(v_max)
        self.v_min = -self.v_max
        self.omega_max = float(omega_max)
        # 轮距：左右驱动轮中心线间距（米）。None ⇒ 取车体宽（0.20），与
        # env.yaml 的 robot.shape.width 和 SimpleBridge 的 WheelOdom 缺省轮距同源。
        self.track_width = float(self.width if track_width is None else track_width)
        if self.track_width <= 0.0:
            raise ValueError(f"track_width 必须是正数，收到 {self.track_width!r}")
        # 动作是归一化量（与阿克曼/麦轮逐位同界），物理范围 [v_min, v_max] /
        # ±omega_max 只在 command() 的映射里施加。
        self._space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action_space(self) -> spaces.Box:
        return self._space

    def command(self, action: np.ndarray) -> ChassisCommand:
        action = self._clip_action(action)
        a0, a1 = (float(x) for x in action)
        # 纵向分段线性（与阿克曼/麦轮同款）：a0 >= 0 → *v_max，a0 < 0 → *(-v_min)
        v = a0 * (self.v_max if a0 >= 0.0 else -self.v_min)
        return ChassisCommand(v=v, omega=a1 * self.omega_max, vy=0.0)

    def wheel_kinematics(self, cmd: ChassisCommand) -> WheelKinematics:
        """左右驱动轮线速度 (m/s)（只读；不参与位姿积分）。

        差速运动学的逆解，b = track_width：

            v_left  = v − ω·b/2
            v_right = v + ω·b/2

        （ω > 0 = 逆时针 = 左转 ⇒ 右轮更快，与 integrate() 的 θ += ω·dt 同号。）
        steering_left / steering_right 恒为 None：差速底盘没有转向轮，转向完全由
        左右轮速差产生。注意这里是**轮缘线速度**；要轮角速度请除以轮半径。
        """
        half = 0.5 * self.track_width
        v = float(cmd.v)
        omega = float(cmd.omega)
        return WheelKinematics(
            kind="diff_drive",
            wheel_left=v - omega * half,
            wheel_right=v + omega * half,
        )

    def _limits(self) -> dict:
        return {
            "v_min": self.v_min,
            "v_max": self.v_max,
            "omega_max": self.omega_max,
            "vy_max": 0.0,
        }

    def _params(self) -> dict:
        return {
            "wheelbase": self.wheelbase,
            "v_max": self.v_max,
            "omega_max": self.omega_max,
            "track_width": self.track_width,
            # 契约键（键名不改）：差速不适用 ⇒ None
            "delta_max": None,
            "curvature_action": None,
            "kappa_max": None,
        }
