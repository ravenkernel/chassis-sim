"""差速底盘：动作 = [v, ω] 的归一化量（[-1,1]^2），物理速度 = 动作 × 各自上限。

与阿克曼、麦轮同一条契约（本仓 A1 裁决）：**原始动作一律是归一化量**，物理量
只在 `command()` 内产生。控制器出口 `to_chassis_action()` 的逆映射
（`a = v/v_max`）因此与本文的正映射逐位互逆 —— 动作空间若暴露物理界
（`[-v_max, v_max]`），控制器与底盘之间的往返会被打上 1/v_max 的偏差。
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .base import Chassis, ChassisCommand


class DiffDriveChassis(Chassis):
    name = "diff_drive"
    action_dim = 2

    def __init__(
        self,
        wheelbase: float = 0.145,
        v_max: float = 1.0,
        omega_max: float = 1.0,
    ) -> None:
        self.wheelbase = float(wheelbase)
        self.v_max = float(v_max)
        self.v_min = -self.v_max
        self.omega_max = float(omega_max)
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
            # 契约键（键名不改）：差速不适用 ⇒ None
            "delta_max": None,
            "curvature_action": None,
            "kappa_max": None,
        }
