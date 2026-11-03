"""麦轮底盘：动作 = [vx, vy, ω] 的归一化量（[-1,1]^3），物理速度 = 动作 × 各自上限。"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .base import Chassis, ChassisCommand


class MecanumChassis(Chassis):
    """全向底盘：保留横向速度，可横移 / 45° 斜行。

    动作语义与阿克曼、差速一致 —— [-1, 1]^3 的归一化量：
    a[0] 映射到 [v_min, v_max]（分段线性：a≥0 乘 v_max、a<0 乘 -v_min，默认对称），
    a[1] 映射到 [-vy_max, vy_max]，a[2] 映射到 [-omega_max, omega_max]。
    物理量由 command() 产出、由 describe() 报告。
    """

    name = "mecanum"
    action_dim = 3

    def __init__(
        self,
        lx: float = 0.19,
        ly: float = 0.10,
        v_max: float = 1.0,
        vy_max: float = 0.5,
        omega_max: float = 1.0,
    ) -> None:
        self.lx = float(lx)
        self.ly = float(ly)
        self.v_max = float(v_max)
        self.v_min = -self.v_max
        self.vy_max = float(vy_max)
        self.omega_max = float(omega_max)
        self._space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def action_space(self) -> spaces.Box:
        return self._space

    def command(self, action: np.ndarray) -> ChassisCommand:
        action = self._clip_action(action)
        a0, a1, a2 = (float(x) for x in action)
        # 纵向映射与阿克曼同款分段线性；默认 v_min = -v_max ⇒ 对称
        v = a0 * (self.v_max if a0 >= 0.0 else -self.v_min)
        return ChassisCommand(v=v, vy=a1 * self.vy_max, omega=a2 * self.omega_max)

    def _limits(self) -> dict:
        return {
            "v_min": self.v_min,
            "v_max": self.v_max,
            "omega_max": self.omega_max,
            "vy_max": self.vy_max,
        }

    def _params(self) -> dict:
        return {
            "lx": self.lx,
            "ly": self.ly,
            "v_max": self.v_max,
            "vy_max": self.vy_max,
            "omega_max": self.omega_max,
            # 契约键（键名不改）：麦轮不适用 ⇒ None
            "wheelbase": None,
            "delta_max": None,
            "curvature_action": None,
            "kappa_max": None,
        }
