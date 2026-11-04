"""麦轮底盘：动作 = [vx, vy, ω] 的归一化量（[-1,1]^3），物理速度 = 动作 × 各自上限。"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .base import Chassis, ChassisCommand, WheelKinematics


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
        wheel_radius: float = 0.05,
    ) -> None:
        self.lx = float(lx)
        self.ly = float(ly)
        self.v_max = float(v_max)
        self.v_min = -self.v_max
        self.vy_max = float(vy_max)
        self.omega_max = float(omega_max)
        # 轮半径（米）：只在轮级运动学里把轮缘线速度换成角速度，不参与位姿积分。
        # 0.05 m 是 100 mm 麦轮（本仓 0.38×0.20 m 车体的常见配套），也是
        # 既有底盘参数表里唯一写明轮径的量级。
        self.wheel_radius = float(wheel_radius)
        if self.wheel_radius <= 0.0:
            raise ValueError(f"wheel_radius 必须是正数，收到 {self.wheel_radius!r}")
        self._space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def action_space(self) -> spaces.Box:
        return self._space

    def command(self, action: np.ndarray) -> ChassisCommand:
        action = self._clip_action(action)
        a0, a1, a2 = (float(x) for x in action)
        # 纵向映射与阿克曼同款分段线性；默认 v_min = -v_max ⇒ 对称
        v = a0 * (self.v_max if a0 >= 0.0 else -self.v_min)
        return ChassisCommand(v=v, vy=a1 * self.vy_max, omega=a2 * self.omega_max)

    def wheel_kinematics(self, cmd: ChassisCommand) -> WheelKinematics:
        """四轮角速度 (rad/s)（只读；不参与位姿积分）。

        逆 Jacobian（车体系：x 前、y 左、ω 逆时针；轮心 FL=(+lx,+ly)、
        FR=(+lx,−ly)、RL=(−lx,+ly)、RR=(−lx,−ly)；辊子按 X 型 45° 布置）：

            ω_FL = (v − vy − (lx+ly)·ω) / r
            ω_FR = (v + vy + (lx+ly)·ω) / r
            ω_RL = (v + vy − (lx+ly)·ω) / r
            ω_RR = (v − vy + (lx+ly)·ω) / r

        正解（测试里逐条对拍，见 tests/test_chassis_kinematics.py）：

            v  = r·( ω_FL + ω_FR + ω_RL + ω_RR) / 4
            vy = r·(−ω_FL + ω_FR + ω_RL − ω_RR) / 4
            ω  = r·(−ω_FL + ω_FR − ω_RL + ω_RR) / (4·(lx+ly))

        ⇒ 纯 v / 纯 vy / 纯 ω 三种输入都能原样解回来（自洽性），且 vy 与 ω 都
        进同一个轮（耦合项 ±(lx+ly)·ω 与 ±vy 同号/异号按轮位不同）。
        steering_* 与 wheel_left/right 恒为 None：麦轮没有转向轮，也不存在
        "左右两轮"这种一维差速抽象。
        """
        v = float(cmd.v)
        vy = float(cmd.vy)
        omega = float(cmd.omega)
        radius = self.wheel_radius
        coupling = (self.lx + self.ly) * omega
        return WheelKinematics(
            kind="mecanum",
            wheel_fl=(v - vy - coupling) / radius,
            wheel_fr=(v + vy + coupling) / radius,
            wheel_rl=(v + vy - coupling) / radius,
            wheel_rr=(v - vy + coupling) / radius,
        )

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
            "wheel_radius": self.wheel_radius,
            "v_max": self.v_max,
            "vy_max": self.vy_max,
            "omega_max": self.omega_max,
            # 契约键（键名不改）：麦轮不适用 ⇒ None
            "wheelbase": None,
            "delta_max": None,
            "curvature_action": None,
            "kappa_max": None,
        }
