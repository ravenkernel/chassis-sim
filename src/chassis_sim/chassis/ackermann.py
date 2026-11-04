"""阿克曼底盘：动作 = [转向, 速度]（归一化到 [-1, 1]，与主仓 env.action_space 一致）。

动作映射**逐字照抄**主仓 /home/zhao/drl-sim/envs/ackermann_env.py 的
`AckermannEnv._step_impl` 第 820~833 行（含 curvature_action 开关与 v 的分段线性映射），
该映射只写在 `_map_action()` 一处，由 `command()` 与 `step()` 共用。

执行器状态：`step()` 把旧桥「转向滞后一拍」做成显式状态（速度当步生效、ω 用上一帧转角，
末帧转角留给下一步）；`command()` 保持当前帧的纯映射，不读也不写这份状态。
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .base import Chassis, ChassisCommand


class AckermannChassis(Chassis):
    """曲率/线性两种转向参数化 + 分段线性速度映射。"""

    name = "ackermann"
    action_dim = 2

    def __init__(
        self,
        wheelbase: float = 0.145,
        delta_max: float = 0.526,
        v_min: float = -1.0,
        v_max: float = 1.0,
        curvature_action: bool = True,
        kappa_max: float = 4.0,
    ) -> None:
        self.wheelbase = float(wheelbase)
        self.delta_max = float(delta_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.curvature_action = bool(curvature_action)
        self.kappa_max = float(kappa_max)
        # 动作是归一化量（主仓 ackermann_env.py:269-273 的 env.action_space 逐位同值），
        # 物理范围 [delta_max] / [v_min, v_max] 在下述映射里施加。
        self._space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # 执行器状态：上一帧转向角（旧桥 numpy_bridge.py:379 的 self._steer），
        # 首帧为 0.0（numpy_bridge.py:228）。只有 step() 读写它，command() 不碰。
        self._prev_delta: float = 0.0

    def action_space(self) -> spaces.Box:
        return self._space

    def _map_action(self, action: np.ndarray) -> tuple[float, float]:
        """原始动作 -> (当前转向角 δ, 纵向速度 v)。command() 与 step() 共用这一份映射。"""
        action = self._clip_action(action)

        # ── 以下照抄 drl-sim/envs/ackermann_env.py:822-833（逐字，字节一致）──
        # fmt: off
        mapped_action = action.copy()
        # 转向映射：曲率模式 (R ∝ 1/|a|) vs 线性模式 (δ ∝ a)
        if self.curvature_action:
            kappa = action[0] * self.kappa_max
            mapped_action[0] = np.arctan(np.clip(kappa, -self.kappa_max, self.kappa_max) * self.wheelbase)
        else:
            mapped_action[0] = action[0] * self.delta_max

        if action[1] >= 0:
            mapped_action[1] = action[1] * self.v_max        # [0, +1] → [0, v_max]
        else:
            mapped_action[1] = action[1] * (-self.v_min)     # [-1, 0] → [v_min, 0]
        # fmt: on
        # ── 照抄结束 ──

        return float(mapped_action[0]), float(mapped_action[1])

    def command(self, action: np.ndarray) -> ChassisCommand:
        """当前动作的稳态命令：ω 用**本帧**转角。纯函数，不改执行器状态。"""
        delta, v = self._map_action(action)
        # 主仓 numpy_bridge.py:383 的横摆角速度；v == 0 时无转向贡献 ⇒ ω = 0
        omega = 0.0 if v == 0.0 else v / self.wheelbase * np.tan(delta)
        return ChassisCommand(v=v, omega=omega, vy=0.0)

    def physical_action(self, action: np.ndarray) -> np.ndarray:
        """归一化动作 -> 外部物理模拟器的 [delta, v]（float64）。

        delta 直接取 `_map_action()` 的**本帧指令值**，不走 `arctan(omega*L/v)`
        反解：反解只在 |v| >= V_EPS 时有定义（零速要沿用上一拍，就成了跨步状态），
        且 200 例随机动作里 64 例差 1 ulp。`_map_action()` 在零速下照给指令 δ，
        因此这里是纯函数：同一 action 永远得到逐位相同的向量。
        """
        delta, v = self._map_action(action)
        return np.array([delta, v], dtype=np.float64)

    def step(self, action: np.ndarray) -> ChassisCommand:
        """推进一步执行器状态：速度立即生效，转向滞后一拍（旧桥 :379-390 的语义）。

        本步 ω 用**上一帧**转角 self._prev_delta，本帧转角只留给下一步；
        返回后 self._prev_delta 才更新为本帧值，故返回值与状态更新次序不可交换。
        """
        delta, v = self._map_action(action)
        omega = 0.0 if v == 0.0 else v / self.wheelbase * np.tan(self._prev_delta)
        self._prev_delta = delta
        return ChassisCommand(v=v, omega=omega, vy=0.0)

    def reset_state(self) -> None:
        """清零转向滞后状态（首帧 prev=0，与 numpy_bridge.py:228 一致）。"""
        self._prev_delta = 0.0

    def steering_angle(self) -> float:
        """当前前轮转角（弧度）：step() 更新后的 δ，即轮子此刻的真实朝向。"""
        return float(self._prev_delta)

    def _limits(self) -> dict:
        # |ω| ≤ |v|·tan|δ| / L，在 v_max 与 delta_max 处取到上界（曲率模式下另有 κ_max 上限）
        return {
            "v_min": self.v_min,
            "v_max": self.v_max,
            "omega_max": self.v_max / self.wheelbase * float(np.tan(self.delta_max)),
            "vy_max": 0.0,
        }

    def _params(self) -> dict:
        return {
            "wheelbase": self.wheelbase,
            "delta_max": self.delta_max,
            "curvature_action": self.curvature_action,
            "kappa_max": self.kappa_max,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }
