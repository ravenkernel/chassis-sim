"""阿克曼底盘：动作 = [转向, 速度]（归一化到 [-1, 1]，与 action_space 同域）。

动作映射的公式（含 curvature_action 开关与 v 的分段线性映射）只写在
`_map_action()` 一处，由 `command()` 与 `step()` 共用；输入归一化到 [-1, 1]。

执行器状态：`step()` 把旧桥「转向滞后一拍」做成显式状态（速度当步生效、ω 用上一帧转角，
末帧转角留给下一步）；`command()` 保持当前帧的纯映射，不读也不写这份状态。
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from .base import Chassis, ChassisCommand, WheelKinematics


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
        track_width: float | None = None,
    ) -> None:
        self.wheelbase = float(wheelbase)
        self.delta_max = float(delta_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.curvature_action = bool(curvature_action)
        self.kappa_max = float(kappa_max)
        # 动作是归一化量（[-1, 1] 与 action_space 逐位同值），
        # 物理范围 [delta_max] / [v_min, v_max] 在下述映射里施加。
        self._space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # 执行器状态：上一帧转向角（旧桥的 _steer 状态），首帧为 0.0。
        # 只有 step() 读写它，command() 不碰。
        self._prev_delta: float = 0.0
        # 轮距：左右**前**轮中心线间距（米）。None ⇒ 取车体宽（0.20），与
        # env.yaml 的 robot.shape.width 和 SimpleBridge 的 WheelOdom 缺省轮距同源。
        self.track_width = float(self.width if track_width is None else track_width)
        if self.track_width <= 0.0:
            raise ValueError(f"track_width 必须是正数，收到 {self.track_width!r}")

    def action_space(self) -> spaces.Box:
        return self._space

    def _map_action(self, action: np.ndarray) -> tuple[float, float]:
        """原始动作 -> (当前转向角 δ, 纵向速度 v)。command() 与 step() 共用这一份映射。"""
        action = self._clip_action(action)

        # ── 动作映射的唯一实现（归一化 → 物理量；command() 与 step() 共用）──
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
        # ── 动作映射结束 ──

        return float(mapped_action[0]), float(mapped_action[1])

    def command(self, action: np.ndarray) -> ChassisCommand:
        """当前动作的稳态命令：ω 用**本帧**转角。纯函数，不改执行器状态。"""
        delta, v = self._map_action(action)
        # 横摆角速度 ω = v/L·tanδ；v == 0 时无转向贡献 ⇒ ω = 0
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

    def front_wheel_angles(self, delta: float) -> tuple[float, float]:
        """由**后轴中心**转向角 δ 解出左右前轮转角 (rad)：返回 (左, 右)。

        阿克曼几何（R = L/tanδ，b = track_width，half = b/2）：

            δ_inner = atan(L / (|R| − half))      δ_outer = atan(L / (|R| + half))

        等价于标准关系 cot δ_outer − cot δ_inner = b/L（符号随 δ）。左转
        （δ > 0）时**左轮在内侧**，故此时 δ_left > δ_right；右转镜像。
        |R| ≤ half 落在阿克曼锥之外（车轮需打超过 90°，物理不可实现）时，
        atan2 给出 ±π/2 附近的连续外推而不是抛 ZeroDivisionError。
        δ == 0（直线）返回 (0, 0)。

        纯只读几何换算：不读也不写执行器状态，**不参与位姿积分**。
        """
        if delta == 0.0:
            return (0.0, 0.0)
        half = 0.5 * self.track_width
        radius = self.wheelbase / math.tan(delta)  # 带符号：左转 R > 0
        inner = math.atan2(self.wheelbase, abs(radius) - half)
        outer = math.atan2(self.wheelbase, abs(radius) + half)
        sign = math.copysign(1.0, delta)
        inner *= sign
        outer *= sign
        return (inner, outer) if delta > 0.0 else (outer, inner)

    def wheel_kinematics(self, cmd: ChassisCommand) -> WheelKinematics:
        """左右前轮转角（只读；不参与位姿积分）。

        δ 由命令反解：tanδ = ω·L / v（v < 0 时 ω 与 δ 反号，比值仍为正 ⇒ 反解
        与前进时逐位同值）。v == 0（此时 ω 恒为 0）或 ω == 0（直行）时转向角取
        0.0 —— 零速的命令里没有转向信息；要看执行器**当前**转角请用
        `steering_angle()`。wheel_left / wheel_right 恒为 None：阿克曼的 ω 由转向
        几何决定，左右轮速只是同一刚体的推论，不是独立指令。
        """
        v = float(cmd.v)
        omega = float(cmd.omega)
        if v == 0.0 or omega == 0.0:
            left = right = 0.0
        else:
            left, right = self.front_wheel_angles(math.atan(omega * self.wheelbase / v))
        return WheelKinematics(
            kind="ackermann", steering_left=left, steering_right=right
        )

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
            "track_width": self.track_width,
        }
