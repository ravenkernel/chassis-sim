"""底盘抽象基类：命令载体、统一位姿积分核、几何与激光雷达安装描述。

设计约束（本仓 A1 验收点）：
1. 位姿积分只有**一份**实现 —— `Chassis.integrate`（闭式），子类不得复制；
2. 每个车型自己暴露 `action_space`（阿克曼/差速 2 维、麦轮 3 维）；
3. 动作入口统一在 `_clip_action` 校验形状并 clip 到 action_space 边界；
4. 轮级运动学（`wheel_kinematics`）是**只读观察层**，绝不参与位姿积分。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

# ── 闭式积分的退化判据 ────────────────────────────────────────────────
# 闭式解把 ΔX 写成 A = [sin(θ₀+φ) − sinθ₀]/ω（φ = ω·dt）。分子是两个几乎相等的
# 正弦之差 ⇒ 减法抵消，A 的绝对误差 ~ eps/|ω|，位移误差 ~ |v|·dt·eps/|φ|；
# 而 Euler 的截断误差 ~ |v|·dt·|φ|/2。两者在 |φ| ≈ sqrt(2·eps) ≈ 2.1e-8 处相等，
# 取 sqrt(eps) ≈ 1.49e-8 作阈值：比交叉点略早退化（偏保守），保证 |φ| ≤ 1.49e-8
# 一律走与旧实现**逐位相同**的 Euler 分支，而不是拿一个抵消到只剩噪声的商去除。
# 用 sqrt(eps) 而不是 eps 的理由：eps 阈值太小 —— ω=1e-12、dt=0.1（|φ|=1e-13）时
# 闭式的每步位移误差约 |v|·dt·eps/|φ| ≈ 2.2e-4 m，比 Euler 的 ~1e-14 m 大十个数量级。
_SMALL_TURN: float = math.sqrt(float(np.finfo(np.float64).eps))


@dataclass(frozen=True)
class WheelKinematics:
    """轮级运动学快照（纯只读；**不参与位姿积分**，也不回写底盘状态）。

    一个车型只填自己那一族字段，其余字段**显式**为 None（不是"忘了填"）：

    kind="ackermann"  steering_left / steering_right = 左右前轮转角 (rad)；
                      wheel_* 为 None —— 阿克曼的 ω 由转向几何决定
                      （ω = v·tanδ/L），左右轮是同一根轴上的刚体，
                      "左右轮线速度"不是独立可控量，报出来会被误当成差速指令。
    kind="diff_drive" wheel_left / wheel_right = 左右驱动轮线速度 (m/s)；
                      steering_* 为 None —— 差速底盘没有转向轮。
    kind="mecanum"    wheel_fl / wheel_fr / wheel_rl / wheel_rr = 四轮角速度 (rad/s)；
                      steering_* 与 wheel_left/right 为 None —— 麦轮没有转向轮，
                      也不存在"左右"两轮这种一维差速抽象。
    """

    kind: str
    steering_left: float | None = None
    steering_right: float | None = None
    wheel_left: float | None = None
    wheel_right: float | None = None
    wheel_fl: float | None = None
    wheel_fr: float | None = None
    wheel_rl: float | None = None
    wheel_rr: float | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        """全部字段的字典快照（含 None；键序 = 字段声明序）。"""
        return {
            "kind": self.kind,
            "steering_left": self.steering_left,
            "steering_right": self.steering_right,
            "wheel_left": self.wheel_left,
            "wheel_right": self.wheel_right,
            "wheel_fl": self.wheel_fl,
            "wheel_fr": self.wheel_fr,
            "wheel_rl": self.wheel_rl,
            "wheel_rr": self.wheel_rr,
        }


@dataclass(frozen=True)
class ChassisCommand:
    """一帧底盘命令，语义固定在**车体系**：

    v     纵向线速度 (m/s)
    omega 横摆角速度 (rad/s)
    vy    横向线速度 (m/s)，仅麦轮非零
    """

    v: float
    omega: float
    vy: float = 0.0


class Chassis(ABC):
    """三种底盘共用的抽象。"""

    name: str = "chassis"
    action_dim: int = 0

    # 几何默认值 = 包内 config/env.yaml 的 robot 块
    # （shape 0.38×0.2、前 lidar offset [0.145,0,0]、后 lidar offset [-0.155,0,π]）。
    length: float = 0.38
    width: float = 0.20
    FRONT_LIDAR: tuple[float, float, float] = (0.145, 0.0, 0.0)
    REAR_LIDAR: tuple[float, float, float] = (-0.155, 0.0, np.pi)

    @abstractmethod
    def action_space(self) -> spaces.Box:
        """本车型的原始动作空间（command() 的输入域）。"""

    @abstractmethod
    def command(self, action: np.ndarray) -> ChassisCommand:
        """原始动作 -> ChassisCommand。入参越界先 clip，形状不符抛 ValueError。"""

    def physical_action(self, action: np.ndarray) -> np.ndarray:
        """归一化 action -> 交给**外部物理模拟器**的物理向量（本车型自己的顺序）。

        ackermann  : [delta(rad), v(m/s)]
        diff_drive : [v(m/s), omega(rad/s)]
        mecanum    : [v(m/s), vy(m/s), omega(rad/s)]

        具体方法（不是 abstractmethod）：第三方子类不必实现它。默认实现基于
        `command()`（纯映射，不读也不写执行器状态）：v 恒在最前；麦轮（action_dim
        >= 3）的顺序是 v, vy, omega（不是 ChassisCommand 的字段声明顺序）。
        阿克曼覆盖它 —— 它的 delta 不在 ChassisCommand 里，只能由车型自己的
        `_map_action()` 直接给出（反解只对 |v| >= V_EPS 有定义）。
        """
        cmd = self.command(action)
        if self.action_dim >= 3:
            return np.array(
                [float(cmd.v), float(cmd.vy), float(cmd.omega)], dtype=np.float64
            )
        return np.array([float(cmd.v), float(cmd.omega)], dtype=np.float64)

    def step(self, action: np.ndarray) -> ChassisCommand:
        """推进执行器状态一步，返回**本积分步生效**的命令。默认无滞后。"""
        return self.command(action)

    def reset_state(self) -> None:
        """清零执行器内部状态（回合/场景重置时调用）。默认无状态。"""
        return

    def integrate(
        self,
        x: float,
        y: float,
        theta: float,
        cmd: ChassisCommand,
        dt: float,
    ) -> tuple[float, float, float]:
        """统一位姿积分核（**闭式**，车体系速度旋到世界系）。

        恒定 (v, vy, ω) 下的解析解（φ = ω·dt）：

            A = [sin(θ₀+φ) − sinθ₀] / ω      B = [cosθ₀ − cos(θ₀+φ)] / ω
            ΔX = v·A − vy·B                  ΔY = v·B + vy·A
            θ  = θ₀ + φ

        这正好是 ∫₀^dt [v·cos(θ₀+ωt) − vy·sin(θ₀+ωt)] dt 与对应 sin 项的积分，
        因此对任意 dt 精确到浮点噪声（旧 Euler 的整圈径向误差是 O(v·dt)：实测
        dt=0.1 时 5.04e-2 m，改闭式后降到 1e-16 量级，见
        tests/test_chassis_integration_closed_form.py）。

        **退化判据（|ω·dt| ≤ sqrt(eps) ≈ 1.49e-8）**：此时 A、B 的分子是两个几乎
        相等的三角函数之差，减法抵消把误差放大成 ~eps/|ω|（见模块级 _SMALL_TURN
        的推导：与 Euler 截断误差的交点在 |φ| ≈ 2.1e-8）。落在阈值内一律走 Euler
        分支 —— 该分支的算数与加乘顺序与改造前**逐字一致**，因此 ω=0 的轨迹
        （以及 |ω·dt| ≤ 1.49e-8 的一切轨迹）与旧实现逐位相同（`tobytes` 相等
        由 tests/test_chassis_integration_closed_form.py::test_omega_zero_bitwise
        守着）。
        """
        v = float(cmd.v)
        vy = float(cmd.vy)
        omega = float(cmd.omega)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        phi = omega * dt
        if abs(phi) <= _SMALL_TURN:
            # 逐位回归分支：表达式、乘加顺序与 v1.0 的 Euler 实现完全相同，
            # 不许改写成 v*dt 之类的"等价形式"（θ≠0 时差 1 ulp）。
            return (
                x + (v * cos_t - vy * sin_t) * dt,
                y + (v * sin_t + vy * cos_t) * dt,
                theta + omega * dt,
            )

        sin_t1 = np.sin(theta + phi)
        cos_t1 = np.cos(theta + phi)
        a = (sin_t1 - sin_t) / omega  # ΔX 的 cos 积分项
        b = (cos_t - cos_t1) / omega  # ΔX 的 sin 积分项
        return (
            x + v * a - vy * b,
            y + v * b + vy * a,
            theta + phi,
        )

    def footprint(self) -> tuple[float, float]:
        """车体矩形外廓 (长, 宽)，单位米。"""
        return (float(self.length), float(self.width))

    def lidar_mounts(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(前, 后) 激光雷达安装位姿 (x, y, 朝向偏移)。"""
        return (self.FRONT_LIDAR, self.REAR_LIDAR)

    def steering_angle(self) -> float:
        """当前前轮转角（弧度）；转向轮恒与车身对齐的底盘返回 0.0。"""
        return 0.0

    def wheel_kinematics(self, cmd: ChassisCommand) -> WheelKinematics | None:
        """轮级运动学快照（**只读**：不改执行器状态、不改位姿、不参与 integrate）。

        这是个纯函数：同一个 cmd 永远得到逐位相同的返回值。它只是"把一条车体系
        命令翻译成轮子上的量"，位姿推进仍然只走 `integrate()`。

        三种内置底盘各自覆盖本方法并返回 kind="ackermann" / "diff_drive" /
        "mecanum" 的快照（字段语义见 `WheelKinematics`）。**基类返回 None**：
        第三方底盘没有轮级几何，这里不做"猜一个空 dict"的静默降级 —— 调用方用
        `is None` 就能区分"没有这一层"，而空 dict/全 None 的 dataclass 会让人误以为
        量算出来了。需要轮级量时请实现本方法（或改用内置三车型）。
        """
        return None

    def describe(self) -> dict:
        """车型描述（键名是与控制器层的契约，**不许改名**）。"""
        space = self.action_space()
        return {
            "name": self.name,
            "action_dim": int(self.action_dim),
            "action_space": {
                "low": [float(v) for v in space.low],
                "high": [float(v) for v in space.high],
            },
            "footprint": list(self.footprint()),
            "lidar_mounts": [list(m) for m in self.lidar_mounts()],
            "limits": self._limits(),
            "params": self._params(),
        }

    def _limits(self) -> dict:
        """物理量上限快照（子类覆盖）：v_min / v_max / omega_max / vy_max，不适用者给数值 0.0。"""
        raise NotImplementedError

    def _params(self) -> dict:
        """构造参数快照（子类覆盖）；契约键不适用时给 None。"""
        return {}

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """形状校验 + clip 到 action_space 边界。"""
        space = self.action_space()
        arr = np.asarray(action, dtype=np.float64)
        if arr.shape != space.shape:
            raise ValueError(
                f"{self.name}.command(): action shape {arr.shape} 与期望 shape "
                f"{space.shape}（action_dim={self.action_dim}）不一致"
            )
        return np.clip(arr, space.low, space.high)
