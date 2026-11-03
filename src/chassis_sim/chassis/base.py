"""底盘抽象基类：命令载体、统一位姿积分核、几何与激光雷达安装描述。

设计约束（本仓 A1 验收点）：
1. 位姿积分只有**一份**实现 —— `Chassis.integrate`，子类不得复制；
2. 每个车型自己暴露 `action_space`（阿克曼/差速 2 维、麦轮 3 维）；
3. 动作入口统一在 `_clip_action` 校验形状并 clip 到 action_space 边界。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


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

    # 几何默认值 = 主仓 config/env.yaml 的 robot 块
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
        """统一位姿积分核（Euler，车体系速度旋到世界系）。

        x += (v·cosθ − vy·sinθ)·dt
        y += (v·sinθ + vy·cosθ)·dt
        θ += ω·dt
        """
        v = float(cmd.v)
        vy = float(cmd.vy)
        omega = float(cmd.omega)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return (
            x + (v * cos_t - vy * sin_t) * dt,
            y + (v * sin_t + vy * cos_t) * dt,
            theta + omega * dt,
        )

    def footprint(self) -> tuple[float, float]:
        """车体矩形外廓 (长, 宽)，单位米。"""
        return (float(self.length), float(self.width))

    def lidar_mounts(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(前, 后) 激光雷达安装位姿 (x, y, 朝向偏移)。"""
        return (self.FRONT_LIDAR, self.REAR_LIDAR)

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
