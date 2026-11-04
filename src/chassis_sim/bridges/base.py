"""SensorProvider 抽象接口 —— 旧桥传感器接口的同构契约。

签名（8 个抽象方法 + 1 个抽象属性）与旧桥逐一对应，只把英文 docstring 译成中文：
消费方（envs / 控制器 / 回放脚本）按位置或关键字调用这些方法，任何签名漂移都会让
旧桥与新桥不可互换。

实现者：
    SimpleBridge —— 本包纯 numpy 的射线投射实现（bridges/simple_bridge.py）
    旧仓 SimBridge / ROS2Bridge / MockBridge —— 同签名，可直接替换

1.0.0 起雷达**只有**一个入口（旧的成对前/后向激光方法与 ROS 式扫描字典都已删）：
:meth:`SensorProvider.get_sensor_reading`（dict[传感器名, 读数]），消费方按名字取值，
不再有"前/后固定两键 + 各 180 线"的隐式契约，也没有第二套（ROS 式）扫描格式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SensorProvider(ABC):
    """传感器数据与动作下发的抽象接口。"""

    @abstractmethod
    def reset(
        self,
        obstacles: list | None = None,
        path_waypoints: np.ndarray | None = None,
        initial_heading: float | None = None,
    ) -> None:
        """复位仿真状态。

        Args:
            obstacles: 可选的障碍物 kwargs 列表（动态创建用）。
            path_waypoints: 可选的 (N, 2) 全局路径航点。
            initial_heading: 可选的起始车头朝向（弧度）。
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """用给定动作推进一个时间步。

        Args:
            action: [delta, v] 转向角与速度
        """
        ...

    @abstractmethod
    def get_odometry(self) -> np.ndarray:
        """返回里程计：[x, y, theta, v, omega]。"""
        ...

    @abstractmethod
    def get_path(self) -> np.ndarray:
        """返回车体系下的全局路径航点：[N, 3] (dx, dy, dtheta)。"""
        ...

    @abstractmethod
    def get_path_global(self) -> np.ndarray:
        """返回世界系下**全部**路径航点，形状 (N, 3) [x, y, θ]。"""
        ...

    @abstractmethod
    def check_collision(self) -> bool:
        """机器人与任何障碍物重叠时返回 True。"""
        ...

    @abstractmethod
    def check_goal_reached(self, threshold: float = 0.3) -> bool:
        """机器人进入末航点 threshold 以内且朝向匹配时返回 True。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""
        ...

    @property
    @abstractmethod
    def goal_heading(self) -> float:
        """目标点的期望朝向（用于终端奖励中的朝向对齐计算）。"""
        ...

    # ── 附加（非抽象）API：动态障碍 ────────────────────────────────────
    def set_dynamic_obstacles(self, specs: list | None = None) -> None:
        """换一组动态障碍（`None` / `[]` = 清空）。**非抽象**：不实现即视为不支持。

        Raises:
            NotImplementedError: 该桥不支持动态障碍（构造它的模拟器只实现了静态
                场景）。这里**故意响亮地报错**而不是静默跳过 —— 一个被静默忽略的
                动态障碍会让"碰撞没发生"看起来像策略变好了。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不支持动态障碍（dynamic_obstacles）："
            "该桥只实现了静态场景；请改用支持动态障碍的桥（如 SimpleBridge）"
        )

    # ── 附加（非抽象）API：可选传感器读数（雷达的唯一入口）──────────────
    def get_sensor_reading(self) -> dict[str, Any] | None:
        """返回当前步的可选传感器读数（**非抽象**：不实现即视为不启用）。

        读数**默认只进 info**：obs 的维度由 ObsSpec 决定（1.0 起默认是
        odom+path+goal+act 的 128 维），与是否启用传感器无关；只有 obs_spec 里
        用 lidar 段显式引用了某个传感器名，它的 scan 才会进观测。

        Returns: dict[传感器名, 读数]，每个读数本身是 dict，实现方只放已启用的项：

        * Lidar:     {"scan": (n_rays,) float32 或 None（延迟预热期 / 丢帧）,
                      "scan_meta": {name, mount, n_rays, fov, angle_min, ...}}
        * Imu:       {ax, ay, yaw_rate, yaw, t}
        * WheelOdom: {v_left, v_right, v, t}

        Returns:
            dict[str, Any]，或 None（未启用任何可选传感器；此时 env 的 info 不新增
            任何键，信息路径与加入该接口前逐位相同）。
        """
        return None
