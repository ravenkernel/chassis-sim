"""把"吃物理命令"的模拟器接到归一化 action 的 env 契约上。

env 的分工是"env 下发归一化动作、桥把它变成物理量再推演"：
`envs/chassis_env.py:1294` 只把 `act` 交给 `bridge.step(act)`，映射由桥负责。
本仓自带的 `SimpleBridge` 一步做完；但外部模拟器（ir-sim、ROS gazebo、真车
接口，以及旧桥的 `NumpySimBridge`）只吃物理量。
直接挂上去，归一化动作会被当成物理量吃掉。

`PhysicalCommandBridge` 补上缺的那一段：

    归一化 action --chassis.physical_action()--> 物理向量

三条约定：

* **映射走 `Chassis.physical_action()`**（纯函数，不读也不写执行器状态）。
  `AckermannChassis.step()` 带一拍转向执行器滞后（`_prev_delta`），而**内层
  物理模拟器自己还有一次**；适配器再用 `step()` 会让滞后施加两次，整车响应
  慢一拍。
* **向量顺序由车型决定**（`chassis/base.py` 的 `physical_action`）：
  ackermann `[delta(rad), v(m/s)]`、diff_drive `[v(m/s), omega(rad/s)]`、
  mecanum `[v(m/s), vy(m/s), omega(rad/s)]`。阿克曼的 delta 由底盘从
  `_map_action()` 直接给出，**不**再做 `arctan(omega*L/v)` 反解：反解在
  |v| < 1e-6 时无定义（只能沿用上一拍，成了跨步状态），200 例随机动作里还有
  64 例差 1 ulp。
* **执行器滞后只由内层模拟器施加恰好一次**：适配器不推进底盘状态、不调
  `chassis.step()`、也不调 `chassis.reset_state()`。内层不带滞后时，它与
  `SimpleBridge` 的位姿等价性会从 1.5e-15 崩到 9.76e-01。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chassis_sim.chassis import Chassis

from .base import SensorProvider


class PhysicalCommandBridge(SensorProvider):
    """把"吃物理命令的模拟器"接到归一化 action 的 env 契约上。

    归一化 action --chassis.physical_action()--> 物理向量 --inner.step()--> 物理桥

    物理向量按车型（`Chassis.physical_action()`）：
    ackermann `[delta, v]`、diff_drive `[v, omega]`、mecanum `[v, vy, omega]`。

    除了 `step` / `reset` / `set_dynamic_obstacles` / `get_sensor_reading`，
    其余成员都是**逐字透传**（同一个返回对象，不拷贝、不重建）：适配器只改动作
    通路，不改传感器通路。`get_sensor_reading` 多一层可选能力判定：内层没有
    该方法时返回 `None`（详见该方法），因为"没有传感器"是合法状态，而不是错误。

    Attributes:
        inner: 内层物理桥；每次 `step` 收到一个 float64 ndarray，顺序见上。
        chassis: 归一化动作的映射器。env 注入桥时要求
            `bridge.chassis is env.chassis`（`envs/chassis_env.py:438`），所以
            这里保存调用方给的**同一个实例**，不自己 new 一个。
    """

    def __init__(self, inner: SensorProvider, chassis: Chassis) -> None:
        """绑定内层物理桥与动作映射器。

        Args:
            inner: 吃物理命令的桥（ir-sim / gazebo / NumpySimBridge 之类）。
                只要求它实现 `SensorProvider` 的契约，不要求是本包类型；它
                收到的向量顺序必须与 `chassis.physical_action()` 一致。
            chassis: 与 env 共享的底盘实例，只用来做归一化 -> 物理的映射。
        """
        self.inner = inner
        self.chassis = chassis

    # ── 动作通路：归一化 action -> 物理向量 ──────────────────────────────
    def to_physical(self, action: np.ndarray) -> np.ndarray:
        """归一化动作 -> 内层桥要的物理向量（float64，长度 = action_dim）。

        顺序由车型决定：ackermann `[delta, v]`、diff_drive `[v, omega]`、
        mecanum `[v, vy, omega]`。**纯函数**：同一 action 永远得到逐位相同的
        向量，与底盘执行器状态（阿克曼的 `_prev_delta`）无关。单独暴露是为了
        让使用方与测试能直接看一条动作对应的物理命令，而不必先推进仿真。
        """
        return np.asarray(self.chassis.physical_action(action), dtype=np.float64)

    def step(self, action: np.ndarray) -> None:
        """把归一化动作映射成物理向量，转交内层桥（积分在内层发生）。"""
        self.inner.step(self.to_physical(action))

    def reset(
        self,
        obstacles: list | None = None,
        path_waypoints: np.ndarray | None = None,
        initial_heading: float | None = None,
    ) -> None:
        """复位内层桥（适配器自身无状态，只转发）。

        刻意**不**调用 `chassis.reset_state()`：适配器从不调用 `chassis.step()`，
        底盘的执行器状态（`_prev_delta`）对纯函数映射没有影响，而不存在的那次
        状态推进也就不需要复位。位姿与执行器状态由 env / 内层桥各自负责。

        Args:
            obstacles: 原样转发给内层桥。
            path_waypoints: 原样转发给内层桥。
            initial_heading: 原样转发给内层桥。
        """
        self.inner.reset(
            obstacles=obstacles,
            path_waypoints=path_waypoints,
            initial_heading=initial_heading,
        )

    # ── 传感器与判定：逐字透传 ──────────────────────────────────────────
    def get_odometry(self) -> np.ndarray:
        """透传内层桥的里程计 `[x, y, theta, v, omega]`。"""
        return self.inner.get_odometry()

    def get_path(self) -> np.ndarray:
        """透传内层桥的车体系路径 `(N, 3)`。"""
        return self.inner.get_path()

    def get_path_global(self) -> np.ndarray:
        """透传内层桥的世界系路径 `(N, 3)`。"""
        return self.inner.get_path_global()

    def check_collision(self) -> bool:
        """透传内层桥的碰撞判定。"""
        return self.inner.check_collision()

    def check_goal_reached(self, threshold: float = 0.3) -> bool:
        """透传内层桥的到点判定。

        Args:
            threshold: 到点距离阈值（米），原样转发。
        """
        return self.inner.check_goal_reached(threshold)

    def close(self) -> None:
        """透传内层桥的资源释放。"""
        self.inner.close()

    @property
    def goal_heading(self) -> float:
        """透传内层桥的目标期望朝向。"""
        return self.inner.goal_heading

    def get_sensor_reading(self) -> dict[str, Any] | None:
        """转发内层桥的可选传感器读数；内层不支持该能力时返回 `None`。

        传感器是**可选能力**（与 `set_dynamic_obstacles` 的"必须响亮报错"
        正相反）：内层桥压根没有这个方法（NumpySimBridge / ir-sim
        SimBridge 就是这样），或只继承了 `SensorProvider` 的非抽象默认实现，
        都视为"未启用可选传感器"，返回 `None` 而不是抛 `AttributeError` ——
        否则 env 侧 `getattr` 兜底（`envs/chassis_env.py:1735`）会被本类的
        具体方法架空，第一步就崩。

        Returns:
            内层读数（`None` 也照原样返回），或内层不支持时的 `None`。
        """
        impl = getattr(self.inner, "get_sensor_reading", None)
        if impl is None or (
            getattr(impl, "__func__", None) is SensorProvider.get_sensor_reading
        ):
            return None
        return impl()

    def set_dynamic_obstacles(self, specs: list | None = None) -> None:
        """转发给内层桥；内层不支持时**响亮地**报错，不静默跳过。

        Args:
            specs: 动态障碍规格列表（`None` / `[]` = 清空），原样转发。

        Raises:
            NotImplementedError: 内层桥只实现了静态场景（没有覆盖
                `SensorProvider.set_dynamic_obstacles`）。
        """
        impl = getattr(self.inner, "set_dynamic_obstacles", None)
        if impl is None or (
            getattr(impl, "__func__", None) is SensorProvider.set_dynamic_obstacles
        ):
            raise NotImplementedError(
                f"内层桥 {type(self.inner).__name__} 不支持动态障碍"
                "（dynamic_obstacles）：适配器只做转发，不自己维护障碍物；"
                "请改用支持动态障碍的内层桥"
            )
        impl(specs)
