"""渲染 / 评估层真正依赖的结构类型（`typing.Protocol`，全部 `@runtime_checkable`）。

1.0.0 起 render 层只读桥的**公开**成员（`render/scene.py` 模块 docstring：不再读
`_x` / `_circle_centers` 这类私有字段），本模块就是那层契约的可执行表述：
不要求继承，`isinstance(obj, Bridge)` 直接判定，第三方桥不必 import 本包基类。

三个协议的分工
--------------
* `Bridge`：渲染 / 录制的输入 —— 推进、位姿、障碍、量程、仿真时间。
* `SensorBridge(Bridge)`：`Bridge` + **可选能力** `get_sensor_reading()`。
* `SceneSource`：只够画一张场景快照的最小面（位姿 + 障碍 + 世界边界），给
  手上只有场景、不推进仿真的调用方用（它比 `Bridge` 严格：多要 `world_bounds`）。

为什么 `get_sensor_reading()` 另立 `SensorBridge` 而不塞进 `Bridge`
---------------------------------------------------------------------
1.0.0 的 `Bridge` 是「没有传感器也能画」的下限：`render.figure._lidar_scans()`
用 `getattr(bridge, "get_sensor_reading", None)` 探测这条通路，探不到就零光束
（HUD 的 d_min 显示 `—`）。把它写进 `Bridge`，所有只画几何的桥都会被
`isinstance(..., Bridge)` 判成不合格 —— 可选能力不该混进最低契约。因此
渲染层的签名收 `Bridge`（运行时用 `getattr` 判定能力），需要强制这条通路的
调用方（自建传感器管线的替身、按能力分流的 isinstance 过滤）改断 `SensorBridge`。

只依赖 numpy 与 typing：`import chassis_sim.protocols` 不拉 matplotlib /
imageio / numba 等可选重依赖（`tests/test_protocols.py` 在干净子进程里守着）。

与实现对齐的三条口径（照着当前桥写的，不是猜测）
----------------------------------------------
* `sim_time` / `max_range` 是**属性**：`SimpleBridge.sim_time` 是 property，
  `SimpleBridge.max_range` 是实例属性。`figure._read_sim_time()` 另外容忍可调用
  的 `sim_time()`，那是宽容读取，不改变契约。
* `reset(**kwargs)`：不同桥的具名参数不同（`SimpleBridge` 收 obstacles= /
  path_waypoints=，适配器收 mapping），协议只固定「能被关键字调用」。
* `step()` 的返回值不参与渲染（`SimpleBridge.step()` 返回 None），协议给 `Any`，
  避免把某个桥的返回值变成公共契约。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["Bridge", "SceneSource", "SensorBridge"]


@runtime_checkable
class SceneSource(Protocol):
    """画一张场景快照所需的最小面：当前位姿 + 障碍几何 + 世界边界。"""

    def pose(self) -> tuple[float, float, float]:
        """世界系位姿 (x, y, theta)，单位米 / 弧度。"""
        ...

    def obstacle_shapes(self) -> list[dict[str, Any]]:
        """障碍物列表，格式与 `SimpleBridge.reset(obstacles=...)` 的输入一致。"""
        ...

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        """世界边界 (xmin, xmax, ymin, ymax)，单位米。"""
        ...


@runtime_checkable
class Bridge(Protocol):
    """渲染 / 录制层对仿真桥的全部要求（见模块 docstring 的分工）。"""

    def reset(self, **kwargs: Any) -> Any:
        """按关键字复位（具体键由各桥自己定）。"""
        ...

    def step(self, action: Any) -> Any:
        """推进一步；返回值不进渲染层。"""
        ...

    def get_odometry(self) -> np.ndarray:
        """里程计读数，前 3 个分量是世界系 (x, y, theta)。"""
        ...

    def pose(self) -> tuple[float, float, float]:
        """世界系位姿 (x, y, theta)；`Bridge` 的权威来源，退化才是 get_odometry()。"""
        ...

    def obstacle_shapes(self) -> list[dict[str, Any]]:
        """障碍物列表（circle / rectangle 两种 name），供场景快照使用。"""
        ...

    @property
    def sim_time(self) -> float:
        """仿真时间，单位秒。"""
        ...

    @property
    def max_range(self) -> float:
        """激光量程，单位米（LiDAR 光束与饱和判定的上限）。"""
        ...


@runtime_checkable
class SensorBridge(Bridge, Protocol):
    """`Bridge` + 可选能力 `get_sensor_reading()`（理由见模块 docstring）。"""

    def get_sensor_reading(self) -> dict[str, Any] | None:
        """传感器读数；带 "scan" 键的条目是本包唯一的雷达数据通路。"""
        ...
