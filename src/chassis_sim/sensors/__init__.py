"""传感器层（0.2.0）：列表式 spec + 注入式几何 + 每支一条派生随机流。

入口二件套：

* `specs`：`Lidar` / `Imu` / `WheelOdom` 三个 frozen dataclass —— 只描述参数；
* `suite`：`SensorSuite(specs, *, seed=0, raycast)` —— 装配、驱动、交付读数。

本包**不 import** `chassis_sim.bridges`：射线求交由构造时注入的 `raycast`
提供，几何归桥。因此 `import chassis_sim.sensors` 只拉起 numpy 与标准库。
"""

from __future__ import annotations

from .specs import Imu, Lidar, Mount, SensorSpec, WheelOdom
from .suite import RaycastFn, SensorSuite, derive_rng

__all__ = [
    "Imu",
    "Lidar",
    "Mount",
    "RaycastFn",
    "SensorSpec",
    "SensorSuite",
    "WheelOdom",
    "derive_rng",
]
