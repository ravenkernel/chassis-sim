"""纯 numpy 的传感器桥（`SensorProvider` 实现）。

对外暴露：抽象接口 `SensorProvider`、实现 `SimpleBridge`、把归一化动作接到吃
物理命令的外部模拟器上的适配器 `PhysicalCommandBridge`。

1.0.0 起雷达只有一条通路：`bridge.get_sensor_reading()` 的 lidar 条目。
模块级的扫描字典 / 点云转换入口随旧接口一起删除，本包不再有第二套扫描格式。
`raycast` 里的两个下划线函数是实现细节，不在此转发（避免把私有符号变成公共 API）。
"""

from __future__ import annotations

from .adapters import PhysicalCommandBridge
from .base import SensorProvider
from .simple_bridge import SimpleBridge

__all__ = [
    "PhysicalCommandBridge",
    "SensorProvider",
    "SimpleBridge",
]
