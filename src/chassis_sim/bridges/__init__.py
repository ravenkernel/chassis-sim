"""纯 numpy 的传感器桥（`SensorProvider` 实现）。

对外暴露：抽象接口 `SensorProvider`、实现 `SimpleBridge`，以及扫描通路的两个
入口 `get_scan` / `scan_to_pointcloud`（后者原本就在 base.py 的模块层）。
`raycast` 里的两个下划线函数是实现细节，不在此转发（避免把私有符号变成公共 API）。

`get_scan` 在 `SensorProvider` 上是**方法**（基类实现直接抛 NotImplementedError，
由 `SimpleBridge` 覆盖），这里转成一个模块级入口，这样
`from chassis_sim.bridges import get_scan, scan_to_pointcloud` 取到的是一对函数，
不必先拿到具体桥的类。转发而非别名：别名会绑到基类那个抛异常的实现上。
"""

from __future__ import annotations

from typing import Any

from .base import SensorProvider, scan_to_pointcloud
from .simple_bridge import SimpleBridge


def get_scan(bridge: SensorProvider, side: str = "front") -> dict[str, Any]:
    """转调 `bridge.get_scan(side)`（模块级入口；语义与实现完全由桥决定）。"""
    return bridge.get_scan(side)


__all__ = [
    "SensorProvider",
    "SimpleBridge",
    "get_scan",
    "scan_to_pointcloud",
]
