"""控制器层：`Controller` 协议 + 两个参考控制器（纯追踪 / DWA）。

三者共用同一份车型逆映射 `Controller.to_chassis_action`：
控制器内部按物理 (v, ω) 规划，出口转成该车型的归一化 action。
"""

from __future__ import annotations

from .base import Controller
from .dwa import DwaController
from .pure_pursuit import PurePursuitController

__all__ = ["Controller", "DwaController", "PurePursuitController"]
