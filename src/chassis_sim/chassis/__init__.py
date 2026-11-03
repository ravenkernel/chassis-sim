"""三种底盘的统一抽象（Ackermann / DiffDrive / Mecanum）。"""

from __future__ import annotations

from .ackermann import AckermannChassis
from .base import Chassis, ChassisCommand
from .diff_drive import DiffDriveChassis
from .mecanum import MecanumChassis
from .registry import CHASSIS_REGISTRY, available_chassis, make_chassis

__all__ = [
    "CHASSIS_REGISTRY",
    "AckermannChassis",
    "Chassis",
    "ChassisCommand",
    "DiffDriveChassis",
    "MecanumChassis",
    "available_chassis",
    "make_chassis",
]
