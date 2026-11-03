"""车型注册表：字符串 name -> Chassis 实现（装饰器注册，运行期可扩展）。

第三方底盘**不必改本仓源码**：在自己的模块里写一次装饰器即可入表，
之后 `available_chassis()` 能看到它、`make_chassis(名字, **kwargs)` 能构造它::

    from chassis_sim.chassis import Chassis
    from chassis_sim.chassis.registry import register_chassis

    @register_chassis("my_rover")
    class MyRover(Chassis):
        ...

注册是**运行期全局状态**（写进 CHASSIS_REGISTRY 这一个字典）：同名重复注册抛
ValueError 而不是静默覆盖——覆盖会让"谁最终生效"取决于 import 顺序。测试若临时
注册假底盘，请自行保存/恢复 CHASSIS_REGISTRY（见 tests/test_chassis_registry.py
的 isolated_registry fixture），否则泄漏的条目会污染别的测试。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ackermann import AckermannChassis
from .base import Chassis
from .diff_drive import DiffDriveChassis
from .mecanum import MecanumChassis

# name -> 底盘类。register_chassis 是唯一写入点，make_chassis 以它为查表源。
CHASSIS_REGISTRY: dict[str, type[Chassis]] = {}


def register_chassis(name: str) -> Callable[[type[Chassis]], type[Chassis]]:
    """返回把底盘类登记为 name 的类装饰器。

    Args:
        name: 车型名（非空字符串），即 make_chassis(name) 的查表键。

    Returns:
        类装饰器。它原样返回被装饰的类（类的身份/名字都不变），因此
        @register_chassis("x") 只是副作用式注册。

    Raises:
        ValueError: name 为空（或不是字符串），或该名字已被注册（含内置三车型）。
            消息里给出已占用者的模块与类名，便于定位是谁先注册的。
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"车型名必须是非空字符串，收到 {name!r}")

    def _decorator(cls: type[Chassis]) -> type[Chassis]:
        existing = CHASSIS_REGISTRY.get(name)
        if existing is not None:
            raise ValueError(
                f"车型名 {name!r} 已注册为 "
                f"{existing.__module__}.{existing.__qualname__}；"
                f"注册表不允许静默覆盖，请换名或先移除旧条目"
            )
        CHASSIS_REGISTRY[name] = cls
        return cls

    return _decorator


def available_chassis() -> list[str]:
    """已注册车型名（字典序，便于报告与断言）。"""
    return sorted(CHASSIS_REGISTRY)


def make_chassis(name: str, **params: Any) -> Chassis:
    """按名构造底盘；未知车型抛 KeyError 并列出可用车型。"""
    if name not in CHASSIS_REGISTRY:
        raise KeyError(f"未知底盘 {name!r}；可用车型: {', '.join(available_chassis())}")
    return CHASSIS_REGISTRY[name](**params)


# ── 内置三车型 ─────────────────────────────────────────────────────────────
# 等价于在三个类各自的定义处写 @register_chassis("...")；因为那三个文件不在本次
# 改动边界内，这里集中登记。顺序无关（available_chassis 自己排序）。
register_chassis("ackermann")(AckermannChassis)
register_chassis("diff_drive")(DiffDriveChassis)
register_chassis("mecanum")(MecanumChassis)
