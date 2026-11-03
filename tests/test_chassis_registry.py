"""@register_chassis 注册表：内置三车型不变、第三方可注册、重名与未知名错误。

注册表是**全局运行期状态**，所以每个会写注册表的测试都套 isolated_registry
fixture（保存/恢复 CHASSIS_REGISTRY），保证测试顺序不影响结果——否则测试里注册的
假底盘会漏进 test_chassis_action_space.py 对 available_chassis() 的等值断言。
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from chassis_sim.chassis import (
    AckermannChassis,
    Chassis,
    ChassisCommand,
    DiffDriveChassis,
    MecanumChassis,
    available_chassis,
    make_chassis,
)
from chassis_sim.chassis.registry import CHASSIS_REGISTRY, register_chassis

BUILTINS = ["ackermann", "diff_drive", "mecanum"]


class FakeChassis(Chassis):
    """最小可构造底盘：只实现两个抽象方法 + 一个可选构造参数。"""

    name = "fake"

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = float(gain)

    def action_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def command(self, action: np.ndarray) -> ChassisCommand:
        return ChassisCommand(v=0.0, omega=0.0)


@pytest.fixture()
def isolated_registry():
    """保存/恢复全局注册表：测试注册的假底盘不许泄漏到别的测试。"""
    saved = dict(CHASSIS_REGISTRY)
    try:
        yield CHASSIS_REGISTRY
    finally:
        CHASSIS_REGISTRY.clear()
        CHASSIS_REGISTRY.update(saved)


# ── 内置三车型：注册方式变了，对外可见行为一字不改 ────────────────────
def test_builtin_chassis_are_registered_via_decorator() -> None:
    assert available_chassis() == BUILTINS
    assert sorted(CHASSIS_REGISTRY) == BUILTINS
    assert CHASSIS_REGISTRY["ackermann"] is AckermannChassis
    assert CHASSIS_REGISTRY["diff_drive"] is DiffDriveChassis
    assert CHASSIS_REGISTRY["mecanum"] is MecanumChassis
    assert isinstance(make_chassis("mecanum"), MecanumChassis)


def test_builtin_make_chassis_still_forwards_kwargs() -> None:
    chassis = make_chassis("ackermann", wheelbase=0.2)
    assert isinstance(chassis, AckermannChassis)
    assert chassis.wheelbase == pytest.approx(0.2)


# ── 第三方注册 ───────────────────────────────────────────────────────
def test_register_chassis_adds_to_available_and_make(isolated_registry) -> None:
    before = available_chassis()
    assert "third_party_rover" not in before

    returned = register_chassis("third_party_rover")(FakeChassis)
    assert returned is FakeChassis  # 装饰器原样返回类

    after = available_chassis()
    assert after == ["ackermann", "diff_drive", "mecanum", "third_party_rover"]
    assert after == sorted([*before, "third_party_rover"])

    chassis = make_chassis("third_party_rover", gain=2.5)
    assert isinstance(chassis, FakeChassis)
    assert chassis.gain == pytest.approx(2.5)


def test_decorator_syntax_keeps_class_identity(isolated_registry) -> None:
    @register_chassis("decorated_dummy")
    class DecoratedDummy(FakeChassis):
        name = "decorated_dummy"

    assert CHASSIS_REGISTRY["decorated_dummy"] is DecoratedDummy
    assert isinstance(make_chassis("decorated_dummy"), DecoratedDummy)
    # 字典序："decorated_dummy" < "diff_drive" ⇒ 排在 ackermann 之后、diff_drive 之前
    assert available_chassis() == [
        "ackermann",
        "decorated_dummy",
        "diff_drive",
        "mecanum",
    ]


# ── 错误路径 ─────────────────────────────────────────────────────────
def test_duplicate_name_raises_value_error_and_keeps_original(
    isolated_registry,
) -> None:
    register_chassis("dup")(FakeChassis)

    class Other(FakeChassis):
        pass

    with pytest.raises(ValueError) as excinfo:
        register_chassis("dup")(Other)
    msg = str(excinfo.value)
    assert "'dup'" in msg
    assert "已注册" in msg
    assert CHASSIS_REGISTRY["dup"] is FakeChassis  # 静默覆盖必须没有发生

    # 内置名同样不可被第三方覆盖
    with pytest.raises(ValueError):
        register_chassis("ackermann")(Other)
    assert CHASSIS_REGISTRY["ackermann"] is AckermannChassis


def test_empty_name_is_rejected(isolated_registry) -> None:
    with pytest.raises(ValueError):
        register_chassis("")
    assert available_chassis() == BUILTINS


def test_unknown_name_raises_keyerror_listing_available() -> None:
    with pytest.raises(KeyError) as excinfo:
        make_chassis("tank")
    msg = str(excinfo.value)
    assert "tank" in msg
    for name in BUILTINS:
        assert name in msg


def test_registry_has_no_leak_from_previous_tests() -> None:
    """顺序无关性自检：前面注册的假底盘已被 fixture 清掉。"""
    assert available_chassis() == BUILTINS
    assert "third_party_rover" not in CHASSIS_REGISTRY
    assert "decorated_dummy" not in CHASSIS_REGISTRY
