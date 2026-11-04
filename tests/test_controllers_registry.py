"""控制器注册表：kind 解析、大小写不敏感、未知 kind 的报错，以及 kwargs 直通。

表是唯一派发点（`_REGISTRY`），所以这里既测"能用"，也测"只登记了一处"。
"""

from __future__ import annotations

import pytest

from chassis_sim.chassis import make_chassis
from chassis_sim.controllers import (
    Controller,
    DwaController,
    PurePursuitController,
    make_controller,
)


@pytest.mark.parametrize("kind", ["pure_pursuit", "pp", "PP", "Pure_Pursuit", "  pp  "])
def test_pure_pursuit_kinds_and_aliases(kind: str) -> None:
    controller = make_controller(kind, chassis=make_chassis("ackermann"))
    assert isinstance(controller, PurePursuitController)
    assert isinstance(controller, Controller)
    assert controller.kind == "ackermann"


@pytest.mark.parametrize("kind", ["dwa", "DWA", "Dwa"])
def test_dwa_kind_is_case_insensitive(kind: str) -> None:
    controller = make_controller(kind, chassis=make_chassis("diff_drive"))
    assert isinstance(controller, DwaController)
    assert controller.kind == "diff_drive"


def test_default_kind_is_pure_pursuit() -> None:
    controller = make_controller(chassis=make_chassis("mecanum"))
    assert isinstance(controller, PurePursuitController)
    assert controller.kind == "mecanum"


def test_kwargs_are_forwarded_to_the_controller_class() -> None:
    pp = make_controller("pp", chassis=make_chassis("ackermann"), lookahead=0.9)
    assert pp.lookahead == pytest.approx(0.9)
    dwa = make_controller(
        "dwa", chassis=make_chassis("ackermann"), horizon=7, w_path=1.5
    )
    assert dwa.horizon == 7
    assert dwa.w_path == pytest.approx(1.5)


def test_unknown_kind_lists_available_kinds() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_controller("nope", chassis=make_chassis("ackermann"))
    message = str(excinfo.value)
    assert "'nope'" in message
    # 消息必须点名可用 kind，而不是只说"未知"
    for kind in ("pure_pursuit", "pp", "dwa"):
        assert kind in message


def test_controller_kind_is_positional_only() -> None:
    """控制器 kind 只能按位置给：kind= 会落到控制器自己的车型 kind 上，被 Controller 拒掉。"""
    with pytest.raises(ValueError, match="未知车型 'pp'"):
        make_controller(kind="pp", chassis=make_chassis("ackermann"))  # type: ignore[call-arg]
    # 即使车型名合法，控制器也必须显式带 limits/params —— make_controller 不猜标定
    with pytest.raises(TypeError, match="limits"):
        make_controller("pp", kind="ackermann")


def test_registry_maps_every_kind_to_a_controller_class() -> None:
    from chassis_sim.controllers import _REGISTRY

    assert _REGISTRY["pp"] is PurePursuitController
    assert _REGISTRY["pure_pursuit"] is PurePursuitController
    assert _REGISTRY["dwa"] is DwaController
    # 归一化是按小写查表 ⇒ 键必须全小写，否则大小写不敏感会漏
    assert [key for key in _REGISTRY if key != key.lower()] == []
    # 每个登记项都真的造得出来
    for kind in _REGISTRY:
        assert isinstance(
            make_controller(kind, chassis=make_chassis("ackermann")), Controller
        )


def test_controller_still_needs_a_chassis_calibration() -> None:
    """make_controller 只解 kind；车型标定仍由 Controller 校验（不静默默认值）。"""
    with pytest.raises(TypeError, match="chassis"):
        make_controller("pp")


class _EnvWithChassis:
    """最小 env 外壳：make_controller(env=...) 只读 unwrapped.chassis / chassis。"""

    def __init__(self, chassis: object) -> None:
        self.unwrapped = self
        self.chassis = chassis


def test_env_argument_is_equivalent_to_the_chassis_argument() -> None:
    """env= 只是"替我从 env 里取底盘"的便利写法，结果必须与 chassis= 同一台。"""
    chassis = make_chassis("ackermann")
    from_env = make_controller("pp", env=_EnvWithChassis(chassis))
    from_chassis = make_controller("pp", chassis=chassis)

    assert type(from_env) is type(from_chassis)
    assert isinstance(from_env, PurePursuitController)
    assert from_env.kind == from_chassis.kind == "ackermann"
    assert from_env.limits == from_chassis.limits
    assert from_env.params == from_chassis.params


def test_env_argument_prefers_unwrapped_chassis_then_plain_chassis() -> None:
    inner = _EnvWithChassis(make_chassis("ackermann"))

    class _Wrapper:
        unwrapped = inner
        chassis = make_chassis("mecanum")  # 取错这一层就会是 mecanum

    assert make_controller("pp", env=_Wrapper()).kind == "ackermann"

    class _PlainEnv:
        chassis = make_chassis("diff_drive")  # 没有 unwrapped ⇒ 直接读自己

    assert make_controller("pp", env=_PlainEnv()).kind == "diff_drive"


def test_env_and_chassis_together_is_a_conflict() -> None:
    chassis = make_chassis("ackermann")
    with pytest.raises(ValueError, match="chassis= 与 env=") as excinfo:
        make_controller("pp", chassis=chassis, env=_EnvWithChassis(chassis))
    message = str(excinfo.value)
    assert 'make_controller("pp", env=env)' in message


def test_missing_calibration_message_names_both_spellings() -> None:
    """既没给 chassis= 也没给 env= ⇒ 报错必须写出两种可抄的写法（不许裸穿透）。"""
    with pytest.raises(TypeError) as excinfo:
        make_controller("pp")
    message = str(excinfo.value)
    assert 'make_controller("pp", chassis=env.unwrapped.chassis)' in message
    assert 'make_controller("pp", env=env)' in message


def test_env_without_a_chassis_is_reported_actionably() -> None:
    with pytest.raises(TypeError, match="chassis") as excinfo:
        make_controller("pp", env=object())
    assert 'make_controller("pp", chassis=env.unwrapped.chassis)' in str(excinfo.value)
