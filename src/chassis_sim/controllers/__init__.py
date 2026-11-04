"""控制器层：`Controller` 协议 + 两个参考控制器（纯追踪 / DWA）。

三者共用同一份车型逆映射 `Controller.to_chassis_action`：
控制器内部按物理 (v, ω) 规划，出口转成该车型的归一化 action。

构造走 `make_controller(kind, **kwargs)`：一张 `_REGISTRY` 表把 kind 映射到类，
kwargs 原样转发给类的 `__init__`。控制器需要车型标定，所以通常要传底盘（给
`env=` 与 `chassis=env.unwrapped.chassis` 等价，前者少写一步取底盘）：

    make_controller("pp", chassis=env.unwrapped.chassis)
    make_controller("pp", env=env)          # 同一个结果
"""

from __future__ import annotations

from typing import Any

from .base import Controller
from .dwa import DwaController
from .pure_pursuit import PurePursuitController

__all__ = ["Controller", "DwaController", "PurePursuitController", "make_controller"]

#: kind（小写）-> 控制器类。kind 只在这一处登记，加控制器 = 加一行。
_REGISTRY: dict[str, type[Controller]] = {
    "pure_pursuit": PurePursuitController,
    "pp": PurePursuitController,
    "dwa": DwaController,
}


def _chassis_from_env(env: Any) -> Any | None:
    """env 的底盘对象：`env.unwrapped.chassis` 优先、退 `env.chassis`。

    只认**带 describe() 的**对象：Controller 构造时读的就是它，取不到就当没有
    （返回 None），由调用方给出可行动的报错。
    """
    unwrapped = getattr(env, "unwrapped", env)
    chassis = getattr(unwrapped, "chassis", None)
    if chassis is None or not hasattr(chassis, "describe"):
        return None
    return chassis


def make_controller(
    kind: str = "pure_pursuit",
    /,
    *,
    chassis: Any = None,
    env: Any = None,
    **kwargs: Any,
) -> Controller:
    """按名字造控制器（大小写不敏感），kwargs 原样转发给类的 `__init__`。

    Args:
        kind: `"pure_pursuit"` / `"pp"` → `PurePursuitController`；`"dwa"` →
            `DwaController`。**位置专用**：控制器自己的 `kind=` 是**车型**
            （ackermann / diff_drive / mecanum），两者同名，所以控制器 kind 只能按
            位置给 —— `make_controller(kind="pp")` 里的 `kind=` 会落到车型上，
            由 `Controller` 报"未知车型 'pp'"。
        chassis: 底盘对象（鸭子类型，只用 `describe()` 与 `action_space()`）。
            与 `env=` 二选一：同时给等于同一样标定说了两遍，直接 ValueError。
        env: 便利参数 —— 从 env 里解出底盘（`env.unwrapped.chassis` 优先，退
            `env.chassis`），等价于 `chassis=env.unwrapped.chassis`。
        **kwargs: 控制器构造参数。控制器按车型标定，要么给底盘（`chassis=` 或
            `env=`），要么显式 `kind=` + `limits=` + `params=`；纯追踪另有 lookahead
            等，DWA 另有 horizon / w_path 等。

    Returns:
        控制器实例。

    Raises:
        ValueError: kind 不在注册表里（消息列出全部可用 kind）；或 `chassis=` 与
            `env=` 同时给。
        TypeError: 既没给底盘也没给 `kind=`（消息点名两种写法）；或控制器类自己的
            其余构造校验。
    """
    key = str(kind).strip().lower()
    try:
        cls = _REGISTRY[key]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"未知控制器 kind {kind!r}；可用 kind: {available}（大小写不敏感）"
        ) from None
    if env is not None:
        if chassis is not None:
            raise ValueError(
                "make_controller() 的 chassis= 与 env= 冲突：两者只能给一个"
                "（env= 会自己从 env 里解出底盘）。\n"
                '  留一个：make_controller("pp", chassis=env.unwrapped.chassis) '
                '或 make_controller("pp", env=env)'
            )
        chassis = _chassis_from_env(env)
        if chassis is None:
            raise TypeError(
                "make_controller(env=...) 需要 env 自带底盘（env.unwrapped.chassis "
                "或 env.chassis，且带 describe()），"
                f"实得 env 类型 {type(env).__name__}。\n"
                '  请显式给底盘：make_controller("pp", chassis=env.unwrapped.chassis)'
            )
    if chassis is None and kwargs.get("kind") is None:
        # 在这一层先拦：否则裸穿透到 Controller 的消息只说了"缺 chassis"，
        # 没告诉读者 make_controller 这边到底该怎么写。
        raise TypeError(
            "make_controller() 需要底盘标定：既没给 chassis= 也没给 env=（也没有"
            "显式 kind=/limits=/params=），控制器因此无法确定车型与 limits。\n"
            "  两种写法任选其一：\n"
            '    1) make_controller("pp", chassis=env.unwrapped.chassis)\n'
            '    2) make_controller("pp", env=env)'
        )
    return cls(chassis=chassis, **kwargs)
