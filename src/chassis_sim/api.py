"""统一入口 make()：一行拿到可复现的 ChassisSimEnv（1.0.0 W1）。

优先级固定为 **kwargs > EnvConfig 字段 > 库默认**，逐字段由
chassis_sim.config.schema._resolve() 解析 —— 与 ChassisSimEnv.__init__ 内部
用的是同一个辅助函数，因此两条构造路径不会分叉。

用法::

    import chassis_sim as cs

    env = cs.make("ackermann")                    # 默认 128 维观测
    env = cs.make("mecanum", v_max=1.2, seed=3)   # kwargs 覆盖
    env = cs.make("ackermann", render_mode="rgb_array")  # 渲染模式透传给 env
    cfg = cs.EnvConfig(v_max=0.8)
    env = cs.make(cfg)                            # 用配置构造
    env = cs.make(cfg, v_max=0.4)                 # kwargs 再覆盖

scenario= 只读调用 chassis_sim.scenarios.load_scenario()，把场景翻译成
reset(options=...) 的键挂在 env.default_reset_options 上。1.1.0 起 env.reset()
在 options=None 时会**自动消费**它，因此场景默认就生效：

    env.reset(seed=0)              # 用 env.default_reset_options
    env.reset(seed=0, options={})  # 显式给 options ⇒ 一律以你为准

优先级只有一条：options 显式给出（含空 dict {}）一律以调用方为准；None 时才回落
到 env.default_reset_options（非空时）。场景名写错时 load_scenario 的
ScenarioNotFoundError 原样抛出，不静默忽略。

domain_rand 是三层解析：显式 kwargs > EnvConfig.domain_rand_enabled > 库默认
False（见 ChassisSimEnv 的同名形参）。库默认**确定性**：不加噪才是默认，
要域随机化必须显式打开（cfg 字段或 kwargs 任一处即可）。

seed 是**运行**种子（1.1.0 D）：env 自造桥时下发给 SimpleBridge(seed=...)，
域随机化噪声因此走桥自持的 Generator ⇒ 同 seed 逐位可复现；seed=None（默认）
仍吃全局 np.random，即 1.0.x 的既有行为、数值逐位不变。sensor_seed 只给传感器
随机流用种，未指定时跟随 seed（规则见 config.schema.resolve_sensor_seed）⇒
make(seed=3) 的传感器流与 1.0.x 的 make(sensor_seed=3) 逐位相同。
"""

from __future__ import annotations

import dataclasses
from typing import Any

from chassis_sim.config import EnvConfig
from chassis_sim.config.schema import _resolve, resolve_sensor_seed
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

__all__ = ["make"]

# EnvConfig 字段里由 ChassisSimEnv.__init__ 同名形参接管的那些：统一走 _resolve
_ENV_PARAM_FIELDS = (
    "chassis",
    "sensors",
    "obs_spec",
    "max_steps",
    "dt",
    "path_lookahead",
    "max_path_points",
    "sensor_seed",
    "seed",
    "path_power",
    "extrapolate_path",
    "path_fill_mode",
    "path_lookahead_floor",
    "emit_reward_components",
    "world_bounds",
)
# 只属于 env 的构造参数（EnvConfig 里没有同名字段）
_EXTRA_ENV_ARGS = (
    "bridge",
    "map_generator",
    "safety_gate",
    "env_index",
    "status_board",
    "reward_weights",
    "stage_index",
    "reward_kwargs_extra",
    "strict_reward_kwargs",
    "reward_cfg",
    # 3.0.0 W1：默认值与合法性校验都在 ChassisSimEnv.__init__ 里，api 层只透传，
    # 免得两处校验漂移（见下面 _EXTRA_ENV_ARGS 的取值循环：None = 没给 ⇒ 不转发）。
    "render_mode",
)
# 别名 -> 正名：obs= 是 obs_spec= 的别名。seed 自 1.1.0 起是 EnvConfig 的**正名**
# 字段（运行种子），不再转投 sensor_seed，故这里是自映射（_check_alias 永不触发）
# —— 保留词条只为让 _ALLOWED 与"可用键"提示里仍有 seed 这个词条。
_ALIASES = {"obs": "obs_spec", "seed": "seed"}


def _field_default(f: dataclasses.Field) -> Any:
    """dataclass 字段的库默认值（default 优先，其次 default_factory）。"""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


_FIELD_NAMES = tuple(f.name for f in dataclasses.fields(EnvConfig))
_FIELD_DEFAULTS = {f.name: _field_default(f) for f in dataclasses.fields(EnvConfig)}
# domain_rand 是 EnvConfig 字段 domain_rand_enabled 的 env 形参名，不在 _FIELD_NAMES
# 也不在 _EXTRA_ENV_ARGS 里（它有自己的三层解析），但必须作为合法键接受。
_ALLOWED = (
    frozenset(_FIELD_NAMES)
    | frozenset(_EXTRA_ENV_ARGS)
    | frozenset(_ALIASES)
    | {"domain_rand"}
)


def _check_alias(alias: str, canonical: str, overrides: dict[str, Any]) -> None:
    """别名与正名同时出现 ⇒ 语义冲突（谁赢都不能靠猜）。"""
    if canonical in overrides:
        raise ValueError(
            f"{alias}= 是 {canonical}= 的别名，两者不能同时给（想覆盖就直接写正名）"
        )


def make(
    chassis_or_cfg: str | EnvConfig = "ackermann",
    *,
    scenario: str | None = None,
    seed: int | None = None,
    sensors: Any = None,
    obs: Any = None,
    **overrides: Any,
) -> ChassisSimEnv:
    """按车型名或 EnvConfig 构造 ChassisSimEnv（kwargs > cfg > 库默认）。

    Args:
        chassis_or_cfg: 车型名（ackermann / diff_drive / mecanum ...）或
            EnvConfig 实例。给 str 时等价于 EnvConfig(chassis=该名)。
        scenario: 场景名或 YAML 路径（chassis_sim.scenarios.load_scenario 的
            入参）。非 None 时翻译成 env.default_reset_options。
        seed: 运行种子（EnvConfig.seed）。env 自造桥时下发给 SimpleBridge，
            域随机化噪声因此可复现；同时是 sensor_seed 未指定时的传感器流
            种子来源（config.schema.resolve_sensor_seed），也是场景地图种子。
            与 sensor_seed= 同时给出是合法的（两者各管各的），不再报冲突。
        sensors: 传感器 spec 序列（Lidar / Imu / WheelOdom 实例），
            等价于 sensors= 覆盖；只在 env 自造桥时生效。
        obs: obs_spec 的别名（ObsSpec 实例）。
        **overrides: EnvConfig 字段名（v_max / dt / domain_rand_enabled ...）或
            env 专属构造参数（map_generator / reward_cfg / safety_gate /
            render_mode ...）。未知键直接 ValueError。

    Returns:
        构造好的 ChassisSimEnv；若给了 scenario，还带 scenario 属性与
        default_reset_options（dict）—— env.reset() 在 options=None 时会自动
        消费它，显式 options（含 {}）仍然覆盖。

    Raises:
        TypeError: 第一个参数既不是 str 也不是 EnvConfig。
        ValueError: 未知 override 键，或别名与正名同时给。
        ScenarioNotFoundError: scenario 名字/路径不存在（不静默忽略）。
    """
    if isinstance(chassis_or_cfg, str):
        cfg = EnvConfig(chassis=chassis_or_cfg)
    elif isinstance(chassis_or_cfg, EnvConfig):
        cfg = chassis_or_cfg
    else:
        raise TypeError(
            "make() 的第一个参数只能是车型名（str）或 EnvConfig 实例，"
            f"得到 {type(chassis_or_cfg).__name__}"
        )

    overrides = dict(overrides)
    unknown = sorted(set(overrides) - _ALLOWED)
    if unknown:
        raise ValueError(
            f"未知的 make() 参数 {unknown}；可用键共 {len(_ALLOWED)} 个"
            "（EnvConfig 字段名 + env 构造参数，别名 obs/seed）"
        )

    if obs is not None:
        _check_alias("obs", "obs_spec", overrides)
        overrides["obs_spec"] = obs
    if seed is not None:
        # seed 是正名字段（运行种子），不再转投 sensor_seed ⇒ 与 sensor_seed=
        # 同时给出是两条独立输入，不是别名冲突。
        overrides["seed"] = seed
    if sensors is not None:
        overrides["sensors"] = sensors

    # EnvConfig 字段：kwargs 覆盖进 cfg（物理参数由 _resolve_chassis 从 cfg 读，
    # 因此 make(cfg, v_max=0.8) 必须落到 cfg 上才真的生效）
    cfg_overrides = {k: v for k, v in overrides.items() if k in _FIELD_NAMES}
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)

    # env 形参：逐字段 kwargs > cfg > 库默认，与 env 内部同一套 _resolve
    env_kwargs: dict[str, Any] = {}
    for name in _ENV_PARAM_FIELDS:
        value = _resolve(overrides.get(name), getattr(cfg, name), _FIELD_DEFAULTS[name])
        if value is not None:
            env_kwargs[name] = value
    # domain_rand 在 EnvConfig 里叫 domain_rand_enabled，语义是同一个开关：
    # kwargs > cfg > False（库默认确定性）。
    env_kwargs["domain_rand"] = bool(
        _resolve(overrides.get("domain_rand"), cfg.domain_rand_enabled, False)
    )
    # env 专属参数只认显式给出（None = 没给，保留 env 自己的默认/判断）
    for name in _EXTRA_ENV_ARGS:
        value = overrides.get(name)
        if value is not None:
            env_kwargs[name] = value

    scenario_name = _resolve(scenario, cfg.scenario, None)
    if scenario_name is not None:
        cfg = dataclasses.replace(cfg, scenario=scenario_name)

    env = ChassisSimEnv(cfg=cfg, **env_kwargs)

    # 场景：只读加载并翻译成 reset(options=...) 的键。1.1.0 起 env.reset() 在
    # options=None 时自动消费它（显式 options 一律以调用方为准），因此
    # make(scenario=...) 造出来的 env 直接 reset() 场景就生效。
    env.scenario = None
    env.default_reset_options = {}
    if scenario_name is not None:
        scenario_obj = load_scenario(scenario_name)
        env.scenario = scenario_obj
        # 地图种子 = **推导后**的传感器种子（sensor_seed 优先，否则跟随 seed，
        # 都没有则 0）：与 1.0.x 的 int(sensor_seed) 在"只给 seed"与"只给
        # sensor_seed"两种写法下逐位相同。
        env.default_reset_options = scenario_to_reset_options(
            scenario_obj,
            seed=resolve_sensor_seed(
                env_kwargs.get("sensor_seed"), env_kwargs.get("seed")
            ),
        )
    return env
