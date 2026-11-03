"""评测器：控制器 × 车型的回合统计 + 95% 置信区间（公开 API）。

设计要点
--------
* **成功率必须带 n 与区间**：40 集、真实成功率 p=0.5 时 Wilson 95% 区间约
  ±15%（30 集约 ±17.5%、60 集约 ±9.8%）。只给一个小数无法判断两组控制器
  是真的有差别还是抽样噪声。
* **区间用 Wilson score interval，不用正态近似**：p 接近 0/1 或 n 很小时，
  正态近似会给出越界（甚至负宽度）的区间，Wilson 不会。公式（z = 1.96）：

      p_hat = k / n
      denom = 1 + z^2 / n
      center = (p_hat + z^2 / (2n)) / denom
      half = z / denom * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4n^2))
      CI = [max(0, center - half), min(1, center + half)]

  n = 0 时返回 (0.0, 1.0)：没有样本就是没有信息，不能伪装成 0% 或 100%。
* **动作空间只有 env.action_space 一个权威**：评测器不产生动作、不读物理界，
  控制器给出的 action 原样交给 env.step（各车型的动作界由 chassis 决定，
  评测器不复制一份）。
* **控制器必须与 env 的底盘同标定**：控制器按**自己的** limits 归一化 (v, ω)
  （`Controller.to_chassis_action`），env 的 `Chassis.command()` 按**底盘**的
  limits 反解。两侧不同源时语义被静默缩放 —— `make_chassis("ackermann")` 的
  v_max=1.0 对上 `ChassisSimEnv` 的 v_max=1.2，请求 0.5 m/s 实跑 0.6 m/s（+20%）。
  `evaluate()` 因此在开跑前比对两侧 limits，不一致直接抛 ValueError；
  `allow_mismatch=True` 可显式跳过（打一条 [WARN][eval]）。
* **n 个回合必须是 n 张不同的地图**：`scenario_to_reset_options()` 在**调用那
  一刻**就把地图（障碍 + 路径）生成好存进 dict，同一条 options 再喂 n 次
  `env.reset` 是同一张图重放 n 次（回合 i 的 `seed + i` 只推进 env 内部 RNG，
  不重掷地图）⇒ Wilson 区间的分母不是 n 个独立样本。因此 `reset_options`
  接受 `callable(episode_index) -> dict`（每回合重掷一张图，由
  `chassis_sim.scenarios.scenario_reset_options_factory` 提供）；传固定 dict
  会打 `[WARN][eval]`，`to_markdown` 的格子标注「⚠固定地图」。
  每回合**实际**跑的那张图由 `EvalResult.map_fingerprints` /
  `distinct_maps` / `maps_independent` 自证。
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Z95",
    "EpisodeMetrics",
    "EvalResult",
    "controller_observation",
    "evaluate",
    "to_markdown",
    "wilson_interval",
]

#: 95% 双侧正态分位数
Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, *, z: float = Z95) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间（公式见模块 docstring）。

    Args:
        successes: 成功次数 k（0 <= k <= n）。
        n: 样本数。n <= 0 时返回 (0.0, 1.0) —— 无样本即无信息。
        z: 分位数，默认 1.95996...（95% 双侧）。

    Returns:
        (low, high)，均已截断到 [0, 1]。
    """
    if n <= 0:
        return (0.0, 1.0)
    p = float(successes) / float(n)
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class EpisodeMetrics:
    """单回合结果。字段全部是 JSON 可序列化的原生类型。"""

    success: bool
    collision: bool
    timeout: bool
    steps: int
    episode_return: float
    min_obstacle_clearance: float | None = None
    termination_reason: str = ""
    #: 本回合**实际**跑的那张地图的指纹（障碍 + 参考路径的 sha1 前 12 位）；
    #: env 既不暴露障碍也没有路径时为 ""（未知，不冒充独立样本）。
    map_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "collision": bool(self.collision),
            "timeout": bool(self.timeout),
            "steps": int(self.steps),
            "episode_return": float(self.episode_return),
            "min_obstacle_clearance": (
                None
                if self.min_obstacle_clearance is None
                else float(self.min_obstacle_clearance)
            ),
            "termination_reason": str(self.termination_reason),
            "map_fingerprint": str(self.map_fingerprint),
        }


#: `EvalResult.reset_options_kind` 的取值（也用于 to_dict / to_markdown 判定）
RESET_FIXED = "fixed"  # 传了固定 dict/Mapping：n 个回合是同一张图重放 n 次
RESET_PER_EPISODE = "per-episode"  # 传了 callable：每回合重新生成 options
RESET_NONE = "none"  # 没传：地图由 env 自己决定（通常按 seed 生成）
RESET_UNSPECIFIED = "unspecified"  # 直接构造 EvalResult（评测器之外的来源）


@dataclass(frozen=True)
class EvalResult:
    """一组同构回合（同一控制器 × 同一车型）的汇总。

    `success_ci` 只在 n 个回合互为**独立样本**时可引用 —— 判据就在本对象上：
    `distinct_maps` / `maps_independent` / 每回合的 `map_fingerprint`。
    `maps_independent` 为 False 时（同一张图重放 n 次）区间只反映"这张图上的
    重放方差"，不反映抽样误差，不要拿它做控制器对比。
    """

    episodes: tuple[EpisodeMetrics, ...]
    controller: str = "controller"
    chassis: str = "chassis"
    seed: int = 0
    #: 调用方喂给 evaluate 的 reset_options 形态（RESET_* 常量之一）
    reset_options_kind: str = RESET_UNSPECIFIED

    def __post_init__(self) -> None:
        if not isinstance(self.episodes, tuple):
            object.__setattr__(self, "episodes", tuple(self.episodes))

    # ── 计数与比率 ────────────────────────────────────────────────
    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def collisions(self) -> int:
        return sum(1 for e in self.episodes if e.collision)

    @property
    def timeouts(self) -> int:
        return sum(1 for e in self.episodes if e.timeout)

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def collision_rate(self) -> float:
        return self.collisions / self.n if self.n else 0.0

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.n if self.n else 0.0

    @property
    def mean_steps(self) -> float:
        return float(np.mean([e.steps for e in self.episodes])) if self.n else 0.0

    @property
    def mean_return(self) -> float:
        return (
            float(np.mean([e.episode_return for e in self.episodes])) if self.n else 0.0
        )

    @property
    def min_obstacle_clearance(self) -> float | None:
        vals = [
            e.min_obstacle_clearance
            for e in self.episodes
            if e.min_obstacle_clearance is not None
        ]
        return min(vals) if vals else None

    @property
    def success_interval(self) -> tuple[float, float]:
        """成功率的 Wilson 95% 区间 (low, high)。"""
        return wilson_interval(self.successes, self.n)

    @property
    def success_ci(self) -> float:
        """成功率的保守半宽：max(p - low, high - p)。"""
        low, high = self.success_interval
        return max(self.success_rate - low, high - self.success_rate)

    # ── 地图独立性（区间能否引用，看这里）────────────────────────
    @property
    def map_fingerprints(self) -> tuple[str, ...]:
        """逐回合的地图指纹（顺序同 `episodes`）。"""
        return tuple(e.map_fingerprint for e in self.episodes)

    @property
    def distinct_maps(self) -> int:
        """不同地图的个数；`0` = 未知（env 不暴露障碍也没有路径）。"""
        return len({fp for fp in self.map_fingerprints if fp})

    @property
    def maps_independent(self) -> bool:
        """n 个回合是否真的各跑一张不同的地图。

        要求 n > 1 时**每个**回合都有非空指纹且两两不同；n <= 1 时无意义，
        恒为 True（一个回合谈不上"独立样本"，也谈不上重复）。
        """
        if self.n <= 1:
            return True
        fingerprints = self.map_fingerprints
        return all(fingerprints) and len(set(fingerprints)) == self.n

    @property
    def same_map_repeated(self) -> bool:
        """n > 1 且所有回合共用同一张地图 —— `success_ci` 只是重放方差。"""
        return self.n > 1 and self.distinct_maps == 1

    # ── 序列化 ────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化的完整结果（含每回合明细与两侧区间端点）。"""
        lo, hi = self.success_interval
        clo, chi = wilson_interval(self.collisions, self.n)
        tlo, thi = wilson_interval(self.timeouts, self.n)
        return {
            "controller": str(self.controller),
            "chassis": str(self.chassis),
            "seed": int(self.seed),
            "reset_options_kind": str(self.reset_options_kind),
            "distinct_maps": int(self.distinct_maps),
            "maps_independent": bool(self.maps_independent),
            "map_fingerprints": list(self.map_fingerprints),
            "n": int(self.n),
            "successes": int(self.successes),
            "success_rate": float(self.success_rate),
            "success_ci": float(self.success_ci),
            "success_ci_low": float(lo),
            "success_ci_high": float(hi),
            "collision_rate": float(self.collision_rate),
            "collision_ci_low": float(clo),
            "collision_ci_high": float(chi),
            "timeout_rate": float(self.timeout_rate),
            "timeout_ci_low": float(tlo),
            "timeout_ci_high": float(thi),
            "mean_steps": float(self.mean_steps),
            "mean_return": float(self.mean_return),
            "min_obstacle_clearance": (
                None
                if self.min_obstacle_clearance is None
                else float(self.min_obstacle_clearance)
            ),
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def to_markdown(
        self,
        rows: Iterable[tuple[str, str, EvalResult]]
        | Mapping[tuple[str, str], EvalResult]
        | None = None,
        *,
        title: str = "chassis-sim · 控制器 × 车型",
    ) -> str:
        """渲染「控制器 × 车型」Markdown 表；rows 省略时只渲染自己这一行。"""
        if rows is None:
            rows = [(self.controller, self.chassis, self)]
        return to_markdown(rows, title=title)


# ── 底盘一致性护栏：控制器必须与 env 的底盘同标定 ─────────────────────
#
# 病因（W11b 实测）：控制器内部按物理量 (v, ω) 规划，出口 to_chassis_action 用
# **控制器自己的** limits 归一化，而 env 的 Chassis.command() 用**底盘自己的**
# limits 反解。两侧不同源时语义被静默缩放：控制器按 make_chassis("ackermann")
# 的 v_max=1.0 标定、env 用 EnvConfig 的 v_max=1.2 ⇒ 请求 0.5 m/s 实跑
# 0.6 m/s（+20%），ω 同理。方向与拓扑没错，契约是假的。
#
# 比对键就是 Controller 构造时从 describe()["limits"] 取的那四把尺子；
# 容差只吸收 float 表示误差（两侧同源时逐位相等，见 examples/eval_matrix.py
# 打印的 0.6 m/s 往返），不吸收参数差异（最小的真实差异是 20% 量级）。
_LIMIT_KEYS: tuple[str, ...] = ("v_min", "v_max", "omega_max", "vy_max")
_LIMIT_RTOL = 1e-12
_LIMIT_ATOL = 1e-12


def _env_chassis(env: Any) -> Any | None:
    """env 的底盘对象；鸭子类型的评测替身没有底盘时返回 None（跳过检查）。"""
    unwrapped = getattr(env, "unwrapped", env)
    chassis = getattr(unwrapped, "chassis", None)
    if chassis is None or not hasattr(chassis, "describe"):
        return None
    return chassis


def _limit_table(limits: Mapping[str, Any]) -> dict[str, float]:
    """按 _LIMIT_KEYS 取限值为 float；vy_max 缺省 0.0（与 Controller 内部同口径）。"""
    return {key: float(limits.get(key, 0.0) or 0.0) for key in _LIMIT_KEYS}


def _format_limits(table: Mapping[str, float]) -> str:
    """'v_min=-1.0 v_max=1.0 omega_max=... vy_max=0.0' —— 两侧都按同一键序打印。"""
    return " ".join(f"{key}={table[key]!r}" for key in _LIMIT_KEYS)


def _mismatch_report(controller: Any, chassis: Any) -> str | None:
    """比对控制器与底盘的车型/limits；一致返回 None，否则返回完整差异描述。"""
    desc = chassis.describe()
    env_limits = _limit_table(desc.get("limits") or {})
    ctrl_limits = _limit_table(controller.limits)
    ctrl_kind = getattr(controller, "kind", None)
    env_kind = desc.get("name", getattr(chassis, "name", None))

    diffs: list[str] = []
    if (
        ctrl_kind is not None
        and env_kind is not None
        and str(ctrl_kind) != str(env_kind)
    ):
        diffs.append(f"kind: 控制器 {ctrl_kind!r} vs env {env_kind!r}")
    for key in _LIMIT_KEYS:
        ctrl_v = ctrl_limits[key]
        env_v = env_limits[key]
        if not math.isclose(ctrl_v, env_v, rel_tol=_LIMIT_RTOL, abs_tol=_LIMIT_ATOL):
            diffs.append(f"{key}: 控制器 {ctrl_v!r} vs env {env_v!r}")
    if not diffs:
        return None

    return (
        "控制器与 env 的底盘不一致 —— 控制器是按另一台底盘标定的：\n"
        f"  controller:  kind={ctrl_kind!r} {_format_limits(ctrl_limits)}\n"
        f"  env.chassis: kind={env_kind!r} {_format_limits(env_limits)}\n"
        f"  不一致项: {'; '.join(diffs)}\n"
        "  后果: to_chassis_action 按控制器的 limits 归一化、Chassis.command() 按\n"
        "  env 底盘的 limits 反解 ⇒ 速度/角速度语义被静默缩放（v_max 1.0→1.2 即\n"
        "  +20% 超速）。\n"
        "  修法: 用 env 的底盘构造控制器（PurePursuitController(\n"
        "  chassis=env.unwrapped.chassis)），或显式 evaluate(..., allow_mismatch=True)。"
    )


def _warn(message: str) -> None:
    """本模块唯一的告警出口：`[WARN][eval]` 前缀 + stderr（便于 grep 与测试断言）。"""
    print(f"[WARN][eval] {message}", file=sys.stderr, flush=True)


def _check_controller_chassis(
    env: Any, controller: Any, *, allow_mismatch: bool
) -> None:
    """开跑前校验控制器与 env 底盘的车型/limits 一致（病因见本节注释）。

    不按物理量归一化的控制器（没有 `limits`，例如直接采 action_space 的 random
    基线）无从比对，直接放行；env 没有底盘对象（鸭子类型的评测替身）同样放行。
    """
    if getattr(controller, "limits", None) is None:
        return
    chassis = _env_chassis(env)
    if chassis is None:
        return
    report = _mismatch_report(controller, chassis)
    if report is None:
        return
    if allow_mismatch:
        _warn(
            "allow_mismatch=True：跳过控制器/底盘一致性检查，"
            f"结果不能与同标定的控制器结果比较\n{report}"
        )
        return
    raise ValueError(report)


# ── 环境适配：控制器吃的是 dict 观测，不是 494 维扁平向量 ──────────────
def controller_observation(env: Any) -> dict[str, Any]:
    """把 env 的状态拼成控制器契约的观测 dict。

    Controller.act() 的契约键（见 controllers/base.py）：
        pose       (3,) 世界系 [x, y, theta] —— 扁平 obs 里没有世界位姿，
                   唯一权威来源是 bridge.get_odometry()[0:3]；
        obstacles  可选，主仓同构的障碍列表（env._obstacles）；
        dt         可选，DWA 用。

    这里**不**使用 env.reset()/step() 返回的扁平 obs：那是给策略网络用的
    归一化视图，控制器需要的是物理量。
    """
    unwrapped = getattr(env, "unwrapped", env)
    bridge = getattr(unwrapped, "bridge", None)
    if bridge is None or not hasattr(bridge, "get_odometry"):
        raise TypeError(
            "env 缺少 bridge.get_odometry()，无法给控制器提供世界位姿 "
            f"（env 类型 {type(env).__name__}）"
        )
    odom = np.asarray(bridge.get_odometry(), dtype=np.float64).reshape(-1)
    if odom.size < 3:
        raise ValueError(f"bridge.get_odometry() 至少要 3 个分量，实得 {odom.size}")
    return {
        "pose": odom[:3].copy(),
        "obstacles": getattr(unwrapped, "_obstacles", None),
        "dt": float(getattr(unwrapped, "dt", 0.1)),
    }


def _env_path(env: Any) -> np.ndarray | None:
    """世界系参考路径 (N, 3)；环境没有路径时返回 None。"""
    unwrapped = getattr(env, "unwrapped", env)
    bridge = getattr(unwrapped, "bridge", None)
    if bridge is None or not hasattr(bridge, "get_path_global"):
        return None
    path = np.asarray(bridge.get_path_global(), dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        return None
    return path


def _obstacle_signature(obstacle: Any) -> str:
    """把一个障碍物压成可比较的字符串（名字 + 位姿前 3 分量 + 尺寸）。"""
    shape = getattr(obstacle, "shape", None)
    if shape is None:
        return f"{type(obstacle).__name__}:{obstacle!r}"
    fields: list[str] = [str(getattr(shape, "name", "") or "")]
    state = np.asarray(getattr(shape, "state", ()), dtype=object).ravel()
    fields.extend(_round_triple(state[:3]))
    for key in ("radius", "length", "width"):
        value = getattr(shape, key, None)
        if value is not None:
            fields.append(f"{key}={_round_value(value)}")
    return "|".join(fields)


def _round_value(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return repr(value)


def _round_triple(values: Any) -> list[str]:
    return [_round_value(v) for v in values]


def _map_fingerprint(env: Any) -> str:
    """回合开跑后对 env **实际**那张图取指纹（12 位 sha1）。

    取自 reset 之后的真实状态（障碍物 + 世界系路径），而不是喂进去的
    reset_options ⇒ 固定 dict、每回合回调、env 自己按 seed 生成三种来源同口径；
    也正因为如此，"传了 callable 但每回合返回同一张图"同样会被看出来。
    两者都拿不到时返回 ""（未知），绝不冒充"独立"。
    """
    unwrapped = getattr(env, "unwrapped", env)
    obstacles = getattr(unwrapped, "_obstacles", None)
    path = _env_path(env)
    if obstacles is None and path is None:
        return ""
    digest = hashlib.sha1()
    for item in sorted(_obstacle_signature(o) for o in (obstacles or ())):
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    if path is not None:
        digest.update(np.round(np.asarray(path, dtype=np.float64), 6).tobytes())
    return digest.hexdigest()[:12]


def _run_episode(
    env: Any,
    controller: Any,
    seed: int,
    max_steps: int | None,
    reset_options: Mapping[str, Any] | None,
) -> EpisodeMetrics:
    """跑一个回合：reset(seed=seed) -> 循环 step(controller.act(...)) -> 汇总。"""
    _, _ = env.reset(seed=seed, options=reset_options)
    map_fingerprint = _map_fingerprint(env)
    if hasattr(controller, "reset"):
        controller.reset()
    path = _env_path(env)
    if path is not None and hasattr(controller, "set_path"):
        controller.set_path(path)

    episode_return = 0.0
    steps = 0
    clearance: float | None = None
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while True:
        action = controller.act(controller_observation(env))
        _, reward, terminated, truncated, info = env.step(action)
        steps += 1
        episode_return += float(reward)
        d_min = info.get("d_min") if isinstance(info, dict) else None
        if d_min is not None:
            d_min = float(d_min)
            if math.isfinite(d_min):
                clearance = d_min if clearance is None else min(clearance, d_min)
        if bool(terminated) or bool(truncated):
            break
        if max_steps is not None and steps >= max_steps:
            break

    info = info if isinstance(info, dict) else {}
    success = bool(info.get("goal_reached", False))
    collision = bool(info.get("collision", False))
    timeout = bool(info.get("timeout", False))
    if not (success or collision or timeout) and (
        bool(truncated) or (max_steps is not None and steps >= max_steps)
    ):
        # 外部截断（TimeLimit / 评测器 max_steps）语义上就是超时：
        # 不算成功、也不算碰撞，回合没走完。
        timeout = True
    return EpisodeMetrics(
        success=success,
        collision=collision,
        timeout=timeout,
        steps=steps,
        episode_return=episode_return,
        min_obstacle_clearance=clearance,
        termination_reason=str(info.get("termination_reason") or ""),
        map_fingerprint=map_fingerprint,
    )


def _labels(
    env: Any, controller: Any, labels: tuple[str, str] | None
) -> tuple[str, str]:
    if labels is not None:
        return str(labels[0]), str(labels[1])
    name = type(controller).__name__
    unwrapped = getattr(env, "unwrapped", env)
    chassis = getattr(unwrapped, "chassis_name", None) or type(unwrapped).__name__
    return name, str(chassis)


def _episode_reset_options(
    reset_options: (
        Mapping[str, Any] | Callable[[int], Mapping[str, Any] | None] | None
    ),
    episode_index: int,
) -> Mapping[str, Any] | None:
    """取第 episode_index 个回合该用的 options：callable 每回合调一次，dict 原样复用。"""
    if reset_options is None or not callable(reset_options):
        return reset_options
    options = reset_options(episode_index)
    if options is not None and not isinstance(options, Mapping):
        raise TypeError(
            "reset_options(episode_index) 必须返回 Mapping（或 None = 本回合不覆盖），"
            f"第 {episode_index} 个回合实得 {type(options).__name__}"
        )
    return options


def evaluate(
    env_factory: Callable[[], Any] | Any,
    controller: Any,
    *,
    n_episodes: int,
    seed: int = 0,
    max_steps: int | None = None,
    reset_options: (
        Mapping[str, Any] | Callable[[int], Mapping[str, Any] | None] | None
    ) = None,
    labels: tuple[str, str] | None = None,
    allow_mismatch: bool = False,
) -> EvalResult:
    """跑 n_episodes 个回合并汇总成 EvalResult。

    回合 i 用 reset(seed=seed + i)（可复现：同一 seed 同一控制器 ⇒ 同一批种子）。
    每回合开始把 env 的世界系路径喂给 controller.set_path()，并 controller.reset()。
    循环 step(controller.act(observation)) 直到 terminated/truncated、或步数触及
    max_steps（此时未结束的回合记为 timeout）。

    Args:
        env_factory: 无参可调用对象（每次调用返回**新** env，由本函数负责 close）
            或一个现成的 env 实例（不会被 close）。
        controller: 具备 act(observation: dict) 的对象；reset()/set_path() 可选。
        n_episodes: 回合数，必须 > 0。
        seed: 起始种子；回合 i 用 seed + i。
        max_steps: 单回合步数上限（None = 只靠 env 自己的终止条件）。
        reset_options: 透传 env.reset(options=...)，两种形态：
            - **callable**（推荐）：`reset_options(episode_index) -> Mapping`，
              每回合调用一次，用于每回合重掷一张**独立**地图
              （`chassis_sim.scenarios.scenario_reset_options_factory`）；
            - **固定 Mapping**（向后兼容）：同一条 options 喂 n 次 ⇒ n 个回合
              是同一张图重放，打一条 `[WARN][eval]`，表格格子里标注
              「⚠固定地图」，此时的 CI 不代表独立样本。
            省略 = 不覆盖，地图由 env 自己按 seed 决定（此时看
            `EvalResult.maps_independent` 才知道是否真的每回合换图）。
        labels: (controller 名, 车型名)，用于表格与 JSON；省略时按类型名/车型自动推断。
        allow_mismatch: 显式允许控制器与 env 底盘的 limits 不一致（默认 False）。
            默认行为是**开跑前**比对两侧的 v_min/v_max/omega_max/vy_max，不一致
            直接抛 ValueError —— 两侧不同源时 to_chassis_action 与 Chassis.command()
            互为逆映射的前提不成立，速度/角速度会被静默缩放（实测 +20%）。

    Returns:
        EvalResult（含每回合明细、成功率与 Wilson 95% 区间）。

    Raises:
        ValueError: n_episodes <= 0，或控制器与 env 底盘的车型/limits 不一致
            （后者可用 allow_mismatch=True 跳过，此时改为打一条 [WARN][eval]）。
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes 必须 > 0，实得 {n_episodes}")

    if reset_options is None:
        options_kind = RESET_NONE
    elif callable(reset_options):
        options_kind = RESET_PER_EPISODE
    else:
        options_kind = RESET_FIXED
        if int(n_episodes) > 1:
            _warn(
                f"reset_options 是固定 {type(reset_options).__name__}："
                f"{int(n_episodes)} 个回合跑的是**同一张地图**（回合 seed 只推进 "
                "env 内部 RNG，不重掷地图）⇒ success_ci 只是该图上的重放方差，"
                "不代表独立样本，不可用于控制器对比。要每回合一张独立地图请传 "
                "callable(episode_index)（例："
                "scenario_reset_options_factory('obstacles_dense', base_seed=0)）。"
            )

    owns_env = callable(env_factory)
    env = env_factory() if owns_env else env_factory
    episodes: list[EpisodeMetrics] = []
    try:
        # 先校验再跑：跑完才发现标定不一致，等于把 10 个回合的假数字留在盘上。
        _check_controller_chassis(env, controller, allow_mismatch=allow_mismatch)
        for i in range(int(n_episodes)):
            options = _episode_reset_options(reset_options, i)
            episodes.append(_run_episode(env, controller, seed + i, max_steps, options))
    finally:
        if owns_env and hasattr(env, "close"):
            env.close()

    controller_label, chassis_label = _labels(env, controller, labels)
    result = EvalResult(
        episodes=tuple(episodes),
        controller=controller_label,
        chassis=chassis_label,
        seed=int(seed),
        reset_options_kind=options_kind,
    )
    if result.n > 1 and result.same_map_repeated and options_kind != RESET_FIXED:
        _warn(
            f"{result.n} 个回合的地图指纹全部相同（{result.map_fingerprints[0]}）："
            "这些回合不是独立样本，success_ci 只是同一张图上的重放方差。"
            "若这是 env 自己按 seed 生成地图，说明它没用上 seed+回合号。"
        )
    return result


# ── Markdown 渲染 ─────────────────────────────────────────────────
def _normalise_rows(
    rows: Iterable[tuple[str, str, EvalResult]] | Mapping[tuple[str, str], EvalResult],
) -> list[tuple[str, str, EvalResult]]:
    if isinstance(rows, Mapping):
        triples: list[tuple[str, str, EvalResult]] = []
        for key, result in rows.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError(
                    "Mapping 形式的 rows 的键必须是 (controller, chassis) 二元组，"
                    f"实得 {key!r}"
                )
            triples.append((str(key[0]), str(key[1]), result))
        return triples

    triples = []
    for item in rows:
        try:
            ctrl, chassis, result = item
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rows 的每一项必须是 (controller, chassis, EvalResult) 三元组，"
                f"实得 {item!r}"
            ) from exc
        triples.append((str(ctrl), str(chassis), result))
    return triples


def _fixed_map(result: EvalResult) -> bool:
    """这一格的成功率是否建立在"同一张地图重放 n 次"之上。"""
    if result.n <= 1:
        return False
    return result.same_map_repeated or result.reset_options_kind == RESET_FIXED


def _success_cell(result: EvalResult) -> str:
    cell = f"{result.success_rate:.1%} ±{result.success_ci:.1%} (n={result.n})"
    if _fixed_map(result):
        # 区间照样打印（旧行为的数字不该被隐藏），但必须一眼看出分母不是 n 张图。
        cell += " ⚠固定地图"
    return cell


def to_markdown(
    rows: Iterable[tuple[str, str, EvalResult]] | Mapping[tuple[str, str], EvalResult],
    *,
    title: str = "chassis-sim · 控制器 × 车型",
) -> str:
    """渲染两张 Markdown 表：

    1. 成功矩阵：行=控制器、列=车型，每格 "成功率 ±CI (n=..)"（Wilson 95%）；
    2. 明细表：每行一个 (控制器, 车型)，给碰撞率 / 超时率 / 平均步数 /
       平均回报 / 回合内最小障碍净空。
    """
    triples = _normalise_rows(rows)
    controllers: list[str] = []
    chassis_names: list[str] = []
    table: dict[tuple[str, str], EvalResult] = {}
    for ctrl, chassis, result in triples:
        if ctrl not in controllers:
            controllers.append(ctrl)
        if chassis not in chassis_names:
            chassis_names.append(chassis)
        table[(ctrl, chassis)] = result

    lines: list[str] = [f"**{title}**", ""]
    header = "| controller | " + " | ".join(chassis_names) + " |"
    sep = "| --- |" + " --- |" * len(chassis_names)
    lines.append(header)
    lines.append(sep)
    for ctrl in controllers:
        cells = [
            _success_cell(table[(ctrl, chassis)]) if (ctrl, chassis) in table else "—"
            for chassis in chassis_names
        ]
        lines.append("| " + ctrl + " | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "| controller | chassis | 碰撞率 | 超时率 | 平均步数 | 平均回报 | 最小净空 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ctrl, chassis, r in triples:
        clearance = (
            "—"
            if r.min_obstacle_clearance is None
            else f"{r.min_obstacle_clearance:.3f} m"
        )
        lines.append(
            f"| {ctrl} | {chassis} | {r.collision_rate:.1%} | {r.timeout_rate:.1%} "
            f"| {r.mean_steps:.1f} | {r.mean_return:.2f} | {clearance} |"
        )

    fixed_rows = [f"{c} × {k}" for c, k, r in triples if _fixed_map(r)]
    if fixed_rows:
        lines.append("")
        lines.append(
            "⚠ **固定地图行**（" + "、".join(fixed_rows) + "）：`reset_options` 传的是固定 "
            "Mapping ⇒ n 个回合是**同一张地图**重放 n 次，Wilson 区间只反映这张图上的重放"
            "方差，分母不是 n 个独立样本，**不可引用**做控制器对比。要独立样本请传 "
            "`callable(episode_index)`（例 `scenario_reset_options_factory(...)`）。"
        )
    unknown_rows = [f"{c} × {k}" for c, k, r in triples if r.n > 1 and not r.distinct_maps]
    if unknown_rows:
        lines.append("")
        lines.append(
            "❔ **地图未知行**（" + "、".join(unknown_rows) + "）：回合没有留下地图指纹，"
            "无法自证 n 个回合是否各跑一张不同的图 ⇒ 区间同样不该按独立样本引用。"
        )
    return "\n".join(lines)
