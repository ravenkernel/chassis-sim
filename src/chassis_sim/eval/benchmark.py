"""基准协议层：策略 × 场景 × seed 的可引用成功率表（`evaluate()` 之上的新层）。

设计要点
--------
* **独立样本是资格闸门，不是脚注**：`BenchmarkCell.success_ci` /
  `BenchmarkSummary.success_ci` 只在 `maps_independent=True` 时返回数字，否则
  返回 `None`（表格里写 —）。判据与 `EvalResult` 同源（每回合的
  `map_fingerprint` 全非空且两两不同），但本层更严：n=1 且没有指纹也不给
  区间（`EvalResult.maps_independent` 对 n<=1 有一句「无从证伪」的短路）。
  宁可少一个数字，也不要一个分母不是 n 的 Wilson 区间。
* **地图种子是纯函数**：`map_seed = seed * episodes + 回合号`。同一
  (场景, seed) 的每张地图在**调用那一刻**生成一次，四个策略共用（配对比较：
  格子之间的差异不是地图差异，也省掉 3 倍地图生成）；不同 seed 的种子区间
  不重叠（起点 seed*episodes）⇒ 跨 seed 汇总时 n 是真样本数。
* **策略只认 `act(observation)`**：与 `evaluate()` 的控制器协议一致。zero /
  random 直接吃 `env.action_space`，不复制一份物理界；pp / dwa 走
  `make_controller(..., env=env)`（控制器与 env 同标定，`evaluate()` 的一致性
  护栏才不炸）。
* **协议要 >= 3 个 seed**：只给 1-2 个 seed 时打一条 `[WARN][eval]` 并把
  `protocol_ok` 置 False（表格照出，但不该被引用为发布数字）。
* **`evaluate()` 不动**：本模块只调它，签名与返回结构保持原样。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from chassis_sim.controllers import make_controller
from chassis_sim.eval import (
    EpisodeMetrics,
    EvalResult,
    _warn,
    evaluate,
    wilson_interval,
)

__all__ = [
    "BASELINES",
    "DEFAULT_EPISODES",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SCENARIO",
    "DEFAULT_SEEDS",
    "MIN_SEEDS",
    "POLICY_KINDS",
    "BenchmarkCell",
    "BenchmarkResult",
    "BenchmarkSummary",
    "PolicySpec",
    "RandomPolicy",
    "ZeroPolicy",
    "benchmark",
    "benchmark_cell",
    "make_policy",
]

#: 默认场景：core 里写死的退化直线 —— 地图不随 seed 变，正好当独立样本闸门的
#: 反面教材（默认配置必须明确报 maps_independent=False）。
DEFAULT_SCENARIO = "straight"
#: 协议要求的 seed 数下限（1-2 个 seed 的「成功率」只是抽样噪声）
MIN_SEEDS = 3
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
#: 每格回合数（CLI 与 API 的默认；越多区间越窄，20 回合 ≈ ±20%）
DEFAULT_EPISODES = 5
#: 单回合步数上限：不会到达目标的基线不该烧掉 env.max_steps（1000）步
DEFAULT_MAX_STEPS = 200
#: 内置基线名（顺序就是表格里的顺序）
POLICY_KINDS: tuple[str, ...] = ("zero", "random", "pp", "dwa")


class ZeroPolicy:
    """基线：全零动作（什么都不做，成功率为 0 的下界）。"""

    def __init__(self, action_space: Any, *, name: str = "zero") -> None:
        self.action_space = action_space
        self.name = name

    def reset(self) -> None:
        """无状态：回合之间不需要复位（evaluate 会调，得有这个方法）。"""

    def act(self, observation: Any = None) -> np.ndarray:
        """返回动作空间原点（不是 low：各车型的 0 都表示「不动」）。"""
        shape = getattr(self.action_space, "shape", (2,))
        dtype = getattr(self.action_space, "dtype", np.float32)
        return np.zeros(shape, dtype=dtype)


class RandomPolicy:
    """基线：在 `env.action_space` 上均匀采样（不看观测）。

    `reset()` 换一条随机流（`default_rng([seed, 复位次数])`）：同一格每次重跑
    逐位可复现，但同一格的第 k 个回合不是第 1 个回合的重放。
    """

    def __init__(
        self, action_space: Any, *, seed: int = 0, name: str = "random"
    ) -> None:
        self.action_space = action_space
        self.name = name
        self.seed = int(seed)
        self._resets = 0
        self._rng = np.random.default_rng([self.seed, self._resets])

    def reset(self) -> None:
        self._resets += 1
        self._rng = np.random.default_rng([self.seed, self._resets])

    def act(self, observation: Any = None) -> np.ndarray:
        low = np.asarray(self.action_space.low, dtype=np.float64)
        high = np.asarray(self.action_space.high, dtype=np.float64)
        dtype = getattr(self.action_space, "dtype", np.float32)
        return self._rng.uniform(low, high).astype(dtype)


@dataclass(frozen=True)
class PolicySpec:
    """一条被测策略：进表的 `name` + `factory(env, seed) -> 策略对象`。"""

    name: str
    factory: Callable[[Any, int], Any]

    def build(self, env: Any, *, seed: int = 0) -> Any:
        return self.factory(env, int(seed))


def _zero_factory(env: Any, seed: int) -> ZeroPolicy:
    return ZeroPolicy(env.action_space)


def _random_factory(env: Any, seed: int) -> RandomPolicy:
    return RandomPolicy(env.action_space, seed=seed)


def _pure_pursuit_factory(env: Any, seed: int) -> Any:
    return make_controller("pure_pursuit", env=env)


def _dwa_factory(env: Any, seed: int) -> Any:
    return make_controller("dwa", env=env)


_BUILTIN: dict[str, PolicySpec] = {
    "zero": PolicySpec("zero", _zero_factory),
    "random": PolicySpec("random", _random_factory),
    "pp": PolicySpec("pp", _pure_pursuit_factory),
    "dwa": PolicySpec("dwa", _dwa_factory),
}

#: 四个内置基线（可直接展开进自定义列表：`benchmark([*BASELINES, mine])`）
BASELINES: tuple[PolicySpec, ...] = tuple(_BUILTIN[name] for name in POLICY_KINDS)


def _resolve_policy(item: str | PolicySpec | Callable[[Any, int], Any]) -> PolicySpec:
    """名字 / PolicySpec / 工厂函数 → PolicySpec（未知名字 ⇒ ValueError + 可用值）。"""
    if isinstance(item, PolicySpec):
        return item
    if isinstance(item, str):
        spec = _BUILTIN.get(item)
        if spec is None:
            raise ValueError(
                f"未知策略 {item!r}；内置基线：{', '.join(POLICY_KINDS)}"
                "（也可以传 PolicySpec 或 (env, seed) -> 策略对象 的工厂）"
            )
        return spec
    if callable(item):
        return PolicySpec(getattr(item, "__name__", "policy"), item)
    raise TypeError(
        "策略只能是内置名字、PolicySpec 或 (env, seed) -> 策略对象 的工厂，"
        f"收到 {type(item).__name__}"
    )


def make_policy(
    item: str | PolicySpec | Callable[[Any, int], Any],
    env: Any,
    *,
    seed: int = 0,
) -> Any:
    """按名字/规格造一条策略（四个基线可以这样单独调用）。"""
    return _resolve_policy(item).build(env, seed=seed)


# ── 统计（格子与汇总共用同一套判据）───────────────────────────────
class _SampleStats:
    """回合统计 + 独立样本闸门；`BenchmarkCell` / `BenchmarkSummary` 共用。

    子类只负责把 `episodes` 交出来（格子委托给 `EvalResult`，汇总是跨 seed
    拼接后的元组）。
    """

    episodes: tuple[EpisodeMetrics, ...]

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
        return sum(e.steps for e in self.episodes) / self.n if self.n else 0.0

    @property
    def mean_return(self) -> float:
        return sum(e.episode_return for e in self.episodes) / self.n if self.n else 0.0

    # ── 独立样本闸门 ─────────────────────────────────────────────
    @property
    def map_fingerprints(self) -> tuple[str, ...]:
        return tuple(e.map_fingerprint for e in self.episodes)

    @property
    def distinct_maps(self) -> int:
        """实际看到的不同地图数（空指纹不计：未知不等于一张新图）。"""
        return len({fp for fp in self.map_fingerprints if fp})

    @property
    def maps_independent(self) -> bool:
        """n 个回合是否各自跑在一张不同的、可辨认的地图上。

        比 `EvalResult.maps_independent` 更严：没有指纹（env 既不给障碍也不给
        路径）时即使 n=1 也判 False —— 没有证据就不发区间。
        """
        fingerprints = self.map_fingerprints
        return (
            bool(fingerprints)
            and "" not in fingerprints
            and len(set(fingerprints)) == len(fingerprints)
        )

    @property
    def success_interval(self) -> tuple[float, float] | None:
        """Wilson 95% 区间；**非独立样本时是 None**（不是 0 宽度的假区间）。"""
        if not self.maps_independent:
            return None
        return wilson_interval(self.successes, self.n)

    @property
    def success_ci(self) -> float | None:
        """区间半宽（与 `EvalResult.success_ci` 同定义）；非独立样本时 None。"""
        interval = self.success_interval
        if interval is None:
            return None
        low, high = interval
        rate = self.success_rate
        return max(rate - low, high - rate)

    def stats_dict(self) -> dict[str, Any]:
        interval = self.success_interval
        return {
            "n": self.n,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "success_ci": self.success_ci,
            "success_ci_low": None if interval is None else interval[0],
            "success_ci_high": None if interval is None else interval[1],
            "collisions": self.collisions,
            "collision_rate": self.collision_rate,
            "timeouts": self.timeouts,
            "timeout_rate": self.timeout_rate,
            "mean_steps": self.mean_steps,
            "mean_return": self.mean_return,
            "distinct_maps": self.distinct_maps,
            "maps_independent": self.maps_independent,
        }


@dataclass(frozen=True)
class BenchmarkCell(_SampleStats):
    """一个 (策略, 场景, seed) 格子：`evaluate()` 的原始结果 + 它来自哪次运行。"""

    policy: str
    scenario: str
    seed: int
    result: EvalResult
    #: 本格第 0 个回合的地图种子（`seed * episodes`）；不知道就是 None
    map_seed_base: int | None = None

    @property
    def episodes(self) -> tuple[EpisodeMetrics, ...]:
        return self.result.episodes

    @property
    def map_seeds(self) -> tuple[int, ...]:
        """每回合实际用的地图种子（纯函数，可复算）；不知道时是空元组。"""
        if self.map_seed_base is None:
            return ()
        return tuple(self.map_seed_base + i for i in range(self.n))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "scenario": self.scenario,
            "seed": int(self.seed),
            "map_seed_base": self.map_seed_base,
            "map_seeds": list(self.map_seeds),
            **self.stats_dict(),
            "map_fingerprints": list(self.map_fingerprints),
            "episodes": [e.to_dict() for e in self.episodes],
        }


@dataclass(frozen=True)
class BenchmarkSummary(_SampleStats):
    """一个 (策略, 场景) 跨 seed 合并后的汇总（每个 seed 一格）。"""

    policy: str
    scenario: str
    episodes: tuple[EpisodeMetrics, ...]
    seeds: tuple[int, ...] = ()
    cells: tuple[BenchmarkCell, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "scenario": self.scenario,
            "seeds": list(self.seeds),
            **self.stats_dict(),
            "map_fingerprints": list(self.map_fingerprints),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """一次 benchmark 的全部格子 + 汇总 + 协议自证（`to_markdown()` 出表）。"""

    cells: tuple[BenchmarkCell, ...]
    policies: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    episodes: int = 0
    chassis: str = "ackermann"
    #: seeds >= MIN_SEEDS（少于 3 个 seed 的表格不该被引用为发布数字）
    protocol_ok: bool = True
    #: `write()` / `benchmark(out=...)` 落盘的路径（空 = 没写文件）
    markdown_path: str = ""
    #: 被控对象旋钮（None = 该旋钮未设置）；和策略、场景一样属于复现这张表的上下文
    dynamics: str | None = None
    sensor_noise_std: float | None = None
    sensor_latency_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cells, tuple):
            object.__setattr__(self, "cells", tuple(self.cells))

    @property
    def plant(self) -> dict[str, Any]:
        """被控对象旋钮：复现一次 benchmark 需要的最小上下文。"""
        return {
            "dynamics": self.dynamics,
            "sensor_noise_std": self.sensor_noise_std,
            "sensor_latency_steps": self.sensor_latency_steps,
        }

    def _plant_label(self) -> str:
        parts = [
            f"dynamics `{self.dynamics}`" if self.dynamics else "dynamics 用 env 默认"
        ]
        if self.sensor_noise_std is not None:
            parts.append(f"LiDAR 噪声 {self.sensor_noise_std} m")
        if self.sensor_latency_steps is not None:
            parts.append(f"LiDAR 延迟 {self.sensor_latency_steps} 步")
        return " · ".join(parts)

    # ── 取格子 / 汇总 ────────────────────────────────────────────
    def cell(self, policy: str, scenario: str, seed: int) -> BenchmarkCell:
        """按 (策略, 场景, seed) 取那一格；没有就是 KeyError（不返回 None 装傻）。"""
        for candidate in self.cells:
            if (
                candidate.policy == policy
                and candidate.scenario == scenario
                and candidate.seed == int(seed)
            ):
                return candidate
        raise KeyError(f"没有这一格：{policy} × {scenario} × seed={seed}")

    @property
    def summaries(self) -> tuple[BenchmarkSummary, ...]:
        """按 (策略, 场景) 分组、跨 seed 合并（顺序 = 首次出现顺序）。"""
        grouped: dict[tuple[str, str], list[BenchmarkCell]] = {}
        for cell in self.cells:
            grouped.setdefault((cell.policy, cell.scenario), []).append(cell)
        summaries: list[BenchmarkSummary] = []
        for (policy, scenario), group in grouped.items():
            summaries.append(
                BenchmarkSummary(
                    policy=policy,
                    scenario=scenario,
                    episodes=tuple(e for cell in group for e in cell.episodes),
                    seeds=tuple(cell.seed for cell in group),
                    cells=tuple(group),
                )
            )
        return tuple(summaries)

    # ── 序列化 ───────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "chassis": self.chassis,
            "episodes": int(self.episodes),
            "seeds": list(self.seeds),
            "policies": list(self.policies),
            "scenarios": list(self.scenarios),
            "protocol_ok": bool(self.protocol_ok),
            "markdown_path": self.markdown_path,
            "plant": self.plant,
            "cells": [cell.to_dict() for cell in self.cells],
            "summary": [summary.to_dict() for summary in self.summaries],
        }

    def to_markdown(
        self, *, title: str = "chassis-sim · benchmark（策略 × 场景 × seed）"
    ) -> str:
        """渲染逐格表 + 跨 seed 汇总表 + 非独立样本脚注。"""
        lines: list[str] = [f"**{title}**", ""]
        lines.append(
            f"车型 `{self.chassis}` · 每格 {self.episodes} 回合 · "
            f"seeds `{list(self.seeds)}` · 场景 `{list(self.scenarios)}`"
        )
        lines.append(f"被控对象 · {self._plant_label()}")
        lines.append("")
        lines.append(
            "地图种子 = `seed × episodes + 回合号`；同一 (场景, seed) 的地图在全部"
            "策略间共享（配对比较）。`success_ci` 是 Wilson 95% 区间半宽，"
            "**只在 `maps_independent=True` 时给数字**。"
        )
        lines.append("")
        lines.append("#### 逐格")
        lines.append("")
        lines.append(
            "| 策略 | 场景 | seed | 地图种子 | n | success | success_ci "
            "| collision | steps | distinct_maps | maps_independent |"
        )
        lines.append(
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        for cell in self.cells:
            lines.append("| " + " | ".join(_cell_row(cell)) + " |")
        lines.append("")
        lines.append("#### 汇总（跨 seed 合并）")
        lines.append("")
        lines.append(
            "| 策略 | 场景 | n | successes | success | success_ci | collision "
            "| steps | distinct_maps | maps_independent |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        for summary in self.summaries:
            lines.append("| " + " | ".join(_summary_row(summary)) + " |")
        lines.extend(self._notes())
        return "\n".join(lines)

    def _notes(self) -> list[str]:
        notes: list[str] = []
        summaries = self.summaries
        replay = [s for s in summaries if s.distinct_maps and not s.maps_independent]
        unknown = [s for s in summaries if not s.distinct_maps]
        if replay:
            notes.append("")
            notes.append(
                "⚠ **非独立样本行**（"
                + "、".join(_tag(s) for s in replay)
                + "）：`distinct_maps` 小于 `n` —— n 个回合不是 n 张不同的地图，"
                "`success_ci` 置 **None**（不给数字）；这些行只能读作「同一张图上的重放」。"
            )
        if unknown:
            notes.append("")
            notes.append(
                "❔ **地图未知行**（"
                + "、".join(_tag(s) for s in unknown)
                + "）：回合没有留下地图指纹，无法自证 n 个回合是否各跑一张不同的图 ⇒ "
                "同样不给 `success_ci`。"
            )
        if not self.protocol_ok:
            notes.append("")
            notes.append(
                f"⚠ **协议未满**：seeds 只有 {len(self.seeds)} 个（< {MIN_SEEDS}）—— "
                "这些成功率还不该被引用为发布数字。"
            )
        return notes

    def write(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        encoding: str = "utf-8",
    ) -> Path:
        """把 Markdown 表写到 `path`（父目录自动创建），返回落盘路径。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_markdown() if title is None else self.to_markdown(title=title)
        target.write_text(text + "\n", encoding=encoding)
        return target


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ci_cell(stats: _SampleStats) -> str:
    half = stats.success_ci
    return "—" if half is None else f"±{half * 100:.1f}%"


def _seed_span(seeds: tuple[int, ...]) -> str:
    if not seeds:
        return "—"
    if len(seeds) == 1:
        return str(seeds[0])
    return f"{seeds[0]}–{seeds[-1]}"


def _flag(ok: bool) -> str:
    return "True" if ok else "**False**"


def _cell_row(cell: BenchmarkCell) -> list[str]:
    return [
        f"`{cell.policy}`",
        cell.scenario,
        str(cell.seed),
        _seed_span(cell.map_seeds),
        str(cell.n),
        _pct(cell.success_rate),
        _ci_cell(cell),
        str(cell.collisions),
        f"{cell.mean_steps:.1f}",
        str(cell.distinct_maps),
        _flag(cell.maps_independent),
    ]


def _summary_row(summary: BenchmarkSummary) -> list[str]:
    return [
        f"`{summary.policy}`",
        summary.scenario,
        str(summary.n),
        str(summary.successes),
        _pct(summary.success_rate),
        _ci_cell(summary),
        str(summary.collisions),
        f"{summary.mean_steps:.1f}",
        str(summary.distinct_maps),
        _flag(summary.maps_independent),
    ]


def _tag(summary: BenchmarkSummary) -> str:
    return f"`{summary.policy}` × `{summary.scenario}`"


# ── 输入归一化 ───────────────────────────────────────────────────
def _policy_specs(
    policies: Sequence[str | PolicySpec | Callable[[Any, int], Any]] | None,
) -> tuple[PolicySpec, ...]:
    items = list(BASELINES) if policies is None else list(policies)
    if not items:
        raise ValueError("policies 不能为空（省略 = 全部四个基线）")
    specs = tuple(_resolve_policy(item) for item in items)
    names = [spec.name for spec in specs]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"策略名重复（表里分不清是哪一条）：{duplicated}")
    return specs


def _scenario_label(scenario: Any) -> str:
    name = getattr(scenario, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(scenario)
    if text.endswith((".yaml", ".yml")):
        return Path(text).stem
    return text


def _scenario_list(scenarios: Any) -> list[Any]:
    if scenarios is None:
        return [DEFAULT_SCENARIO]
    if isinstance(scenarios, (str, Path)):
        return [scenarios]
    items = list(scenarios)
    if not items:
        raise ValueError("scenarios 不能为空（至少一个场景名或 YAML 路径）")
    return items


def _seed_list(seeds: Sequence[int]) -> tuple[int, ...]:
    try:
        values = tuple(int(seed) for seed in seeds)
    except (TypeError, ValueError):
        raise ValueError(f"seeds 必须是整数序列，收到 {seeds!r}") from None
    if not values:
        raise ValueError("seeds 不能为空（独立样本闸门至少要一个 seed）")
    if len(set(values)) != len(values):
        raise ValueError(f"seeds 不能重复（重复等于把同一批地图再跑一遍）：{values}")
    if any(value < 0 for value in values):
        raise ValueError(f"seeds 必须是非负整数：{values}")
    return values


def _episode_count(episodes: int) -> int:
    count = int(episodes)
    if count < 1:
        raise ValueError(f"episodes 必须是正整数，收到 {episodes!r}")
    return count


def _env_scenario_arg(scenario: Any) -> str | None:
    if isinstance(scenario, (str, Path)):
        return str(scenario)
    name = getattr(scenario, "name", None)
    return str(name) if name else None


#: dynamics 的合法取值（与 EnvConfig 同源，早失败好过第一个 reset 才炸）
DYNAMICS_KINDS: tuple[str, ...] = ("none", "limits", "wheel")

#: 传感器旋钮打开时装的 LiDAR：前向 + 后向各 60 线，参数与默认场景一致
_BENCH_LIDAR_RAYS = 60


def _bench_sensors(
    noise_std: float | None, latency_steps: int | None
) -> tuple[Any, ...] | None:
    """传感器旋钮 → 前/后两支 Lidar；两个旋钮都没给就返回 None（不挂传感器）。"""
    if noise_std is None and latency_steps is None:
        return None
    from chassis_sim.sensors import Lidar

    extra: dict[str, Any] = {}
    if noise_std is not None:
        extra["noise_std"] = float(noise_std)
    if latency_steps is not None:
        extra["latency_steps"] = int(latency_steps)
    return (
        Lidar(
            name="front",
            mount=(0.12, 0.0, 0.0),
            fov=float(np.pi),
            n_rays=_BENCH_LIDAR_RAYS,
            **extra,
        ),
        Lidar(
            name="rear",
            mount=(-0.12, 0.0, float(np.pi)),
            fov=float(np.pi),
            n_rays=_BENCH_LIDAR_RAYS,
            **extra,
        ),
    )


def _check_plant(
    dynamics: str | None,
    sensor_noise_std: float | None,
    sensor_latency_steps: int | None,
) -> None:
    """三个工厂旋钮的取值在开跑前就校验完。"""
    if dynamics is not None and dynamics not in DYNAMICS_KINDS:
        raise ValueError(f"dynamics 只能是 {list(DYNAMICS_KINDS)}，收到 {dynamics!r}")
    if sensor_noise_std is not None and float(sensor_noise_std) < 0.0:
        raise ValueError(f"sensor_noise_std 必须 >= 0，收到 {sensor_noise_std!r}")
    if sensor_latency_steps is not None and int(sensor_latency_steps) < 0:
        raise ValueError(
            f"sensor_latency_steps 必须 >= 0，收到 {sensor_latency_steps!r}"
        )


def _default_env_factory(
    chassis: str,
    scenario: Any,
    *,
    dynamics: str | None = None,
    sensor_noise_std: float | None = None,
    sensor_latency_steps: int | None = None,
) -> Callable[[int], Any]:
    """默认 env 工厂：每次调用造一台**新的** env（benchmark 负责 close）。

    三个旋钮只改这台 env 的**被控对象**（执行器模型与传感器退化），不动策略、
    不动地图，因此同一 seed 的对照实验可以直接读差异。
    """
    target = _env_scenario_arg(scenario)
    sensors = _bench_sensors(sensor_noise_std, sensor_latency_steps)

    def factory(seed: int) -> Any:
        from chassis_sim.api import make

        kwargs: dict[str, Any] = {}
        if dynamics is not None:
            kwargs["dynamics"] = dynamics
        if sensors is not None:
            kwargs["sensors"] = sensors
        return make(chassis, scenario=target, seed=int(seed), **kwargs)

    return factory


def _scenario_options_factory(scenario: Any, seed: int, episodes: int) -> Any:
    """默认每回合重掷地图：地图种子 = `seed * episodes + 回合号`。"""
    from chassis_sim.scenarios import scenario_reset_options_factory

    return scenario_reset_options_factory(scenario, base_seed=int(seed) * episodes)


def _memo(provider: Callable[[int], Any]) -> Callable[[int], Any]:
    """同一 (场景, seed) 的地图只生成一次 —— 四个策略共用同一批图。"""
    cache: dict[int, Any] = {}

    def cached(index: int) -> Any:
        key = int(index)
        if key not in cache:
            cache[key] = provider(key)
        return cache[key]

    return cached


def _check_scenario_loads(scenario: Any) -> None:
    """场景现在就得能加载（不要等第一个 reset 才炸）。"""
    from chassis_sim.scenarios import load_scenario

    load_scenario(scenario)


# ── 跑格子 ───────────────────────────────────────────────────────
def _run_cell(
    spec: PolicySpec,
    scenario_label: str,
    seed: int,
    *,
    episodes: int,
    chassis: str,
    max_steps: int | None,
    env_factory: Callable[[int], Any],
    reset_options: Any,
    map_seed_base: int | None,
    allow_mismatch: bool,
) -> BenchmarkCell:
    """一台 env、一条策略、一个 seed：跑完就 close（env 的生命周期不外泄）。"""
    env = env_factory(seed)
    try:
        policy = spec.build(env, seed=seed)
        result = evaluate(
            env,
            policy,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
            reset_options=reset_options,
            labels=(spec.name, chassis),
            allow_mismatch=allow_mismatch,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return BenchmarkCell(
        policy=spec.name,
        scenario=scenario_label,
        seed=int(seed),
        result=result,
        map_seed_base=map_seed_base,
    )


def benchmark_cell(
    policy: str | PolicySpec | Callable[[Any, int], Any],
    scenario: Any = DEFAULT_SCENARIO,
    *,
    seed: int = 0,
    episodes: int = DEFAULT_EPISODES,
    chassis: str = "ackermann",
    max_steps: int | None = DEFAULT_MAX_STEPS,
    dynamics: str | None = None,
    sensor_noise_std: float | None = None,
    sensor_latency_steps: int | None = None,
    env_factory: Callable[[int], Any] | None = None,
    reset_options: Any = None,
    allow_mismatch: bool = False,
) -> BenchmarkCell:
    """跑一格：单策略 × 单场景 × 单 seed（排查与单元测试的最小入口）。

    `reset_options` 原样交给 `evaluate()`（要每回合重掷地图就传
    `callable(episode_index)`）；地图种子未知 ⇒ `map_seed_base=None`，Markdown
    那一列写 —。
    """
    spec = _resolve_policy(policy)
    count = _episode_count(episodes)
    _check_plant(dynamics, sensor_noise_std, sensor_latency_steps)
    return _run_cell(
        spec,
        _scenario_label(scenario),
        int(seed),
        episodes=count,
        chassis=chassis,
        max_steps=max_steps,
        env_factory=env_factory
        or _default_env_factory(
            chassis,
            scenario,
            dynamics=dynamics,
            sensor_noise_std=sensor_noise_std,
            sensor_latency_steps=sensor_latency_steps,
        ),
        reset_options=reset_options,
        map_seed_base=None,
        allow_mismatch=allow_mismatch,
    )


def benchmark(
    policies: Sequence[str | PolicySpec | Callable[[Any, int], Any]] | None = None,
    scenarios: Sequence[Any] | str | Path | None = None,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    episodes: int = DEFAULT_EPISODES,
    chassis: str = "ackermann",
    max_steps: int | None = DEFAULT_MAX_STEPS,
    dynamics: str | None = None,
    sensor_noise_std: float | None = None,
    sensor_latency_steps: int | None = None,
    env_factory: Callable[[int], Any] | None = None,
    reset_options_factory: Callable[[Any, int], Callable[[int], Any]] | None = None,
    allow_mismatch: bool = False,
    out: str | Path | None = None,
    title: str = "",
) -> BenchmarkResult:
    """在 held-out 场景 × 多个 seed 上跑策略表。

    Args:
        policies: 策略列表（名字 / PolicySpec / `(env, seed) -> 策略对象`）；
            省略 = 四个内置基线 zero/random/pp/dwa。
        scenarios: 场景名或 YAML 路径列表；省略 = `straight`（退化固定直线）。
        seeds: seed 列表（非负、不重复）。`len(seeds) < 3` 时打
            `[WARN][eval]` 并把 `protocol_ok` 置 False（不抛异常）。
        episodes: 每格回合数（>= 1）。
        chassis: 车型（ackermann / diff_drive / mecanum ...）。
        max_steps: 单回合步数上限（None = 用 env 自己的 max_steps）。
        dynamics: 执行器模型 none/limits/wheel；省略 = env 默认（纯运动学）。
        sensor_noise_std: LiDAR 测距噪声标准差（米）；给了就挂前/后两支 Lidar。
            **只有读 scan 的策略才会受影响**：pp/dwa 读的是路径几何，噪声对它们
            没有作用（表格里看不出差异是正确行为，不是旋钮失效）。
        sensor_latency_steps: LiDAR 延迟步数（>= 0）；同样只在有策略读 scan 时生效。
        env_factory: `seed -> env`；省略时按 (chassis, scenario) 造新 env。
            工厂**每次调用必须返回一台新 env**（benchmark 负责 close）。
        reset_options_factory: `(scenario, seed) -> callable(episode_index) -> options`；
            省略时用 `scenario_reset_options_factory(..., base_seed=seed*episodes)`。
        allow_mismatch: 透传给 `evaluate()`（控制器与底盘标定不一致时只告警）。
        out: 非空时把 Markdown 表写到这个路径（父目录自动创建）。
        title: 表标题（空 = 默认标题）。

    Returns:
        `BenchmarkResult`：逐格 `cells` + 跨 seed 的 `summaries`，以及
        `protocol_ok` / `markdown_path` 两项自证。

    Raises:
        ValueError: 空策略/空场景/空 seeds、重复 seed、重复策略名、episodes < 1。
    """
    specs = _policy_specs(policies)
    scenario_items = _scenario_list(scenarios)
    seed_values = _seed_list(seeds)
    count = _episode_count(episodes)
    _check_plant(dynamics, sensor_noise_std, sensor_latency_steps)

    protocol_ok = len(seed_values) >= MIN_SEEDS
    if not protocol_ok:
        _warn(
            f"benchmark：seeds 只有 {len(seed_values)} 个（协议要求 >= {MIN_SEEDS}）"
            "；表格照出，但 protocol_ok=False，成功率不该被引用为发布数字"
        )

    cells: list[BenchmarkCell] = []
    for scenario in scenario_items:
        label = _scenario_label(scenario)
        if reset_options_factory is None:
            _check_scenario_loads(scenario)
        make_env = env_factory or _default_env_factory(
            chassis,
            scenario,
            dynamics=dynamics,
            sensor_noise_std=sensor_noise_std,
            sensor_latency_steps=sensor_latency_steps,
        )
        for seed in seed_values:
            if reset_options_factory is None:
                provider = _scenario_options_factory(scenario, seed, count)
            else:
                provider = reset_options_factory(scenario, seed)
            reset_options = _memo(provider)
            for spec in specs:
                cells.append(
                    _run_cell(
                        spec,
                        label,
                        seed,
                        episodes=count,
                        chassis=chassis,
                        max_steps=max_steps,
                        env_factory=make_env,
                        reset_options=reset_options,
                        map_seed_base=seed * count,
                        allow_mismatch=allow_mismatch,
                    )
                )

    result = BenchmarkResult(
        cells=tuple(cells),
        policies=tuple(spec.name for spec in specs),
        scenarios=tuple(_scenario_label(item) for item in scenario_items),
        seeds=seed_values,
        episodes=count,
        chassis=chassis,
        protocol_ok=protocol_ok,
        dynamics=dynamics,
        sensor_noise_std=sensor_noise_std,
        sensor_latency_steps=sensor_latency_steps,
    )
    if out is not None:
        written = result.write(out) if not title else result.write(out, title=title)
        result = replace(result, markdown_path=str(written))
    return result
