"""场景 YAML 的加载与**严格**校验。

YAML schema（硬契约；任何偏离都抛 :class:`ScenarioValidationError`，消息里必带
**文件路径**与**键名**）::

    name: str                                # 必填；data/ 下按名字加载时须 == 文件名 stem
    description: str                         # 必填
    world_bounds: [xmin, xmax, ymin, ymax]   # 必填；xmin < xmax、ymin < ymax
    obstacles:                               # 必填，可为 []（planning 场景写 []）
      - type: circle                         # x, y, radius(>0), theta(可选, 默认 0)
      - type: rectangle                      # x, y, length(>0), width(>0), theta(可选)
    path:                                    # 必填
      type: straight | waypoints | stage
      params:                                # 按 type 取下列之一
        straight:  {start: [x, y(, theta)], goal: [x, y(, theta)], spacing: > 0}
        waypoints: {points: [[x, y(, theta)], ...], spacing: > 0}   # >=2 点，列数一致
        stage:     {stage: <curriculum.yaml 里存在的 stage 名>}
    dynamic_obstacles:                       # 可选，默认 []；逐项按 type 分派
      - type: linear                         # 匀速直线
        start: [x, y]                        #   必填，t=0 时的圆心（米）
        velocity: [vx, vy]                   #   必填，米/秒；有限值（可为 0）
        radius: > 0                          #   必填，本体半径（米）
        period: >= 0                         #   可选，默认 0 = 不循环；> 0 = 周期（秒）
        pingpong: bool                       #   可选，默认 false；需 period > 0
                                             #   false=锯齿 / true=三角波折返
      - type: circular                       # 匀速圆周
        center: [x, y]                       #   必填，轨道圆心（米）
        radius_orbit: > 0                    #   必填，**轨道半径**（米）
        omega: rad/s                         #   必填，角速度；有限值（负=顺时针）
        phase: rad                           #   可选，默认 0；t=0 的初相位
        radius: > 0                          #   必填，**本体半径**（米）
    seed: int                                # 必填
    requires: planning | null                # 可选，默认 null；planning ⇔ path.type == stage

未知键一律报错，并用 `difflib.get_close_matches` 给出 `did you mean ...` 提示。

两条由本模块强制的**跨字段**规则：

1. `requires: planning` ⇔ `path.type: stage`（planning 场景的地图必须由
   MapGenerator 现场生成，YAML 不重复写障碍与路径点）；
2. `dynamic_obstacles` 的 `radius` / `radius_orbit` 必须 > 0，`omega` / `velocity` /
   `phase` 必须是**有限**数值（`inf`/`nan` 一律拒绝）；`pingpong: true` 需要
   `period > 0`（`period == 0` 表示「不循环」，没有可折返的周期）。
"""

from __future__ import annotations

import difflib
import importlib.resources
import math
import os
from pathlib import Path
from typing import Any

import yaml

from chassis_sim.scenarios.scenario import (
    CircleObstacle,
    CircularDynamicObstacle,
    DynamicObstacle,
    LinearDynamicObstacle,
    PathSpec,
    RectangleObstacle,
    Scenario,
    ScenarioError,
    ScenarioNotFoundError,
    ScenarioValidationError,
)

__all__ = [
    "DATA_DIR",
    "curriculum_stages",
    "list_scenarios",
    "load_curriculum",
    "load_scenario",
]

#: 场景 YAML 目录（随 wheel 以 package-data 发布）。
DATA_DIR = Path(__file__).resolve().parent / "data"

_SUFFIXES = (".yaml", ".yml")

_TOP_KEYS = (
    "name",
    "description",
    "world_bounds",
    "obstacles",
    "path",
    "dynamic_obstacles",
    "seed",
    "requires",
)
_TOP_REQUIRED = ("name", "description", "world_bounds", "obstacles", "path", "seed")

_OBSTACLE_KEYS = {
    "circle": ("type", "x", "y", "radius", "theta"),
    "rectangle": ("type", "x", "y", "length", "width", "theta"),
}

_PATH_PARAMS = {
    "straight": ("start", "goal", "spacing"),
    "waypoints": ("points", "spacing"),
    "stage": ("stage",),
}

_DYNAMIC_KEYS = {
    "linear": ("type", "start", "velocity", "radius", "period", "pingpong"),
    "circular": ("type", "center", "radius_orbit", "omega", "phase", "radius"),
}

_CURRICULUM_RESOURCE = "curriculum.yaml"


# ═════════════════════════════════════════════════════════════════════
# 课程配置（**不 import chassis_sim.planning**：core 安装没有 numba 也要能校验
# planning 场景的 stage 名，且 planning 内部对 numba 缺失是恒等降级，import 本
# 身不会失败——所以 numba 守卫只能落在 reset_options 的生成分支里）
# ═════════════════════════════════════════════════════════════════════
def load_curriculum() -> dict[str, Any]:
    """读包内 `config/curriculum.yaml`（与 planning 同一资源、同一 schema）。"""
    resource = importlib.resources.files("chassis_sim.config").joinpath(
        _CURRICULUM_RESOURCE
    )
    return yaml.safe_load(resource.read_text(encoding="utf-8"))


def curriculum_stages(curriculum: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """返回 curriculum 的 stage 列表（`None` 时现场读包内 YAML）。"""
    data = load_curriculum() if curriculum is None else curriculum
    stages = data.get("stages") or []
    return [stage for stage in stages if isinstance(stage, dict)]


def _stage_names(curriculum: dict[str, Any] | None = None) -> list[str]:
    return [str(stage.get("name")) for stage in curriculum_stages(curriculum)]


# ═════════════════════════════════════════════════════════════════════
# 基础校验原语
# ═════════════════════════════════════════════════════════════════════
def _prefix(path: Path, where: str) -> str:
    return f"{path}: {where}"


def _fail(path: Path, where: str, message: str) -> ScenarioValidationError:
    return ScenarioValidationError(f"{_prefix(path, where)}: {message}")


def _did_you_mean(value: str, allowed: tuple[str, ...] | list[str]) -> str:
    guess = difflib.get_close_matches(str(value), list(allowed), n=1, cutoff=0.5)
    return f" — did you mean '{guess[0]}'?" if guess else ""


def _check_mapping(value: Any, path: Path, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(path, where, f"期望映射（dict），实际是 {type(value).__name__}")
    return value


def _check_keys(
    mapping: dict[str, Any],
    allowed: tuple[str, ...],
    required: tuple[str, ...],
    path: Path,
    where: str,
) -> None:
    """未知键（带 did-you-mean）与缺必填键的统一检查。"""
    for key in mapping:
        if key in allowed:
            continue
        raise _fail(
            path,
            where,
            f"未知键 '{key}'{_did_you_mean(str(key), allowed)}"
            f" 允许的键: {', '.join(allowed)}",
        )
    for key in required:
        if key not in mapping:
            raise _fail(
                path,
                where,
                f"缺少必填键 '{key}'（允许的键: {', '.join(allowed)}）",
            )


def _sequence(value: Any, path: Path, where: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise _fail(path, where, f"期望列表，实际是 {type(value).__name__}")
    return list(value)


def _text(value: Any, path: Path, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, where, f"期望非空字符串，实际是 {value!r}")
    return value


def _number(value: Any, path: Path, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(
            path, where, f"期望数值，实际是 {type(value).__name__}（{value!r}）"
        )
    out = float(value)
    if not math.isfinite(out):
        raise _fail(path, where, f"必须是有限数值，实际是 {value!r}")
    return out


def _positive(value: Any, path: Path, where: str) -> float:
    out = _number(value, path, where)
    if out <= 0.0:
        raise _fail(path, where, f"必须 > 0，实际是 {out!r}")
    return out


def _integer(value: Any, path: Path, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(
            path, where, f"期望整数，实际是 {type(value).__name__}（{value!r}）"
        )
    return int(value)


# ═════════════════════════════════════════════════════════════════════
# 各字段解析
# ═════════════════════════════════════════════════════════════════════
def _parse_world_bounds(value: Any, path: Path) -> tuple[float, float, float, float]:
    seq = _sequence(value, path, "world_bounds")
    if len(seq) != 4:
        raise _fail(
            path,
            "world_bounds",
            f"必须正好 4 个数 [xmin, xmax, ymin, ymax]，实际 {len(seq)} 个: {seq!r}",
        )
    xmin, xmax, ymin, ymax = (
        _number(item, path, f"world_bounds[{index}]") for index, item in enumerate(seq)
    )
    if not xmin < xmax:
        raise _fail(
            path,
            "world_bounds",
            f"要求 xmin < xmax，实际 xmin={xmin!r} xmax={xmax!r}"
            "（顺序固定为 [xmin, xmax, ymin, ymax]）",
        )
    if not ymin < ymax:
        raise _fail(
            path,
            "world_bounds",
            f"要求 ymin < ymax，实际 ymin={ymin!r} ymax={ymax!r}"
            "（顺序固定为 [xmin, xmax, ymin, ymax]）",
        )
    return (xmin, xmax, ymin, ymax)


def _parse_point(value: Any, path: Path, where: str) -> tuple[float, ...]:
    seq = _sequence(value, path, where)
    if len(seq) not in (2, 3):
        raise _fail(
            path,
            where,
            f"路径点必须是 [x, y] 或 [x, y, theta]，实际 {len(seq)} 个: {seq!r}",
        )
    return tuple(
        _number(item, path, f"{where}[{index}]") for index, item in enumerate(seq)
    )


def _parse_obstacles(
    value: Any, path: Path
) -> tuple[CircleObstacle | RectangleObstacle, ...]:
    seq = _sequence(value, path, "obstacles")
    obstacles: list[CircleObstacle | RectangleObstacle] = []
    for index, item in enumerate(seq):
        where = f"obstacles[{index}]"
        mapping = _check_mapping(item, path, where)
        if "type" not in mapping:
            raise _fail(path, where, "缺少必填键 'type'（circle 或 rectangle）")
        kind = _text(mapping["type"], path, f"{where}.type")
        if kind not in _OBSTACLE_KEYS:
            raise _fail(
                path,
                f"{where}.type",
                f"未知障碍类型 '{kind}'{_did_you_mean(kind, tuple(_OBSTACLE_KEYS))}"
                f" 支持: {', '.join(_OBSTACLE_KEYS)}",
            )
        allowed = _OBSTACLE_KEYS[kind]
        required = (
            ("type", "x", "y", "radius")
            if kind == "circle"
            else ("type", "x", "y", "length", "width")
        )
        _check_keys(mapping, allowed, required, path, where)
        x = _number(mapping["x"], path, f"{where}.x")
        y = _number(mapping["y"], path, f"{where}.y")
        theta = _number(mapping.get("theta", 0.0), path, f"{where}.theta")
        if kind == "circle":
            radius = _positive(mapping["radius"], path, f"{where}.radius")
            obstacles.append(CircleObstacle(x=x, y=y, radius=radius, theta=theta))
        else:
            length = _positive(mapping["length"], path, f"{where}.length")
            width = _positive(mapping["width"], path, f"{where}.width")
            obstacles.append(
                RectangleObstacle(x=x, y=y, length=length, width=width, theta=theta)
            )
    return tuple(obstacles)


def _parse_path(value: Any, path: Path) -> PathSpec:
    mapping = _check_mapping(value, path, "path")
    _check_keys(mapping, ("type", "params"), ("type", "params"), path, "path")
    path_type = _text(mapping["type"], path, "path.type")
    if path_type not in _PATH_PARAMS:
        raise _fail(
            path,
            "path.type",
            f"未知路径类型 '{path_type}'"
            f"{_did_you_mean(path_type, tuple(_PATH_PARAMS))}"
            f" 支持: {', '.join(_PATH_PARAMS)}",
        )
    raw_params = _check_mapping(mapping["params"], path, "path.params")
    allowed = _PATH_PARAMS[path_type]
    _check_keys(raw_params, allowed, allowed, path, "path.params")

    if path_type == "straight":
        start = _parse_point(raw_params["start"], path, "path.params.start")
        goal = _parse_point(raw_params["goal"], path, "path.params.goal")
        if len(start) != len(goal):
            raise _fail(
                path,
                "path.params",
                f"start 与 goal 必须同为 [x, y] 或 [x, y, theta]，"
                f"实际 {len(start)} 列 vs {len(goal)} 列",
            )
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) <= 0.0:
            raise _fail(path, "path.params", "start 与 goal 重合，直线路径长度为 0")
        spacing = _positive(raw_params["spacing"], path, "path.params.spacing")
        return PathSpec(
            type="straight",
            params={"start": list(start), "goal": list(goal), "spacing": spacing},
        )

    if path_type == "waypoints":
        raw_points = _sequence(raw_params["points"], path, "path.params.points")
        if len(raw_points) < 2:
            raise _fail(
                path,
                "path.params.points",
                f"至少需要 2 个路径点，实际 {len(raw_points)} 个",
            )
        points = [
            _parse_point(item, path, f"path.params.points[{index}]")
            for index, item in enumerate(raw_points)
        ]
        widths = sorted({len(point) for point in points})
        if len(widths) != 1:
            raise _fail(
                path,
                "path.params.points",
                f"所有点必须同为 [x, y] 或 [x, y, theta]，检测到混合列数 {widths}",
            )
        spacing = _positive(raw_params["spacing"], path, "path.params.spacing")
        return PathSpec(
            type="waypoints",
            params={"points": [list(point) for point in points], "spacing": spacing},
        )

    stage = _text(raw_params["stage"], path, "path.params.stage")
    available = _stage_names()
    if stage not in available:
        raise _fail(
            path,
            "path.params.stage",
            f"未知课程阶段 '{stage}'{_did_you_mean(stage, available)}"
            f" 可用: {', '.join(available)}",
        )
    return PathSpec(type="stage", params={"stage": stage})


def _parse_vec2(value: Any, path: Path, where: str) -> tuple[float, float]:
    """`[x, y]`：正好两个有限数值。"""
    seq = _sequence(value, path, where)
    if len(seq) != 2:
        raise _fail(path, where, f"必须是 [x, y] 两个数，实际 {len(seq)} 个: {seq!r}")
    x, y = (_number(item, path, f"{where}[{index}]") for index, item in enumerate(seq))
    return (x, y)


def _parse_dynamic_obstacles(value: Any, path: Path) -> tuple[DynamicObstacle, ...]:
    """按 `type` 分表解析动态障碍（与 `_parse_obstacles` 同一范式）。"""
    seq = _sequence(value, path, "dynamic_obstacles")
    dynamics: list[DynamicObstacle] = []
    for index, item in enumerate(seq):
        where = f"dynamic_obstacles[{index}]"
        mapping = _check_mapping(item, path, where)
        if "type" not in mapping:
            raise _fail(path, where, "缺少必填键 'type'（linear 或 circular）")
        kind = _text(mapping["type"], path, f"{where}.type")
        if kind not in _DYNAMIC_KEYS:
            raise _fail(
                path,
                f"{where}.type",
                f"未知动态障碍类型 '{kind}'"
                f"{_did_you_mean(kind, tuple(_DYNAMIC_KEYS))}"
                f" 支持: {', '.join(_DYNAMIC_KEYS)}",
            )
        allowed = _DYNAMIC_KEYS[kind]
        required = (
            ("type", "start", "velocity", "radius")
            if kind == "linear"
            else ("type", "center", "radius_orbit", "omega", "radius")
        )
        _check_keys(mapping, allowed, required, path, where)
        radius = _positive(mapping["radius"], path, f"{where}.radius")

        if kind == "linear":
            start = _parse_vec2(mapping["start"], path, f"{where}.start")
            velocity = _parse_vec2(mapping["velocity"], path, f"{where}.velocity")
            period = _number(mapping.get("period", 0.0), path, f"{where}.period")
            if period < 0.0:
                raise _fail(
                    path,
                    f"{where}.period",
                    f"必须 >= 0（0 = 不循环），实际是 {period!r}",
                )
            pingpong = mapping.get("pingpong", False)
            if not isinstance(pingpong, bool):
                raise _fail(
                    path,
                    f"{where}.pingpong",
                    f"期望布尔值，实际是 {type(pingpong).__name__}（{pingpong!r}）",
                )
            if pingpong and period == 0.0:
                raise _fail(
                    path,
                    f"{where}.pingpong",
                    "pingpong 需要 period > 0（period=0 表示「不循环」，没有可折返的周期）",
                )
            dynamics.append(
                LinearDynamicObstacle(
                    start=start,
                    velocity=velocity,
                    radius=radius,
                    period=period,
                    pingpong=pingpong,
                )
            )
            continue

        center = _parse_vec2(mapping["center"], path, f"{where}.center")
        radius_orbit = _positive(mapping["radius_orbit"], path, f"{where}.radius_orbit")
        omega = _number(mapping["omega"], path, f"{where}.omega")
        phase = _number(mapping.get("phase", 0.0), path, f"{where}.phase")
        dynamics.append(
            CircularDynamicObstacle(
                center=center,
                radius_orbit=radius_orbit,
                omega=omega,
                phase=phase,
                radius=radius,
            )
        )
    return tuple(dynamics)


def _parse_scenario(raw: Any, path: Path) -> Scenario:
    """把 YAML 解析结果转成 :class:`Scenario`（严格校验，失败即抛）。"""
    mapping = _check_mapping(raw, path, "根")
    _check_keys(mapping, _TOP_KEYS, _TOP_REQUIRED, path, "根")

    name = _text(mapping["name"], path, "name")
    description = _text(mapping["description"], path, "description")
    world_bounds = _parse_world_bounds(mapping["world_bounds"], path)
    obstacles = _parse_obstacles(mapping["obstacles"], path)
    path_spec = _parse_path(mapping["path"], path)
    dynamics = _parse_dynamic_obstacles(mapping.get("dynamic_obstacles", []), path)
    seed = _integer(mapping["seed"], path, "seed")

    requires = mapping.get("requires", None)
    if requires is not None and requires != "planning":
        raise _fail(path, "requires", f"只能是 null 或 'planning'，实际是 {requires!r}")
    if requires == "planning" and path_spec.type != "stage":
        raise _fail(
            path,
            "requires",
            "'planning' 只能配 path.type == 'stage'，"
            f"实际 path.type={path_spec.type!r}",
        )
    if path_spec.type == "stage" and requires != "planning":
        raise _fail(
            path,
            "path.type",
            f"'stage' 必须同时声明 requires: planning（当前 requires={requires!r}）",
        )

    return Scenario(
        name=name,
        description=description,
        world_bounds=world_bounds,
        obstacles=obstacles,
        path=path_spec,
        dynamic_obstacles=dynamics,
        seed=seed,
        requires=requires,
        source=path,
    )


# ═════════════════════════════════════════════════════════════════════
# 公开 API
# ═════════════════════════════════════════════════════════════════════
def list_scenarios() -> list[str]:
    """返回包内 `data/` 目录下的全部场景名（稳定排序）。"""
    if not DATA_DIR.is_dir():
        return []
    names = [
        item.stem
        for item in DATA_DIR.iterdir()
        if item.is_file() and item.suffix in _SUFFIXES
    ]
    return sorted(names)


def _resolve_path(name_or_path: str | os.PathLike[str]) -> Path:
    """把"场景名"或"文件路径"解析成存在的 YAML 路径（失败即抛）。"""
    raw = os.fspath(name_or_path)
    candidate = Path(raw)
    looks_like_path = (
        isinstance(name_or_path, Path)
        or os.sep in raw
        or (os.altsep is not None and os.altsep in raw)
    )
    if looks_like_path:
        resolved = candidate.expanduser()
        if not resolved.is_file():
            raise ScenarioNotFoundError(
                f"场景文件不存在: {resolved}（可用场景: {', '.join(list_scenarios())}）"
            )
        return resolved

    for option in (
        candidate,
        DATA_DIR / raw,
        DATA_DIR / f"{raw}.yaml",
        DATA_DIR / f"{raw}.yml",
    ):
        if option.is_file() and (option is not candidate or option.suffix in _SUFFIXES):
            return option
    raise ScenarioNotFoundError(
        f"未知场景 '{raw}'（可用场景: {', '.join(list_scenarios())}）"
    )


def load_scenario(name_or_path: str | os.PathLike[str]) -> Scenario:
    """按场景名（`data/` 目录）或 YAML 路径加载并校验一个场景。

    Args:
        name_or_path: 场景名（如 `"corridor"`）或 YAML 文件路径。

    Returns:
        校验通过的 :class:`Scenario`。

    Raises:
        ScenarioNotFoundError: 名字/路径不存在（消息里列出可用场景）。
        ScenarioValidationError: schema 校验失败（消息含文件路径与键名）。
        NotImplementedError: `dynamic_obstacles` 非空。
    """
    path = _resolve_path(name_or_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"{path}: 读取失败: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioValidationError(f"{path}: YAML 解析失败: {exc}") from exc

    scenario = _parse_scenario(raw, path)
    if path.parent.resolve() == DATA_DIR.resolve() and scenario.name != path.stem:
        raise _fail(
            path,
            "name",
            f"data/ 目录下的场景名必须等于文件名 stem '{path.stem}'，"
            f"实际是 '{scenario.name}'",
        )
    return scenario
