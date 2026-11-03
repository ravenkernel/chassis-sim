"""场景数据模型（W4b：静态场景集；W11b：动态障碍）。

本模块只定义**数据结构与异常类型**：YAML 解析与严格校验在 `loader.py`，
env 接线在 `reset_options.py`。全部数据类 `frozen=True`——场景是只读契约，
`scenario_to_reset_options` 会把这些数值原样交给 `bridge.reset`。

障碍的 env 侧格式（见 `simple_bridge.reset`）::

    {"shape": {"name": "circle", "radius": r}, "state": [x, y, theta]}
    {"shape": {"name": "rectangle", "length": l, "width": w}, "state": [x, y, theta]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PathType = Literal["straight", "waypoints", "stage"]
Requirement = Literal["planning"]
DynamicKind = Literal["linear", "circular"]


class ScenarioError(Exception):
    """场景相关错误的基类。"""


class ScenarioValidationError(ScenarioError, ValueError):
    """YAML 不符合 schema（消息必带**文件路径**与**键名**）。"""


class ScenarioNotFoundError(ScenarioError, LookupError):
    """按名字找不到场景（消息里列出可用场景）。"""


@dataclass(frozen=True)
class CircleObstacle:
    """圆形静态障碍：世界系中心 (x, y) + 半径。"""

    x: float
    y: float
    radius: float
    theta: float = 0.0

    @property
    def kind(self) -> str:
        """障碍类型标签（诊断/日志用）。"""
        return "circle"

    def to_env_dict(self) -> dict[str, Any]:
        """转成 `bridge.reset(obstacles=...)` 接受的 dict。"""
        return {
            "shape": {"name": "circle", "radius": float(self.radius)},
            "state": [float(self.x), float(self.y), float(self.theta)],
        }


@dataclass(frozen=True)
class RectangleObstacle:
    """矩形静态障碍：世界系中心 + 全尺寸 length×width + 朝向 theta。"""

    x: float
    y: float
    length: float
    width: float
    theta: float = 0.0

    @property
    def kind(self) -> str:
        """障碍类型标签（诊断/日志用）。"""
        return "rectangle"

    def to_env_dict(self) -> dict[str, Any]:
        """转成 `bridge.reset(obstacles=...)` 接受的 dict。"""
        return {
            "shape": {
                "name": "rectangle",
                "length": float(self.length),
                "width": float(self.width),
            },
            "state": [float(self.x), float(self.y), float(self.theta)],
        }


Obstacle = CircleObstacle | RectangleObstacle


@dataclass(frozen=True)
class PathSpec:
    """路径规格：`type` + `params`（schema 见 `chassis_sim.scenarios` 的包 docstring）。"""

    type: PathType
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_name(self) -> str | None:
        """`type == "stage"` 时返回 curriculum 的 stage 名，否则 None。"""
        if self.type != "stage":
            return None
        value = self.params.get("stage")
        return None if value is None else str(value)


@dataclass(frozen=True)
class LinearDynamicObstacle:
    """匀速直线运动的动态障碍：`p(t) = start + velocity·phase(t)`。

    运动学**纯解析、零随机**：给定 t 唯一确定位置，因此"同 seed 同轨迹"是构造性
    成立的（本类不含任何随机状态，reset 只把 t 归零）。

    Attributes:
        start: 世界系起点 `(x, y)`（米），即 t=0 时的圆心。
        velocity: 世界系速度 `(vx, vy)`（米/秒），有限值；允许为 0（静止的退化情形）。
        radius: 本体半径（米），必须 > 0。
        period: 循环周期（秒），0 = 不循环（沿 velocity 一直前进）。> 0 时按下面的
            `phase(t)` 折叠，**两种模式的 period 都表示一个完整来回/循环的总时长**。
        pingpong: 仅当 period > 0 有意义。False = 锯齿（走到相位末尾瞬移回起点）；
            True = 三角波（走到中点半程后原路折返，位置连续）。

    phase(t) 的精确语义（两位实现必须与此一致，见 `SimpleBridge`）::

        period == 0        ⇒ phase = t                      # 不循环
        period > 0, 锯齿   ⇒ phase = t mod period
        period > 0, 折返   ⇒ phase = period/2 - |(t mod period) - period/2|

    t=0 时三种模式的 phase 都恰好是 0（起点即 start）。
    """

    start: tuple[float, float]
    velocity: tuple[float, float]
    radius: float
    period: float = 0.0
    pingpong: bool = False

    @property
    def kind(self) -> str:
        """障碍类型标签（诊断/日志用）。"""
        return "linear"

    def to_env_dict(self) -> dict[str, Any]:
        """转成 `SimpleBridge(dynamic_obstacles=...)` 接受的 dict（默认值显式写全）。"""
        return {
            "type": "linear",
            "start": [float(self.start[0]), float(self.start[1])],
            "velocity": [float(self.velocity[0]), float(self.velocity[1])],
            "radius": float(self.radius),
            "period": float(self.period),
            "pingpong": bool(self.pingpong),
        }


@dataclass(frozen=True)
class CircularDynamicObstacle:
    """绕固定圆心做匀速圆周运动的动态障碍：`p(t) = center + radius_orbit·(cos φ, sin φ)`。

    其中 `φ = phase + omega·t`。同样纯解析、零随机。

    Attributes:
        center: 轨道圆心 `(x, y)`（米）。
        radius_orbit: **轨道半径**（米），必须 > 0。注意与下面的 `radius` 不是一回事。
        omega: 角速度（弧度/秒），有限值；负值 = 顺时针，0 = 停在初相位。
        phase: t=0 时的初相位（弧度），有限值，不归一化（直接进 cos/sin）。
        radius: **本体半径**（米），必须 > 0（参与 raycast 与 SAT/圆-矩形检测）。
    """

    center: tuple[float, float]
    radius_orbit: float
    omega: float
    phase: float
    radius: float

    @property
    def kind(self) -> str:
        """障碍类型标签（诊断/日志用）。"""
        return "circular"

    def to_env_dict(self) -> dict[str, Any]:
        """转成 `SimpleBridge(dynamic_obstacles=...)` 接受的 dict。"""
        return {
            "type": "circular",
            "center": [float(self.center[0]), float(self.center[1])],
            "radius_orbit": float(self.radius_orbit),
            "omega": float(self.omega),
            "phase": float(self.phase),
            "radius": float(self.radius),
        }


#: 动态障碍的判别联合（与静态的 `Obstacle` 同构：按 `kind` 分派）。
DynamicObstacle = LinearDynamicObstacle | CircularDynamicObstacle


@dataclass(frozen=True)
class Scenario:
    """一个场景的只读快照。

    Attributes:
        name: 场景名；从 `data/` 目录按名字加载时必须等于文件名 stem。
        description: 一句话说明（给人看，不参与几何计算）。
        world_bounds: `(xmin, xmax, ymin, ymax)`，顺序固定。
        obstacles: 静态障碍元组；planning 场景为空（地图由 MapGenerator 生成）。
        path: 路径规格。
        dynamic_obstacles: 动态障碍元组；非空时 reset options 抛 NotImplementedError。
        seed: 场景默认随机种子（planning 场景决定地图；core 场景只记录）。
        requires: `None` 或 `"planning"`（需要 numba + MapGenerator）。
        source: 来源 YAML 路径（程序内构造时为 None），仅用于诊断。
    """

    name: str
    description: str
    world_bounds: tuple[float, float, float, float]
    obstacles: tuple[Obstacle, ...]
    path: PathSpec
    dynamic_obstacles: tuple[DynamicObstacle, ...] = ()
    seed: int = 0
    requires: Requirement | None = None
    source: Path | None = None

    @property
    def requires_planning(self) -> bool:
        """是否走 MapGenerator（`requires: planning`）。"""
        return self.requires == "planning"

    def summary(self) -> str:
        """一行诊断摘要：名字 + 路径类型 + 障碍数 + 是否 planning。"""
        return (
            f"{self.name}: path={self.path.type}"
            f" obstacles={len(self.obstacles)}"
            f" dynamic={len(self.dynamic_obstacles)}"
            f" requires={self.requires}"
        )
