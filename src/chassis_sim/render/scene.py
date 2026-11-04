"""渲染场景快照：与 matplotlib 完全解耦的只读数据视图。

场景只走桥的**公开只读方法**：`pose()`、`obstacle_shapes()`（语义与
`reset(obstacles=...)` 的输入格式一致），位姿最后退回 `get_odometry()`。
1.0.0 起渲染层**不再**读私有字段（`_x` / `_y` / `_theta` / `_circle_centers` ...）：
那样会把桥的内部实现变成事实上的公共契约。渲染层只读取、不写回，也不改变桥的任何行为。

参考路径的第 2 列是**车头鼻向**（不是运动方向），所以倒车段上鼻向与运动方向相差 π；
`split_path_by_direction` 据此把路径切成前进 / 倒车两组折线，供着色使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from chassis_sim.protocols import Bridge

_DEFAULT_WORLD: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0)
_DEFAULT_FOOTPRINT: tuple[float, float] = (0.38, 0.20)
_DEFAULT_MOUNTS: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.145, 0.0, 0.0),
    (-0.155, 0.0, math.pi),
)


def normalize_path(value: object) -> np.ndarray:
    """把路径整理成 (N, 2) 或 (N, 3) 的 float64 数组；空输入给 (0, 3)。"""
    if value is None:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] not in (2, 3):
            raise ValueError(f"路径形状 {arr.shape} 不是 (N, 2) / (N, 3)")
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"路径形状 {arr.shape} 不是 (N, 2) / (N, 3)")
    return arr


@dataclass(frozen=True)
class Scene:
    """一帧渲染所需的静态场景 + 当前位姿（纯 numpy，可序列化）。"""

    world_bounds: tuple[float, float, float, float]
    footprint: tuple[float, float]
    pose: tuple[float, float, float]
    circle_centers: np.ndarray
    circle_radii: np.ndarray
    rect_centers: np.ndarray
    rect_half_dims: np.ndarray
    rect_thetas: np.ndarray
    path: np.ndarray
    lidar_mounts: tuple[tuple[float, float, float], tuple[float, float, float]]
    max_range: float = 10.0
    wheelbase: float | None = None  # 轴距（米）；拿不到时为 None，渲染层按 footprint 推

    @property
    def n_obstacles(self) -> int:
        """圆 + 矩形障碍物总数。"""
        return int(len(self.circle_radii) + len(self.rect_thetas))

    def with_pose(self, pose: object) -> Scene:
        """返回替换位姿后的新场景（其余字段共享，均为只读使用）。"""
        arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            raise ValueError(f"位姿至少需要 3 个分量，收到 {arr.size} 个")
        return replace(self, pose=(float(arr[0]), float(arr[1]), float(arr[2])))


def _wheelbase_from(chassis: object) -> float | None:
    """读取底盘轴距（米）；没有 / 不是正有限数时给 None。"""
    value = getattr(chassis, "wheelbase", None)
    try:
        wheelbase = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(wheelbase) or wheelbase <= 0.0:
        return None
    return wheelbase


def _arrays_from_shapes(
    shapes: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """把 `obstacle_shapes()` 的 dict 列表转成五组障碍物数组。

    输入格式与 `SimpleBridge.reset(obstacles=...)` 一致；遇到未知形状返回 None，
    调用方据此退回空场景（不静默丢障碍物、也不读私有字段）。
    """
    circles: list[list[float]] = []
    radii: list[float] = []
    rect_centers: list[list[float]] = []
    rect_half_dims: list[list[float]] = []
    rect_thetas: list[float] = []
    for item in shapes or ():  # type: ignore[union-attr]
        try:
            shape = dict(item["shape"])
            state = np.asarray(item["state"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return None
        if state.size < 2:
            return None
        name = str(shape.get("name", "")).strip().lower()
        if name == "circle":
            circles.append([float(state[0]), float(state[1])])
            radii.append(float(shape.get("radius", 0.0)))
        elif name in ("rectangle", "rect"):
            rect_centers.append([float(state[0]), float(state[1])])
            rect_half_dims.append(
                [
                    float(shape.get("length", 0.0)) * 0.5,
                    float(shape.get("width", 0.0)) * 0.5,
                ]
            )
            rect_thetas.append(float(state[2]) if state.size > 2 else 0.0)
        else:
            return None
    return (
        np.asarray(circles, dtype=np.float64).reshape(-1, 2),
        np.asarray(radii, dtype=np.float64).reshape(-1),
        np.asarray(rect_centers, dtype=np.float64).reshape(-1, 2),
        np.asarray(rect_half_dims, dtype=np.float64).reshape(-1, 2),
        np.asarray(rect_thetas, dtype=np.float64).reshape(-1),
    )


def _obstacle_arrays(
    bridge: Bridge,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读公开只读方法 obstacle_shapes()；桥没有/读不出时给空数组。"""
    empty = (
        np.empty((0, 2)),
        np.empty((0,)),
        np.empty((0, 2)),
        np.empty((0, 2)),
        np.empty((0,)),
    )
    public = getattr(bridge, "obstacle_shapes", None)
    if not callable(public):
        return empty
    try:
        arrays = _arrays_from_shapes(public())
    except (TypeError, ValueError, AttributeError):
        return empty
    return empty if arrays is None else arrays


def steering_angle(bridge: Bridge) -> float | None:
    """读取当前前轮转角（弧度）；拿不到就给 None（渲染层随即使四轮对齐车身）。"""
    owners = [bridge, getattr(bridge, "chassis", None)]
    names = ("steering_angle", "steer_angle")
    for owner in owners:
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if callable(value):
                try:
                    value = value()
                except (TypeError, ValueError, AttributeError):
                    continue
            if value is None:
                continue
            try:
                angle = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(angle):
                return angle
    return None


def _point_to_box_distance(px: float, py: float, half_l: float, half_w: float) -> float:
    """车体中心到（已平移到局部系）轴对齐矩形的表面距离；内部给 0。"""
    dx = max(abs(px) - half_l, 0.0)
    dy = max(abs(py) - half_w, 0.0)
    return math.hypot(dx, dy)


def nearest_obstacle_distance(scene: Scene) -> float | None:
    """车体中心到最近障碍物表面的距离（米）；没有障碍物给 None。"""
    px, py = float(scene.pose[0]), float(scene.pose[1])
    best: float | None = None
    for center, radius in zip(scene.circle_centers, scene.circle_radii):
        gap = math.hypot(px - float(center[0]), py - float(center[1])) - float(radius)
        gap = max(gap, 0.0)
        best = gap if best is None else min(best, gap)
    for center, half_dims, theta in zip(
        scene.rect_centers, scene.rect_half_dims, scene.rect_thetas
    ):
        dx = px - float(center[0])
        dy = py - float(center[1])
        cos_t = math.cos(-float(theta))
        sin_t = math.sin(-float(theta))
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t
        gap = _point_to_box_distance(
            local_x, local_y, float(half_dims[0]), float(half_dims[1])
        )
        best = gap if best is None else min(best, gap)
    return best


def pose_from_bridge(bridge: Bridge) -> tuple[float, float, float]:
    """读取桥的真实位姿：优先公开 `pose()`，其次 `get_odometry()`。"""
    public = getattr(bridge, "pose", None)
    if callable(public):
        try:
            arr = np.asarray(public(), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            arr = np.empty((0,), dtype=np.float64)
        if arr.size >= 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
    getter = getattr(bridge, "get_odometry", None)
    if getter is None:
        raise TypeError("bridge 既没有 (_x, _y, _theta)，也没有 get_odometry()")
    odom = np.asarray(getter(), dtype=np.float64).reshape(-1)
    if odom.size < 3:
        raise ValueError(f"get_odometry() 至少需要 3 个分量，收到 {odom.size} 个")
    return (float(odom[0]), float(odom[1]), float(odom[2]))


def scene_from_bridge(bridge: Bridge) -> Scene:
    """从 `SimpleBridge`（或同构桥）抽取只读场景快照。"""
    chassis = getattr(bridge, "chassis", None)
    if chassis is not None:
        footprint = tuple(float(v) for v in chassis.footprint())
        mounts = tuple(
            (float(m[0]), float(m[1]), float(m[2])) for m in chassis.lidar_mounts()
        )
    else:
        footprint = _DEFAULT_FOOTPRINT
        mounts = _DEFAULT_MOUNTS

    world_bounds = (
        float(getattr(bridge, "_world_xmin", _DEFAULT_WORLD[0])),
        float(getattr(bridge, "_world_xmax", _DEFAULT_WORLD[1])),
        float(getattr(bridge, "_world_ymin", _DEFAULT_WORLD[2])),
        float(getattr(bridge, "_world_ymax", _DEFAULT_WORLD[3])),
    )

    get_path = getattr(bridge, "get_path_global", None)
    if get_path is not None:
        path = normalize_path(get_path())
    else:
        path = normalize_path(getattr(bridge, "_path_waypoints_global", None))

    circles, radii, rect_centers, rect_half_dims, rect_thetas = _obstacle_arrays(bridge)

    return Scene(
        world_bounds=world_bounds,
        footprint=(float(footprint[0]), float(footprint[1])),
        pose=pose_from_bridge(bridge),
        circle_centers=circles,
        circle_radii=radii,
        rect_centers=rect_centers,
        rect_half_dims=rect_half_dims,
        rect_thetas=rect_thetas,
        path=path,
        lidar_mounts=mounts,  # type: ignore[arg-type]
        max_range=float(getattr(bridge, "max_range", 10.0)),
        wheelbase=_wheelbase_from(chassis),
    )


def split_path_by_direction(
    path: object,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """按鼻向把路径切成（前进折线, 倒车折线）。

    判定依据：第 i 段运动方向 `atan2(dy, dx)` 与第 i+1 个航点的鼻向之差取
    `[-π, π]` 归一化后绝对值 > π/2 即为倒车。只有 (N, 2) 路径（无鼻向列）时
    全部按前进处理。每组折线都是 `(k, 2)` 的 x/y 数组，相邻组共享端点。
    """
    arr = normalize_path(path)
    if arr.shape[0] < 2:
        return ([], [])

    flags: list[bool] = []
    for i in range(arr.shape[0] - 1):
        dx = float(arr[i + 1, 0] - arr[i, 0])
        dy = float(arr[i + 1, 1] - arr[i, 1])
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            # 重复航点没有运动方向，沿用上一段的判定
            flags.append(flags[-1] if flags else False)
            continue
        if arr.shape[1] < 3:
            flags.append(False)
            continue
        motion = math.atan2(dy, dx)
        delta = float(arr[i + 1, 2]) - motion
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        flags.append(abs(delta) > math.pi / 2.0)

    forward: list[np.ndarray] = []
    reverse: list[np.ndarray] = []
    start = 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[start]:
            line = arr[start : i + 1, :2].copy()
            (reverse if flags[start] else forward).append(line)
            start = i
    return (forward, reverse)
