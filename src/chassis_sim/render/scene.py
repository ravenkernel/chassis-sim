"""渲染场景快照：与 matplotlib 完全解耦的只读数据视图。

本任务边界禁止修改 `chassis_sim.bridges`，而 `SimpleBridge` 没有公开障碍物访问器，
因此这里以**只读**方式读取桥的私有数组（`_circle_centers` / `_circle_radii` /
`_rect_centers` / `_rect_half_dims` / `_rect_thetas`）与真实位姿
（`_x` / `_y` / `_theta`）。渲染层只读取、不写回，也不改变桥的任何行为。

参考路径的第 2 列是**车头鼻向**（不是运动方向），所以倒车段上鼻向与运动方向相差 π；
`split_path_by_direction` 据此把路径切成前进 / 倒车两组折线，供着色使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

_DEFAULT_WORLD: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0)
_DEFAULT_FOOTPRINT: tuple[float, float] = (0.38, 0.20)
_DEFAULT_MOUNTS: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.145, 0.0, 0.0),
    (-0.155, 0.0, math.pi),
)


def _as_pairs(value: object, width: int) -> np.ndarray:
    """把任意输入整理成 (N, width) 的 float64 数组；None / 空输入给 (0, width)。"""
    if value is None:
        return np.empty((0, width), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, width), dtype=np.float64)
    return arr.reshape(-1, width)


def _as_scalars(value: object) -> np.ndarray:
    """把任意输入整理成 (N,) 的 float64 数组；None / 空输入给 (0,)。"""
    if value is None:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


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


def pose_from_bridge(bridge: object) -> tuple[float, float, float]:
    """读取桥的真实位姿；没有私有位姿字段时退回 `get_odometry()`。"""
    if all(hasattr(bridge, name) for name in ("_x", "_y", "_theta")):
        return (
            float(bridge._x),
            float(bridge._y),
            float(bridge._theta),
        )
    getter = getattr(bridge, "get_odometry", None)
    if getter is None:
        raise TypeError("bridge 既没有 (_x, _y, _theta)，也没有 get_odometry()")
    odom = np.asarray(getter(), dtype=np.float64).reshape(-1)
    if odom.size < 3:
        raise ValueError(f"get_odometry() 至少需要 3 个分量，收到 {odom.size} 个")
    return (float(odom[0]), float(odom[1]), float(odom[2]))


def scene_from_bridge(bridge: object) -> Scene:
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

    return Scene(
        world_bounds=world_bounds,
        footprint=(float(footprint[0]), float(footprint[1])),
        pose=pose_from_bridge(bridge),
        circle_centers=_as_pairs(getattr(bridge, "_circle_centers", None), 2),
        circle_radii=_as_scalars(getattr(bridge, "_circle_radii", None)),
        rect_centers=_as_pairs(getattr(bridge, "_rect_centers", None), 2),
        rect_half_dims=_as_pairs(getattr(bridge, "_rect_half_dims", None), 2),
        rect_thetas=_as_scalars(getattr(bridge, "_rect_thetas", None)),
        path=path,
        lidar_mounts=mounts,  # type: ignore[arg-type]
        max_range=float(getattr(bridge, "max_range", 10.0)),
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
