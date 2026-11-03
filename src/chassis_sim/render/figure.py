"""matplotlib 渲染核心：延迟导入、Agg 后端、确定性 PNG。

约束（与任务书一致）：

* 模块顶层**不**导入 matplotlib；所有 matplotlib 符号都在函数内部导入。
  缺失时抛 `ImportError`，消息里带 `pip install chassis-sim[viz]`。
* 固定 Agg 后端，不调用 `pyplot.show`，无显示器可直接出图。
* PNG 固定 dpi，写入前把 `Software` 元数据置空，同一状态两次渲染像素逐位相同。

图层顺序（zorder 由小到大）：世界边界 → 障碍物 → 参考路径 → 尾迹 / LiDAR →
车体 footprint → 终点标记。参考路径按鼻向分为前进（绿）与倒车（紫）两色。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .scene import (
    Scene,
    normalize_path,
    pose_from_bridge,
    scene_from_bridge,
    split_path_by_direction,
)

# ── 调色板（测试直接引用这些常量做像素断言）──────────────────────────
CAR_FACE_COLOR = "#2f6fb3"
CAR_EDGE_COLOR = "#123c66"
CIRCLE_FACE_COLOR = "#d1495b"
CIRCLE_EDGE_COLOR = "#7d1f2c"
RECT_FACE_COLOR = "#e08a3c"
RECT_EDGE_COLOR = "#8a4a12"
PATH_FORWARD_COLOR = "#2a9d8f"
PATH_REVERSE_COLOR = "#9b5de5"
TRAIL_COLOR = "#f4a261"
LIDAR_COLOR = "#8d99ae"
BOUNDARY_COLOR = "#333333"
GOAL_COLOR = "#ffd166"
AXES_FACE_COLOR = "#f7f7f7"

_DEFAULT_DPI = 100.0
_DEFAULT_FIGSIZE: tuple[float, float] = (6.0, 6.0)
_MAX_LIDAR_RAYS = 120
_MAX_TRAIL_POINTS = 500
_LIDAR_LINES_PER_SCAN = 180


def _import_matplotlib() -> tuple[Any, Any, Any, Any, Any, Any]:
    """唯一的 matplotlib 导入点：强制 Agg 后端，返回绘图层需要的符号。"""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, Polygon, Rectangle
    except ImportError as exc:  # pragma: no cover - 缺依赖时由降级测试覆盖
        raise ImportError(
            "chassis_sim.render 需要 matplotlib，请安装可视化依赖："
            "pip install chassis-sim[viz]"
        ) from exc
    return matplotlib, Figure, FigureCanvasAgg, Circle, Polygon, Rectangle


def _rect_corners(
    cx: float, cy: float, half_l: float, half_w: float, theta: float
) -> np.ndarray:
    """矩形局部系四角旋转到世界系，返回 (4, 2)（闭合顺序 CCW）。"""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    local = np.array(
        [
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def _setup_axes(ax: Any, world_bounds: tuple[float, float, float, float]) -> None:
    """固定世界坐标范围与等比例坐标轴（关闭刻度，保证像素映射稳定）。"""
    xmin, xmax, ymin, ymax = world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(AXES_FACE_COLOR)
    ax.set_axis_off()


def _draw_boundary(ax: Any, scene: Scene, rectangle_cls: Any) -> None:
    xmin, xmax, ymin, ymax = scene.world_bounds
    ax.add_patch(
        rectangle_cls(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=BOUNDARY_COLOR,
            linewidth=1.2,
            zorder=1,
        )
    )


def _draw_obstacles(ax: Any, scene: Scene, circle_cls: Any, polygon_cls: Any) -> None:
    for center, radius in zip(scene.circle_centers, scene.circle_radii):
        ax.add_patch(
            circle_cls(
                (float(center[0]), float(center[1])),
                float(radius),
                facecolor=CIRCLE_FACE_COLOR,
                edgecolor=CIRCLE_EDGE_COLOR,
                linewidth=0.8,
                zorder=2,
            )
        )
    for center, half_dims, theta in zip(
        scene.rect_centers, scene.rect_half_dims, scene.rect_thetas
    ):
        corners = _rect_corners(
            float(center[0]),
            float(center[1]),
            float(half_dims[0]),
            float(half_dims[1]),
            float(theta),
        )
        ax.add_patch(
            polygon_cls(
                corners,
                closed=True,
                facecolor=RECT_FACE_COLOR,
                edgecolor=RECT_EDGE_COLOR,
                linewidth=0.8,
                zorder=2,
            )
        )


def _draw_path(ax: Any, scene: Scene) -> None:
    forward, reverse = split_path_by_direction(scene.path)
    for line in forward:
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=PATH_FORWARD_COLOR,
            linewidth=1.8,
            alpha=0.9,
            zorder=3,
            solid_capstyle="round",
        )
    for line in reverse:
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=PATH_REVERSE_COLOR,
            linewidth=1.8,
            alpha=0.9,
            zorder=3,
            solid_capstyle="round",
        )
    path = normalize_path(scene.path)
    if path.shape[0] > 0:
        goal = path[-1, :2]
        ax.plot(
            [float(goal[0])],
            [float(goal[1])],
            marker="*",
            markersize=12,
            color=GOAL_COLOR,
            markeredgecolor="#7a4f00",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=6,
        )


def _draw_trail(ax: Any, trail: object) -> None:
    points = np.asarray(trail, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] == 0:
        return
    points = points[-_MAX_TRAIL_POINTS:, :2]
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=TRAIL_COLOR,
        linewidth=1.4,
        alpha=0.9,
        zorder=4,
        solid_capstyle="round",
    )


def _draw_lidar(ax: Any, rays: np.ndarray) -> None:
    if rays.ndim != 2 or rays.shape[1] != 4 or rays.shape[0] == 0:
        return
    ax.plot(
        rays[:, [0, 2]].T,
        rays[:, [1, 3]].T,
        color=LIDAR_COLOR,
        linewidth=0.35,
        alpha=0.55,
        zorder=4,
    )


def _draw_vehicle(ax: Any, scene: Scene, polygon_cls: Any) -> None:
    x, y, theta = scene.pose
    half_l = scene.footprint[0] / 2.0
    half_w = scene.footprint[1] / 2.0
    corners = _rect_corners(x, y, half_l, half_w, theta)
    ax.add_patch(
        polygon_cls(
            corners,
            closed=True,
            facecolor=CAR_FACE_COLOR,
            edgecolor=CAR_EDGE_COLOR,
            linewidth=1.0,
            zorder=5,
        )
    )
    # 车头方向短线：从中心指向车头，便于判断鼻向（倒车时与运动方向相反）
    nose_x = x + math.cos(theta) * half_l
    nose_y = y + math.sin(theta) * half_l
    ax.plot(
        [x, nose_x],
        [y, nose_y],
        color=CAR_EDGE_COLOR,
        linewidth=1.0,
        zorder=6,
    )


def _collect_lidar_rays(
    bridge: object, scene: Scene, *, max_rays: int = _MAX_LIDAR_RAYS
) -> np.ndarray:
    """把前后两个 180 线扫描等间隔抽样成一条 (M, 4) 线段数组 [x0, y0, x1, y1]。

    抽样是**总数为上限**：前后雷达同时存在时各 60 条，只有一个时 120 条。
    LiDAR 安装位姿与扫描角度公式和 `SimpleBridge._simulate_lidar` 保持一致。
    """
    getters: list[tuple[int, Any]] = []
    front = getattr(bridge, "get_lidar_front", None)
    rear = getattr(bridge, "get_lidar_rear", None)
    if front is not None:
        getters.append((0, front))
    if rear is not None:
        getters.append((1, rear))
    if not getters or max_rays <= 0:
        return np.empty((0, 4), dtype=np.float64)

    per_scan = max(1, max_rays // len(getters))
    x, y, theta = scene.pose
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    chunks: list[np.ndarray] = []
    for mount_index, getter in getters:
        scan = np.asarray(getter(), dtype=np.float64).reshape(-1)
        if scan.size == 0:
            continue
        count = min(per_scan, scan.size)
        indices = np.unique(
            np.linspace(0, scan.size - 1, count).round().astype(np.int64)
        )
        off_x, off_y, angle_off = scene.lidar_mounts[mount_index]
        origin_x = x + off_x * cos_t - off_y * sin_t
        origin_y = y + off_x * sin_t + off_y * cos_t
        angles = (
            theta + angle_off - math.pi / 2.0 + np.linspace(0.0, math.pi, scan.size)
        )
        dist = scan[indices]
        chunks.append(
            np.column_stack(
                [
                    np.full(indices.size, origin_x),
                    np.full(indices.size, origin_y),
                    origin_x + dist * np.cos(angles[indices]),
                    origin_y + dist * np.sin(angles[indices]),
                ]
            )
        )
    if not chunks:
        return np.empty((0, 4), dtype=np.float64)
    rays = np.concatenate(chunks, axis=0)
    return rays[:max_rays]


def render_scene(
    scene: Scene,
    *,
    ax: Any = None,
    lidar_rays: np.ndarray | None = None,
    show_trail: bool = False,
    trail: object = None,
    dpi: float = _DEFAULT_DPI,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Any:
    """把场景画到一张新 Figure（或给定 axes）上，返回 Figure。

    `ax` 非空时复用之（dpi / figsize 由调用方的 Figure 决定），否则新建
    指定 dpi / figsize 的 Figure；LiDAR 线段需由调用方预先算好传入。
    """
    _, figure_cls, canvas_cls, circle_cls, polygon_cls, rectangle_cls = (
        _import_matplotlib()
    )
    if ax is None:
        fig = figure_cls(figsize=figsize, dpi=dpi, facecolor="white")
        canvas_cls(fig)
        ax = fig.add_axes((0.015, 0.015, 0.97, 0.97))
    else:
        fig = ax.figure

    _setup_axes(ax, scene.world_bounds)
    _draw_boundary(ax, scene, rectangle_cls)
    _draw_obstacles(ax, scene, circle_cls, polygon_cls)
    _draw_path(ax, scene)
    if show_trail and trail is not None:
        _draw_trail(ax, trail)
    if lidar_rays is not None:
        _draw_lidar(ax, lidar_rays)
    _draw_vehicle(ax, scene, polygon_cls)
    return fig


def render_frame(
    bridge: object,
    ax: Any = None,
    *,
    show_lidar: bool = False,
    show_trail: bool = False,
    trail: object = None,
    dpi: float = _DEFAULT_DPI,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Any:
    """渲染桥的当前状态，返回 matplotlib Figure。

    Args:
        bridge: `SimpleBridge` 或同构桥（只读访问位姿 / 障碍 / 路径）。
        ax: 复用的 axes；为空时新建 Figure。
        show_lidar: 是否叠加 LiDAR 射线（默认关闭，总计最多 120 条）。
        show_trail: 是否绘制轨迹尾迹（默认关闭，最多保留最后 500 点）。
        trail: 尾迹点序列，形状 (N, 2) / (N, 3)；`show_trail=False` 时忽略。
        dpi: 新建 Figure 的分辨率；复用 ax 时由原 Figure 决定。
        figsize: 新建 Figure 的英寸尺寸。
    """
    scene = scene_from_bridge(bridge)
    lidar_rays = _collect_lidar_rays(bridge, scene) if show_lidar else None
    return render_scene(
        scene,
        ax=ax,
        lidar_rays=lidar_rays,
        show_trail=show_trail,
        trail=trail,
        dpi=dpi,
        figsize=figsize,
    )


def save_figure(fig: Any, path: str | Path, *, dpi: float | None = None) -> Path:
    """把 Figure 写成确定性 PNG（去掉 `Software` 元数据，无时间戳）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        target,
        format="png",
        dpi=dpi if dpi is not None else fig.dpi,
        facecolor="white",
        metadata={"Software": None},
    )
    return target


def render_to_png(
    bridge: object, path: str | Path, *, dpi: float = _DEFAULT_DPI, **kwargs: Any
) -> Path:
    """渲染桥的当前状态并写成 PNG，返回文件路径（父目录自动创建）。"""
    fig = render_frame(bridge, dpi=dpi, **kwargs)
    return save_figure(fig, path, dpi=dpi)


def render_episode_frames(
    bridge: object,
    actions: Sequence[np.ndarray],
    out_dir: str | Path,
    *,
    every: int = 1,
    prefix: str = "frame",
    show_trail: bool = False,
    dpi: float = _DEFAULT_DPI,
    **kwargs: Any,
) -> list[Path]:
    """逐步推进桥并在每 `every` 步写一帧 PNG，返回写出的路径列表。

    最后一帧总会被写出（即使步数不是 `every` 的整数倍）。`show_trail=True`
    时自动累积已经走过的位姿作为尾迹。
    """
    if every < 1:
        raise ValueError(f"every 必须是正整数，收到 {every}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    trail_points: list[tuple[float, float, float]] = []
    steps = list(actions)
    total = len(steps)
    for step, action in enumerate(steps, start=1):
        bridge.step(np.asarray(action, dtype=np.float64))
        trail_points.append(pose_from_bridge(bridge))
        if step % every == 0 or step == total:
            target = out / f"{prefix}_{step:05d}.png"
            frames.append(
                render_to_png(
                    bridge,
                    target,
                    dpi=dpi,
                    show_trail=show_trail,
                    trail=np.asarray(trail_points, dtype=np.float64),
                    **kwargs,
                )
            )
    return frames
