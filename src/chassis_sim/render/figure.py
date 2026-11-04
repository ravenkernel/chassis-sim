"""matplotlib 演示风渲染核心：深色高对比、延迟导入、Agg 后端、确定性 PNG。

约束（与任务书一致）：

* 模块顶层**不**导入 matplotlib；所有 matplotlib 符号都在
  `_import_matplotlib()` 里延迟导入。缺失时抛 `ImportError`，消息里带
  `pip install chassis-sim[viz]`。
* 固定 Agg 后端，不调用 `pyplot.show`，无显示器可直接出图。
* PNG 固定 dpi，写入前把 `Software` 元数据置空；同一状态两次渲染逐字节相同。

画面只有一个「演示风」主题，配色全部集中在模块级 `PALETTE`（换风格改一处）：

* 画布满幅（`add_axes((0, 0, 1, 1))`，等价于 subplots_adjust(0, 0, 1, 1) 但
  不依赖已有 subplot 布局）、无坐标轴边框、深色底；
* 0.5 m 细格 / 1 m 粗格铺满世界边界；
* 障碍物暖色填充 + 2 层外发光（纯 patch 叠 alpha，不用 imshow / 渐变图）；
* 参考路径青色 + 宽光晕线，终点金色星形 + 光晕圆；
* 车体按真实 footprint / 朝向画圆角矩形，四轮按轴距与前轮转角摆放，车头楔形箭头；
* LiDAR 扇形光束（总计 ≤36 条，LineCollection + scatter）、尾迹渐隐（LineCollection）；
* 右下角缩放内嵌窗（`zoom` 指定窗口边长，默认关闭）：放大车体细节而不是放大几何；
* 左上角 HUD 显示**真实读数**（`get_odometry()` / `sim_time` / 场景几何），
  拿不到就显示 `—`，从不编数字。

图层顺序（zorder 由小到大）：栅格 → 世界边界 → 障碍物 → 参考路径 → 尾迹 →
LiDAR → 车体 → 终点 → HUD。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .scene import (
    Scene,
    normalize_path,
    pose_from_bridge,
    scene_from_bridge,
    split_path_by_direction,
    steering_angle,
)

# ── 演示风调色板（浅色）：唯一定义处，换风格只改这里 ──────────────
PALETTE: dict[str, str] = {
    "canvas": "#FAFBFC",
    "grid_fine": "#E9EDF2",
    "grid_coarse": "#D5DBE4",
    "boundary": "#C7CFD9",
    "obstacle": "#F2A43C",
    "obstacle_edge": "#A9691E",
    "path": "#0E8F9E",
    "goal": "#E6A700",
    "car": "#2E86DE",
    "car_edge": "#12456E",
    "wheel": "#2B2F36",
    "wheel_edge": "#9AA4B2",
    "nose": "#FFFFFF",
    "lidar": "#0F9D6B",
    "trail": "#D9467A",
    "hud_bg": "#FFFFFF",
    "hud_edge": "#D5DBE4",
    "hud_text": "#111827",
    "hud_muted": "#6B7280",
}

_DEFAULT_DPI = 100.0
_DEFAULT_FIGSIZE: tuple[float, float] = (6.0, 6.0)
# 36 条（前 18 + 后 18）：条数少一点、单条更实，浅底上才读得出「雷达」而不是底纹
_MAX_LIDAR_RAYS = 36
_MAX_TRAIL_POINTS = 500
_GLOW_SCALE = 1.18  # 浅底上光晕半径放大会糊成脏灰：只外扩一小圈
_GLOW_ALPHA = 0.15
_OBSTACLE_ALPHA = 0.9
_CAR_GLOW_SCALE = 1.3
_CAR_GLOW_ALPHA = 0.15
_PATH_HALO_ALPHA = 0.18
_GOAL_GLOW_ALPHA = 0.25
_LIDAR_ALPHA = 0.22
_LIDAR_HIT_SIZE = 8.0  # 命中点：比光束实，加白描边在浅底上才立得住
_LIDAR_HIT_ALPHA = 0.55
_LIDAR_HIT_EDGE = 0.6
_LIDAR_SATURATED = 0.999  # ≥ 量程此比例 ⇒ 饱和 / 无回波：画到量程但不画命中点
_FRAME_PAD = 0.016  # 画面留白（世界跨度的比例）：让场地边界成为画面里唯一的「墙」
_FRAME_LW = 1.2
_DETAIL_TARGET_PX = 50.0  # 车体短于此像素数时按比例放大轮子 / 楔形 / 描边
_DETAIL_MAX = 2.2
_TRAIL_ALPHA: tuple[float, float] = (0.08, 0.85)
_HUD_ALPHA = 0.88
_HUD_FONT_SIZE = 8.5
_HUD_LINE_SPACING = 1.45
_HUD_MARGIN = 0.012
_HUD_PADDING_IN = 0.11
_MONO_FAMILY = "DejaVu Sans Mono"
_ZOOM_BOUNDS: tuple[float, float, float, float] = (0.705, 0.018, 0.278, 0.278)
_ZOOM_FONT_SIZE = 7.0
_ZOOM_ROUND = 0.012
_ZOOM_DETAIL = 1.6  # 窗口内车体细节补偿：四轮 / 车头楔形在这个尺寸下才看得清
_MISSING = "—"


# ── matplotlib 延迟导入 ───────────────────────────────────────────
@dataclass(frozen=True)
class _Mpl:
    """一次延迟导入得到的符号集合（避免每个绘图函数各写一遍 import）。"""

    matplotlib: Any
    Figure: Any
    FigureCanvasAgg: Any
    Circle: Any
    Polygon: Any
    Rectangle: Any
    FancyBboxPatch: Any
    PatchCollection: Any
    LineCollection: Any
    to_rgb: Any


def _import_matplotlib() -> _Mpl:
    """唯一的 matplotlib 导入点：强制 Agg 后端，返回绘图层需要的符号。"""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import LineCollection, PatchCollection
        from matplotlib.colors import to_rgb
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
    except ImportError as exc:  # pragma: no cover - 缺依赖时由降级测试覆盖
        raise ImportError(
            "chassis_sim.render 需要 matplotlib，请安装可视化依赖："
            "pip install chassis-sim[viz]"
        ) from exc
    return _Mpl(
        matplotlib=matplotlib,
        Figure=Figure,
        FigureCanvasAgg=FigureCanvasAgg,
        Circle=Circle,
        Polygon=Polygon,
        Rectangle=Rectangle,
        FancyBboxPatch=FancyBboxPatch,
        PatchCollection=PatchCollection,
        LineCollection=LineCollection,
        to_rgb=to_rgb,
    )


# ── 纯几何小工具（不含 matplotlib）────────────────────────────────
def _rot(angle: float) -> np.ndarray:
    """二维旋转矩阵（逆时针）。"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)


def _oriented_corners(
    local: np.ndarray, x: float, y: float, theta: float
) -> np.ndarray:
    """把车体系点列按朝向旋转后平移到世界系。"""
    return local @ _rot(theta).T + np.array([x, y], dtype=np.float64)


def _rect_corners(
    cx: float, cy: float, half_l: float, half_w: float, theta: float
) -> np.ndarray:
    """矩形局部系四角旋转到世界系，返回 (4, 2)（闭合顺序 CCW）。"""
    local = np.array(
        [
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    return _oriented_corners(local, cx, cy, theta)


def _rounded_body(
    half_l: float, half_w: float, radius: float, *, segments: int = 4
) -> np.ndarray:
    """车体系圆角矩形轮廓（CCW 闭合点列，不含首点重复）。"""
    r = max(min(radius, half_l * 0.9, half_w * 0.9), 1e-9)
    inner_l = half_l - r
    inner_w = half_w - r
    corners = (
        (inner_l, -inner_w, -math.pi / 2.0),
        (inner_l, inner_w, 0.0),
        (-inner_l, inner_w, math.pi / 2.0),
        (-inner_l, -inner_w, math.pi),
    )
    points: list[tuple[float, float]] = []
    for corner_x, corner_y, start in corners:
        for index in range(segments + 1):
            angle = start + (math.pi / 2.0) * index / segments
            points.append(
                (corner_x + r * math.cos(angle), corner_y + r * math.sin(angle))
            )
    return np.asarray(points, dtype=np.float64)


def _grid_range(low: float, high: float, step: float) -> np.ndarray:
    """[low, high] 内所有 step 的整数倍（端点含入，按浮点容差判定）。"""
    if step <= 0.0 or high < low:
        return np.empty(0, dtype=np.float64)
    start = math.ceil(low / step - 1e-9) * step
    count = math.floor((high - start) / step + 1e-9) + 1
    return start + step * np.arange(max(count, 0), dtype=np.float64)


def _grid_step(xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    """细格步长：默认 0.5 m；超大世界按倍数放大，避免铺出上万条线。"""
    span = max(xmax - xmin, ymax - ymin)
    if span <= 40.0:
        return 0.5
    return 0.5 * math.ceil(span / 80.0)


# ── 各图层 ───────────────────────────────────────────────────────
def _setup_axes(ax: Any, world_bounds: tuple[float, float, float, float]) -> None:
    """满幅深色画布 + 等比例坐标轴 + 关掉刻度与边框（像素映射稳定）。"""
    xmin, xmax, ymin, ymax = world_bounds
    pad = _FRAME_PAD * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(PALETTE["canvas"])
    ax.set_axis_off()
    ax.margins(0.0)


def _draw_grid(ax: Any, scene: Scene, mpl: _Mpl) -> None:
    """0.5 m 细格 + 1 m 粗格，铺满世界边界。"""
    xmin, xmax, ymin, ymax = scene.world_bounds
    step = _grid_step(xmin, xmax, ymin, ymax)
    layers = (
        (step, PALETTE["grid_fine"], 0.4, 0.1),
        (step * 2.0, PALETTE["grid_coarse"], 0.6, 0.2),
    )
    for size, color, width, zorder in layers:
        segments = [((x, ymin), (x, ymax)) for x in _grid_range(xmin, xmax, size)]
        segments += [((xmin, y), (xmax, y)) for y in _grid_range(ymin, ymax, size)]
        if not segments:
            continue
        ax.add_collection(
            mpl.LineCollection(segments, colors=color, linewidths=width, zorder=zorder)
        )


def _draw_boundary(ax: Any, scene: Scene, mpl: _Mpl) -> Any:
    """世界边界淡描边（场景内容，不是坐标轴边框），并作为场内绘制的裁切路径。"""
    xmin, xmax, ymin, ymax = scene.world_bounds
    frame = mpl.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor=PALETTE["boundary"],
        linewidth=_FRAME_LW,
        zorder=4.5,  # 压在激光之上：射线到这里就停，画面里的「墙」看得见
    )
    ax.add_patch(frame)
    return frame


def _draw_obstacles(ax: Any, scene: Scene, mpl: _Mpl, *, clip: Any = None) -> None:
    """暖色障碍物：一层外发光 + 一层实体（2 层 patch 叠 alpha 造体积感）。"""
    glow: list[Any] = []
    body: list[Any] = []
    for center, radius in zip(scene.circle_centers, scene.circle_radii):
        center_x, center_y = float(center[0]), float(center[1])
        radius = float(radius)
        glow.append(mpl.Circle((center_x, center_y), radius * _GLOW_SCALE))
        body.append(mpl.Circle((center_x, center_y), radius))
    for center, half_dims, theta in zip(
        scene.rect_centers, scene.rect_half_dims, scene.rect_thetas
    ):
        center_x, center_y = float(center[0]), float(center[1])
        half_l, half_w = float(half_dims[0]), float(half_dims[1])
        pad = min(half_l, half_w) * (_GLOW_SCALE - 1.0) + 0.01
        glow.append(
            mpl.Polygon(
                _rect_corners(
                    center_x, center_y, half_l + pad, half_w + pad, float(theta)
                ),
                closed=True,
            )
        )
        body.append(
            mpl.Polygon(
                _rect_corners(center_x, center_y, half_l, half_w, float(theta)),
                closed=True,
            )
        )
    layers: list[Any] = []
    if glow:
        layers.append(
            mpl.PatchCollection(
                glow,
                facecolor=PALETTE["obstacle"],
                edgecolor="none",
                alpha=_GLOW_ALPHA,
                zorder=2.0,
            )
        )
    if body:
        layers.append(
            mpl.PatchCollection(
                body,
                facecolor=PALETTE["obstacle"],
                edgecolor=PALETTE["obstacle_edge"],
                linewidths=1.2,
                alpha=_OBSTACLE_ALPHA,
                zorder=2.1,
            )
        )
    for collection in layers:
        if clip is not None:
            collection.set_clip_path(clip)
        ax.add_collection(collection)


def _draw_path(ax: Any, scene: Scene, mpl: _Mpl) -> None:
    """青色参考路径（下垫宽光晕线）+ 终点金色星形；倒车段用虚线区分。"""
    forward, reverse = split_path_by_direction(scene.path)
    for lines, dashed in ((forward, False), (reverse, True)):
        for line in lines:
            style: dict[str, Any] = {"dashes": (3.0, 2.4)} if dashed else {}
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=PALETTE["path"],
                linewidth=6.0,
                alpha=_PATH_HALO_ALPHA,
                solid_capstyle="round",
                zorder=3.0,
                **style,
            )
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=PALETTE["path"],
                linewidth=2.0,
                alpha=0.65 if dashed else 1.0,
                solid_capstyle="round",
                zorder=3.1,
                **style,
            )
    path = normalize_path(scene.path)
    if path.shape[0] == 0:
        return
    goal_x, goal_y = float(path[-1, 0]), float(path[-1, 1])
    ax.add_patch(
        mpl.Circle(
            (goal_x, goal_y),
            0.3,
            facecolor=PALETTE["goal"],
            edgecolor="none",
            alpha=_GOAL_GLOW_ALPHA,
            zorder=5.0,
        )
    )
    ax.plot(
        [goal_x],
        [goal_y],
        marker="*",
        markersize=18,
        color=PALETTE["goal"],
        markeredgecolor=PALETTE["canvas"],
        markeredgewidth=1.0,
        linestyle="none",
        zorder=5.1,
    )


def _draw_trail(ax: Any, trail: object, mpl: _Mpl) -> None:
    """尾迹渐隐：越早越淡（_TRAIL_ALPHA 由淡到浓），单个 LineCollection 一次画完。"""
    points = np.asarray(trail, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 2:
        return
    points = points[-_MAX_TRAIL_POINTS:, :2]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(_TRAIL_ALPHA[0], _TRAIL_ALPHA[1], len(segments))
    colors = np.empty((len(segments), 4), dtype=np.float64)
    colors[:, :3] = np.asarray(mpl.to_rgb(PALETTE["trail"]), dtype=np.float64)
    colors[:, 3] = alphas
    ax.add_collection(
        mpl.LineCollection(
            segments, colors=colors, linewidths=1.8, capstyle="round", zorder=4.0
        )
    )


def _draw_lidar(
    ax: Any, rays: np.ndarray, max_range: float, mpl: _Mpl, *, clip: Any = None
) -> None:
    """LiDAR 光束（≤36 条，长度 = 该射线真实距离）+ 命中点小圆点，全程向量化。"""
    if rays.ndim != 2 or rays.shape[1] != 4 or rays.shape[0] == 0:
        return
    rays = rays[:_MAX_LIDAR_RAYS]
    segments = np.stack([rays[:, :2], rays[:, 2:4]], axis=1)
    beams = mpl.LineCollection(
        segments,
        colors=PALETTE["lidar"],
        linewidths=0.6,
        alpha=_LIDAR_ALPHA,
        zorder=4.3,
    )
    if clip is not None:
        beams.set_clip_path(clip)
    ax.add_collection(beams)
    lengths = np.hypot(rays[:, 2] - rays[:, 0], rays[:, 3] - rays[:, 1])
    hits = rays[lengths < max_range * _LIDAR_SATURATED]
    if hits.size:
        dots = ax.scatter(
            hits[:, 2],
            hits[:, 3],
            s=_LIDAR_HIT_SIZE,
            color=PALETTE["lidar"],
            alpha=_LIDAR_HIT_ALPHA,
            edgecolors=PALETTE["hud_bg"],
            linewidths=_LIDAR_HIT_EDGE,
            zorder=4.4,
        )
        if clip is not None:
            dots.set_clip_path(clip)


def _wheel_layout(
    scene: Scene, steering: float | None, detail: float = 1.0
) -> tuple[list[tuple[float, float, float]], float, float]:
    """四轮车体系布局 [(x, y, 偏转角), ...] 与 (轮长, 轮宽)。

    位置由 footprint 与轴距推出：前/后轴各在 ±轴距/2，左右各贴车宽内侧；
    只有前轮吃当前转角（后轮恒对齐车身）。轴距拿不到时退化为 footprint 的一半。
    detail 是小尺度补偿：世界大 / dpi 低时放大轮子，否则 600 px 下糊成一团。
    """
    length, width = scene.footprint
    wheelbase = scene.wheelbase if scene.wheelbase else length * 0.5
    axle = float(wheelbase) / 2.0
    stretch = min(math.sqrt(detail), 1.5)
    wheel_len = min(float(wheelbase) * 0.55 * stretch, length * 0.32)
    wheel_th = min(width * 0.16 * detail, width * 0.26)
    offset_y = width * 0.465
    delta = 0.0 if steering is None else float(steering)
    wheels = [
        (axle, offset_y, delta),
        (axle, -offset_y, delta),
        (-axle, offset_y, 0.0),
        (-axle, -offset_y, 0.0),
    ]
    return wheels, wheel_len, wheel_th


def _detail_scale(fig: Any, scene: Scene) -> float:
    """车体细节的像素补偿：车体越小，轮子 / 楔形 / 描边放大得越多。

    按车体在画布上的像素长度反算（10×10 m 世界 + 600 px 时车体约 22 px，不补偿就
    看不出「四个轮子 + 车头」），上限 _DETAIL_MAX 防止 5×5 m 世界里过度放大。
    """
    xmin, xmax, ymin, ymax = scene.world_bounds
    span = max(xmax - xmin, ymax - ymin)
    width_px = float(fig.get_figwidth()) * float(fig.dpi)
    if span <= 0.0 or width_px <= 0.0:
        return 1.0
    car_px = float(scene.footprint[0]) * width_px / (span * (1.0 + 2.0 * _FRAME_PAD))
    if car_px <= 0.0:
        return 1.0
    return min(max(_DETAIL_TARGET_PX / car_px, 1.0), _DETAIL_MAX)


def _draw_vehicle(
    ax: Any,
    scene: Scene,
    mpl: _Mpl,
    *,
    steering: float | None = None,
    detail: float = 1.0,
    clip: Any = None,
) -> None:
    """车身（圆角矩形 + 发光）、四个轮子、车头楔形箭头。

    轮子 / 楔形 / 描边按 detail 放大（见 _detail_scale），保证 10×10 m 场景下仍能
    一眼分辨四轮与前向楔形。
    """
    x, y, theta = scene.pose
    half_l = scene.footprint[0] / 2.0
    half_w = scene.footprint[1] / 2.0
    radius = min(half_l, half_w) * 0.45
    body_lw = 1.2 * min(detail, 1.6)
    wheel_lw = 0.7 * min(detail, 1.8)
    nose_lw = 0.8 * min(detail, 1.6)

    glow_body = _rounded_body(
        half_l * _CAR_GLOW_SCALE, half_w * _CAR_GLOW_SCALE, radius * _CAR_GLOW_SCALE
    )
    parts: list[Any] = []
    parts.append(
        mpl.Polygon(
            _oriented_corners(glow_body, x, y, theta),
            closed=True,
            facecolor=PALETTE["car"],
            edgecolor="none",
            alpha=_CAR_GLOW_ALPHA,
            zorder=4.6,
        )
    )

    body = _rounded_body(half_l, half_w, radius)
    parts.append(
        mpl.Polygon(
            _oriented_corners(body, x, y, theta),
            closed=True,
            facecolor=PALETTE["car"],
            edgecolor=PALETTE["car_edge"],
            linewidth=body_lw,
            zorder=4.8,
        )
    )

    wheels, wheel_len, wheel_th = _wheel_layout(scene, steering, detail)
    base = np.array(
        [
            [-wheel_len / 2.0, -wheel_th / 2.0],
            [wheel_len / 2.0, -wheel_th / 2.0],
            [wheel_len / 2.0, wheel_th / 2.0],
            [-wheel_len / 2.0, wheel_th / 2.0],
        ],
        dtype=np.float64,
    )
    for wheel_x, wheel_y, wheel_angle in wheels:
        local = base @ _rot(wheel_angle).T + np.array([wheel_x, wheel_y])
        parts.append(
            mpl.Polygon(
                _oriented_corners(local, x, y, theta),
                closed=True,
                facecolor=PALETTE["wheel"],
                edgecolor=PALETTE["wheel_edge"],
                linewidth=wheel_lw,
                zorder=4.85,
            )
        )

    growth = min(detail, 1.9) - 1.0
    nose_base_x = half_l * min(0.42 + 0.10 * growth, 0.62)
    nose_base_y = half_w * min(0.42 + 0.16 * growth, 0.80)
    nose = np.array(
        [
            [half_l * 0.98, 0.0],
            [nose_base_x, nose_base_y],
            [nose_base_x, -nose_base_y],
        ],
        dtype=np.float64,
    )
    parts.append(
        mpl.Polygon(
            _oriented_corners(nose, x, y, theta),
            closed=True,
            facecolor=PALETTE["nose"],
            edgecolor=PALETTE["car_edge"],
            linewidth=nose_lw,
            zorder=4.9,
        )
    )

    for patch in parts:
        if clip is not None:
            patch.set_clip_path(clip)
        ax.add_patch(patch)


def _fmt_number(value: object, pattern: str, suffix: str = "") -> str:
    """按格式显示读数；拿不到 / 非有限值时显示 `—`。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        return _MISSING
    number = float(value)
    if not math.isfinite(number):
        return _MISSING
    return pattern.format(number) + suffix


def _hud_lines(telemetry: dict[str, Any]) -> list[str]:
    """HUD 文本行：全部来自真实读数，缺项显示 `—`。"""
    return [
        f"{'step':<6}{_fmt_number(telemetry.get('step'), '{:.0f}')}",
        f"{'t':<6}{_fmt_number(telemetry.get('t'), '{:.2f}', ' s')}",
        f"{'v':<6}{_fmt_number(telemetry.get('v'), '{:+.2f}', ' m/s')}",
        f"{'ω':<6}{_fmt_number(telemetry.get('omega'), '{:+.2f}', ' rad/s')}",
        f"{'d_min':<6}{_fmt_number(telemetry.get('d_min'), '{:.2f}', ' m')}",
    ]


def _draw_hud(
    ax: Any,
    mpl: _Mpl,
    telemetry: dict[str, Any],
    *,
    title: str | None = None,
) -> None:
    """左上角半透明读数面板 + 右上角 chassis / title 小字。"""
    lines = _hud_lines(telemetry)
    fig = ax.figure
    fig_width, fig_height = (float(v) for v in fig.get_size_inches())
    longest = max(len(line) for line in lines)
    # 等宽字形前进宽度 ≈ 0.6 em；面板尺寸由字体磅值换算成 figure 比例。
    width = (0.6 * _HUD_FONT_SIZE * longest / 72.0 + 2.0 * _HUD_PADDING_IN) / fig_width
    height = (
        _HUD_FONT_SIZE * _HUD_LINE_SPACING * len(lines) / 72.0 + 2.0 * _HUD_PADDING_IN
    ) / fig_height
    left = _HUD_MARGIN
    top = 1.0 - _HUD_MARGIN
    ax.add_patch(
        mpl.FancyBboxPatch(
            (left, top - height),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.014",
            transform=ax.transAxes,
            facecolor=PALETTE["hud_bg"],
            edgecolor=PALETTE["hud_edge"],
            linewidth=0.8,
            alpha=_HUD_ALPHA,
            zorder=6.0,
        )
    )
    ax.text(
        left + _HUD_PADDING_IN / fig_width,
        top - _HUD_PADDING_IN / fig_height,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        family=_MONO_FAMILY,
        fontsize=_HUD_FONT_SIZE,
        color=PALETTE["hud_text"],
        linespacing=_HUD_LINE_SPACING,
        zorder=6.1,
    )

    meta = [str(telemetry.get("chassis") or _MISSING)]
    if title:
        meta.append(str(title))
    ax.text(
        1.0 - _HUD_MARGIN,
        top,
        "\n".join(meta),
        transform=ax.transAxes,
        ha="right",
        va="top",
        family=_MONO_FAMILY,
        fontsize=7.5,
        color=PALETTE["hud_muted"],
        linespacing=1.3,
        zorder=6.1,
    )


def _draw_zoom_inset(
    fig: Any,
    ax: Any,
    scene: Scene,
    mpl: _Mpl,
    *,
    zoom: float,
    lidar_rays: np.ndarray | None = None,
    steering: float | None = None,
) -> Any:
    """右下角缩放内嵌窗：以车体为中心、边长 zoom 米，用同一套绘制函数画放大视野。

    10×10 m / 900 px 时车体只有约 30 px，四轮与车头楔形落不到像素上；内嵌窗放大的是
    视野而不是几何——车身轮廓、轮胎位置仍按真实尺寸画，因此不会失真。
    """
    if zoom <= 0.0:
        raise ValueError(f"zoom 必须是正数（米），收到 {zoom}")
    pose_x, pose_y, _ = scene.pose
    half = float(zoom) / 2.0
    inset = ax.inset_axes(list(_ZOOM_BOUNDS), zorder=8.0)
    inset.set_facecolor(PALETTE["hud_bg"])
    inset.set_xlim(pose_x - half, pose_x + half)
    inset.set_ylim(pose_y - half, pose_y + half)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)  # 方角边框交给下面的圆角描边，免得两套角叠在一起

    _draw_obstacles(inset, scene, mpl)
    _draw_path(inset, scene, mpl)
    if lidar_rays is not None:
        _draw_lidar(
            inset, np.asarray(lidar_rays, dtype=np.float64), scene.max_range, mpl
        )
    # 窗口里车体约 47 px（2 m 窗）：沿用主图的小尺度补偿把四轮与楔形撑到能分辨
    _draw_vehicle(inset, scene, mpl, steering=steering, detail=_ZOOM_DETAIL)

    # 比例尺：窗口够宽才是 1 m，否则按半窗宽缩短，免得压到车体
    bar = 1.0 if zoom >= 1.6 else zoom * 0.5
    bar_x = pose_x - half + 0.07 * zoom
    bar_y = pose_y - half + 0.09 * zoom
    inset.plot(
        [bar_x, bar_x + bar],
        [bar_y, bar_y],
        color=PALETTE["hud_text"],
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=6.0,
    )
    inset.text(
        bar_x + bar + 0.04 * zoom,
        bar_y,
        f"{bar:g} m",
        color=PALETTE["hud_muted"],
        fontsize=_ZOOM_FONT_SIZE,
        family=_MONO_FAMILY,
        ha="left",
        va="center",
        zorder=6.0,
    )

    # 圆角描边压在上面：方角被圆角盖住，读起来像一块浮起来的窗（不用引线）
    position = ax.get_position()
    fig.patches.append(
        mpl.FancyBboxPatch(
            (
                position.x0 + _ZOOM_BOUNDS[0] * position.width,
                position.y0 + _ZOOM_BOUNDS[1] * position.height,
            ),
            _ZOOM_BOUNDS[2] * position.width,
            _ZOOM_BOUNDS[3] * position.height,
            boxstyle=f"round,pad=0,rounding_size={_ZOOM_ROUND}",
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=PALETTE["boundary"],
            linewidth=_FRAME_LW,
            zorder=9.0,
        )
    )
    return inset


# ── 读数（全部走公开方法，拿不到就给 None）────────────────────────
def _read_odometry(bridge: object) -> np.ndarray | None:
    """`get_odometry()` 的只读快照；不可用 / 形状不符时返回 None。"""
    getter = getattr(bridge, "get_odometry", None)
    if not callable(getter):
        return None
    try:
        odom = np.asarray(getter(), dtype=np.float64).reshape(-1)
    except (AttributeError, TypeError, ValueError):  # 读数失败只让 HUD 退化成 "—"
        return None
    return odom if odom.size >= 5 else None


def _read_sim_time(bridge: object) -> float | None:
    """`sim_time`（属性或方法）的只读快照；拿不到返回 None。"""
    value = getattr(bridge, "sim_time", None)
    if callable(value):
        try:
            value = value()
        except (AttributeError, TypeError, ValueError):
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _chassis_label(bridge: object) -> str | None:
    """底盘名（`chassis.name`，退化到类名）。"""
    chassis = getattr(bridge, "chassis", None)
    if chassis is None:
        return None
    name = getattr(chassis, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(chassis).__name__


def _bridge_telemetry(bridge: object, scene: Scene) -> dict[str, Any]:
    """HUD 读数：odometry 的 v / ω、sim_time 推的 t 与 step、真实激光回波给的 d_min。

    d_min 取前后雷达最近的真实回波；桥没有激光能力时留空（HUD 显示「—」）——
    宁可不显示，也不拿几何估算冒充分米波读数。
    """
    telemetry: dict[str, Any] = {
        "chassis": _chassis_label(bridge),
        "d_min": _lidar_min_range(bridge, scene),
    }
    odom = _read_odometry(bridge)
    if odom is not None:
        telemetry["v"] = float(odom[3])
        telemetry["omega"] = float(odom[4])
    sim_time = _read_sim_time(bridge)
    if sim_time is not None:
        telemetry["t"] = sim_time
        dt = getattr(bridge, "dt", None)
        if (
            isinstance(dt, (int, float, np.floating))
            and not isinstance(dt, bool)
            and float(dt) > 0.0
        ):
            telemetry["step"] = round(sim_time / float(dt))
    return telemetry


def _read_scan(bridge: object, side: str) -> np.ndarray | None:
    """单侧真实扫描：优先 get_scan(side)（饱和读数报 +inf），其次 get_lidar_front()。

    返回 None 表示这条数据通路不存在或读失败——渲染层据此一条光束都不画。
    """
    reader = getattr(bridge, "get_scan", None)
    if callable(reader):
        try:
            payload = reader(side)
        except (AttributeError, TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and payload.get("ranges") is not None:
            scan = np.asarray(payload["ranges"], dtype=np.float64).reshape(-1)
            if scan.size:
                return scan
    getter = getattr(bridge, "get_lidar_" + side, None)
    if not callable(getter):
        return None
    try:
        raw = getter()
    except (AttributeError, TypeError, ValueError):
        return None
    # 返回 None / 标量说明这条数据通路没有扫描数据，不能当成一帧 nan 射线
    if raw is None or np.ndim(raw) == 0:
        return None
    try:
        scan = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return None
    return scan if scan.size else None


def _lidar_scans(bridge: object) -> list[tuple[int, np.ndarray]]:
    """[(雷达安装位编号, 真实距离数组)]；桥没有激光能力时是空列表。"""
    scans: list[tuple[int, np.ndarray]] = []
    for mount_index, side in ((0, "front"), (1, "rear")):
        scan = _read_scan(bridge, side)
        if scan is not None:
            scans.append((mount_index, scan))
    return scans


def _ray_indices(size: int, count: int) -> np.ndarray:
    """等间隔抽 count 条射线的下标（含首尾、去重升序）；count >= size 时全取。"""
    picks = np.linspace(0.0, float(size - 1), min(count, size)).round()
    return np.unique(picks.astype(np.int64))


def _lidar_min_range(bridge: object, scene: Scene) -> float | None:
    """最近的真实回波距离（米）；没有激光数据或全部饱和时给 None。"""
    best: float | None = None
    for _mount, scan in _lidar_scans(bridge):
        finite = scan[np.isfinite(scan) & (scan > 0.0)]
        if finite.size == 0:
            continue
        value = float(np.min(finite))
        if value >= scene.max_range * _LIDAR_SATURATED:
            continue
        best = value if best is None else min(best, value)
    return best


def _collect_lidar_rays(
    bridge: object, scene: Scene, *, max_rays: int = _MAX_LIDAR_RAYS
) -> np.ndarray:
    """把真实扫描等间隔抽样成 (M, 4) 线段 [x0, y0, x1, y1]，长度 = 该射线真实距离。

    抽样是总数为上限：前后雷达同时存在时各一半（默认 36）。数据全部来自桥自己的
    求交结果：get_scan(side) 用 +inf 表示无回波，get_lidar_front() / get_lidar_rear()
    给出含饱和值的距离。无回波整条丢弃，绝不把「没打到」画成「打到很远」；桥没有
    任何激光能力时返回空数组，渲染层据此零光束。LiDAR 安装位姿与扫描角度公式和
    SimpleBridge._simulate_lidar 保持一致。
    """
    scans = _lidar_scans(bridge)
    if not scans or max_rays <= 0:
        return np.empty((0, 4), dtype=np.float64)
    per_scan = max(1, max_rays // len(scans))
    x, y, theta = scene.pose
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    chunks: list[np.ndarray] = []
    for mount_index, scan in scans:
        usable = np.flatnonzero(np.isfinite(scan) & (scan > 0.0))
        if usable.size == 0:
            continue
        indices = usable[_ray_indices(usable.size, per_scan)]
        off_x, off_y, angle_off = scene.lidar_mounts[mount_index]
        origin_x = x + off_x * cos_t - off_y * sin_t
        origin_y = y + off_x * sin_t + off_y * cos_t
        angles = (
            theta + angle_off - math.pi / 2.0 + np.linspace(0.0, math.pi, scan.size)
        )
        dist = np.minimum(scan[indices], scene.max_range)
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
    return np.concatenate(chunks, axis=0)[:max_rays]


# ── 对外入口 ─────────────────────────────────────────────────────
def render_scene(
    scene: Scene,
    *,
    ax: Any = None,
    lidar_rays: np.ndarray | None = None,
    show_trail: bool = False,
    trail: object = None,
    dpi: float = _DEFAULT_DPI,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    steering: float | None = None,
    telemetry: dict[str, Any] | None = None,
    title: str | None = None,
    zoom: float | None = None,
) -> Any:
    """把场景画到一张新 Figure（或给定 axes）上，返回 Figure。

    `ax` 非空时复用之（dpi / figsize 由调用方的 Figure 决定），否则新建
    满幅 Figure（画布色取自 PALETTE）；LiDAR 线段需由调用方预先算好传入。`telemetry` 是 HUD
    读数（缺省时 HUD 全部显示 `—`），`title` 是右上角附注。
    """
    mpl = _import_matplotlib()
    if ax is None:
        fig = mpl.Figure(figsize=figsize, dpi=dpi, facecolor=PALETTE["canvas"])
        mpl.FigureCanvasAgg(fig)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    else:
        fig = ax.figure

    hud = dict(telemetry) if telemetry else {}
    _setup_axes(ax, scene.world_bounds)
    _draw_grid(ax, scene, mpl)
    frame = _draw_boundary(ax, scene, mpl)
    detail = _detail_scale(fig, scene)
    _draw_obstacles(ax, scene, mpl, clip=frame)
    _draw_path(ax, scene, mpl)
    if show_trail and trail is not None:
        _draw_trail(ax, trail, mpl)
    if lidar_rays is not None:
        _draw_lidar(
            ax,
            np.asarray(lidar_rays, dtype=np.float64),
            scene.max_range,
            mpl,
            clip=frame,
        )
    _draw_vehicle(ax, scene, mpl, steering=steering, detail=detail, clip=frame)
    _draw_hud(ax, mpl, hud, title=title)
    if zoom is not None:
        _draw_zoom_inset(
            fig,
            ax,
            scene,
            mpl,
            zoom=float(zoom),
            lidar_rays=lidar_rays,
            steering=steering,
        )
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
    title: str | None = None,
    zoom: float | None = None,
) -> Any:
    """渲染桥的当前状态，返回 matplotlib Figure。

    Args:
        bridge: `SimpleBridge` 或同构桥（只读访问位姿 / 障碍 / 路径）。
        ax: 复用的 axes；为空时新建满幅 Figure（无白边、无轴框）。
        show_lidar: 是否叠加 LiDAR 扇形光束（默认关闭，总计最多 36 条）。
        show_trail: 是否绘制渐隐轨迹尾迹（默认关闭，最多保留最后 500 点）。
        trail: 尾迹点序列，形状 (N, 2) / (N, 3)；`show_trail=False` 时忽略。
        dpi: 新建 Figure 的分辨率；复用 ax 时由原 Figure 决定。
        figsize: 新建 Figure 的英寸尺寸。
        title: 右上角附注（可选）。
        zoom: 右下角缩放内嵌窗的边长（米）；None（默认）不出窗，输出与旧版逐像素一致。
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
        steering=steering_angle(bridge),
        telemetry=_bridge_telemetry(bridge, scene),
        title=title,
        zoom=zoom,
    )


def save_figure(fig: Any, path: str | Path, *, dpi: float | None = None) -> Path:
    """把 Figure 写成确定性 PNG（去掉 `Software` 元数据，无时间戳）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        target,
        format="png",
        dpi=dpi if dpi is not None else fig.dpi,
        facecolor=fig.get_facecolor(),
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
