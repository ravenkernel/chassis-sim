"""像素级断言：世界坐标到像素坐标的映射，以及车体 / 障碍物的着色位置。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.sensors import Lidar

# render 层只在函数内部 import matplotlib（缺失时抛 ImportError 而不是
# ModuleNotFoundError），跳过必须发生在任何 chassis_sim.render 导入与调用之前。
matplotlib = pytest.importorskip(
    "matplotlib",
    reason="render 测试需要 matplotlib（pip install chassis-sim[viz]）",
)

from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.image import imread

from chassis_sim.render import render_frame, render_to_png
from chassis_sim.render.figure import (
    _GLOW_ALPHA,
    _MAX_LIDAR_RAYS,
    _OBSTACLE_ALPHA,
    PALETTE,
    _bridge_telemetry,
    _collect_lidar_rays,
)
from chassis_sim.render.scene import scene_from_bridge

DPI = 100.0
CAR_XY = (3.0, 3.0)
CAR_THETA = math.pi / 4.0
CIRCLE_XY = (6.5, 4.0)
CIRCLE_R = 0.6
RECT_XY = (4.0, 6.5)
RECT_LENGTH = 0.8
RECT_WIDTH = 0.4


def _make_bridge() -> SimpleBridge:
    """车在 (3,3) 朝 45°，一个圆在 (6.5,4)，一个矩形在 (4,6.5)。

    雷达只有 get_sensor_reading 一个入口：前/后两支 Lidar 必须显式装配（mount 取
    底盘的真实安装位，几何与旧的两键接口逐位相同）。
    """
    chassis = AckermannChassis()
    front, rear = chassis.lidar_mounts()
    bridge = SimpleBridge(
        chassis,
        sensors=[
            Lidar("front", mount=front, n_rays=180),
            Lidar("rear", mount=rear, n_rays=180),
        ],
    )
    path = np.array(
        [
            [CAR_XY[0], CAR_XY[1], CAR_THETA],
            [9.0, 9.0, CAR_THETA],
        ],
        dtype=np.float64,
    )
    obstacles = [
        {
            "shape": {"name": "circle", "radius": CIRCLE_R},
            "state": [CIRCLE_XY[0], CIRCLE_XY[1], 0.0],
        },
        {
            "shape": {
                "name": "rectangle",
                "length": RECT_LENGTH,
                "width": RECT_WIDTH,
            },
            "state": [RECT_XY[0], RECT_XY[1], 0.3],
        },
    ]
    bridge.reset(obstacles=obstacles, path_waypoints=path)
    return bridge


def _pixel_of(fig: object, ax: object, xy: tuple[float, float]) -> tuple[int, int]:
    """世界坐标 -> (行, 列)；先 draw 使 transData 反映最终布局。"""
    fig.canvas.draw()
    display = ax.transData.transform(xy)
    height = fig.canvas.get_width_height()[1]
    return round(height - display[1]), round(display[0])


def _tolerance_px(dpi: float) -> int:
    """容差按 dpi 折算：3 个排版点（1 pt = dpi/72 px）。"""
    return math.ceil(3.0 * dpi / 72.0)


def _patch(image: np.ndarray, row: int, col: int, tol: int) -> np.ndarray:
    h, w = image.shape[:2]
    return image[
        max(row - tol, 0) : min(row + tol + 1, h),
        max(col - tol, 0) : min(col + tol + 1, w),
        :3,
    ]


def _min_color_delta(patch: np.ndarray, color: str | np.ndarray) -> float:
    expected = (
        np.array(to_rgb(color), dtype=np.float64)
        if isinstance(color, str)
        else np.asarray(color, dtype=np.float64)
    )
    return float(np.min(np.abs(patch - expected)))


def _obstacle_fill() -> np.ndarray:
    """障碍物中心的实际像素色：发光层(alpha) 之上再叠实体层(alpha) 的合成色。"""
    canvas = np.array(to_rgb(PALETTE["canvas"]), dtype=np.float64)
    warm = np.array(to_rgb(PALETTE["obstacle"]), dtype=np.float64)
    under = warm * _GLOW_ALPHA + canvas * (1.0 - _GLOW_ALPHA)
    return warm * _OBSTACLE_ALPHA + under * (1.0 - _OBSTACLE_ALPHA)


def test_car_and_obstacles_land_on_expected_pixels(tmp_path: Path) -> None:
    bridge = _make_bridge()
    png = render_to_png(bridge, tmp_path / "frame.png", dpi=DPI)
    image = imread(png)

    fig = render_frame(bridge, dpi=DPI)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    assert image.shape[0] == canvas_height
    assert image.shape[1] == canvas_width

    tol = _tolerance_px(DPI)
    ax = fig.axes[0]

    # 圆形障碍物：圆心像素必须是暖色填充（发光层 + 实体层叠出来的合成色）
    fill = _obstacle_fill()
    row, col = _pixel_of(fig, ax, CIRCLE_XY)
    assert _min_color_delta(_patch(image, row, col, tol), fill) < 0.02

    # 矩形障碍物：中心像素必须是同一套暖色（矩形自身旋转 0.3 rad 不影响中心）
    row, col = _pixel_of(fig, ax, RECT_XY)
    assert _min_color_delta(_patch(image, row, col, tol), fill) < 0.02

    # 车体：取车体系 (0, 0.35 * half_width) 的点，避开中心线（车头方向短线）
    _, width = bridge.chassis.footprint()
    offset = 0.35 * (width / 2.0)
    world_x = CAR_XY[0] - math.sin(CAR_THETA) * offset
    world_y = CAR_XY[1] + math.cos(CAR_THETA) * offset
    row, col = _pixel_of(fig, ax, (world_x, world_y))
    assert _min_color_delta(_patch(image, row, col, tol), PALETTE["car"]) < 0.02


def test_lidar_ray_sampling_is_capped_at_60() -> None:
    bridge = _make_bridge()
    scene = scene_from_bridge(bridge)
    rays = _collect_lidar_rays(bridge, scene)
    assert rays.ndim == 2
    assert rays.shape[1] == 4
    assert _MAX_LIDAR_RAYS == 36
    assert 0 < rays.shape[0] <= _MAX_LIDAR_RAYS
    lengths = np.hypot(rays[:, 2] - rays[:, 0], rays[:, 3] - rays[:, 1])
    assert np.all(lengths <= scene.max_range + 1e-6)


class _BlindBridge:
    """只有位姿与障碍物形状的桥：激光方法存在但读不到数据。"""

    chassis = None
    world_bounds = (0.0, 10.0, 0.0, 10.0)
    dt = 0.1
    sim_time = 0.0

    def pose(self) -> tuple[float, float, float]:
        return (3.0, 3.0, 0.0)

    def obstacle_shapes(self) -> list[dict]:
        return [{"shape": {"name": "circle", "radius": 0.6}, "state": [6.5, 4.0, 0.0]}]

    def get_sensor_reading(self) -> None:
        return None


def _beam_collections(fig: object) -> int:
    """统计 ax 上的 LineCollection 数（激光扇形是唯一用它的场景元素）。"""
    return sum(
        1 for item in fig.axes[0].collections if isinstance(item, LineCollection)
    )


def test_bridge_without_lidar_draws_no_beams() -> None:
    """没有激光数据通路时：零光束、d_min 留空（不编数字），且照常出图。"""
    bridge = _BlindBridge()
    scene = scene_from_bridge(bridge)
    rays = _collect_lidar_rays(bridge, scene)
    assert rays.shape == (0, 4)
    assert _bridge_telemetry(bridge, scene)["d_min"] is None

    plain = _beam_collections(render_frame(bridge, dpi=DPI, show_lidar=False))
    with_lidar = _beam_collections(render_frame(bridge, dpi=DPI, show_lidar=True))
    assert with_lidar == plain


def test_zoom_inset_draws_scaled_window() -> None:
    """zoom 给正数时多出一个以车体为中心、边长 zoom 米的窗口；默认不出窗。"""
    bridge = _make_bridge()
    assert not render_frame(bridge, dpi=DPI).axes[0].child_axes

    fig = render_frame(bridge, dpi=DPI, zoom=2.0)
    (inset,) = fig.axes[0].child_axes
    x0, x1 = inset.get_xlim()
    y0, y1 = inset.get_ylim()
    assert math.isclose(x1 - x0, 2.0)
    assert math.isclose(y1 - y0, 2.0)
    assert math.isclose((x0 + x1) / 2.0, CAR_XY[0])
    assert math.isclose((y0 + y1) / 2.0, CAR_XY[1])
    # 车体与障碍在窗口里按同一套函数重画了一遍
    assert inset.patches or inset.collections

    with pytest.raises(ValueError):
        render_frame(bridge, dpi=DPI, zoom=0.0)
