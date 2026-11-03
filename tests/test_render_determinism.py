"""确定性：同一状态渲染两次，PNG 像素数组必须逐位相等且无 Software 元数据。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis

# render 层只在函数内部 import matplotlib（缺失时抛 ImportError 而不是
# ModuleNotFoundError），跳过必须发生在任何 chassis_sim.render 导入与调用之前。
matplotlib = pytest.importorskip(
    "matplotlib",
    reason="render 测试需要 matplotlib（pip install chassis-sim[viz]）",
)

from matplotlib.image import imread

from chassis_sim.render import render_to_png

DPI = 100.0


def _make_bridge() -> SimpleBridge:
    bridge = SimpleBridge(AckermannChassis())
    path = np.array(
        [[2.0, 2.0, math.pi / 6.0], [8.0, 7.0, math.pi / 6.0]],
        dtype=np.float64,
    )
    obstacles = [
        {
            "shape": {"name": "circle", "radius": 0.35},
            "state": [4.0, 5.0, 0.0],
        },
        {
            "shape": {"name": "rectangle", "length": 0.7, "width": 0.3},
            "state": [6.0, 3.5, -0.4],
        },
    ]
    bridge.reset(obstacles=obstacles, path_waypoints=path)
    bridge.step(np.array([0.2, 0.6], dtype=np.float64))
    return bridge


def test_two_renders_are_bit_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bridge = _make_bridge()
    first = render_to_png(bridge, tmp_path / "first.png", dpi=DPI)
    second = render_to_png(bridge, tmp_path / "second.png", dpi=DPI)

    image_a = imread(first)
    image_b = imread(second)
    assert image_a.shape == image_b.shape
    max_delta = float(np.max(np.abs(image_a - image_b)))
    print(f"max|Δpixel| = {max_delta}")
    assert np.array_equal(image_a, image_b)
    assert max_delta == 0.0

    # 去掉 Software 元数据后，PNG 里不应再出现 matplotlib 版本字样或时间戳
    for png in (first, second):
        raw = png.read_bytes()
        assert b"Software" not in raw
        assert b"Creation Time" not in raw
