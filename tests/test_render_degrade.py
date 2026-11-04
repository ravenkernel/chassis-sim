"""imageio 缺失时的降级路径：写帧目录 + 一条 [WARN][render] 日志，不抛异常。"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis

# 这个模块遮的是 imageio（降级路径），正向路径仍要 matplotlib 才能出帧；
# 跳过必须发生在任何 chassis_sim.render 导入与调用之前。
matplotlib = pytest.importorskip(
    "matplotlib",
    reason="render 测试需要 matplotlib（pip install chassis-sim[viz]）",
)

from chassis_sim.render import render_to_png, save_gif


def _make_frames(tmp_path: Path) -> list[Path]:
    bridge = SimpleBridge(AckermannChassis())
    bridge.reset(
        obstacles=[
            {
                "shape": {"name": "circle", "radius": 0.3},
                "state": [5.0, 5.0, 0.0],
            }
        ],
        path_waypoints=np.array([[1.0, 1.0, 0.0], [9.0, 1.0, 0.0]]),
    )
    frames = []
    for index in range(2):
        bridge.step(np.array([0.0, 0.5], dtype=np.float64))
        frames.append(render_to_png(bridge, tmp_path / f"src_{index}.png", dpi=48.0))
    return frames


def test_save_gif_falls_back_when_imageio_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    frames = _make_frames(tmp_path)
    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "imageio.v2", None)

    with caplog.at_level(logging.WARNING, logger="chassis_sim.render"):
        result = save_gif(frames, tmp_path / "episode.gif", fps=5)

    assert result.is_dir()
    written = sorted(result.glob("*.png"))
    assert [path.name for path in written] == [
        "frame_00000.png",
        "frame_00001.png",
    ]
    assert [path.read_bytes() for path in written] == [
        frame.read_bytes() for frame in frames
    ]

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "chassis_sim.render"
        and record.getMessage().startswith("[WARN][render]")
    ]
    assert warnings, [record.getMessage() for record in caplog.records]
    print(warnings[0])


def test_save_gif_writes_gif_when_imageio_present(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    frames = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.full((16, 16, 3), 255, dtype=np.uint8),
    ]
    result = save_gif(frames, tmp_path / "anim.gif", fps=4)
    assert result.is_file()
    assert result.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def _matplotlib_frames() -> list[np.ndarray]:
    """用 matplotlib 画 3 帧小图，返回 (H, W, 3) uint8 数组。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    frames: list[np.ndarray] = []
    for index in range(3):
        figure = Figure(figsize=(1.0, 1.0), dpi=32.0)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        axes.plot([0.0, 1.0], [0.0, float(index)])
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(np.ascontiguousarray(rgba[..., :3]))
    return frames


def test_save_gif_duration_has_no_fps_deprecation(tmp_path: Path) -> None:
    """fps 必须换算成 duration(ms)，且 imageio 不再打 fps 弃用告警。"""
    pytest.importorskip("imageio")
    image = pytest.importorskip("PIL.Image")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = save_gif(_matplotlib_frames(), tmp_path / "anim.gif", fps=10)

    assert result.suffix == ".gif"
    assert result.is_file()

    with image.open(result) as gif:
        assert getattr(gif, "n_frames", 1) == 3

    offenders = [
        str(record.message)
        for record in caught
        if issubclass(record.category, DeprecationWarning)
        and "fps" in str(record.message)
    ]
    assert offenders == []
