"""imageio 缺失时的降级路径：写帧目录 + 一条 [WARN][render] 日志，不抛异常。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.render import render_to_png, save_gif

matplotlib = pytest.importorskip("matplotlib")


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
