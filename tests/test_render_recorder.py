"""回放层：npz 往返、无桥回放成帧，以及 episode 帧抽取。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis

# 回放层只在函数内部 import matplotlib（缺失时抛 ImportError 而不是
# ModuleNotFoundError），跳过必须发生在任何 chassis_sim.render 导入与调用之前。
pytest.importorskip(
    "matplotlib",
    reason="render 测试需要 matplotlib（pip install chassis-sim[viz]）",
)

from chassis_sim.render import ReplayRecorder, render_episode_frames


def _make_bridge() -> SimpleBridge:
    bridge = SimpleBridge(AckermannChassis())
    bridge.reset(
        obstacles=[
            {
                "shape": {"name": "circle", "radius": 0.25},
                "state": [6.0, 6.0, 0.0],
            },
            {
                "shape": {"name": "rectangle", "length": 0.6, "width": 0.3},
                "state": [4.0, 7.0, 0.2],
            },
        ],
        path_waypoints=np.array([[1.0, 1.0, 0.0], [8.0, 2.0, 0.0]], dtype=np.float64),
    )
    return bridge


def test_npz_roundtrip_and_replay_frames(tmp_path: Path) -> None:
    bridge = _make_bridge()
    recorder = ReplayRecorder(bridge)
    action = np.array([0.15, 0.5], dtype=np.float64)
    for step in range(8):
        recorder.record_step(bridge, action, reward=-float(step))

    npz = recorder.dump(tmp_path / "episode.npz")
    assert npz.is_file()
    assert npz.suffix == ".npz"

    loaded = ReplayRecorder.load(npz)
    assert len(loaded) == 8
    assert np.allclose(loaded.actions, recorder.actions)
    assert np.allclose(loaded.poses, recorder.poses)
    assert np.allclose(loaded.rewards, recorder.rewards)
    assert loaded.initial_pose == recorder.initial_pose

    frames = loaded.render_frames(tmp_path / "replay", every=2, dpi=48.0)
    assert len(frames) == 5
    for frame in frames:
        assert frame.is_file()
        assert frame.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_episode_frames_writes_last_frame(tmp_path: Path) -> None:
    bridge = _make_bridge()
    actions = [np.array([0.0, 0.5], dtype=np.float64) for _ in range(5)]
    frames = render_episode_frames(
        bridge, actions, tmp_path / "episode", every=2, dpi=48.0, show_trail=True
    )
    # 第 2、4 步 + 最后一步（第 5 步），共 3 帧
    assert [path.name for path in frames] == [
        "frame_00002.png",
        "frame_00004.png",
        "frame_00005.png",
    ]
