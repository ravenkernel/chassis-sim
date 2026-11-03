"""端到端渲染演示：AckermannChassis + SimpleBridge 跑 50 步，输出 5 帧 PNG。

用法（仓库根目录）：
    python examples/render_demo.py

产物：/tmp/render_demo/frame_00010.png ... frame_00050.png（共 5 帧，600x600）。
同时打印总字节数、单帧尺寸与 7 次渲染耗时中位数（毫秒）。
"""

from __future__ import annotations

import statistics
import struct
import time
from pathlib import Path

import numpy as np

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.render import render_episode_frames, render_to_png

OUT_DIR = Path("/tmp/render_demo")
N_STEPS = 50
N_FRAMES = 5
EVERY = N_STEPS // N_FRAMES
DPI = 100.0
BENCH_ROUNDS = 7


def _make_path() -> np.ndarray:
    s = np.linspace(0.0, 1.0, 80)
    x = 1.0 + 8.0 * s
    y = 1.0 + 8.0 * s
    theta = np.full_like(s, np.pi / 4.0)
    return np.column_stack([x, y, theta])


def _make_obstacles() -> list[dict]:
    """20 个障碍：沿对角线两侧各偏移 2.2 m，避开参考路径。"""
    obstacles: list[dict] = []
    perpendicular = np.array([-np.sqrt(0.5), np.sqrt(0.5)])
    for index, t in enumerate(np.linspace(2.0, 8.0, 10)):
        for side in (1.0, -1.0):
            cx, cy = np.array([t, t]) + side * 2.2 * perpendicular
            if (index + int(side)) % 2 == 0:
                obstacles.append(
                    {
                        "shape": {"name": "circle", "radius": 0.25},
                        "state": [float(cx), float(cy), 0.0],
                    }
                )
            else:
                obstacles.append(
                    {
                        "shape": {
                            "name": "rectangle",
                            "length": 0.6,
                            "width": 0.3,
                        },
                        "state": [float(cx), float(cy), 0.4],
                    }
                )
    return obstacles


def _make_actions() -> np.ndarray:
    steps = np.arange(N_STEPS, dtype=np.float64)
    steer = 0.25 * np.sin(2.0 * np.pi * steps / 25.0)
    speed = np.full(N_STEPS, 0.5, dtype=np.float64)
    return np.column_stack([steer, speed])


def _png_size(path: Path) -> tuple[int, int]:
    """读 PNG IHDR：宽高各 4 字节大端，偏移 16 和 20。"""
    header = path.read_bytes()[16:24]
    width, height = struct.unpack(">II", header)
    return int(width), int(height)


def main() -> None:
    bridge = SimpleBridge(AckermannChassis())
    bridge.reset(obstacles=_make_obstacles(), path_waypoints=_make_path())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("frame_*.png"):
        stale.unlink()

    frames = render_episode_frames(
        bridge,
        _make_actions(),
        OUT_DIR,
        every=EVERY,
        dpi=DPI,
        show_trail=True,
    )

    bench_path = OUT_DIR / "_bench.png"
    timings_ms: list[float] = []
    for _ in range(BENCH_ROUNDS):
        start = time.perf_counter()
        render_to_png(bridge, bench_path, dpi=DPI, show_trail=True)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    bench_path.unlink()

    total_bytes = sum(frame.stat().st_size for frame in frames)
    width, height = _png_size(frames[0])
    print(f"frames={len(frames)}")
    print("files=" + ",".join(frame.name for frame in frames))
    print(f"total_bytes={total_bytes}")
    print(f"per_frame_bytes={[frame.stat().st_size for frame in frames]}")
    print(f"frame_size={width}x{height}")
    print(f"median_frame_ms={statistics.median(timings_ms):.2f}")
    print(f"samples_ms={[round(value, 2) for value in timings_ms]}")


if __name__ == "__main__":
    main()
