"""端到端渲染演示：10x10 m 密集场景里 Ackermann 底盘沿规划路径跑 60 步。

用法（仓库根目录）：
    python examples/render_demo.py

产物（/tmp/render_demo/）：
    hero.png           900x900 末状态（LiDAR 扇形 + 尾迹渐隐 + 右下 2 m 缩放窗）
    hero_plain.png     同图但 zoom=None（默认输出，证明旧行为未被改动）
    hero_nolidar.png   同图关掉 LiDAR（对照：光束长度来自真实回波）
    frames/frame_*.png 60 张 900x900 过程图（dpi=150，也是耗时基准）
    episode.gif        过程图每 10 帧取一张串成的 GIF（缺 imageio 时降级成帧目录）

同时打印 60 帧全量耗时、单帧字节数、GIF 字节数。

这里刻意用 chassis_sim.render 的画图原语（可控 hero / trail / zoom）：要“录一个
回合”或“存一张静图”用一键入口 cs.record(env, ctrl, out=...) / cs.snapshot(env, path)。
"""

from __future__ import annotations

import statistics
import struct
import time
from pathlib import Path

import numpy as np

import chassis_sim as cs
from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.render import render_episode_frames, render_to_png, save_gif
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

OUT_DIR = Path("/tmp/render_demo")
SCENARIO = "obstacles_dense"
N_STEPS = 60
GIF_STRIDE = 10
DPI = 150.0
GIF_FPS = 8
TITLE = "chassis_sim · obstacles_dense"
ZOOM = 2.0  # 右下角缩放窗边长（米）：10x10 m 世界里车体只有 ~30 px，靠它看清四轮与车头


def _build() -> SimpleBridge:
    """10x10 m 密集障碍场景 + 阿克曼底盘（世界边界、起终点都来自 scenario）。"""
    scenario = load_scenario(SCENARIO)
    bridge = SimpleBridge(AckermannChassis(), world_bounds=tuple(scenario.world_bounds))
    bridge.reset(**scenario_to_reset_options(scenario, seed=0))
    return bridge


def _drive(bridge: SimpleBridge, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """纯追踪闭环跑 steps 步，返回 (动作序列, 位姿序列)。"""
    controller = cs.make_controller(
        "pp",
        chassis=bridge.chassis,
        lookahead=0.55,
        speed_gain=0.9,
        slowdown_distance=0.45,
    )
    controller.set_path(np.asarray(bridge.get_path_global(), dtype=np.float64))
    actions: list[np.ndarray] = []
    poses: list[tuple[float, float, float]] = []
    for _ in range(steps):
        action = controller.act({"pose": bridge.pose()})
        bridge.step(np.asarray(action, dtype=np.float64))
        actions.append(action)
        poses.append(bridge.pose())
    return np.asarray(actions, dtype=np.float64), np.asarray(poses, dtype=np.float64)


def _png_size(path: Path) -> tuple[int, int]:
    """读 PNG IHDR：宽高各 4 字节大端，偏移 16 和 20。"""
    width, height = struct.unpack(">II", path.read_bytes()[16:24])
    return int(width), int(height)


def main() -> None:
    bridge = _build()
    actions, poses = _drive(bridge, N_STEPS)
    collided = bridge.check_collision()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("frame_*.png"):
        stale.unlink()

    hero = render_to_png(
        bridge,
        OUT_DIR / "hero.png",
        dpi=DPI,
        show_lidar=True,
        show_trail=True,
        trail=poses,
        title=TITLE,
        zoom=ZOOM,
    )
    render_to_png(
        bridge,
        OUT_DIR / "hero_plain.png",
        dpi=DPI,
        show_lidar=True,
        show_trail=True,
        trail=poses,
        title=TITLE,
    )
    render_to_png(
        bridge,
        OUT_DIR / "hero_nolidar.png",
        dpi=DPI,
        show_lidar=False,
        show_trail=True,
        trail=poses,
        title=TITLE,
        zoom=ZOOM,
    )

    start = time.perf_counter()
    frames = render_episode_frames(
        bridge,
        actions,
        OUT_DIR / "frames",
        dpi=DPI,
        show_lidar=True,
        show_trail=True,
        title=TITLE,
        zoom=ZOOM,
    )
    elapsed = time.perf_counter() - start
    gif = save_gif(frames[::GIF_STRIDE], OUT_DIR / "episode.gif", fps=GIF_FPS)

    width, height = _png_size(hero)
    sizes = [frame.stat().st_size for frame in frames]
    print(
        f"scenario={SCENARIO} steps={N_STEPS} frames={len(frames)} collided={collided}"
    )
    print(f"hero={hero} size={width}x{height} bytes={hero.stat().st_size}")
    print(f"frames_seconds={elapsed:.2f} per_frame_s={elapsed / len(frames):.3f}")
    print(
        f"frame_bytes_min={min(sizes)} frame_bytes_median={int(statistics.median(sizes))}"
        f" frame_bytes_max={max(sizes)}"
    )
    print(f"gif={gif} is_file={gif.is_file()} bytes={gif.stat().st_size}")


if __name__ == "__main__":
    main()
