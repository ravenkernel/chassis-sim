"""录一个回合成 GIF / 把当前画面存成 PNG：`record()` 与 `snapshot()`。

两个入口都只在**函数体内** import matplotlib / imageio，因此
`import chassis_sim` 与 `import chassis_sim.recording` 不会拉起重模块。
缺依赖时抛 `ImportError`，消息里带 `pip install "chassis-sim[viz]"`。

静态图入口叫 `snapshot()` 而不是 `render()`：本包已有 `chassis_sim.render`
子包，顶层再挂同名函数必然被撞掉 —— `record()`/本模块内部 `from .render import`
触发子模块导入时，import 系统会把父包属性 `chassis_sim.render` 重绑成 module，
同名的函数对象就此失效。这是真冲突，不是命名口味。
"""

from __future__ import annotations

import importlib
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .eval import controller_observation

__all__ = ["RecordingResult", "record", "snapshot"]

_VIZ_HINT = 'pip install "chassis-sim[viz]"'
#: 可视化依赖的导入点（顺序即消息里的顺序）；缺哪个就报哪个，缺几个报几个
_VIZ_MODULES: tuple[str, ...] = ("matplotlib", "imageio.v2")


@dataclass(frozen=True)
class RecordingResult:
    """一次 `record()` 的结果。

    Attributes:
        path: 实际写出的文件（通常是 GIF；极端降级下是帧目录）。
        frames: 写进 GIF 的帧数（PNG 张数）。
        steps: 回合里 env.step() 的次数。
        total_reward: 逐帧累加的 reward。
        terminated: env 自己报的终止（到达目标 / 碰撞）。
        truncated: env 报的截断，或调用方 max_steps 提前收工（外部截断）。
    """

    path: Path
    frames: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool


def _require_viz(who: str) -> None:
    """提前确认可视化依赖可用（两个 import 都发生在函数体内）。

    缺几个就一次报几个：只报第一个命中的模块，在"core 安装（matplotlib 与
    imageio 都没有）"下会漏掉另一半，用户按提示装完再撞第二次。
    """
    missing: list[str] = []
    for name in _VIZ_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name.split(".")[0])
    if missing:
        raise ImportError(
            f"{who} 需要 {'、'.join(missing)}；安装可视化依赖：{_VIZ_HINT}"
        )


def _bridge_of(target: Any) -> Any:
    """从 env 取公开属性 `bridge`；没有该属性时认为 target 本身就是 bridge。"""
    if target is None:
        raise TypeError("需要一个 env（公开属性 bridge）或 bridge 实例，收到 None")
    bridge = getattr(target, "bridge", None)
    return bridge if bridge is not None else target


def _world_path(bridge: Any) -> np.ndarray | None:
    """桥的世界系参考路径 (N, 3)；没有路径时返回 None（控制器就不 set_path）。"""
    getter = getattr(bridge, "get_path_global", None)
    if not callable(getter):
        return None
    path = np.asarray(getter(), dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        return None
    return path


def snapshot(target: Any, path: str | Path = "snapshot.png", **kwargs: Any) -> str:
    """把 env（或 bridge）的当前画面存成 PNG，返回写出的路径字符串。

    Args:
        target: env 实例（取其公开属性 `bridge`）或 bridge 本身。
        path: 目标 PNG 路径；父目录自动创建。
        **kwargs: 透传给 `render.render_to_png`（`show_lidar` / `title` /
            `zoom` / `dpi` / `figsize` ...）。

    Returns:
        写出的 PNG 路径（str）。

    Raises:
        ImportError: 缺 matplotlib / imageio 等可视化依赖。
    """
    _require_viz("snapshot()")
    from .render import render_to_png

    out = Path(path)
    render_to_png(_bridge_of(target), out, **kwargs)
    return str(out)


def record(
    env: Any,
    controller: Any = None,
    *,
    out: str | Path = "demo.gif",
    every: int = 1,
    fps: int = 12,
    max_steps: int | None = None,
    seed: int = 0,
    reset_options: Mapping[str, Any] | None = None,
    save_frames: bool = False,
    **render_kwargs: Any,
) -> RecordingResult:
    """跑一个回合，每 `every` 步渲染一帧，最后合成 GIF。

    回合跑法与 `evaluate()` 同口径：`reset(seed=seed, options=...)` →
    循环 `step(controller.act(controller_observation(env)))` →
    env 报 terminated/truncated 或到达步数上限时停。最后一帧总会被写出
    （步数不是 `every` 整数倍时补一帧）。

    Args:
        env: env 实例（需要一个公开的 `bridge` 用来渲染）。
        controller: 控制器实例，必须给（本函数不提供 `actions=` 回放入口）。
            控制器按车型标定，通常传 `make_controller("pp", chassis=env.unwrapped.chassis)`。
        out: GIF 目标路径，父目录自动创建。
        every: 每几步渲染一帧（>= 1）。
        fps: GIF 帧率（> 0）。
        max_steps: 步数上限；`None` 用 env 自己的上限（`env.max_steps`）。
        seed: 传给 `env.reset(seed=seed)` 的种子。
        reset_options: 传给 `env.reset(options=...)`；`None` 时用
            `env.default_reset_options`（`make()` 造出来的 env 才有）。
        save_frames: True 时把逐帧 PNG 留在 `<out 名>_frames/`，False 用完即删。
        **render_kwargs: 透传给 `render.render_to_png`（`show_lidar` / `title` ...）。

    Returns:
        `RecordingResult`（GIF 路径、帧数、步数、累计 reward、终止/截断标志）。

    Raises:
        TypeError: 没给 `controller`，或 env 取不到 bridge。
        ValueError: `every` / `fps` 非法。
        ImportError: 缺 matplotlib / imageio 等可视化依赖。
    """
    if controller is None:
        raise TypeError(
            "record() 需要 controller=（1.0 不提供 actions= 回放入口）；"
            '控制器按车型标定，例如 make_controller("pp", chassis=env.unwrapped.chassis)'
        )
    if int(every) < 1:
        raise ValueError(f"every 必须是正整数，收到 {every}")
    if int(fps) <= 0:
        raise ValueError(f"fps 必须是正数，收到 {fps}")

    _require_viz("record()")
    from .render import render_to_png, save_gif

    bridge = _bridge_of(env)
    gif_path = Path(out)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    if save_frames:
        # 复用同名目录前先清空：否则上一回合的旧帧会留在盘上冒充这一回合
        frames_dir = gif_path.with_name(f"{gif_path.stem}_frames")
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = Path(tempfile.mkdtemp(prefix="chassis_sim_rec_"))
    steps_limit = (
        max_steps if max_steps is not None else getattr(env, "max_steps", None)
    )
    options = (
        reset_options
        if reset_options is not None
        else getattr(env, "default_reset_options", None)
    )

    env.reset(seed=seed, options=options)
    if hasattr(controller, "reset"):
        controller.reset()
    path = _world_path(bridge)
    if path is not None and hasattr(controller, "set_path"):
        controller.set_path(path)

    written: list[Path] = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    try:
        while True:
            action = controller.act(controller_observation(env))
            _, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            total_reward += float(reward)
            if steps % int(every) == 0:
                written.append(
                    render_to_png(
                        bridge, frames_dir / f"frame_{steps:05d}.png", **render_kwargs
                    )
                )
            if bool(terminated) or bool(truncated):
                break
            if steps_limit is not None and steps >= int(steps_limit):
                truncated = True
                break
        if steps % int(every) != 0:
            written.append(
                render_to_png(
                    bridge, frames_dir / f"frame_{steps:05d}.png", **render_kwargs
                )
            )
        target = save_gif(written, gif_path, fps=int(fps))
    finally:
        if not save_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

    return RecordingResult(
        path=Path(target),
        frames=len(written),
        steps=steps,
        total_reward=total_reward,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )
