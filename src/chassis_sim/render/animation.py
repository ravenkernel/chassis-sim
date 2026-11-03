"""GIF 导出：imageio 为可选依赖，缺失时降级为帧目录 + 一条 [WARN] 日志。

降级路径不抛异常：把每一帧复制（或写成 PNG）到 `<gif 名>_frames/` 目录，
并用 `logging.getLogger("chassis_sim.render")` 打一条 WARNING。核心依赖
（numpy / gymnasium / pyyaml）不受影响。
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER = logging.getLogger("chassis_sim.render")
_WARN_PREFIX = "[WARN][render]"


def _import_imageio() -> Any:
    """导入 imageio.v2；缺失或被 `sys.modules[...] = None` 屏蔽时抛 ImportError。"""
    for name in ("imageio", "imageio.v2"):
        if name in sys.modules and sys.modules[name] is None:
            raise ImportError(f"{name} 已被屏蔽")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("imageio 未安装") from exc
    return imageio


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """把任意图像数组统一成 (H, W, 3) uint8，供 GIF 编码。"""
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0.0, 255.0).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.ascontiguousarray(arr[..., :3])
    raise ValueError(f"无法把形状 {arr.shape} 的图像转成 RGB")


def _load_frame(frame: object, imageio: Any) -> np.ndarray:
    """读取一帧（路径交给 imageio，数组直接归一化）。"""
    if isinstance(frame, (str, Path)):
        return _to_uint8_rgb(imageio.imread(str(frame)))
    return _to_uint8_rgb(np.asarray(frame))


def _write_array_png(array: np.ndarray, target: Path) -> None:
    """imageio 缺失时用 matplotlib 写 PNG（matplotlib 是渲染层已有依赖）。"""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.image import imsave
    except ImportError as exc:
        raise ImportError(
            "chassis_sim.render 需要 matplotlib，请安装可视化依赖："
            "pip install chassis-sim[viz]"
        ) from exc

    imsave(target, _to_uint8_rgb(array))


def _write_frames_dir(frames: Sequence[object], gif_path: Path) -> Path:
    """降级路径：把帧写进 `<gif 名>_frames/` 目录，返回该目录。"""
    directory = gif_path.with_name(f"{gif_path.stem}_frames")
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        if isinstance(frame, (str, Path)):
            source = Path(frame)
            suffix = source.suffix or ".png"
            shutil.copyfile(source, directory / f"frame_{index:05d}{suffix}")
        else:
            _write_array_png(np.asarray(frame), directory / f"frame_{index:05d}.png")
    return directory


def save_gif(
    frames: Sequence[object],
    path: str | Path,
    fps: int = 10,
) -> Path:
    """把帧序列写成 GIF；imageio 缺失时降级为帧目录并返回目录路径。

    Args:
        frames: 帧序列，元素可以是 PNG 路径（str / Path）或 (H, W, 3/4) 数组。
        path: 目标 .gif 路径（后缀不是 .gif 时自动补上）。
        fps: 帧率，必须是正数。

    Returns:
        正常路径返回 GIF 文件路径；降级路径返回帧目录路径。
    """
    if fps <= 0:
        raise ValueError(f"fps 必须是正数，收到 {fps}")
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("save_gif 至少需要一帧")

    target = Path(path)
    if target.suffix.lower() != ".gif":
        target = target.with_suffix(".gif")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        imageio = _import_imageio()
    except ImportError:
        directory = _write_frames_dir(frame_list, target)
        _LOGGER.warning(
            "%s imageio 不可用（pip install chassis-sim[viz] 可启用 GIF 导出），"
            "已将 %d 帧写入目录 %s",
            _WARN_PREFIX,
            len(frame_list),
            directory,
        )
        return directory

    arrays = [_load_frame(frame, imageio) for frame in frame_list]
    imageio.mimsave(str(target), arrays, fps=fps, loop=0)
    return target
