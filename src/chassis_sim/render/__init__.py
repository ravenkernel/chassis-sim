"""chassis_sim.render —— 渲染与录制层（matplotlib 延迟导入，无显示器可跑）。

公开接口
--------
```
render_frame(bridge, ax=None, *, show_lidar=False, show_trail=False, trail=None,
             dpi=100.0, figsize=(6.0, 6.0)) -> matplotlib.figure.Figure
render_to_png(bridge, path, *, dpi=100.0, **kwargs) -> pathlib.Path
render_episode_frames(bridge, actions, out_dir, *, every=1, prefix="frame",
                      show_trail=False, dpi=100.0, **kwargs) -> list[pathlib.Path]
save_gif(frames, path, fps=10) -> pathlib.Path
ReplayRecorder(bridge=None, *, scene=None)
    .record_step(bridge, action, reward=None)
    .record(action, pose=None, reward=None)
    .dump(path) -> pathlib.Path
    .load(path) -> ReplayRecorder          （别名 .from_npz）
    .render_frames(out_dir, *, every=1, prefix="replay",
                   show_trail=True, dpi=100.0) -> list[pathlib.Path]
```

约定
----
* matplotlib 只在函数内部 import；缺失时抛 `ImportError`，提示
  `pip install chassis-sim[viz]`。核心依赖仍是 numpy + gymnasium + pyyaml。
* 固定 Agg 后端、固定 dpi、PNG 元数据去掉 `Software` 字段；同一状态渲染两次
  像素逐位相同（测试比较 numpy 像素数组，不比较文件字节）。
* 场景读取 `SimpleBridge` 的 `_circle_centers` / `_circle_radii` /
  `_rect_centers` / `_rect_half_dims` / `_rect_thetas` 与 `_x, _y, _theta`
  的只读快照；渲染层不写回，也不改变桥行为。
* LiDAR 默认关闭；开启时前后两个 180 线扫描合计最多等间隔抽样 120 条。
* 轨迹尾迹最多保留最后 500 个位姿点。
* 参考路径按第 2 列鼻向区分前进（绿色）与倒车（紫色）段。

`render_scene`（Scene 输入，无需桥）与 `save_figure` 一并导出，供回放层复用。
"""

from __future__ import annotations

from .animation import save_gif
from .figure import (
    render_episode_frames,
    render_frame,
    render_scene,
    render_to_png,
    save_figure,
)
from .recorder import ReplayRecorder
from .scene import Scene, scene_from_bridge, split_path_by_direction

__all__ = [
    "ReplayRecorder",
    "Scene",
    "render_episode_frames",
    "render_frame",
    "render_scene",
    "render_to_png",
    "save_figure",
    "save_gif",
    "scene_from_bridge",
    "split_path_by_direction",
]
