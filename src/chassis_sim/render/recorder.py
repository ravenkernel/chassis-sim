"""回放记录器：逐步记录 (action, pose, reward)，dump 成 .npz 并脱离桥回放成帧。

`.npz` 里同时保存了场景快照（障碍物、路径、世界边界、车体尺寸、LiDAR 安装位姿），
所以 `ReplayRecorder.load(...).render_frames(...)` 不需要原始 bridge，也不需要
matplotlib 之外的任何东西。LiDAR 扫描不随轨迹记录，回放帧默认不画射线。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .figure import render_scene, save_figure
from .scene import Scene, pose_from_bridge, scene_from_bridge

_DEFAULT_DPI = 100.0


def _stack_actions(actions: Sequence[np.ndarray]) -> np.ndarray:
    """把动作列表堆成 (N, A)；空列表给 (0, 0)。"""
    if not actions:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(actions, dtype=np.float64)


def _stored_path(value: object) -> np.ndarray:
    """从 npz 读回路径，兼容 (N, 2) / (N, 3) / 单点 / 空数组。"""
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class ReplayRecorder:
    """记录 (action, pose, reward)，支持 .npz 往返与无桥回放。"""

    def __init__(
        self, bridge: object | None = None, *, scene: Scene | None = None
    ) -> None:
        if scene is None:
            if bridge is None:
                raise ValueError("ReplayRecorder 需要 bridge 或 scene 之一")
            scene = scene_from_bridge(bridge)
        self._scene = scene
        self._bridge = bridge
        self._initial_pose: tuple[float, float, float] = scene.pose
        self._actions: list[np.ndarray] = []
        self._poses: list[tuple[float, float, float]] = []
        self._rewards: list[float] = []

    # ── 记录 ──────────────────────────────────────────────────────

    def record_step(
        self, bridge: object, action: object, reward: float | None = None
    ) -> None:
        """推进桥一步并记录 `(action, 步后位姿, reward)`；首步前捕获初始位姿。"""
        if not self._actions:
            self._initial_pose = pose_from_bridge(bridge)
            self._scene = self._scene.with_pose(self._initial_pose)
        bridge.step(np.asarray(action, dtype=np.float64))
        self._actions.append(np.asarray(action, dtype=np.float64).copy())
        self._poses.append(pose_from_bridge(bridge))
        self._rewards.append(float("nan") if reward is None else float(reward))

    def record(
        self,
        action: object,
        pose: object | None = None,
        reward: float | None = None,
    ) -> None:
        """记录一条已经执行完的动作与位姿；`pose=None` 时读构造时桥的当前位姿。"""
        if pose is None:
            if self._bridge is None:
                raise ValueError("未提供 pose，且构造 ReplayRecorder 时没有 bridge")
            pose = pose_from_bridge(self._bridge)
        arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            raise ValueError(f"位姿至少需要 3 个分量，收到 {arr.size} 个")
        self._actions.append(np.asarray(action, dtype=np.float64).copy())
        self._poses.append((float(arr[0]), float(arr[1]), float(arr[2])))
        self._rewards.append(float("nan") if reward is None else float(reward))

    # ── 序列化 ────────────────────────────────────────────────────

    def dump(self, path: str | Path) -> Path:
        """把轨迹与场景快照写成压缩 .npz（后缀自动补 .npz），返回文件路径。"""
        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "world_bounds": [float(v) for v in self._scene.world_bounds],
            "footprint": [float(v) for v in self._scene.footprint],
            "lidar_mounts": [
                [float(v) for v in mount] for mount in self._scene.lidar_mounts
            ],
            "max_range": float(self._scene.max_range),
        }
        np.savez_compressed(
            target,
            actions=_stack_actions(self._actions),
            poses=np.asarray(self._poses, dtype=np.float64).reshape(-1, 3),
            rewards=np.asarray(self._rewards, dtype=np.float64),
            initial_pose=np.asarray(self._initial_pose, dtype=np.float64),
            meta=np.array(json.dumps(meta, sort_keys=True)),
            circle_centers=np.asarray(self._scene.circle_centers, dtype=np.float64),
            circle_radii=np.asarray(self._scene.circle_radii, dtype=np.float64),
            rect_centers=np.asarray(self._scene.rect_centers, dtype=np.float64),
            rect_half_dims=np.asarray(self._scene.rect_half_dims, dtype=np.float64),
            rect_thetas=np.asarray(self._scene.rect_thetas, dtype=np.float64),
            path=np.asarray(self._scene.path, dtype=np.float64),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ReplayRecorder:
        """从 .npz 恢复记录器（场景快照一并恢复，无需原始桥）。"""
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            scene = Scene(
                world_bounds=tuple(float(v) for v in meta["world_bounds"]),
                footprint=tuple(float(v) for v in meta["footprint"]),
                pose=tuple(
                    float(v) for v in np.asarray(data["initial_pose"]).reshape(3)
                ),
                circle_centers=np.array(
                    data["circle_centers"], dtype=np.float64
                ).reshape(-1, 2),
                circle_radii=np.array(data["circle_radii"], dtype=np.float64).reshape(
                    -1
                ),
                rect_centers=np.array(data["rect_centers"], dtype=np.float64).reshape(
                    -1, 2
                ),
                rect_half_dims=np.array(
                    data["rect_half_dims"], dtype=np.float64
                ).reshape(-1, 2),
                rect_thetas=np.array(data["rect_thetas"], dtype=np.float64).reshape(-1),
                path=_stored_path(data["path"]),
                lidar_mounts=tuple(
                    (float(m[0]), float(m[1]), float(m[2]))
                    for m in meta["lidar_mounts"]
                ),
                max_range=float(meta["max_range"]),
            )
            actions = np.array(data["actions"], dtype=np.float64)
            poses = np.array(data["poses"], dtype=np.float64).reshape(-1, 3)
            rewards = np.array(data["rewards"], dtype=np.float64).reshape(-1)

        recorder = cls(scene=scene)
        recorder._actions = [row.copy() for row in actions] if actions.size else []
        recorder._poses = [tuple(float(v) for v in row) for row in poses]
        recorder._rewards = [float(v) for v in rewards]
        return recorder

    from_npz = load

    # ── 回放 ──────────────────────────────────────────────────────

    def render_frames(
        self,
        out_dir: str | Path,
        *,
        every: int = 1,
        prefix: str = "replay",
        show_trail: bool = True,
        dpi: float = _DEFAULT_DPI,
        **kwargs: Any,
    ) -> list[Path]:
        """把初始位姿与每 `every` 个记录位姿渲染成 PNG，返回路径列表。

        帧文件名按输出序号编号（`prefix_00000.png` ...）；`show_trail=True`
        时尾迹取到当前帧为止的全部回放位姿。回放没有 LiDAR 扫描，不画射线。
        """
        if every < 1:
            raise ValueError(f"every 必须是正整数，收到 {every}")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        all_poses = [self._initial_pose, *self._poses]
        indices = [0, *range(every, len(all_poses), every)]
        if indices[-1] != len(all_poses) - 1:
            indices.append(len(all_poses) - 1)

        frames: list[Path] = []
        for output_index, pose_index in enumerate(indices):
            scene = self._scene.with_pose(all_poses[pose_index])
            trail = (
                np.asarray(all_poses[: pose_index + 1], dtype=np.float64)
                if show_trail
                else None
            )
            fig = render_scene(
                scene,
                show_trail=show_trail,
                trail=trail,
                dpi=dpi,
                **kwargs,
            )
            target = out / f"{prefix}_{output_index:05d}.png"
            frames.append(save_figure(fig, target, dpi=dpi))
        return frames

    # ── 只读属性 ──────────────────────────────────────────────────

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def initial_pose(self) -> tuple[float, float, float]:
        return self._initial_pose

    @property
    def actions(self) -> np.ndarray:
        return _stack_actions(self._actions)

    @property
    def poses(self) -> np.ndarray:
        return np.asarray(self._poses, dtype=np.float64).reshape(-1, 3)

    @property
    def rewards(self) -> np.ndarray:
        return np.asarray(self._rewards, dtype=np.float64).reshape(-1)

    def __len__(self) -> int:
        return len(self._actions)
