"""可组合的原始观测段 —— 有序段列表即拼接顺序。

旧 ObsLayout（src/chassis_sim/envs/observation.py:35）把 9 段顺序写死在 ORDER 里，
宽度只能靠 max_path_points / action_dim 两个旋钮缩放。本模块把观测拆成**段对象**：
每段自己声明 name / dim 并负责 build(ctx)，ObsSpec 只按给定顺序拼接。

设计要点：
  · 段顺序即拼接顺序；同名段在 ObsSpec 构造期直接报错（切片会互相覆盖）。
  · 段宽在 ObsSpec.resolved() 时固定：lidar 段的射线数可由 sensor_meta
    （形如 {"front": {"n_rays": 180}}）补齐，显式 ray_counts 优先。
  · 拼接后总宽必须等于 total_dim，**不静默 pad / truncate**。
  · spec 只支持 Python 对象（不进 YAML）：自定义段是 callable，YAML 表达不了。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

# lidar 段的多传感器合并方式
_POOL_MODES = ("none", "max", "mean")


@dataclass
class ObsContext:
    """段构建的输入；env 侧负责填。字段语义沿用旧 ObservationBuilder。"""

    lidar: Mapping[str, np.ndarray]  # 传感器名 -> 原始 ranges
    odom_rel: np.ndarray  # [4] dx_body*10, dy_body*10, v, ω
    path_flat: np.ndarray  # [points*4] rho, sin_phi, cos_phi, dtheta
    goal: np.ndarray  # [2] 终点在车体系中的 (x, y)
    act: np.ndarray  # [action_dim] 上一拍动作（已归一化）
    err: np.ndarray  # [2] CTE 归一化, 航向误差
    goal_sig: np.ndarray  # [2] sin/cos(终点航向误差)
    path_valid: float  # 真实路点占比 [0,1]
    path_progress: float  # env 真值 arc_completion [0,1]


def _flat_f32(value, *, dim: int, who: str) -> np.ndarray:
    """转 float32 一维并校验宽度（旧 build 对每段都做 ravel）。"""
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.size != dim:
        raise ValueError(f"{who}: 期望 {dim} 维，实际 {arr.size} 维")
    return arr


class Segment:
    """段协议：name + dim + build(ctx) -> float32[dim]。"""

    name: str
    dim: int

    def _resolve(self, sensor_meta: Mapping[str, Mapping[str, int]] | None) -> Segment:
        """返回把外部元数据填好的段；宽度固定的段原样返回。"""
        return self

    def build(self, ctx: ObsContext) -> np.ndarray:
        raise NotImplementedError


class Odom(Segment):
    """里程计相对位姿：dim=4。"""

    name = "odom"
    dim = 4

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.odom_rel, dim=self.dim, who="Odom.build")


class Path(Segment):
    """车体系参考路径（极坐标四通道），每点 4 维。"""

    name = "path"

    def __init__(self, points: int = 30) -> None:
        if not isinstance(points, (int, np.integer)) or points <= 0:
            raise ValueError(f"Path: points 必须为正整数，得到 {points!r}")
        self.points = int(points)
        self.dim = self.points * 4  # rho, sin_phi, cos_phi, dtheta

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(
            ctx.path_flat, dim=self.dim, who=f"Path(points={self.points}).build"
        )


class Goal(Segment):
    """终点在车体系中的位置：dim=2。"""

    name = "goal"
    dim = 2

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.goal, dim=self.dim, who="Goal.build")


class Act(Segment):
    """上一拍动作：dim=action_dim。"""

    name = "act"

    def __init__(self, action_dim: int = 2) -> None:
        if not isinstance(action_dim, (int, np.integer)) or action_dim <= 0:
            raise ValueError(f"Act: action_dim 必须为正整数，得到 {action_dim!r}")
        self.action_dim = int(action_dim)
        self.dim = self.action_dim

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(
            ctx.act, dim=self.dim, who=f"Act(action_dim={self.action_dim}).build"
        )


class Err(Segment):
    """CTE 归一化 + 航向误差：dim=2。"""

    name = "err"
    dim = 2

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.err, dim=self.dim, who="Err.build")


class GoalSig(Segment):
    """sin/cos(终点航向误差)：dim=2。"""

    name = "goal_sig"
    dim = 2

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.goal_sig, dim=self.dim, who="GoalSig.build")


class PathValid(Segment):
    """真实路点占比：dim=1。"""

    name = "path_valid"
    dim = 1

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.path_valid, dim=self.dim, who="PathValid.build")


class PathProgress(Segment):
    """沿参考路径已完成比例（env 真值 arc_completion）：dim=1。"""

    name = "path_progress"
    dim = 1

    def build(self, ctx: ObsContext) -> np.ndarray:
        arr = _flat_f32(ctx.path_progress, dim=self.dim, who="PathProgress.build")
        # 与旧 ObsLayout.build 同款夹紧：真值本身已在 [0,1]，这里只把 NaN/越界
        # 挡在观测之外（观测一写进 buffer 就会污染整个训练）。
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(arr, 0.0, 1.0)


class LidarSegment(Segment):
    """激光段：多传感器按 pool 合并成一条原始扫描。

    pool="none"：dim = Σ ray_counts[s]，按 sensors 顺序直接拼接；
    pool="max"/"mean"：所有引用传感器的 n_rays 必须相等，逐元素 max/mean，
    dim = 该相等值。

    ray_counts 未给（或只给了部分）时，由 ObsSpec.resolved(sensor_meta) 从
    sensor_meta 补齐；显式 ray_counts 优先。缺元数据时 resolved() 报错并列出
    缺哪些传感器名。
    """

    def __init__(
        self,
        name: str = "lidar",
        sensors: Iterable[str] = ("front", "rear"),
        pool: str = "none",
        ray_counts: Mapping[str, int] | None = None,
    ) -> None:
        if pool not in _POOL_MODES:
            raise ValueError(
                f"LidarSegment({name!r}): pool 必须是 {list(_POOL_MODES)} 之一，得到 {pool!r}"
            )
        if not isinstance(name, str) or not name:
            raise ValueError(f"LidarSegment: name 必须为非空字符串，得到 {name!r}")
        self.name = name
        self.sensors = tuple(str(s) for s in sensors)
        if not self.sensors:
            raise ValueError(f"LidarSegment({name!r}): sensors 不能为空")
        dup = sorted({s for s in self.sensors if self.sensors.count(s) > 1})
        if dup:
            # 同名传感器写两遍会把宽度静默翻倍，直接拒绝
            raise ValueError(f"LidarSegment({name!r}): sensors 重复 {dup}")
        self.pool = pool
        self.ray_counts = (
            None
            if ray_counts is None
            else {str(k): int(v) for k, v in ray_counts.items()}
        )

    def _counts(self, sensor_meta: Mapping[str, Mapping[str, int]] | None = None):
        """本段引用传感器的射线数（显式 ray_counts 优先，缺的从元数据补）。"""
        counts = dict(self.ray_counts or {})
        for s in self.sensors:
            if s in counts or not sensor_meta:
                continue
            meta = sensor_meta.get(s)
            if isinstance(meta, Mapping) and "n_rays" in meta:
                counts[s] = int(meta["n_rays"])
        missing = [s for s in self.sensors if s not in counts]
        if missing:
            raise ValueError(
                f"LidarSegment({self.name!r}): 缺少传感器射线数 {missing}；"
                f"请传入 ray_counts 或调用 ObsSpec.resolved(sensor_meta)"
            )
        return {s: counts[s] for s in self.sensors}

    def _pooled_width(self, counts: Mapping[str, int]) -> int:
        widths = {s: counts[s] for s in self.sensors}
        uniq = set(widths.values())
        if len(uniq) != 1:
            raise ValueError(
                f"LidarSegment({self.name!r}): pool={self.pool!r} 要求所有传感器射线数相等，"
                f"实际 {widths}"
            )
        return next(iter(uniq))

    @property
    def dim(self) -> int:
        counts = self._counts()
        if self.pool == "none":
            return int(sum(counts.values()))
        return self._pooled_width(counts)

    def _resolve(self, sensor_meta=None) -> LidarSegment:
        return LidarSegment(
            name=self.name,
            sensors=self.sensors,
            pool=self.pool,
            ray_counts=self._counts(sensor_meta),
        )

    def build(self, ctx: ObsContext) -> np.ndarray:
        counts = self._counts()
        if self.pool != "none":
            self._pooled_width(counts)  # 先报"射线数不等"，而不是后面的 stack
        missing = [s for s in self.sensors if s not in ctx.lidar]
        if missing:
            raise ValueError(
                f"LidarSegment({self.name!r}).build: ctx.lidar 缺少传感器 {missing}"
            )
        parts = []
        for s in self.sensors:
            arr = np.asarray(ctx.lidar[s], dtype=np.float32).ravel()
            if arr.size != counts[s]:
                raise ValueError(
                    f"LidarSegment({self.name!r}).build: 传感器 {s!r} 期望 {counts[s]} 根射线，"
                    f"实际 {arr.size}"
                )
            parts.append(arr)
        if self.pool == "none":
            return np.concatenate(parts)
        stacked = np.stack(parts, axis=0)
        pooled = stacked.max(axis=0) if self.pool == "max" else stacked.mean(axis=0)
        return np.asarray(pooled, dtype=np.float32)


class Custom(Segment):
    """自定义段：fn(ctx) -> 一维 float32[dim]。

    dim 在构造时声明；运行时宽度不等于 dim、或 ndim != 1 一律抛 ValueError
    （不静默 reshape/pad），dtype 统一转 float32。
    """

    def __init__(
        self, name: str, dim: int, fn: Callable[[ObsContext], np.ndarray]
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Custom: name 必须为非空字符串，得到 {name!r}")
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError(f"Custom({name!r}): dim 必须为正整数，得到 {dim!r}")
        if not callable(fn):
            raise TypeError(
                f"Custom({name!r}): fn 必须可调用，得到 {type(fn).__name__}"
            )
        self.name = name
        self.dim = int(dim)
        self.fn = fn

    def build(self, ctx: ObsContext) -> np.ndarray:
        out = np.asarray(self.fn(ctx), dtype=np.float32)
        if out.ndim != 1:
            raise ValueError(
                f"Custom({self.name!r}).build: 期望一维数组，实际 ndim={out.ndim} "
                f"shape={out.shape}"
            )
        if out.size != self.dim:
            raise ValueError(
                f"Custom({self.name!r}).build: 期望宽度 {self.dim}，实际 {out.size}"
            )
        return out


class ObsSpec:
    """有序段列表。构造期只检查段名（lidar 的宽度要等 resolved 才有）。"""

    def __init__(self, segments: Iterable[Segment]) -> None:
        segs = tuple(segments)
        if not segs:
            raise ValueError("ObsSpec: 至少需要一个段")
        for s in segs:
            if not isinstance(getattr(s, "name", None), str) or not s.name:
                raise ValueError(f"ObsSpec: 段 {s!r} 缺少非空 name")
        names = [s.name for s in segs]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(
                f"ObsSpec: 段名重复 {dup}；同名段会让 describe() 的切片互相覆盖"
            )
        self.segments = segs

    def resolved(
        self, sensor_meta: Mapping[str, Mapping[str, int]] | None = None
    ) -> ResolvedObsSpec:
        """固定各段宽度（lidar 从 sensor_meta 补射线数）。"""
        return ResolvedObsSpec(tuple(s._resolve(sensor_meta) for s in self.segments))

    @staticmethod
    def default(action_dim: int = 2, path_points: int = 30) -> ObsSpec:
        """默认 spec = odom + path + goal + act（不含 lidar）。"""
        return ObsSpec(
            (Odom(), Path(points=path_points), Goal(), Act(action_dim=action_dim))
        )

    def __repr__(self) -> str:
        return f"ObsSpec(segments={[s.name for s in self.segments]})"


class ResolvedObsSpec:
    """宽度已固定的 spec：total_dim / describe() / build()。"""

    def __init__(self, segments: Iterable[Segment]) -> None:
        segs = tuple(segments)
        if not segs:
            raise ValueError("ResolvedObsSpec: 至少需要一个段")
        slices: dict[str, slice] = {}
        offset = 0
        for s in segs:
            dim = int(s.dim)
            if dim <= 0:
                raise ValueError(
                    f"ResolvedObsSpec: 段 {s.name!r} 的 dim 必须为正，得到 {dim}"
                )
            slices[s.name] = slice(offset, offset + dim)
            offset += dim
        self.segments = segs
        self._slices = slices
        self._total = offset

    @property
    def total_dim(self) -> int:
        return self._total

    def describe(self) -> dict[str, slice]:
        """name -> slice，键顺序即段顺序、相邻切片首尾相接。"""
        return dict(self._slices)

    def build(self, ctx: ObsContext) -> np.ndarray:
        """逐段拼接；总宽或 dtype 与契约不一致直接抛错，不 pad。"""
        fused = np.concatenate([s.build(ctx) for s in self.segments])
        if fused.size != self._total:
            widths = {s.name: s.dim for s in self.segments}
            raise ValueError(
                f"ObsSpec.build: 期望 {self._total} 维，实际拼接出 {fused.size} 维；"
                f"段宽度 {widths}"
            )
        if fused.dtype != np.float32:
            raise ValueError(f"ObsSpec.build: 拼接结果应为 float32，实际 {fused.dtype}")
        return fused

    def __repr__(self) -> str:
        return (
            f"ResolvedObsSpec(total_dim={self._total}, "
            f"segments={[s.name for s in self.segments]})"
        )
