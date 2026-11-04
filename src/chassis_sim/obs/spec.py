"""可组合的原始观测段 —— 有序段列表即拼接顺序。

1.0.0 之前库内有一份把 9 段顺序写死的定长布局（已删），
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
    """段构建的输入；env 侧由 ContextBuilder 填（见 obs/context.py）。

    path_valid / path_progress 标量或 1 元素数组都接受（ContextBuilder 实际
    传的是 float32[1]），段侧统一经 _flat_f32 ravel 到 1 维后再用。
    """

    lidar: Mapping[str, np.ndarray]  # 传感器名 -> 原始 ranges
    odom_rel: np.ndarray  # [4] dx_body*10, dy_body*10, v, ω
    path_flat: np.ndarray  # [points*4] rho, sin_phi, cos_phi, dtheta
    goal: np.ndarray  # [2] 终点在车体系中的 (x, y)
    act: np.ndarray  # [action_dim] 上一拍动作（已归一化）
    err: np.ndarray  # [2] CTE 归一化, 航向误差
    goal_sig: np.ndarray  # [2] sin/cos(终点航向误差)
    path_valid: float | np.ndarray  # 真实路点占比 [0,1]（标量或 1 元素数组）
    path_progress: float | np.ndarray  # env 真值 arc_completion [0,1]（同上）


def _flat_f32(value, *, dim: int, who: str) -> np.ndarray:
    """转 float32 一维并校验宽度（旧 build 对每段都做 ravel）。"""
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.size != dim:
        raise ValueError(f"{who}: 期望 {dim} 维，实际 {arr.size} 维")
    return arr


@dataclass(frozen=True)
class BoundContext:
    """观测段物理界的推导输入（SI 单位，env 构造期算一次）。

    只收**能推出硬界**的环境量：往这里加字段，就是给某一段发"可以宣称有界"的
    许可证。拿不到依据的段保持 ±inf（见 Segment.bounds），由 check_env 点名。
    """

    dt: float
    v_max: float
    v_min: float
    vy_max: float
    omega_max: float
    lookahead: float
    world_diag: float
    #: 传感器名 -> (range_min, max_range)，只有 lidar 段会读
    lidar: Mapping[str, tuple[float, float]]


def _bounds_inf(dim: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(int(dim), -np.inf, dtype=np.float64)
    return low, np.full(int(dim), np.inf, dtype=np.float64)


class Segment:
    """段协议：name + dim + build(ctx) -> float32[dim] + bounds(ctx) -> (low, high)。"""

    name: str
    dim: int

    def _resolve(self, sensor_meta: Mapping[str, Mapping[str, int]] | None) -> Segment:
        """返回把外部元数据填好的段；宽度固定的段原样返回。"""
        return self

    def build(self, ctx: ObsContext) -> np.ndarray:
        raise NotImplementedError

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """本段每维的物理上下界，形状各自 (dim,)。

        默认 ±inf：没有推导依据的段（Custom、未声明量程的 lidar 传感器）**不猜**。
        Box 因此保持非有限，check_env 会点名 —— 这是信号，不是缺陷。
        """
        return _bounds_inf(self.dim)


class Odom(Segment):
    """里程计相对位姿：dim=4。"""

    name = "odom"
    dim = 4

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.odom_rel, dim=self.dim, who="Odom.build")

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """dx/dy：单步体坐标位移 ×10；v：执行值量程；ω：角速度量程。

        v/ω 两维是桥回报的**执行值**，已被底盘夹在 [v_min, v_max] / ±ω_max 内。
        位移两维 = 10·Δs，Δs ≤ dt·(|v|max + vy_max)（麦轮横移与纵移同一拍发生，
        按和取上界）；1.5 倍余量留给里程计漂移、量化与 float32 表示误差。
        """
        span = max(abs(ctx.v_max), abs(ctx.v_min)) + max(float(ctx.vy_max), 0.0)
        disp = 10.0 * 1.5 * float(ctx.dt) * span
        low = np.array([-disp, -disp, float(ctx.v_min), -float(ctx.omega_max)])
        high = np.array([disp, disp, float(ctx.v_max), float(ctx.omega_max)])
        return low, high


class PathSegment(Segment):
    """车体系参考路径（极坐标四通道），每点 4 维。

    1.0.0 起段类名是 `PathSegment`（旧名 `Path` 与 `pathlib.Path` 撞名）。
    段名仍是 "path"，切片与段序不变。
    """

    name = "path"

    def __init__(self, points: int = 30) -> None:
        if not isinstance(points, (int, np.integer)) or points <= 0:
            raise ValueError(f"PathSegment: points 必须为正整数，得到 {points!r}")
        self.points = int(points)
        self.dim = self.points * 4  # rho, sin_phi, cos_phi, dtheta

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(
            ctx.path_flat,
            dim=self.dim,
            who=f"PathSegment(points={self.points}).build",
        )

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """ρ ∈ [0, diag+lookahead+0.1]，sinφ/cosφ ∈ [-1,1]，dθ ∈ [-π,π]。

        ρ 是**车体系欧氏距离**（paths.cartesian_to_polar: ρ = hypot(dx, dy)），
        不是沿路径的弧长。PathExtractor 只保留"最近路点之后 lookahead 弧长内"
        的点，末端不足时按 0.1 m 步长外推（_fill_lookahead），于是
        ρ ≤ ρ_nearest + lookahead + 0.1；车与路径都在 world_bounds 内
        ⇒ ρ_nearest ≤ diag（越界位姿由 env 计入 info["obs_clipped"]）。
        整条路径短于 lookahead 时末点被复用（padding），仍落在同一界内。
        """
        rho_max = float(ctx.world_diag) + float(ctx.lookahead) + 0.1
        low = np.empty(self.dim, dtype=np.float64)
        high = np.empty(self.dim, dtype=np.float64)
        low[0::4], high[0::4] = 0.0, rho_max  # rho
        low[1::4], high[1::4] = -1.0, 1.0  # sin_phi
        low[2::4], high[2::4] = -1.0, 1.0  # cos_phi
        low[3::4], high[3::4] = -np.pi, np.pi  # dtheta
        return low, high


class Goal(Segment):
    """终点在车体系中的位置：dim=2。"""

    name = "goal"
    dim = 2

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.goal, dim=self.dim, who="Goal.build")

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """终点在车体系中的 (x, y)：车与终点都在 world_bounds 内 ⇒ |·| ≤ diag。"""
        d = float(ctx.world_diag)
        return np.array([-d, -d]), np.array([d, d])


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

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """上一拍动作是**归一化**量，恒在 [-1,1]（见 env._physical_action）。"""
        return np.full(self.dim, -1.0), np.full(self.dim, 1.0)


class Err(Segment):
    """CTE 归一化 + 航向误差：dim=2。"""

    name = "err"
    dim = 2

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.err, dim=self.dim, who="Err.build")

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """CTE 在 ContextBuilder 里已 clip 到 ±1；航向误差经 wrap 落在 ±π。"""
        return np.array([-1.0, -np.pi]), np.array([1.0, np.pi])


class GoalSig(Segment):
    """sin/cos(终点航向误差)：dim=2。"""

    name = "goal_sig"
    dim = 2

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.goal_sig, dim=self.dim, who="GoalSig.build")

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """sin/cos(终点航向误差)：解析上界就是 ±1。"""
        return np.full(self.dim, -1.0), np.full(self.dim, 1.0)


class PathValid(Segment):
    """真实路点占比：dim=1。"""

    name = "path_valid"
    dim = 1

    def build(self, ctx: ObsContext) -> np.ndarray:
        return _flat_f32(ctx.path_valid, dim=self.dim, who="PathValid.build")

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """真实路点占比是分数 ⇒ [0,1]。"""
        return np.zeros(self.dim), np.ones(self.dim)


class PathProgress(Segment):
    """沿参考路径已完成比例（env 真值 arc_completion）：dim=1。"""

    name = "path_progress"
    dim = 1

    def build(self, ctx: ObsContext) -> np.ndarray:
        arr = _flat_f32(ctx.path_progress, dim=self.dim, who="PathProgress.build")
        # 夹紧是观测契约的一部分：真值本身已在 [0,1]，这里只把 NaN/越界
        # 挡在观测之外（观测一写进 buffer 就会污染整个训练）。
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(arr, 0.0, 1.0)

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """build() 已经把这一维夹进 [0,1]，界就是它自己。"""
        return np.zeros(self.dim), np.ones(self.dim)


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

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """每根射线 ∈ 该传感器的 [range_min, max_range]。

        SensorSuite 的处理顺序是 原始量程 → clip → 加噪 → 再 clip → 量化
        （sensors/suite.py），所以加噪也不会把射线推出量程。
        pool="max"/"mean" 取所引用传感器的包络；任一传感器没在 env 侧声明量程
        就整段退回 ±inf —— 不猜。
        """
        counts = self._counts()
        per = [
            (float(ctx.lidar[s][0]), float(ctx.lidar[s][1]))
            if s in ctx.lidar
            else (-np.inf, np.inf)
            for s in self.sensors
        ]
        if self.pool != "none":
            width = self._pooled_width(counts)
            lo = min(p[0] for p in per)
            hi = max(p[1] for p in per)
            return np.full(width, lo), np.full(width, hi)
        lows = [np.full(counts[s], lo) for s, (lo, _hi) in zip(self.sensors, per)]
        highs = [np.full(counts[s], hi) for s, (_lo, hi) in zip(self.sensors, per)]
        return np.concatenate(lows), np.concatenate(highs)


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
            (
                Odom(),
                PathSegment(points=path_points),
                Goal(),
                Act(action_dim=action_dim),
            )
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

    def bounds(self, ctx: BoundContext) -> tuple[np.ndarray, np.ndarray]:
        """逐段拼出 observation_space 的上下界（float32，与 build() 同宽同序）。

        宽度按段校验：任何一段给出错误宽度都在这里报错，而不是先构造出一个
        "能 contains 但错位"的 Box。结果再向两侧各扩 1 ulp —— float64 → float32
        的舍入若是向内的，一个恰好等于界的合法值会变成"越界 1 ulp"，让
        _clip_obs 白记一次；扩展只动有限数，±inf 原样保留。
        """
        lows, highs = [], []
        for seg in self.segments:
            low, high = seg.bounds(ctx)
            low = np.asarray(low, dtype=np.float64).ravel()
            high = np.asarray(high, dtype=np.float64).ravel()
            if low.size != seg.dim or high.size != seg.dim:
                raise ValueError(
                    f"ObsSpec.bounds: 段 {seg.name!r} 期望 {seg.dim} 维界，"
                    f"实际 low={low.size} high={high.size}"
                )
            lows.append(low)
            highs.append(high)
        low32 = np.concatenate(lows).astype(np.float32)
        high32 = np.concatenate(highs).astype(np.float32)
        # 只扩非零的有限值：0.0 转 float32 精确，不会向内舍入；而 0 扩 1 ulp 会落到
        # denormal（下游 np.allclose 之类的相乘会报 underflow，纯噪声）。
        with np.errstate(under="ignore"):
            low32 = np.where(
                np.isfinite(low32) & (low32 != 0.0),
                np.nextafter(low32, np.float32(-np.inf)),
                low32,
            )
            high32 = np.where(
                np.isfinite(high32) & (high32 != 0.0),
                np.nextafter(high32, np.float32(np.inf)),
                high32,
            )
        return low32, high32

    def __repr__(self) -> str:
        return (
            f"ResolvedObsSpec(total_dim={self._total}, "
            f"segments={[s.name for s in self.segments]})"
        )
