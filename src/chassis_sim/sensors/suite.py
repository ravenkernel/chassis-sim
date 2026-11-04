"""列表式传感器套件：任意数量、任意位姿，每支传感器一条独立随机流。

设计约束
========
1. **纯 numpy、零新依赖**（只 import numpy 与标准库），不破坏 import 纯净性。
2. **随机源隔离**（硬约束）：每支传感器各用一条由 `(seed, name)` 派生的独立
   `np.random.Generator`，**绝不触碰模块级 `np.random` 序列**。派生式为::

       np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(name.encode())]))

   它带来两条性质：**跨进程稳定**（不依赖 Python 的 hash 随机化）与**增删无关**
   （新增一支传感器不会平移任何其它支的随机序列——`spawn` 做不到这一点，
   因为 spawn 的子流按位置分配）。
3. **几何外置**：本包不 import 任何桥，射线求交由构造时注入的 `raycast` 提供::

       raycast(origin_xy: tuple[float, float], angle_rad: float, max_range: float) -> float

   `origin_xy` 与 `angle_rad` 都是**车体系**（车头 x 轴起、逆时针为正）：
   套件不知道车在世界里的位姿，桥知道。返回该射线的命中距离（米）。

IMU 模型假设
============
输出 `{ax, ay, yaw_rate, yaw, t}`，车体系（x 前、y 左）：

* `ax = (v_k - v_{k-1}) / dt` —— 纵向加速度，一阶后向差分（不建模轮地滑移、
  悬挂俯仰、重力分量）。首步没有历史，取 0；`dt` 取相邻两次 update 的 `t` 之差，
  因此与仿真步长天然一致，不需要再传一遍步长。
* `ay = v_k * omega_k + (vy_k - vy_{k-1}) / dt` —— 横向加速度。第一项是单轨
  模型的向心加速度（后轴中心绕瞬心），第二项只对麦轮非零（横移速度差分）；
  阿克曼/差速的 `vy` 恒为 0。
* `yaw_rate = omega` —— 直接取底盘命令的横摆角速度（不做陀螺仪量化/偏置）。
* `yaw = theta` —— 桥归一化到 [-pi, pi] 的航向角（不是从 yaw_rate 积分得来）。
* `t` —— 自 reset 起的模拟时间 = 步数 × dt（乘法，不累加浮点）。

轮速计模型假设
==============
输出 `{v_left, v_right, v, t}`：

* 阿克曼用**差速近似**（把左右轮当作差速驱动的两轮）：
  `v_left = v - omega * W / 2`、`v_right = v + omega * W / 2`，其中 `W` 是轮距。
* `v = (v_left + v_right) / 2` —— 差速中点速度（无噪声时恒等于真值 v）。
* 差速底盘该式就是其运动学定义；麦轮只给轮组的**等效纵向**速度，不建模辊子
  横移分量（`vy` 不进轮速计）。
* `t` —— 同 IMU。

参数化 LiDAR
============
方位角约定（端点包含）::

    angle_increment = fov / (n_rays - 1)
    angle_min       = mount.yaw - fov / 2
    angle_max       = angle_min + (n_rays - 1) * angle_increment

处理顺序（固定，不得调换）：原始量程 -> clip[range_min, max_range] -> 加性高斯
噪声 -> 再 clip -> 量化（round(r/q)·q，q=0 关闭）-> 压成 float32 -> 进延迟环形
缓冲 -> 丢帧抽样 -> 交付。噪声在"采集时刻"施加（即延迟的是**已经带噪声**的帧），
丢帧抽样在"交付时刻"施加（被丢的是**该步要交付的那一帧**）。

读数结构
========
`SensorSuite.update(...)` 返回 `dict[name, reading]`，键序与 specs 列表一致::

    lidar: {"scan": (n_rays,) float32 或 None, "scan_meta": {...}}
    imu:   {"ax", "ay", "yaw_rate", "yaw", "t"}
    wheel: {"v_left", "v_right", "v", "t"}
"""

from __future__ import annotations

import zlib
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .specs import Imu, Lidar, SensorSpec, WheelOdom

# 车体系射线求交：raycast(origin_xy, angle_rad, max_range) -> 距离（米）
RaycastFn = Callable[[tuple[float, float], float, float], float]


def derive_rng(seed: int, name: str) -> np.random.Generator:
    """按 `(seed, name)` 派生一支传感器专属的随机流。

    用 `zlib.crc32` 而不是 `hash()`：后者在 CPython 里对 str 默认加盐
    （PYTHONHASHSEED），跨进程不稳定；crc32 是纯函数，同 (seed, name) 恒得同一条流。
    """
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), zlib.crc32(name.encode("utf-8"))])
    )


class _LatencyBuffer:
    """定长环形缓冲：交付 `latency_steps` 步之前采集的那一帧。

    长度 = latency + 1：追加第 k 帧后，deque[0] 恰好是第 k-latency 帧；未填满
    （前 latency 步）时返回 None（预热期，交付方应在 scan_meta 里标记 valid=False）。
    零延迟时退化为"直接交付当前帧"。
    """

    def __init__(self, latency_steps: int) -> None:
        self.latency_steps = int(latency_steps)
        self._frames: deque[tuple[float, np.ndarray]] = deque(
            maxlen=self.latency_steps + 1
        )

    def reset(self) -> None:
        self._frames.clear()

    def push(
        self, t: float, frame: np.ndarray
    ) -> tuple[np.ndarray | None, float | None]:
        """压入当前帧，返回 (交付帧, 该帧的采集时刻)；预热期返回 (None, None)。"""
        self._frames.append((float(t), frame))
        if len(self._frames) <= self.latency_steps:
            return None, None
        frame_t, stored = self._frames[0]
        return stored, frame_t


class _LidarSensor:
    """单支参数化 LiDAR：射线扫描 + 噪声 / 量化 / 延迟 / 丢帧。"""

    def __init__(self, spec: Lidar, rng: np.random.Generator) -> None:
        self.spec = spec
        self._rng = rng
        self._buffer = _LatencyBuffer(spec.latency_steps)
        self._origin_xy = (float(spec.mount[0]), float(spec.mount[1]))
        self._angle_min = float(spec.angle_min)
        self._angle_increment = float(spec.angle_increment)

    def reset(self) -> None:
        """清空延迟缓冲（不重置随机流：种子只在构造时派生一次）。"""
        self._buffer.reset()

    def update(self, *, raycast: RaycastFn, t: float) -> dict[str, Any]:
        """采集并交付一帧。"""
        p = self.spec
        n_rays = int(p.n_rays)
        bearings = self._angle_min + np.arange(n_rays, dtype=np.float64) * (
            self._angle_increment
        )
        raw = np.fromiter(
            (raycast(self._origin_xy, float(a), float(p.max_range)) for a in bearings),
            dtype=np.float64,
            count=n_rays,
        )
        ranges = np.clip(raw, p.range_min, p.max_range)
        if float(p.noise_std) > 0.0:
            ranges = ranges + self._rng.normal(0.0, p.noise_std, size=n_rays)
            ranges = np.clip(ranges, p.range_min, p.max_range)
        if float(p.quant_step) > 0.0:
            ranges = np.round(ranges / p.quant_step) * p.quant_step
            ranges = np.clip(ranges, p.range_min, p.max_range)

        delivered, frame_t = self._buffer.push(t, ranges.astype(np.float32))
        # 只在真的会丢帧时才抽签：全关闭的 LiDAR 一个随机数都不消耗
        dropped = False
        if float(p.dropout_prob) > 0.0:
            dropped = bool(self._rng.random() < float(p.dropout_prob))
        scan = None if dropped else delivered

        meta: dict[str, Any] = {
            "name": p.name,
            "mount": tuple(p.mount),
            "n_rays": n_rays,
            "fov": float(p.fov),
            "angle_min": float(self._angle_min),
            "angle_max": float(self._angle_min + p.angle_span),
            "angle_increment": float(self._angle_increment),
            "range_min": float(p.range_min),
            "range_max": float(p.max_range),
            "latency_steps": int(p.latency_steps),
            "t": float(t),
            "frame_t": None if scan is None else float(frame_t),
            "valid": scan is not None,
            "dropped": bool(dropped),
        }
        return {"scan": scan, "scan_meta": meta}


class _ImuSensor:
    """单支车体系 IMU（模型假设见模块 docstring）。"""

    def __init__(self, spec: Imu, rng: np.random.Generator) -> None:
        self.spec = spec
        self._rng = rng
        self._prev_v: float | None = None
        self._prev_vy: float | None = None
        self._prev_t: float | None = None

    def reset(self) -> None:
        self._prev_v = None
        self._prev_vy = None
        self._prev_t = None

    def update(
        self, *, v: float, vy: float, omega: float, theta: float, t: float
    ) -> dict[str, float]:
        v = float(v)
        vy = float(vy)
        omega = float(omega)
        t = float(t)
        if self._prev_v is None or self._prev_vy is None or self._prev_t is None:
            ax = 0.0
            ay = v * omega
        else:
            dt = t - self._prev_t
            if dt <= 0.0:
                raise ValueError(
                    f"IMU {self.spec.name!r} 的 t 必须严格递增，收到 {self._prev_t!r} -> {t!r}"
                )
            ax = (v - self._prev_v) / dt
            ay = v * omega + (vy - self._prev_vy) / dt
        self._prev_v = v
        self._prev_vy = vy
        self._prev_t = t

        yaw_rate = omega
        if float(self.spec.noise_std) > 0.0:
            noise = self._rng.normal(0.0, float(self.spec.noise_std), size=3)
            ax = ax + float(noise[0])
            ay = ay + float(noise[1])
            yaw_rate = yaw_rate + float(noise[2])

        return {
            "ax": float(ax),
            "ay": float(ay),
            "yaw_rate": float(yaw_rate),
            "yaw": float(theta),
            "t": t,
        }


class _WheelSensor:
    """单支轮速计（阿克曼用差速近似；公式见模块 docstring）。"""

    def __init__(
        self, spec: WheelOdom, rng: np.random.Generator, *, track_width: float
    ) -> None:
        self.spec = spec
        self._rng = rng
        self._track_width = float(track_width)

    def reset(self) -> None:
        """无内部状态（保留统一接口）。"""

    @property
    def track_width(self) -> float:
        return self._track_width

    def update(self, *, v: float, omega: float, t: float) -> dict[str, float]:
        half = 0.5 * self._track_width
        v_left = float(v) - float(omega) * half
        v_right = float(v) + float(omega) * half
        if float(self.spec.noise_std) > 0.0:
            noise = self._rng.normal(0.0, float(self.spec.noise_std), size=2)
            v_left = v_left + float(noise[0])
            v_right = v_right + float(noise[1])
        return {
            "v_left": float(v_left),
            "v_right": float(v_right),
            "v": 0.5 * (v_left + v_right),
            "t": float(t),
        }


class SensorSuite:
    """按 spec 列表装配并驱动任意多支传感器。

    每支传感器各用一条由 `(seed, name)` 派生的独立子流，因此**新增/删除任何一支
    都不会平移其它支的随机序列**（见模块 docstring 的派生式）。几何完全外置：
    LiDAR 的每条射线都调调用方注入的 `raycast`。
    """

    def __init__(
        self,
        specs: Iterable[SensorSpec],
        *,
        seed: int = 0,
        raycast: RaycastFn | None = None,
        default_track_width: float | None = None,
    ) -> None:
        """
        Args:
            specs: 传感器 spec 序列（`Lidar` / `Imu` / `WheelOdom` 任意混合，
                顺序即读数键序）；空序列 = 没有任何传感器。
            seed: 随机流种子（>= 0）。与 domain_rand 的全局 np.random **无关**。
            raycast: 车体系射线求交（有 Lidar 时必填，否则构造即报错）。
            default_track_width: `WheelOdom.track_width=None` 时用的轮距（米）。
                桥会传底盘的 `footprint()[1]`；直接装配时也可显式给。
        """
        if int(seed) < 0:
            raise ValueError(f"seed 必须 >= 0，收到 {seed!r}")
        self._seed = int(seed)
        self._specs: tuple[SensorSpec, ...] = tuple(specs)
        self._raycast = raycast
        self._sensors: dict[str, _LidarSensor | _ImuSensor | _WheelSensor] = {}

        for spec in self._specs:
            if isinstance(spec, Lidar):
                if raycast is None:
                    raise ValueError(
                        f"Lidar {spec.name!r} 需要注入 raycast："
                        "sensors 包不做几何求交（几何归桥）"
                    )
                self._add(spec, _LidarSensor(spec, derive_rng(self._seed, spec.name)))
            elif isinstance(spec, Imu):
                self._add(spec, _ImuSensor(spec, derive_rng(self._seed, spec.name)))
            elif isinstance(spec, WheelOdom):
                track_width = (
                    float(spec.track_width)
                    if spec.track_width is not None
                    else default_track_width
                )
                if track_width is None:
                    raise ValueError(
                        f"WheelOdom {spec.name!r} 没有 track_width，也没有给 "
                        "default_track_width（桥上取底盘 footprint()[1]）"
                    )
                if float(track_width) <= 0.0:
                    raise ValueError(f"track_width 必须是正数，收到 {track_width!r}")
                self._add(
                    spec,
                    _WheelSensor(
                        spec,
                        derive_rng(self._seed, spec.name),
                        track_width=float(track_width),
                    ),
                )
            else:
                raise TypeError(
                    "specs 只能包含 Lidar / Imu / WheelOdom，收到 "
                    f"{type(spec).__name__}"
                )

    def _add(
        self, spec: SensorSpec, sensor: _LidarSensor | _ImuSensor | _WheelSensor
    ) -> None:
        if spec.name in self._sensors:
            raise ValueError(
                f"传感器名重复：{spec.name!r}（随机流按 name 派生，读数也按 name 取，"
                "因此必须唯一）"
            )
        self._sensors[spec.name] = sensor

    # ── 只读视图 ────────────────────────────────────────────────────

    @property
    def specs(self) -> tuple[SensorSpec, ...]:
        """装配用的 spec（顺序 = 读数键序）。"""
        return self._specs

    @property
    def names(self) -> tuple[str, ...]:
        """所有传感器名（顺序 = 读数键序）。"""
        return tuple(self._sensors)

    @property
    def track_width(self) -> float | None:
        """轮速计实际使用的轮距（米）；没有轮速计时为 None。"""
        for sensor in self._sensors.values():
            if isinstance(sensor, _WheelSensor):
                return sensor.track_width
        return None

    def __len__(self) -> int:
        return len(self._sensors)

    # ── 驱动 ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """清空所有传感器的内部状态（延迟缓冲、IMU 速度历史）；不重播种。"""
        for sensor in self._sensors.values():
            sensor.reset()

    def update(
        self, *, v: float, vy: float, omega: float, theta: float, t: float
    ) -> dict[str, dict[str, Any]]:
        """推进一拍的全部传感器，返回 `dict[name, reading]`（键序 = specs 顺序）。"""
        raycast = self._raycast
        reading: dict[str, dict[str, Any]] = {}
        for name, sensor in self._sensors.items():
            if isinstance(sensor, _LidarSensor):
                if raycast is None:  # pragma: no cover - 构造期已拦住
                    raise ValueError(f"Lidar {name!r} 没有 raycast")
                reading[name] = sensor.update(raycast=raycast, t=t)
            elif isinstance(sensor, _ImuSensor):
                reading[name] = sensor.update(v=v, vy=vy, omega=omega, theta=theta, t=t)
            else:
                reading[name] = sensor.update(v=v, omega=omega, t=t)
        return reading
