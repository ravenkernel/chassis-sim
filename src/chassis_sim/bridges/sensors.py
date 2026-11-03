"""参数化传感器模型：IMU / 轮速计 / 可配置 LiDAR（噪声、量化、延迟、丢帧）。

设计约束
========
1. **纯 numpy、零新依赖**（只 import numpy 与标准库），因此不破坏 import 纯净性。
2. **随机源隔离**（硬约束）：本模块的每一路随机数都来自构造时从
   `SensorConfig.sensor_seed` **派生**的独立 `np.random.Generator`，
   **绝不触碰模块级 `np.random` 序列**。因此"打开传感器"不会改变地图 /
   domain_rand（LiDAR 噪声、里程计漂移）等吃全局 `np.random` 的序列。
3. **默认全关**：`SensorConfig()` 的各个开关都是 False / None，桥在关闭时
   完全不构造本模块的对象，info 不新增任何键（信息路径零变化）。
4. **不进 obs**：读数只经 `SimpleBridge.get_sensor_reading()` 进 info。

IMU 模型假设
============
输出 `{ax, ay, yaw_rate, yaw, t}`，车体系（x 前、y 左）：

* `ax = (v_k - v_{k-1}) / dt` —— 纵向加速度，一阶后向差分（不建模轮地滑移、
  悬挂俯仰、重力分量）。首步没有历史，取 0。
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
  `v_left = v - omega * W / 2`、`v_right = v + omega * W / 2`，
  其中 `W` 是轮距（默认取 `chassis.footprint()[1]`，即车体宽度 0.20 m）。
* `v = (v_left + v_right) / 2` —— 差速中点速度（无噪声时恒等于真值 v）。
* 差速底盘该式就是其运动学定义；麦轮只给轮组的**等效纵向**速度，不建模辊子
  横移分量（`vy` 不进轮速计）。
* `t` —— 同 IMU。

参数化 LiDAR
============
方位角约定与 `SensorProvider.get_scan()` 逐字相同（端点包含）：

    angle_increment = fov / (n_rays - 1)
    angle_min       = 安装朝向偏移 - fov / 2
    angle_max       = angle_min + (n_rays - 1) * angle_increment

默认 `LidarParams()`（180 线 / pi / 10 m / 0.05 m、零噪声、零量化、零延迟、
零丢帧）与历史的 `get_lidar_front()` / `get_lidar_rear()` **逐位相同**
（由 `tests/test_bridges_sensor_integration.py` 的对拍断言守着）。

处理顺序（固定，不得调换）：原始量程 -> clip[range_min, range_max] -> 加性高斯
噪声 -> 再 clip -> 量化（round(r/q)·q，q=0 关闭）-> 压成 float32 -> 进延迟环形
缓冲 -> 丢帧抽样 -> 交付。噪声在"采集时刻"施加（即延迟的是**已经带噪声**的帧），
丢帧抽样在"交付时刻"施加（被丢的是**该步要交付的那一帧**）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LidarParams:
    """一帧参数化 LiDAR 的几何与退化参数（默认值与历史 180 线雷达逐位一致）。

    Args:
        n_rays: 单侧射线数（>= 2）。
        fov: 单侧视场角（弧度，> 0）；射线覆盖 [angle_min, angle_min + fov]。
        max_range: 最大量程（米）。
        range_min: 最小量程（米）：更近的命中被截到该值。
        noise_std: 加性高斯噪声标准差（米），0 = 关闭。
        quant_step: 量程量化步长（米），0 = 关闭；量化式为 round(r/q)*q。
        latency_steps: 延迟步数（>= 0）：第 k 步交付第 k-latency 步采集的帧。
        dropout_prob: 丢帧概率（[0, 1]）：对**交付帧**独立抽样，命中则交付 None。
        side: "front" 或 "rear"（决定安装位姿与法线朝向）。
    """

    n_rays: int = 180
    fov: float = float(np.pi)
    max_range: float = 10.0
    range_min: float = 0.05
    noise_std: float = 0.0
    quant_step: float = 0.0
    latency_steps: int = 0
    dropout_prob: float = 0.0
    side: str = "front"

    def __post_init__(self) -> None:
        if int(self.n_rays) < 2:
            raise ValueError(f"n_rays 必须 >= 2，收到 {self.n_rays!r}")
        if not np.isfinite(self.fov) or float(self.fov) <= 0.0:
            raise ValueError(f"fov 必须是有限正数，收到 {self.fov!r}")
        if not np.isfinite(self.max_range) or float(self.max_range) <= 0.0:
            raise ValueError(f"max_range 必须是有限正数，收到 {self.max_range!r}")
        if not (0.0 < float(self.range_min) <= float(self.max_range)):
            raise ValueError(
                f"range_min 必须满足 0 < range_min <= max_range，收到 {self.range_min!r}"
            )
        if float(self.noise_std) < 0.0:
            raise ValueError(f"noise_std 必须 >= 0，收到 {self.noise_std!r}")
        if float(self.quant_step) < 0.0:
            raise ValueError(f"quant_step 必须 >= 0，收到 {self.quant_step!r}")
        if int(self.latency_steps) < 0:
            raise ValueError(f"latency_steps 必须 >= 0，收到 {self.latency_steps!r}")
        if not (0.0 <= float(self.dropout_prob) <= 1.0):
            raise ValueError(f"dropout_prob 必须位于 [0, 1]，收到 {self.dropout_prob!r}")
        if self.side not in ("front", "rear"):
            raise ValueError(
                f"side 必须是 'front' 或 'rear'，收到 {self.side!r}"
                "（合法值: 'front', 'rear'）"
            )

    @property
    def angle_increment(self) -> float:
        """相邻射线的方位角增量（端点包含 ⇒ fov/(n_rays-1)）。"""
        return float(self.fov) / (int(self.n_rays) - 1)

    @property
    def angle_span(self) -> float:
        """首尾射线之间的方位角跨度（= (n_rays-1) * angle_increment = fov）。"""
        return (int(self.n_rays) - 1) * self.angle_increment


@dataclass(frozen=True)
class SensorConfig:
    """可选传感器总开关（默认全关 ⇒ 与加入本模块前逐位相同）。

    Args:
        lidar: 参数化 LiDAR 参数；None = 关闭（默认）。
        imu: 是否输出 IMU 读数。
        wheel: 是否输出轮速计读数。
        imu_noise_std: IMU 的 ax/ay/yaw_rate 加性高斯噪声标准差，0 = 关闭。
        wheel_noise_std: 左右轮速的加性高斯噪声标准差（m/s），0 = 关闭。
        track_width: 轮距（米）；None = 取底盘的 `footprint()[1]`（车体宽度）。
        sensor_seed: 传感器随机流种子。与 domain_rand 的全局 np.random **无关**。
    """

    lidar: LidarParams | None = None
    imu: bool = False
    wheel: bool = False
    imu_noise_std: float = 0.0
    wheel_noise_std: float = 0.0
    track_width: float | None = None
    sensor_seed: int = 0

    def __post_init__(self) -> None:
        if float(self.imu_noise_std) < 0.0:
            raise ValueError(f"imu_noise_std 必须 >= 0，收到 {self.imu_noise_std!r}")
        if float(self.wheel_noise_std) < 0.0:
            raise ValueError(
                f"wheel_noise_std 必须 >= 0，收到 {self.wheel_noise_std!r}"
            )
        if self.track_width is not None and float(self.track_width) <= 0.0:
            raise ValueError(f"track_width 必须是正数，收到 {self.track_width!r}")
        if int(self.sensor_seed) < 0:
            raise ValueError(f"sensor_seed 必须 >= 0，收到 {self.sensor_seed!r}")

    @property
    def enabled(self) -> bool:
        """是否至少启用了一路传感器。"""
        return self.lidar is not None or bool(self.imu) or bool(self.wheel)


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

    def push(self, t: float, frame: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        """压入当前帧，返回 (交付帧, 该帧的采集时刻)；预热期返回 (None, None)。"""
        self._frames.append((float(t), frame))
        if len(self._frames) <= self.latency_steps:
            return None, None
        frame_t, stored = self._frames[0]
        return stored, frame_t


class ParametricLidar:
    """参数化单侧 LiDAR：噪声 / 量化 / 延迟 / 丢帧（用独立 Generator）。"""

    def __init__(
        self,
        params: LidarParams,
        rng: np.random.Generator,
        *,
        mount_yaw: float = 0.0,
    ) -> None:
        self.params = params
        self._rng = rng
        # 安装朝向偏移（前雷达 0、后雷达 pi）：只用于推导 angle_min，射线本身由桥投射。
        self._mount_yaw = float(mount_yaw)
        self._buffer = _LatencyBuffer(params.latency_steps)

    def reset(self) -> None:
        """清空延迟缓冲（不重置随机流：种子只在构造时派生一次）。"""
        self._buffer.reset()

    def update(self, raw_ranges: np.ndarray, *, t: float) -> tuple[np.ndarray | None, dict[str, Any]]:
        """处理一帧原始量程。

        Args:
            raw_ranges: 当前步的**无噪声**量程 (n_rays,)（由桥的射线投射给出）。
            t: 当前步的模拟时间（秒）。

        Returns:
            (scan, meta)。scan 为 (n_rays,) float32 或 None（预热 / 丢帧）。
        """
        p = self.params
        ranges = np.asarray(raw_ranges, dtype=np.float64).reshape(-1)
        if ranges.size != int(p.n_rays):
            raise ValueError(
                f"raw_ranges 长度必须是 {p.n_rays}，收到 {ranges.size}"
            )
        ranges = np.clip(ranges, p.range_min, p.max_range)
        if float(p.noise_std) > 0.0:
            ranges = ranges + self._rng.normal(0.0, p.noise_std, size=ranges.size)
            ranges = np.clip(ranges, p.range_min, p.max_range)
        if float(p.quant_step) > 0.0:
            ranges = np.round(ranges / p.quant_step) * p.quant_step
            ranges = np.clip(ranges, p.range_min, p.max_range)

        delivered, frame_t = self._buffer.push(t, ranges.astype(np.float32))
        dropped = False
        if float(p.dropout_prob) > 0.0:
            dropped = bool(self._rng.random() < float(p.dropout_prob))
        scan = None if dropped else delivered

        angle_min = self.angle_min
        meta: dict[str, Any] = {
            "n_rays": int(p.n_rays),
            "fov": float(p.fov),
            "angle_min": float(angle_min),
            "angle_max": float(angle_min + p.angle_span),
            "angle_increment": float(p.angle_increment),
            "range_min": float(p.range_min),
            "range_max": float(p.max_range),
            "side": str(p.side),
            "latency_steps": int(p.latency_steps),
            "t": float(t),
            "frame_t": None if scan is None else float(frame_t),
            "valid": scan is not None,
            "dropped": bool(dropped),
        }
        return scan, meta

    @property
    def angle_min(self) -> float:
        """首条射线相对车头的方位角（= 安装朝向偏移 - fov/2）。"""
        return self._mount_yaw - 0.5 * float(self.params.fov)

class Imu:
    """车体系 IMU（模型假设见模块 docstring）。

    状态只有上一拍的 (v, vy)，`reset()` 清掉它 —— 否则新回合首步会拿上一
    回合末的速度差分出一个假加速度。
    """

    def __init__(
        self,
        *,
        dt: float,
        rng: np.random.Generator,
        noise_std: float = 0.0,
    ) -> None:
        self.dt = float(dt)
        self._rng = rng
        self._noise_std = float(noise_std)
        self._prev_v: float | None = None
        self._prev_vy: float | None = None

    def reset(self) -> None:
        self._prev_v = None
        self._prev_vy = None

    def update(
        self,
        *,
        v: float,
        vy: float,
        omega: float,
        theta: float,
        t: float,
    ) -> dict[str, float]:
        """推进一拍的 IMU 读数（ax/ay/yaw_rate/yaw/t）。"""
        v = float(v)
        vy = float(vy)
        omega = float(omega)
        if self._prev_v is None or self._prev_vy is None:
            ax = 0.0
            ay = v * omega
        else:
            ax = (v - self._prev_v) / self.dt
            ay = v * omega + (vy - self._prev_vy) / self.dt
        self._prev_v = v
        self._prev_vy = vy

        yaw_rate = omega
        if self._noise_std > 0.0:
            noise = self._rng.normal(0.0, self._noise_std, size=3)
            ax = ax + float(noise[0])
            ay = ay + float(noise[1])
            yaw_rate = yaw_rate + float(noise[2])

        return {
            "ax": float(ax),
            "ay": float(ay),
            "yaw_rate": float(yaw_rate),
            "yaw": float(theta),
            "t": float(t),
        }


class WheelOdometry:
    """轮速计（阿克曼用差速近似；公式见模块 docstring）。"""

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        track_width: float,
        noise_std: float = 0.0,
    ) -> None:
        self.track_width = float(track_width)
        self._rng = rng
        self._noise_std = float(noise_std)

    def reset(self) -> None:
        """无内部状态（保留统一接口）。"""

    def update(self, *, v: float, omega: float, t: float) -> dict[str, float]:
        """推进一拍的左右轮速（v_left/v_right/v/t）。"""
        half = 0.5 * self.track_width
        v_left = float(v) - float(omega) * half
        v_right = float(v) + float(omega) * half
        if self._noise_std > 0.0:
            noise = self._rng.normal(0.0, self._noise_std, size=2)
            v_left = v_left + float(noise[0])
            v_right = v_right + float(noise[1])
        return {
            "v_left": float(v_left),
            "v_right": float(v_right),
            "v": 0.5 * (v_left + v_right),
            "t": float(t),
        }


class SensorSuite:
    """按 `SensorConfig` 组装并驱动各路传感器。

    关闭的项**不构造对象**，因此连随机数都不消耗；每路传感器各用一条从
    `sensor_seed` 派生的独立子流（SeedSequence.spawn），打开/关闭某一路不会
    平移另一路的随机序列。
    """

    def __init__(
        self,
        config: SensorConfig,
        *,
        dt: float,
        fallback_track_width: float,
        lidar_mounts: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        self.config = config
        seeds = np.random.SeedSequence(int(config.sensor_seed)).spawn(3)

        self._imu: Imu | None = None
        if config.imu:
            self._imu = Imu(
                dt=dt,
                rng=np.random.default_rng(seeds[0]),
                noise_std=config.imu_noise_std,
            )

        self._track_width = (
            float(config.track_width)
            if config.track_width is not None
            else float(fallback_track_width)
        )
        self._wheel: WheelOdometry | None = None
        if config.wheel:
            self._wheel = WheelOdometry(
                rng=np.random.default_rng(seeds[1]),
                track_width=self._track_width,
                noise_std=config.wheel_noise_std,
            )

        self._lidar: ParametricLidar | None = None
        if config.lidar is not None:
            mount_yaw = (
                float(lidar_mounts[0][2])
                if config.lidar.side == "front"
                else float(lidar_mounts[1][2])
            )
            self._lidar = ParametricLidar(
                config.lidar,
                np.random.default_rng(seeds[2]),
                mount_yaw=mount_yaw,
            )

    @property
    def track_width(self) -> float:
        """轮速计使用的轮距（米）。"""
        return self._track_width

    @property
    def lidar_enabled(self) -> bool:
        """是否启用了参数化 LiDAR。"""
        return self._lidar is not None

    @property
    def lidar_params(self) -> LidarParams | None:
        """参数化 LiDAR 的配置（未启用为 None）。"""
        return None if self._lidar is None else self._lidar.params

    @property
    def scan_side(self) -> str:
        """参数化 LiDAR 的侧别（未启用时给 'front'）。"""
        return "front" if self._lidar is None else str(self._lidar.params.side)

    def reset(self) -> None:
        """清空所有传感器的内部状态（延迟缓冲、IMU 速度历史）。"""
        for sensor in (self._imu, self._wheel, self._lidar):
            if sensor is not None:
                sensor.reset()

    def update(
        self,
        *,
        v: float,
        vy: float,
        omega: float,
        theta: float,
        t: float,
        raw_scan: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """推进一拍的全部传感器，返回只含已启用项的读数字典。"""
        reading: dict[str, Any] = {}
        if self._imu is not None:
            reading["imu"] = self._imu.update(
                v=v, vy=vy, omega=omega, theta=theta, t=t
            )
        if self._wheel is not None:
            reading["wheel"] = self._wheel.update(v=v, omega=omega, t=t)
        if self._lidar is not None:
            if raw_scan is None:
                raise ValueError("启用了 lidar 但没有传入 raw_scan")
            scan, meta = self._lidar.update(raw_scan, t=t)
            reading["scan"] = scan
            reading["scan_meta"] = meta
        return reading

