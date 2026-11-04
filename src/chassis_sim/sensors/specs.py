"""传感器规格（spec）：纯数据、声明式的"这支传感器长什么样"。

一条 spec 只描述**参数**——安装位姿、视场、量程、退化模型。它

* **不含**随机数生成器（随机流由 `suite.SensorSuite` 按 `(seed, name)` 派生）；
* **不含**任何几何求解（射线求交由调用方注入，见 `suite.SensorSuite`）；
* **不引用**底盘：因此同一份 spec 可以搬到任意场景、任意数量的同型传感器上。

位姿约定（与全库一致）：车体系 **x 前、y 左**，角度以 x 轴逆时针为正。
`Lidar.mount=(x, y, yaw)` 三个分量都可任取，不再受底盘两个固定安装位限制。

列表式装配（0.2.0）：一个场景里的传感器数量与位姿都由调用方给的列表决定，
"没有传感器"就是空列表——不再有总开关对象。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 车体系安装位姿（米, 米, 弧度）
Mount = tuple[float, float, float]


def _check_name(name: str) -> None:
    """传感器名必须是非空字符串：随机流按 name 派生，且读数以 name 为键。"""
    if not isinstance(name, str) or not name:
        raise ValueError(f"name 必须是非空字符串，收到 {name!r}")


def _check_mount(mount: tuple[float, float, float]) -> Mount:
    """校验并规范化车体系安装位姿 (x, y, yaw)。"""
    try:
        x, y, yaw = (float(v) for v in mount)
    except (TypeError, ValueError):
        raise ValueError(f"mount 必须是 (x, y, yaw) 三元组，收到 {mount!r}") from None
    if not all(np.isfinite(v) for v in (x, y, yaw)):
        raise ValueError(f"mount 的三个分量必须是有限数，收到 {mount!r}")
    return (x, y, yaw)


def _check_noise(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} 必须 >= 0，收到 {value!r}")


@dataclass(frozen=True)
class Lidar:
    """一帧参数化 LiDAR 的几何与退化参数。

    Args:
        name: 传感器名（唯一；读数键名，也是随机流派生输入）。
        mount: 车体系安装位姿 (x, y, yaw)，米 / 弧度；可任取。
        fov: 视场角（弧度，> 0）；射线覆盖 [angle_min, angle_min + fov]。
        n_rays: 单帧射线数（>= 2）；端点包含，angle_increment = fov/(n_rays-1)。
        max_range: 最大量程（米）。
        range_min: 最小量程（米）：更近的命中被截到该值。
        noise_std: 加性高斯噪声标准差（米），0 = 关闭。
        quant_step: 量程量化步长（米），0 = 关闭；量化式为 round(r/q)*q。
        latency_steps: 延迟步数（>= 0）：第 k 步交付第 k-latency 步采集的帧。
        dropout_prob: 丢帧概率（[0, 1]）：对**交付帧**独立抽样，命中则交付 None。
    """

    name: str
    mount: Mount = (0.0, 0.0, 0.0)
    fov: float = float(np.pi)
    n_rays: int = 180
    max_range: float = 10.0
    range_min: float = 0.05
    noise_std: float = 0.0
    quant_step: float = 0.0
    latency_steps: int = 0
    dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        _check_name(self.name)
        object.__setattr__(self, "mount", _check_mount(self.mount))
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
        _check_noise("noise_std", self.noise_std)
        _check_noise("quant_step", self.quant_step)
        if int(self.latency_steps) < 0:
            raise ValueError(f"latency_steps 必须 >= 0，收到 {self.latency_steps!r}")
        if not (0.0 <= float(self.dropout_prob) <= 1.0):
            raise ValueError(
                f"dropout_prob 必须位于 [0, 1]，收到 {self.dropout_prob!r}"
            )
        # 规范化整数/浮点字段，下游不必再做类型转换
        object.__setattr__(self, "n_rays", int(self.n_rays))
        object.__setattr__(self, "latency_steps", int(self.latency_steps))
        object.__setattr__(self, "fov", float(self.fov))
        object.__setattr__(self, "max_range", float(self.max_range))
        object.__setattr__(self, "range_min", float(self.range_min))
        object.__setattr__(self, "noise_std", float(self.noise_std))
        object.__setattr__(self, "quant_step", float(self.quant_step))
        object.__setattr__(self, "dropout_prob", float(self.dropout_prob))

    @property
    def angle_increment(self) -> float:
        """相邻射线的方位角增量（端点包含 ⇒ fov/(n_rays-1)）。"""
        return float(self.fov) / (int(self.n_rays) - 1)

    @property
    def angle_min(self) -> float:
        """首条射线相对车头的方位角（= mount 的 yaw - fov/2）。"""
        return float(self.mount[2]) - 0.5 * float(self.fov)

    @property
    def angle_span(self) -> float:
        """首尾射线之间的方位角跨度（= fov）。"""
        return (int(self.n_rays) - 1) * self.angle_increment


@dataclass(frozen=True)
class Imu:
    """车体系 IMU 规格（模型假设见 `suite` 模块说明）。

    Args:
        name: 传感器名（唯一）。
        noise_std: ax/ay/yaw_rate 的加性高斯噪声标准差，0 = 关闭。
    """

    name: str
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_noise("noise_std", self.noise_std)
        object.__setattr__(self, "noise_std", float(self.noise_std))


@dataclass(frozen=True)
class WheelOdom:
    """轮速计规格（左右轮差速；公式见 `suite` 模块说明）。

    Args:
        name: 传感器名（唯一）。
        track_width: 轮距（米）；None = 由装配方给缺省值（桥上取底盘
            `footprint()[1]`）。
        noise_std: 左右轮速的加性高斯噪声标准差（m/s），0 = 关闭。
    """

    name: str
    track_width: float | None = None
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        _check_name(self.name)
        if self.track_width is not None and float(self.track_width) <= 0.0:
            raise ValueError(f"track_width 必须是正数，收到 {self.track_width!r}")
        _check_noise("noise_std", self.noise_std)
        if self.track_width is not None:
            object.__setattr__(self, "track_width", float(self.track_width))
        object.__setattr__(self, "noise_std", float(self.noise_std))


# 一支传感器的合法 spec 类型（装配顺序即 specs 列表顺序）
SensorSpec = Lidar | Imu | WheelOdom
