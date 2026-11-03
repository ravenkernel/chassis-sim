"""SensorProvider 抽象接口 —— 与主仓 ros_bridge/sensor_provider.py 逐字同构。

签名（9 个抽象方法 + 1 个抽象属性）按字节照抄
`/home/zhao/drl-sim/ros_bridge/sensor_provider.py`，只把英文 docstring 译成中文：
消费方（envs / 控制器 / 回放脚本）按位置或关键字调用这些方法，任何签名漂移都会让
旧桥与新桥不可互换。

实现者：
    SimpleBridge —— 本包纯 numpy 的射线投射实现（bridges/simple_bridge.py）
    旧仓 SimBridge / ROS2Bridge / MockBridge —— 同签名，可直接替换

在上面的**抽象**接口之外，本模块还提供两项**附加（非抽象）**能力，供需要 ROS
式扫描元数据的消费方使用；不实现它们也不影响已有桥的可用性：

    SensorProvider.get_scan() —— 单侧扫描字典（字段见 SCAN_FIELDS，对齐 ROS LaserScan）
    scan_to_pointcloud()      —— 扫描字典 -> 世界系 (N, 2) float64 点云
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

# ROS sensor_msgs/LaserScan 的十个字段，**顺序即字典键序**（消费方按 ROS 顺序读）。
SCAN_FIELDS: tuple[str, ...] = (
    "angle_min",
    "angle_max",
    "angle_increment",
    "time_increment",
    "scan_time",
    "range_min",
    "range_max",
    "ranges",
    "intensities",
    "frame_id",
)


class SensorProvider(ABC):
    """传感器数据与动作下发的抽象接口。"""

    @abstractmethod
    def reset(
        self,
        obstacles: list | None = None,
        path_waypoints: np.ndarray | None = None,
        initial_heading: float | None = None,
    ) -> None:
        """复位仿真状态。

        Args:
            obstacles: 可选的障碍物 kwargs 列表（动态创建用）。
            path_waypoints: 可选的 (N, 2) 全局路径航点。
            initial_heading: 可选的起始车头朝向（弧度）。
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """用给定动作推进一个时间步。

        Args:
            action: [delta, v] 转向角与速度
        """
        ...

    @abstractmethod
    def get_lidar_front(self) -> np.ndarray:
        """返回前向 LiDAR 扫描：[180] 个距离（米）。"""
        ...

    @abstractmethod
    def get_lidar_rear(self) -> np.ndarray:
        """返回后向 LiDAR 扫描：[180] 个距离（米）。"""
        ...

    @abstractmethod
    def get_odometry(self) -> np.ndarray:
        """返回里程计：[x, y, theta, v, omega]。"""
        ...

    @abstractmethod
    def get_path(self) -> np.ndarray:
        """返回车体系下的全局路径航点：[N, 3] (dx, dy, dtheta)。"""
        ...

    @abstractmethod
    def get_path_global(self) -> np.ndarray:
        """返回世界系下**全部**路径航点，形状 (N, 3) [x, y, θ]。"""
        ...

    @abstractmethod
    def check_collision(self) -> bool:
        """机器人与任何障碍物重叠时返回 True。"""
        ...

    @abstractmethod
    def check_goal_reached(self, threshold: float = 0.3) -> bool:
        """机器人进入末航点 threshold 以内且朝向匹配时返回 True。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""
        ...

    @property
    @abstractmethod
    def goal_heading(self) -> float:
        """目标点的期望朝向（用于终端奖励中的朝向对齐计算）。"""
        ...

    # ── 附加（非抽象）API：动态障碍 ────────────────────────────────────
    def set_dynamic_obstacles(self, specs: list | None = None) -> None:
        """换一组动态障碍（`None` / `[]` = 清空）。**非抽象**：不实现即视为不支持。

        Raises:
            NotImplementedError: 该桥不支持动态障碍（构造它的模拟器只实现了静态
                场景）。这里**故意响亮地报错**而不是静默跳过 —— 一个被静默忽略的
                动态障碍会让"碰撞没发生"看起来像策略变好了。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不支持动态障碍（dynamic_obstacles）："
            "该桥只实现了静态场景；请改用支持动态障碍的桥（如 SimpleBridge）"
        )

    # ── 附加（非抽象）API：ROS 式扫描元数据 ────────────────────────────
    def get_scan(self, side: str = "front") -> dict[str, Any]:
        """返回带元数据的单侧扫描字典（**非抽象**：不实现即视为不支持）。

        Returns:
            含 SCAN_FIELDS 十个键的字典，键序与 ROS sensor_msgs/LaserScan 一致：
            angle_min, angle_max, angle_increment, time_increment, scan_time,
            range_min, range_max, ranges, intensities, frame_id。

        角度约定（实现方必须遵守，scan_to_pointcloud 依赖它）：
            第 i 条射线的方位角 = angle_min + i * angle_increment，单位为弧度、
            以**车头**为 0；且
            angle_increment == (angle_max - angle_min) / (len(ranges) - 1)。
            即 angle_max 是**最后一条射线**的方位角（不是"末尾再加一个步长"），
            与 np.linspace(angle_min, angle_max, len(ranges)) 端点包含的语义一致。

        数值约定：ranges 必须是 float32 一维数组；**无返回/超出量程的射线用
        +inf** 表示，不得用 0 或 range_max 冒充有效测距（scan_to_pointcloud 依赖
        np.isfinite 丢弃这些射线）。intensities 允许为 None（不建模回波强度）。

        Args:
            side: 传感器侧别，实现方各自的合法取值。

        Raises:
            NotImplementedError: 本实现不提供扫描元数据。
            ValueError: side 不是合法取值（消息里必须列出合法值）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 get_scan()；只有提供扫描元数据的桥才支持"
        )

    # ── 附加（非抽象）API：可选传感器读数 ──────────────────────────────
    def get_sensor_reading(self) -> dict[str, Any] | None:
        """返回当前步的可选传感器读数（**非抽象**：不实现即视为不启用）。

        读数**只进 info、绝不进 obs**：obs 的维度契约（494 / 495）与传感器
        是否启用无关。键按传感器命名，实现方只放已启用的项：

        * "imu":       {ax, ay, yaw_rate, yaw, t}
        * "wheel":     {v_left, v_right, v, t}
        * "scan":      (n_rays,) float32，或 None（延迟预热期 / 丢帧）
        * "scan_meta": 参数化 LiDAR 的角度与有效性元数据

        Returns:
            dict[str, Any]，或 None（未启用任何可选传感器；此时 env 的 info 不新增
            任何键，信息路径与加入该接口前逐位相同）。
        """
        return None


def scan_to_pointcloud(
    scan: Mapping[str, Any],
    origin: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
) -> np.ndarray:
    """把 get_scan() 的字典转换为世界系点云 (N, 2) float64。

    第 i 条射线的方位角按 angle_min + i * angle_increment 还原，它是**相对车头**
    的 bearing；世界系方向 = heading + bearing。

    Args:
        scan: 至少含 angle_min / angle_increment / ranges 的扫描字典。
        origin: 传感器在**世界系**的 (x, y) 位置（米）；注意不是车体后轴中心，
            前/后雷达各有安装偏移，需要由调用方叠加。
        heading: 传感器在**世界系**的朝向（弧度），即车头朝向。

    Returns:
        (N, 2) float64 世界系坐标。ranges 里的非有限值（inf/nan，即"无返回"）
        对应的射线整条丢弃，因此返回行数 <= len(ranges)；全部无返回时为 (0, 2)。
    """
    ranges = np.asarray(scan["ranges"], dtype=np.float64).reshape(-1)
    angle_min = float(scan["angle_min"])
    angle_increment = float(scan["angle_increment"])
    angles = angle_min + np.arange(ranges.size, dtype=np.float64) * angle_increment
    angles = angles + float(heading)

    finite = np.isfinite(ranges)
    if not np.any(finite):
        return np.zeros((0, 2), dtype=np.float64)

    r = ranges[finite]
    a = angles[finite]
    ox = float(origin[0])
    oy = float(origin[1])
    return np.column_stack([ox + r * np.cos(a), oy + r * np.sin(a)])
