"""SimpleBridge：纯 numpy 的传感器桥，几何与运动学由注入的 `Chassis` 决定。

移植自 `/home/zhao/drl-sim/ros_bridge/numpy_bridge.py`（678 行），**逐行照搬**，
不重写任何公式。旧桥行号 ↔ 内容：

=============  =========================================================
旧桥位置         内容
=============  =========================================================
:30-132        射线求交（圆解析解 / 矩形局部系 slab test）→ raycast.py
:286-305       障碍物解析（circle / rectangle → numpy 数组）
:317-357       起始位姿 + 终点鼻向（B5）
:369-408       步进 + 碰撞 / 到达判定
:412-513       LiDAR（180 线 × 每侧 π 视场）
:515-525       里程计（先返回后累积漂移）
:527-549       get_path / get_path_global
:582-678       碰撞（4 轴 SAT + 按朝向的边界 AABB）
=============  =========================================================

与旧桥**有意**不同的三处（都是"注入底盘"带来的，不是放宽阈值）：

1. 运动学：每步调用 `chassis.step(action)` 拿 `ChassisCommand`，再交给唯一的积分核
   `Chassis.integrate`。旧桥里"用上一帧转角算 ω"的滞后是
   `AckermannChassis.step` 的职责，本桥**不**重复实现（否则滞后会被算两次）。
2. 几何：车体半长/半宽来自 `chassis.footprint()`，LiDAR 安装位姿来自
   `chassis.lidar_mounts()`，不再读 env.yaml。
3. 世界边界 / 量程 / 两个 margin 提升为构造参数；默认值与旧桥字面量逐位相同
   （`[0, 10, 0, 10]` / `10.0` / `0.03` / `0.05`），因此默认构造下与旧桥
   位级一致。

域随机化：两个开关分工不同。
  · `domain_rand=True` 打开 LiDAR 加性高斯噪声 + 里程计随机游走漂移；
  · `seed` 决定这些噪声从哪条随机流里取：`seed` 非 None ⇒ 桥自持
    `np.random.default_rng(seed)`（同一 seed 的两次构造逐位相同，且不扰动
    全局 `np.random`）；`seed=None`（默认）⇒ 仍旧吃**全局**
    `np.random`，即 1.0.x 的既有行为、数值逐位不变（要复现就由调用方固定全局种子）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from chassis_sim.chassis import Chassis
from chassis_sim.sensors import Imu, Lidar, SensorSuite, WheelOdom

from .base import SensorProvider
from .raycast import _ray_circles_intersection, _ray_rects_intersection


def _linear_phase(t: float, period: float, pingpong: bool) -> float:
    """linear 动态障碍在时刻 `t` 的运动相位（秒）。

    语义与 `scenarios.scenario.LinearDynamicObstacle` 的 docstring 逐字对应::

        period == 0        ⇒ t
        period > 0, 锯齿   ⇒ t mod period
        period > 0, 折返   ⇒ period/2 - |(t mod period) - period/2|
    """
    if period <= 0.0:
        return t
    u = t % period
    if not pingpong:
        return u
    half = 0.5 * period
    return half - abs(u - half)


class SimpleBridge(SensorProvider):
    """纯 numpy 的 `SensorProvider`：向量化射线投射 + 底盘无关的运动学。

    参数
    ----------
    chassis:
        决定几何（`footprint`）、LiDAR 安装（`lidar_mounts`）与运动学
        （`step` + `integrate`）的底盘实例。
    dt:
        积分步长（秒），默认 0.1（= 主仓 config/env.yaml 的 world.step_time）。
    world_bounds:
        `(xmin, xmax, ymin, ymax)`，默认 `(0, 10, 0, 10)`，与主仓 world.width /
        height = 10 推导出的四面边界墙一致。
    max_range:
        LiDAR 最大量程（米），默认 10.0（= 主仓 env.yaml sensors.range_max）。
    domain_rand:
        是否启用 LiDAR 加性高斯噪声 + 里程计随机游走漂移。随机源由 seed 决定：
        seed 给定时走本桥自持的 Generator，否则吃全局 np.random（历史行为）。
    seed:
        域随机化噪声的随机源种子。非 None ⇒ 本桥自持一条
        np.random.default_rng(seed)，LiDAR 噪声与里程计漂移都从它取 ⇒ 同一 seed
        的两次构造逐位相同，且不动全局 np.random。None（默认）⇒ 与 1.0.x 逐位
        相同（吃全局 np.random，可复现性由调用方固定全局种子决定）。
    lidar_noise_std:
        LiDAR 高斯噪声标准差（米）。
    odom_drift_xy / odom_drift_theta:
        每步 xy / theta 漂移比例系数（乘 dt 后作为高斯标准差）。
    goal_dist_threshold:
        判到达的距离阈值（米），与 `check_goal_reached` 的默认 threshold 同值，
        二者相等时直接返回 `step()` 缓存的布尔值（旧桥 :557-558 的行为）。
    goal_end_wp_count / goal_end_wp_margin:
        "最近航点必须落在末尾窗口内"的两个整数旋钮，防止路径折返时路过 goal
        提前判到达。
    collision_margin:
        碰撞安全裕度（米），对应旧桥 :588 的 `margin = 0.03`。
    boundary_margin:
        世界边界内缩量（米），对应旧桥 :669 的 `margin = 0.05`。
    dynamic_obstacles:
        动态障碍规格列表（`chassis_sim.scenarios.DynamicObstacle.to_env_dict()` 的
        输出格式）；默认为空 = 逐位不变。**纯运动学、零随机**：第 k 步后的模拟时间
        恒为 `t = k * dt`，位置是 t 的解析函数（不累加浮点），因此同一 reset 序列
        必然给出逐位相同的轨迹。运行中换一组用 `set_dynamic_obstacles()`。
    sensors:
        传感器 spec 列表（`chassis_sim.sensors.Lidar` / `Imu` / `WheelOdom` 任意
        混合、任意数量与位姿）；`None` 或空列表 = 完全关闭（不构造对象、不消耗
        随机数、info 零新键）。几何由本桥注入（`_raycast_body_ray`），sensors 包
        自己不碰世界；每支传感器一条按 `(sensor_seed, name)` 派生的独立随机流。
    sensor_seed:
        传感器随机流种子。与 domain_rand 噪声用的随机源（见 seed）**无关**：传感器
        绝不消耗 domain_rand 的序列，因此开关它们不会平移域随机化的噪声。
        env 侧传入的是**已推导**的整数种子（见 config.schema.resolve_sensor_seed：
        sensor_seed 优先，否则跟随 seed，都没有则 0）。
    """

    # ── 扫描几何常量（1.0 起雷达只有 get_sensor_reading 一个入口）──────────
    N_LIDAR_RAYS: int = 180  # 射线缓冲区预分配宽度（传感器射线数可任意）
    RANGE_MIN: float = 0.05  # 最小量程（米）：更近的命中被截到该值（旧桥 :390-392）

    def __init__(
        self,
        chassis: Chassis,
        *,
        dt: float = 0.1,
        world_bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
        max_range: float = 10.0,
        domain_rand: bool = False,
        seed: int | None = None,
        lidar_noise_std: float = 0.03,
        odom_drift_xy: float = 0.01,
        odom_drift_theta: float = 0.005,
        goal_dist_threshold: float = 0.3,
        goal_end_wp_count: int = 3,
        goal_end_wp_margin: int = 2,
        collision_margin: float = 0.03,
        boundary_margin: float = 0.05,
        dynamic_obstacles: list | None = None,
        sensors: Sequence[Lidar | Imu | WheelOdom] | None = None,
        sensor_seed: int = 0,
    ) -> None:
        self.chassis = chassis
        self.dt = float(dt)
        self.max_range = float(max_range)
        self.collision_margin = float(collision_margin)
        self.boundary_margin = float(boundary_margin)

        self._domain_rand = bool(domain_rand)
        # 域随机化噪声的随机源（1.1.0 D）：seed 非 None ⇒ 自持一条 Generator，
        # 同 seed 逐位可复现且不动全局 np.random；None ⇒ 仍旧吃全局 np.random
        # （1.0.x 的既有行为，数值逐位不变）。
        self._seed = None if seed is None else int(seed)
        self._rng = None if self._seed is None else np.random.default_rng(self._seed)
        self._lidar_noise_std = float(lidar_noise_std)
        self._odom_drift_xy = float(odom_drift_xy)
        self._odom_drift_theta = float(odom_drift_theta)
        self._goal_dist_threshold = float(goal_dist_threshold)
        self._goal_end_wp_count = int(goal_end_wp_count)
        self._goal_end_wp_margin = int(goal_end_wp_margin)

        # ── 世界边界（旧桥由 env.yaml world.bounds / width,height 推导）──────
        self._world_xmin = float(world_bounds[0])
        self._world_xmax = float(world_bounds[1])
        self._world_ymin = float(world_bounds[2])
        self._world_ymax = float(world_bounds[3])

        # ── 车体几何（旧桥从 env.yaml robot.shape 读，这里从底盘取）─────────
        _length, _width = chassis.footprint()
        self._robot_half_l = float(_length) / 2.0
        self._robot_half_w = float(_width) / 2.0
        # LiDAR 安装位姿：(x, y, 朝向偏移)，前 = (0.145, 0, 0)、后 = (-0.155, 0, π)
        self._front_lidar, self._rear_lidar = chassis.lidar_mounts()

        # ── 机器人状态 ────────────────────────────────────────────
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._v = 0.0  # 当前前进速度（米/秒）
        self._vy = 0.0  # 当前横向速度（米/秒；仅麦轮非零，IMU 的 ay 用它做差分项）
        self._omega = 0.0  # 当前角速度（弧度/秒）
        self._collision = False
        self._goal_reached = False

        # ── 路径 ───────────────────────────────────────────────────
        self._path_waypoints_global: np.ndarray = np.zeros((0, 3))
        self._goal_heading = 0.0

        # ── 里程计漂移累积量 ────────────────────────────
        self._odom_drift_x = 0.0
        self._odom_drift_y = 0.0
        self._odom_drift_t = 0.0

        # ── 障碍物（用于快速 broadcasting 的 numpy 数组）─────────
        self._circle_centers: np.ndarray = np.empty((0, 2))
        self._circle_radii: np.ndarray = np.empty((0,))
        self._rect_centers: np.ndarray = np.empty((0, 2))
        self._rect_half_dims: np.ndarray = np.empty((0, 2))
        self._rect_thetas: np.ndarray = np.empty((0,))

        # ── 预分配的 LiDAR 缓冲区（避免每次调用时的 malloc）────
        n_rays = int(self.N_LIDAR_RAYS)
        self._lidar_origins = np.empty((n_rays, 2), dtype=np.float64)
        self._lidar_directions = np.empty((n_rays, 2), dtype=np.float64)
        self._lidar_distances = np.empty(n_rays, dtype=np.float64)
        self._lidar_angles = np.empty(n_rays, dtype=np.float64)

        # ── 动态障碍（解析运动；空列表 ⇒ 零额外计算，逐位不变）────────
        self._steps = 0  # 已推进的步数；模拟时间 = _steps * dt（乘法，不累加浮点）
        self._dyn_specs: list[dict[str, Any]] = []
        self._dyn_centers: np.ndarray = np.empty((0, 2))
        self._dyn_radii: np.ndarray = np.empty((0,))
        self.set_dynamic_obstacles(dynamic_obstacles)

        # ── 可选传感器（默认全关：不构造对象、不消耗随机数、info 零新键）──────
        # 几何注入：传感器包只给车体系 (原点, 方位角, 量程)，世界系变换与求交留在桥上。
        self._sensor_seed = int(sensor_seed)
        self._sensors: SensorSuite | None = None
        if sensors is not None and len(sensors) > 0:
            self._sensors = SensorSuite(
                sensors,
                seed=self._sensor_seed,
                raycast=self._raycast_body_ray,
                default_track_width=float(_width),
            )
        self._sensor_reading: dict[str, Any] | None = None

        # ── 预分配的世界系角点缓冲 ────────────────────
        # 注意：局部角点**不**在此缓存。旧桥 :259-261 缓存过一份不含 margin 的
        # 局部角点，但 :598-599 又按 hl/hw（含 margin）重建 —— 缓存那份是死代码。
        # 本实现只保留世界系缓冲，局部角点每帧按 margin 重建。
        self._collision_corners_world = np.empty((4, 2), dtype=np.float64)

    def _noise(self, scale: float, size: Any = None) -> Any:
        """域随机化的高斯噪声 N(0, scale)：随机源只有一个判据 —— 有没有 seed。

        * seed 非 None ⇒ 从本桥自持的 Generator 取（同 seed 逐位可复现）；
        * seed 为 None ⇒ 从全局 np.random 取（1.0.x 的既有行为）。

        `size=None` 返回标量（里程计漂移），给 `size` 返回数组（LiDAR 噪声）。
        两条路径的调用形状与 1.0.x 逐字一致，因此 seed=None 时逐位不变。
        """
        if self._rng is None:
            return np.random.normal(0.0, scale, size=size)
        return self._rng.normal(0.0, scale, size=size)

    # ═════════════════════════════════════════════════════════════════
    # SensorProvider 接口
    # ═════════════════════════════════════════════════════════════════

    @property
    def goal_heading(self) -> float:
        """终点处的**车头朝向（鼻向）** ∈ [-π, π]（B5：不是运动方向）。

        消费方是 env 的 goal_sig（sin/cos(robot_theta − goal_heading)）与终端奖励的
        朝向对齐项；语义为"车到达终点时应让车头对齐路径终点的鼻向"——倒车到达时
        车头朝向恰好等于路径末点鼻向，因此该项在末段倒车的地图上不再反向。
        """
        return self._goal_heading

    def reset(
        self,
        obstacles: list | None = None,
        path_waypoints: np.ndarray | None = None,
        initial_heading: float | None = None,
    ) -> None:
        # ── 将障碍物解析为带类型的 numpy 数组 ────────────────
        circles_c, circles_r = [], []
        rects_c, rects_hd, rects_t = [], [], []
        if obstacles:
            for obs in obstacles:
                shape = obs["shape"]
                state = obs["state"]
                cx, cy, ct = float(state[0]), float(state[1]), float(state[2])
                if shape["name"] == "circle":
                    circles_c.append([cx, cy])
                    circles_r.append(float(shape["radius"]))
                elif shape["name"] == "rectangle":
                    rects_c.append([cx, cy])
                    rects_hd.append(
                        [float(shape["length"]) / 2, float(shape["width"]) / 2]
                    )
                    rects_t.append(ct)
        self._circle_centers = np.array(circles_c, dtype=np.float64).reshape(-1, 2)
        self._circle_radii = np.array(circles_r, dtype=np.float64)
        self._rect_centers = np.array(rects_c, dtype=np.float64).reshape(-1, 2)
        self._rect_half_dims = np.array(rects_hd, dtype=np.float64).reshape(-1, 2)
        self._rect_thetas = np.array(rects_t, dtype=np.float64)

        # ── 动态障碍相位复位（t = 0 ⇒ 场景 seed 决定的初态；无随机时即起点）──
        self._steps = 0
        self._refresh_dynamic()

        # ── 根据路径初始化机器人 ─────────────────────────────
        self._collision = False
        self._goal_reached = False
        self._odom_drift_x = 0.0
        self._odom_drift_y = 0.0
        self._odom_drift_t = 0.0
        self._v = 0.0
        self._vy = 0.0
        self._omega = 0.0
        # 执行器状态清零：旧桥在此处 self._steer = 0.0（:315），对应本仓底盘里的
        # 转向滞后状态。不清零则新回合第一步沿用上一回合末的转角。
        self.chassis.reset_state()
        # 传感器的延迟缓冲与 IMU 速度历史同样不能跨回合残留
        self._sensor_reading = None
        if self._sensors is not None:
            self._sensors.reset()

        if path_waypoints is not None and len(path_waypoints) > 0:
            self._path_waypoints_global = path_waypoints.astype(np.float64)
            start_pt = path_waypoints[0]
            self._x = float(start_pt[0])
            self._y = float(start_pt[1])

            if initial_heading is not None:
                self._theta = float(initial_heading)
            elif len(path_waypoints) >= 2:
                dx = path_waypoints[1][0] - start_pt[0]
                dy = path_waypoints[1][1] - start_pt[1]
                self._theta = float(np.arctan2(dy, dx))
            else:
                self._theta = 0.0

            # ── 终点朝向 = 路径末点自带的**鼻向**（B5 修复，2026-10-05）──
            # 路径第 2 列是车头朝向（RS 车辆积分所得），不是运动方向；倒车段上
            # 二者相差 π。旧实现取最后两点 XY 差（atan2(dy, dx) = 运动方向），
            # 于是末段倒车的地图上：
            #   1. env 用它算 goal_sig（sin/cos(robot_theta - goal_heading)）→ 目标
            #      朝向被指反；
            #   2. 车"正确地倒车到达"时 reward 的 10×cos(goal_heading_error) 给 −10。
            # 与地图生成侧的 _compute_goal_heading 保持同一语义，两处必须一致，
            # 否则"地图给的终点朝向"与"bridge 报的终点朝向"打架。
            # 注意：本 bridge 的 check_goal_reached() 只判距离 + 末段航点窗口，
            # 不含朝向容差；朝向容差判定只在 ROS2 桥使用。
            if len(path_waypoints) >= 2 and path_waypoints.shape[1] >= 3:
                self._goal_heading = float(
                    (float(path_waypoints[-1][2]) + np.pi) % (2 * np.pi) - np.pi
                )
            elif len(path_waypoints) >= 2:
                # 2 列路径无鼻向信息 → 退回运动方向（保持旧行为）
                dx = path_waypoints[-1][0] - path_waypoints[-2][0]
                dy = path_waypoints[-1][1] - path_waypoints[-2][1]
                self._goal_heading = float(np.arctan2(dy, dx))
            else:
                self._goal_heading = 0.0
        else:
            # 回退方案：从当前朝向向前生成直线路径（10 米）
            num_pts = 20
            t = np.linspace(0, 10.0, num_pts)
            path_x = self._x + t * np.cos(self._theta)
            path_y = self._y + t * np.sin(self._theta)
            self._path_waypoints_global = np.column_stack(
                [
                    path_x,
                    path_y,
                    np.full(num_pts, self._theta),
                ]
            )
            self._goal_heading = float(self._theta)

        # ── 首帧传感器读数（S3b）──
        # 必须排在位姿 / 朝向 / 时间都复位之后：射线求交用的就是 _x/_y/_theta，
        # 提前调用会拿**上一回合**的位姿算。v/vy/omega 已在上方清零，所以这一帧
        # 是"静止于起点"的真实读数，reset 返回的观测里 lidar 段不再谎报量程外。
        self._update_sensors()

    # ── 动态障碍 ────────────────────────────────────────────────────

    @property
    def sim_time(self) -> float:
        """自上次 `reset()` 起的模拟时间（秒）= 步数 × dt。"""
        return self._steps * self.dt

    def set_dynamic_obstacles(self, specs: list | None = None) -> None:
        """换一组动态障碍（`None` / `[]` = 清空）。**不**改变模拟时间。

        `reset()` 会把时间归零，因此"相位复位到初态"由 `reset()` 负责；本方法
        只替换规格，位置按**当前**时间重新求值（运行中换障碍即在下一步生效）。
        """
        parsed: list[dict[str, Any]] = []
        for index, raw in enumerate(specs or ()):
            kind = raw.get("type")
            radius = float(raw.get("radius", 0.0))
            if radius <= 0.0:
                raise ValueError(
                    f"dynamic_obstacles[{index}].radius 必须 > 0，实际是 {radius!r}"
                )
            if kind == "linear":
                parsed.append(
                    {
                        "type": "linear",
                        "start": tuple(float(v) for v in raw["start"]),
                        "velocity": tuple(float(v) for v in raw["velocity"]),
                        "radius": radius,
                        "period": float(raw.get("period", 0.0)),
                        "pingpong": bool(raw.get("pingpong", False)),
                    }
                )
            elif kind == "circular":
                parsed.append(
                    {
                        "type": "circular",
                        "center": tuple(float(v) for v in raw["center"]),
                        "radius_orbit": float(raw["radius_orbit"]),
                        "omega": float(raw["omega"]),
                        "phase": float(raw.get("phase", 0.0)),
                        "radius": radius,
                    }
                )
            else:
                raise ValueError(
                    f"dynamic_obstacles[{index}].type 未知: {kind!r}"
                    "（支持: linear, circular）"
                )
        self._dyn_specs = parsed
        self._dyn_centers = np.zeros((len(parsed), 2), dtype=np.float64)
        self._dyn_radii = np.array([p["radius"] for p in parsed], dtype=np.float64)
        self._refresh_dynamic()

    def _refresh_dynamic(self) -> None:
        """按当前 `t = _steps * dt` 重算动态障碍圆心（解析式，零随机）。"""
        if not self._dyn_specs:
            return
        t = self._steps * self.dt
        for index, spec in enumerate(self._dyn_specs):
            if spec["type"] == "linear":
                sx, sy = spec["start"]
                vx, vy = spec["velocity"]
                phase = _linear_phase(t, spec["period"], spec["pingpong"])
                self._dyn_centers[index, 0] = sx + vx * phase
                self._dyn_centers[index, 1] = sy + vy * phase
            else:
                cx, cy = spec["center"]
                phi = spec["phase"] + spec["omega"] * t
                self._dyn_centers[index, 0] = cx + spec["radius_orbit"] * np.cos(phi)
                self._dyn_centers[index, 1] = cy + spec["radius_orbit"] * np.sin(phi)

    def dynamic_positions(self) -> np.ndarray:
        """`(D, 3)` 的 `[x, y, radius]` 快照（当前时刻；无动态障碍时 `(0, 3)`）。"""
        if not self._dyn_specs:
            return np.empty((0, 3))
        return np.column_stack([self._dyn_centers, self._dyn_radii])

    def step(self, action: np.ndarray) -> None:
        """把仿真推进一个时间步（三步顺序固定，不能变）。

        1. `cmd = chassis.step(action)` —— 执行器状态推进（阿克曼的转向滞后一拍
           在底盘里发生；默认底盘等价于 `command()`）；
        2. `chassis.integrate(...)` —— 唯一的位姿积分核；
        3. `θ` 归一化到 [-π, π] —— 与旧桥 :390 逐字同式。
        """
        cmd = self.chassis.step(action)
        self._v = float(cmd.v)
        self._omega = float(cmd.omega)
        self._x, self._y, self._theta = self.chassis.integrate(
            self._x, self._y, self._theta, cmd, self.dt
        )
        self._theta = (self._theta + np.pi) % (2 * np.pi) - np.pi
        self._vy = float(cmd.vy)

        # 动态障碍先推进到与本步位姿**同一时刻**（t = _steps * dt），再判碰撞，
        # 否则第 k 步会用上一时刻的障碍位置（错一拍）。
        self._steps += 1
        self._refresh_dynamic()

        # 碰撞 / 目标检测
        self._collision = self._check_collision()

        goal_pt = (
            self._path_waypoints_global[-1, :2]
            if len(self._path_waypoints_global) > 0
            else None
        )
        if goal_pt is not None:
            dist = np.hypot(self._x - goal_pt[0], self._y - goal_pt[1])
            # 最近路径点必须在末尾 N 个以内，防止路径折返时路过 goal 提前触发
            wp_dists = np.hypot(
                self._path_waypoints_global[:, 0] - self._x,
                self._path_waypoints_global[:, 1] - self._y,
            )
            nearest_idx = int(np.argmin(wp_dists))
            n_waypoints = len(self._path_waypoints_global)
            end_margin = min(self._goal_end_wp_count, max(n_waypoints - 1, 1))
            near_end = (
                nearest_idx >= n_waypoints - end_margin - self._goal_end_wp_margin
            )
            # 显式 bool()：SensorProvider 声明 -> bool，旧桥此处会漏出 np.bool_
            self._goal_reached = bool(dist <= self._goal_dist_threshold and near_end)
        else:
            self._goal_reached = False

        # 传感器在本步位姿/速度确定之后推进（t = _steps * dt，与 sim_time 同源）
        self._update_sensors()

    # ── 传感器读数（雷达只有 get_sensor_reading 一个入口）────────────────

    def _ensure_lidar_buffers(self, n_rays: int) -> None:
        """按射线数准备预分配缓冲区（默认 180 时**不**重分配）。"""
        if self._lidar_distances.shape[0] == n_rays:
            return
        self._lidar_origins = np.empty((n_rays, 2), dtype=np.float64)
        self._lidar_directions = np.empty((n_rays, 2), dtype=np.float64)
        self._lidar_distances = np.empty(n_rays, dtype=np.float64)
        self._lidar_angles = np.empty(n_rays, dtype=np.float64)

    def _cast_rays(
        self,
        *,
        front: bool,
        n_rays: int,
        fov: float,
        max_range: float,
        range_min: float,
        domain_rand: bool,
        mount: tuple[float, float] | None = None,
        angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """从前方或后方 LiDAR 安装位发射 `n_rays` 条射线。

        几何与默认雷达共用同一份算术（圆 / 矩形 / 动态圆 / 四面墙求交 + clip），
        参数化只体现在射线数与视场角：

            angles = linspace(θ + mount_yaw - fov/2, + fov, n_rays, endpoint=True)

        因此第 i 条射线相对车头的方位角 = (mount_yaw - fov/2) + i·fov/(n_rays-1)：
        视场以安装朝向为中心（fov = π 时与历史的 -π/2 起点逐位相同）。

        Args:
            front: True 用前安装位，False 用后安装位。
            n_rays: 射线数（>= 2）。
            fov: 视场角（弧度）。
            max_range: 最大量程（米）。
            range_min: 最小量程（米）。
            domain_rand: True 时按 `lidar_noise_std` 叠加**全局** np.random 噪声
                （历史行为）；参数化传感器传 False —— 它有自己的独立 Generator。
            mount: 车体系射线原点 (x, y)；给定时忽略 front 所指的安装位（安装朝向
                已经烘进 angles），供传感器包注入的单射线调用使用。
            angles: 世界系射线方位角（长度必须等于 n_rays）；给定时替代
                `linspace(θ + mount_yaw - fov/2, +fov)` 的均分。两条新参同时缺省
                时本函数的算术与缓冲区用法与加入参数化传感器前逐位相同。

        Returns:
            (n_rays,) float32 距离，已 clip 到 [range_min, max_range]。
        """
        self._ensure_lidar_buffers(n_rays)

        # 传感器在世界坐标系中的位姿
        if mount is None:
            off_x, off_y, angle_off = self._front_lidar if front else self._rear_lidar
        else:
            # 车体系原点直给：世界系变换照旧，朝向由 angles 自己带
            off_x, off_y, angle_off = float(mount[0]), float(mount[1]), 0.0

        cos_t = np.cos(self._theta)
        sin_t = np.sin(self._theta)
        sx = self._x + off_x * cos_t - off_y * sin_t
        sy = self._y + off_x * sin_t + off_y * cos_t

        # 射线方向——复用预分配的缓冲区
        if angles is None:
            start_angle = self._theta + angle_off - 0.5 * fov
            self._lidar_angles[:] = np.linspace(
                start_angle, start_angle + fov, n_rays, endpoint=True
            )
        else:
            self._lidar_angles[:] = np.asarray(angles, dtype=np.float64)
        np.cos(self._lidar_angles, out=self._lidar_directions[:, 0])
        np.sin(self._lidar_angles, out=self._lidar_directions[:, 1])
        self._lidar_origins[:, 0] = sx
        self._lidar_origins[:, 1] = sy

        # 对所有障碍物进行射线求交
        self._lidar_distances.fill(max_range)

        if len(self._circle_radii) > 0:
            d_c = _ray_circles_intersection(
                self._lidar_origins,
                self._lidar_directions,
                self._circle_centers,
                self._circle_radii,
                max_range,
            )
            np.minimum(self._lidar_distances, d_c, out=self._lidar_distances)

        # 动态障碍（当作圆参与同一套射线求交；空则跳过 ⇒ 默认路径逐位不变）
        if len(self._dyn_radii) > 0:
            d_dyn = _ray_circles_intersection(
                self._lidar_origins,
                self._lidar_directions,
                self._dyn_centers,
                self._dyn_radii,
                max_range,
            )
            np.minimum(self._lidar_distances, d_dyn, out=self._lidar_distances)

        if len(self._rect_thetas) > 0:
            d_r = _ray_rects_intersection(
                self._lidar_origins,
                self._lidar_directions,
                self._rect_centers,
                self._rect_half_dims,
                self._rect_thetas,
                max_range,
            )
            np.minimum(self._lidar_distances, d_r, out=self._lidar_distances)

        # 墙壁边界
        directions_y = self._lidar_directions[:, 1]
        directions_x = self._lidar_directions[:, 0]

        # 上墙
        mask = directions_y > 1e-10
        if np.any(mask):
            d_top = (self._world_ymax - sy) / directions_y[mask]
            better = (d_top > 0) & (d_top < self._lidar_distances[mask])
            self._lidar_distances[mask] = np.where(
                better, d_top, self._lidar_distances[mask]
            )

        # 下墙
        mask = directions_y < -1e-10
        if np.any(mask):
            d_bot = (self._world_ymin - sy) / directions_y[mask]
            better = (d_bot > 0) & (d_bot < self._lidar_distances[mask])
            self._lidar_distances[mask] = np.where(
                better, d_bot, self._lidar_distances[mask]
            )

        # 右墙
        mask = directions_x > 1e-10
        if np.any(mask):
            d_right = (self._world_xmax - sx) / directions_x[mask]
            better = (d_right > 0) & (d_right < self._lidar_distances[mask])
            self._lidar_distances[mask] = np.where(
                better, d_right, self._lidar_distances[mask]
            )

        # 左墙
        mask = directions_x < -1e-10
        if np.any(mask):
            d_left = (self._world_xmin - sx) / directions_x[mask]
            better = (d_left > 0) & (d_left < self._lidar_distances[mask])
            self._lidar_distances[mask] = np.where(
                better, d_left, self._lidar_distances[mask]
            )

        # 地板（最小量程）
        np.clip(
            self._lidar_distances,
            range_min,
            max_range,
            out=self._lidar_distances,
        )

        # Domain randomisation（随机源见 _noise：有 seed 走自持 Generator，否则走
        # 全局 np.random；参数化传感器不走这里，它们有自己的 Generator）
        if domain_rand:
            noise = self._noise(self._lidar_noise_std, n_rays).astype(np.float64)
            self._lidar_distances += noise
            np.clip(
                self._lidar_distances,
                range_min,
                max_range,
                out=self._lidar_distances,
            )

        return self._lidar_distances.copy().astype(np.float32)

    def _raycast_body_ray(
        self, origin_xy: tuple[float, float], angle_rad: float, max_range: float
    ) -> float:
        """车体系单射线求交——注入给 `chassis_sim.sensors.SensorSuite` 的 `raycast`。

        与 `_cast_rays` 共用同一份算术（圆 / 矩形 / 动态圆 / 四面墙求交 + clip）：
        几何只有一份实现，因此不同的传感器之间不会各自漂移。区别只在
        把"一次 n_rays 条均分射线"换成"一条指定方位角、指定原点的射线"，因此本方法
        每次都从头车变换出发（theta / x / y 取自当前步状态）。

        Args:
            origin_xy: 车体系射线原点 (x, y)，单位米。
            angle_rad: 车体系方位角（车头 x 轴起、逆时针为正），单位弧度。
            max_range: 本次求交的最大量程（米）。

        Returns:
            float：命中距离（米），已 clip 到 [RANGE_MIN, max_range]；无命中时返回
            max_range（饱和），即"无回波"与"恰好命中量程"不可区分。
        """
        angles = np.array([self._theta + float(angle_rad)], dtype=np.float64)
        distances = self._cast_rays(
            front=True,
            n_rays=1,
            fov=0.0,
            max_range=float(max_range),
            range_min=self.RANGE_MIN,
            domain_rand=False,
            mount=(float(origin_xy[0]), float(origin_xy[1])),
            angles=angles,
        )
        return float(distances[0])

    def get_sensor_reading(self) -> dict[str, Any] | None:
        """当前步的传感器读数（未启用任何传感器时为 None）。

        结构是 `dict[传感器名, 读数]`，键序 = 构造时传入的 spec 列表顺序，只包含
        **已装配**的传感器（没给 spec 就没有键）：

        * LiDAR:      {"scan": (n_rays,) float32 或 None, "scan_meta": {...}}
                      scan 为 None = 延迟预热期 / 本步丢帧（见 scan_meta 的
                      valid / dropped）；scan_meta 含 name / mount / 角度几何 /
                      range_min / range_max / latency_steps / t / frame_t。
        * IMU:        {"ax", "ay", "yaw_rate", "yaw", "t"}
        * 轮速计:     {"v_left", "v_right", "v", "t"}

        同名传感器不可能共存（随机流与读数都按 name 索引，构造时即报错）。
        消费方只有 env 的 info 组装：本桥从不把读数放进 obs，obs 维度保持不变。
        返回浅拷贝，调用方改字典不会污染桥的内部状态（内层读数与数组仍是引用）。
        """
        return None if self._sensor_reading is None else dict(self._sensor_reading)

    def _update_sensors(self) -> None:
        """推进可选传感器；未启用时是零开销空操作（不消耗任何随机数）。"""
        if self._sensors is None:
            return
        # LiDAR 的每条射线通过构造时注入的 _raycast_body_ray 回调求交；那里一律
        # domain_rand=False —— 传感器的噪声来自自己的 Generator，绝不消耗全局
        # np.random（否则会平移 domain_rand 的 LiDAR/里程计序列）。
        self._sensor_reading = self._sensors.update(
            v=self._v,
            vy=self._vy,
            omega=self._omega,
            theta=self._theta,
            t=self.sim_time,
        )

    def get_odometry(self) -> np.ndarray:
        # 注意: 里程计漂移在返回后累积 (与 ir-sim 桥的内部漂移时序差 1 步)。
        # 初始化时漂移为 0，训练中无实际影响；对拍精度测试可能出现微小偏差。
        # 顺序即契约：先产出含当前漂移的值，再把本步漂移累加进状态。
        x = self._x + self._odom_drift_x
        y = self._y + self._odom_drift_y
        t = self._theta + self._odom_drift_t
        if self._domain_rand:
            self._odom_drift_x += self._noise(self._odom_drift_xy * self.dt)
            self._odom_drift_y += self._noise(self._odom_drift_xy * self.dt)
            self._odom_drift_t += self._noise(self._odom_drift_theta * self.dt)
        return np.array([x, y, t, self._v, self._omega], dtype=np.float32)

    # ── 只读快照访问器（渲染层用；不改变任何既有行为）────────────────

    def pose(self) -> tuple[float, float, float]:
        """当前真实位姿 (x, y, θ)：世界系，单位米 / 弧度（不含里程计漂移）。"""
        return (float(self._x), float(self._y), float(self._theta))

    def obstacle_shapes(self) -> list[dict[str, Any]]:
        """静态障碍物快照，格式与 `reset(obstacles=...)` 的输入一致。

        返回**新建**的 list / dict，调用方改动不会污染桥内部状态；动态障碍物
        不在其中（它们随时间变化，需走 `dynamic_positions()`）。
        """
        shapes: list[dict[str, Any]] = []
        for (cx, cy), radius in zip(self._circle_centers, self._circle_radii):
            shapes.append(
                {
                    "shape": {"name": "circle", "radius": float(radius)},
                    "state": [float(cx), float(cy), 0.0],
                }
            )
        for (cx, cy), (half_l, half_w), theta in zip(
            self._rect_centers, self._rect_half_dims, self._rect_thetas
        ):
            shapes.append(
                {
                    "shape": {
                        "name": "rectangle",
                        "length": float(2.0 * half_l),
                        "width": float(2.0 * half_w),
                    },
                    "state": [float(cx), float(cy), float(theta)],
                }
            )
        return shapes

    def get_path(self) -> np.ndarray:
        if len(self._path_waypoints_global) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 向量化的车体坐标系变换
        dx = self._path_waypoints_global[:, 0] - self._x
        dy = self._path_waypoints_global[:, 1] - self._y
        cos_t = np.cos(-self._theta)
        sin_t = np.sin(-self._theta)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t

        if self._path_waypoints_global.shape[1] >= 3:
            rtheta = self._path_waypoints_global[:, 2] - self._theta
            rtheta = (rtheta + np.pi) % (2 * np.pi) - np.pi
            return np.column_stack([rx, ry, rtheta]).astype(np.float32)
        return np.column_stack([rx, ry]).astype(np.float32)

    def get_path_global(self) -> np.ndarray:
        """返回世界系下全部路径航点，形状 (N, 3) [x, y, θ]。"""
        if len(self._path_waypoints_global) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self._path_waypoints_global.astype(np.float32).copy()

    def check_collision(self) -> bool:
        return self._collision

    def check_goal_reached(self, threshold: float = 0.3) -> bool:
        # _goal_reached 在 step() 中用 _goal_dist_threshold 计算。
        # 若调用者传入不同 threshold，用新阈值重新判断距离部分。
        if threshold == self._goal_dist_threshold:
            return bool(self._goal_reached)
        if not self._goal_reached and threshold <= self._goal_dist_threshold:
            return False  # 更严格阈值，原本就未达成，无需复算
        goal_pt = (
            self._path_waypoints_global[-1, :2]
            if len(self._path_waypoints_global) > 0
            else None
        )
        if goal_pt is not None:
            dist = np.hypot(self._x - goal_pt[0], self._y - goal_pt[1])
            # 最近路径点必须在末尾 N 个以内，防止路径折返时路过 goal 提前触发
            wp_dists = np.hypot(
                self._path_waypoints_global[:, 0] - self._x,
                self._path_waypoints_global[:, 1] - self._y,
            )
            nearest_idx = int(np.argmin(wp_dists))
            n_waypoints = len(self._path_waypoints_global)
            end_margin = min(self._goal_end_wp_count, max(n_waypoints - 1, 1))
            near_end = (
                nearest_idx >= n_waypoints - end_margin - self._goal_end_wp_margin
            )
            return bool(dist < threshold and near_end)
        return bool(self._goal_reached)

    def close(self) -> None:
        pass  # 无需清理资源

    # ═════════════════════════════════════════════════════════════════
    # 内部方法
    # ═════════════════════════════════════════════════════════════════

    def _circles_overlap_robot(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        hl: float,
        hw: float,
        cos_t: float,
        sin_t: float,
        cw: np.ndarray,
    ) -> bool:
        """圆（静态/动态共用）与机器人矩形的重叠判定。

        两条独立的检查相或：(a) 机器人四个角点落在圆内（`r + margin`）；
        (b) 圆心落在机器人矩形内（`hl/hw` 已含 margin，故半径**不**再加 margin，
        避免 margin 双重计数）。静态圆与动态圆走同一份算数，只是喂进来的数组不同。
        """
        margin = self.collision_margin
        diff = cw[:, None, :] - centers[None, :, :]  # (4, C, 2)
        r_eff = radii + margin  # (C,)
        if np.any(np.sum(diff * diff, axis=-1) < r_eff[None, :] ** 2):
            return True
        local_c = centers - np.array([self._x, self._y])  # (C, 2)
        lx_c = cos_t * local_c[:, 0] + sin_t * local_c[:, 1]
        ly_c = -sin_t * local_c[:, 0] + cos_t * local_c[:, 1]
        return bool(np.any((np.abs(lx_c) < hl + radii) & (np.abs(ly_c) < hw + radii)))

    def _check_collision(self) -> bool:
        """机器人体与任何障碍物重叠则返回 True。

        在障碍物维度上完全向量化——没有 Python for 循环。
        """
        # 碰撞检测安全裕度：防止雷达噪声导致轻微触碰即判碰撞
        margin = self.collision_margin
        hl = self._robot_half_l + margin
        hw = self._robot_half_w + margin
        cos_t = np.cos(self._theta)
        sin_t = np.sin(self._theta)

        # ── 机器人角点在世界坐标系中的位置（含安全裕度）───────
        # Bug fix: 旧桥缓存的局部角点不含 margin，直接用则 margin 形同虚设——
        # 机器人角点判定仍是紧贴车身的物理极限，"幽灵车角"。此处用 hl/hw（已加
        # margin）重建局部角点，确保四个角正确外扩。
        cl_m = np.array(
            [[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]], dtype=np.float64
        )  # (4, 2) with margin
        self._collision_corners_world[:, 0] = (
            self._x + cos_t * cl_m[:, 0] - sin_t * cl_m[:, 1]
        )
        self._collision_corners_world[:, 1] = (
            self._y + sin_t * cl_m[:, 0] + cos_t * cl_m[:, 1]
        )
        cw = self._collision_corners_world  # (4, 2)

        # ── 机器人 vs 圆形（静态 + 动态共用同一判定）──────────────
        if len(self._circle_radii) > 0 and self._circles_overlap_robot(
            self._circle_centers, self._circle_radii, hl, hw, cos_t, sin_t, cw
        ):
            return True
        if len(self._dyn_radii) > 0 and self._circles_overlap_robot(
            self._dyn_centers, self._dyn_radii, hl, hw, cos_t, sin_t, cw
        ):
            return True

        # ── SAT: OBB(机器人) vs OBB(矩形障碍物) ──────────────────
        # 旧代码仅检查"机器人角点是否在矩形内部"——单向检查。小障碍物
        # (10cm 锥桶 AABB) 穿过车身中心时四个角全在盒外，幽灵穿模。
        # SAT 在 4 根轴上检验投影重叠：任一根轴分离 → 无碰撞。
        if len(self._rect_thetas) > 0:
            h_rect = self._rect_half_dims + margin  # (M, 2)
            cos_rt = np.cos(self._rect_thetas)  # (M,)
            sin_rt = np.sin(self._rect_thetas)  # (M,)
            # 中心差: rect - robot
            dx_c = self._rect_centers[:, 0] - self._x  # (M,)
            dy_c = self._rect_centers[:, 1] - self._y  # (M,)

            # 轴 1: 机器人局部 X = (cos_t, sin_t)
            d1 = np.abs(cos_t * dx_c + sin_t * dy_c)
            r_h1 = np.abs(h_rect[:, 0] * (cos_t * cos_rt + sin_t * sin_rt)) + np.abs(
                h_rect[:, 1] * (-cos_t * sin_rt + sin_t * cos_rt)
            )
            sep1 = d1 > hl + r_h1

            # 轴 2: 机器人局部 Y = (-sin_t, cos_t)
            d2 = np.abs(-sin_t * dx_c + cos_t * dy_c)
            r_h2 = np.abs(h_rect[:, 0] * (-sin_t * cos_rt + cos_t * sin_rt)) + np.abs(
                h_rect[:, 1] * (sin_t * sin_rt + cos_t * cos_rt)
            )
            sep2 = d2 > hw + r_h2

            # 轴 3: 矩形的局部 X = (cos_rt, sin_rt)
            d3 = np.abs(cos_rt * dx_c + sin_rt * dy_c)
            robot_h3 = np.abs(hl * (cos_rt * cos_t + sin_rt * sin_t)) + np.abs(
                hw * (-cos_rt * sin_t + sin_rt * cos_t)
            )
            sep3 = d3 > robot_h3 + h_rect[:, 0]

            # 轴 4: 矩形的局部 Y = (-sin_rt, cos_rt)
            d4 = np.abs(-sin_rt * dx_c + cos_rt * dy_c)
            robot_h4 = np.abs(hl * (-sin_rt * cos_t + cos_rt * sin_t)) + np.abs(
                hw * (sin_rt * sin_t + cos_rt * cos_t)
            )
            sep4 = d4 > robot_h4 + h_rect[:, 1]

            # 四轴全重叠 → 碰撞（无轴分离）
            if np.any(~(sep1 | sep2 | sep3 | sep4)):
                return True

        # ── 世界边界：按朝向的轴对齐包围盒（P2-16）──────────────
        # 旧实现用 hl 判 x、hw 判 y，与 θ 无关：θ=90° 时车身沿 y 的半宽其实
        # 是 hl=0.22（含 margin）却按 hw=0.13 判 ⇒ 车尾可侵入墙 0.09 m 而不判
        # 碰撞。改用旋转后的 AABB 半宽/半高（|cosθ|·hl+|sinθ|·hw 等）。
        # ⚠ 地图生成侧的 _check_path_footprint 是同一份逻辑（"照抄"关系，参考
        #   路径复验用），**两处必须同改**，否则复验会放行 env 判碰撞的位姿。
        margin = self.boundary_margin
        ext_x = abs(cos_t) * hl + abs(sin_t) * hw
        ext_y = abs(sin_t) * hl + abs(cos_t) * hw
        return bool(
            self._x - ext_x < self._world_xmin + margin
            or self._x + ext_x > self._world_xmax - margin
            or self._y - ext_y < self._world_ymin + margin
            or self._y + ext_y > self._world_ymax - margin
        )
