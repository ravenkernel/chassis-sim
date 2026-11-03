"""PhysicalCommandBridge：归一化 action -> 底盘公开契约 physical_action() -> 内层物理桥。

五条主线：

1. 映射**逐位**等于 chassis.physical_action()（三车型各 200 例），阿克曼的 delta
   与 _map_action() 的第一个返回值 0 ulp 相等（不再 arctan 反解）；
2. 映射是**纯函数**：连续两次（中间穿插 chassis.step()）逐位相同；
3. 与 SimpleBridge 位姿等价到 1e-12（内层替身自己承担一拍转向滞后）；
4. 差速/麦轮收到的是 [v, omega] / [v, vy, omega]，与 command() 逐位相等；
5. env 的 info["mapped_action"][0] 走新契约后不再恒 0（外部桥不推进底盘状态）。

外加 SensorProvider 每个成员都被**显式**实现并转发、扫描/读数原样透传。
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from chassis_sim.bridges import PhysicalCommandBridge, SensorProvider, SimpleBridge
from chassis_sim.chassis import AckermannChassis, ChassisCommand, make_chassis
from chassis_sim.envs import ChassisSimEnv

N_STEPS = 200
SEED = 20261104
TOL_POSE = 1e-12
DT = 0.1
CHASSIS_NAMES = ("ackermann", "diff_drive", "mecanum")

#: SensorProvider 的全部成员（10 抽象方法 + 1 抽象属性 + 3 个非抽象附加 API）。
REQUIRED_MEMBERS = {
    "reset",
    "step",
    "get_lidar_front",
    "get_lidar_rear",
    "get_odometry",
    "get_path",
    "get_path_global",
    "check_collision",
    "check_goal_reached",
    "close",
    "goal_heading",
    "set_dynamic_obstacles",
    "get_scan",
    "get_sensor_reading",
}


class _RecordingBridge(SensorProvider):
    """记录型内层桥：不推进物理，只把收到的物理命令与调用次数记下来。

    刻意**不**覆盖 set_dynamic_obstacles：这类桥代表"只实现静态场景"的内层。
    """

    def __init__(
        self,
        scan: dict[str, Any] | None = None,
        sensor_reading: dict[str, Any] | None = None,
    ) -> None:
        self.steps: list[np.ndarray] = []
        self.calls: dict[str, int] = {}
        self.scan = {} if scan is None else scan
        self.scan_sides: list[str] = []
        self.sensor_reading = sensor_reading
        self.reset_args: tuple[Any, Any, Any] | None = None

    def _hit(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def reset(
        self,
        obstacles: list | None = None,
        path_waypoints: np.ndarray | None = None,
        initial_heading: float | None = None,
    ) -> None:
        self._hit("reset")
        self.reset_args = (obstacles, path_waypoints, initial_heading)

    def step(self, action: np.ndarray) -> None:
        self._hit("step")
        self.steps.append(np.array(action, dtype=np.float64, copy=True))

    def get_lidar_front(self) -> np.ndarray:
        self._hit("get_lidar_front")
        return np.full(180, 1.5)

    def get_lidar_rear(self) -> np.ndarray:
        self._hit("get_lidar_rear")
        return np.full(180, 2.5)

    def get_odometry(self) -> np.ndarray:
        self._hit("get_odometry")
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    def get_path(self) -> np.ndarray:
        self._hit("get_path")
        return np.zeros((3, 3))

    def get_path_global(self) -> np.ndarray:
        self._hit("get_path_global")
        return np.ones((4, 3))

    def check_collision(self) -> bool:
        self._hit("check_collision")
        return False

    def check_goal_reached(self, threshold: float = 0.3) -> bool:
        self._hit("check_goal_reached")
        return True

    def close(self) -> None:
        self._hit("close")

    @property
    def goal_heading(self) -> float:
        self._hit("goal_heading")
        return 1.25

    def get_scan(self, side: str = "front") -> dict[str, Any]:
        self._hit("get_scan")
        self.scan_sides.append(side)
        return self.scan[side]

    def get_sensor_reading(self) -> dict[str, Any] | None:
        self._hit("get_sensor_reading")
        return self.sensor_reading


class _DynamicRecordingBridge(_RecordingBridge):
    """覆盖了 set_dynamic_obstacles 的记录型内层（支持动态障碍的那种）。"""

    def set_dynamic_obstacles(self, specs: list | None = None) -> None:
        self._hit("set_dynamic_obstacles")
        self.dynamic_specs = specs


class _PhysicalAckermannBridge(SensorProvider):
    """测试替身：真正吃 [delta, v] 物理命令的阿克曼内层桥。

    与 SimpleBridge 的唯一区别是**自己**承担一拍转向执行器滞后
    （_prev_delta，公式与 chassis/ackermann.py 的 step() 逐字一致）——
    真实物理模拟器本来就是这么干的，适配器则完全不带滞后。位姿积分复用
    chassis.integrate，所以两者只可能在"omega 的算法差异"上分开。
    """

    def __init__(self, chassis: AckermannChassis, dt: float = DT) -> None:
        self.chassis = chassis
        self.dt = dt
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._v = 0.0
        self._omega = 0.0
        self._prev_delta = 0.0
        self.received: list[np.ndarray] = []

    def reset(
        self,
        obstacles: list | None = None,
        path_waypoints: np.ndarray | None = None,
        initial_heading: float | None = None,
    ) -> None:
        self._prev_delta = 0.0
        self._v = 0.0
        self._omega = 0.0
        self.received.clear()
        if path_waypoints is not None and len(path_waypoints) > 0:
            self._x = float(path_waypoints[0][0])
            self._y = float(path_waypoints[0][1])
            if initial_heading is not None:
                self._theta = float(initial_heading)
            elif len(path_waypoints) >= 2:
                dx = float(path_waypoints[1][0]) - self._x
                dy = float(path_waypoints[1][1]) - self._y
                self._theta = float(np.arctan2(dy, dx))

    def step(self, action: np.ndarray) -> None:
        self.received.append(np.array(action, dtype=np.float64, copy=True))
        delta = float(action[0])
        v = float(action[1])
        omega = (
            0.0 if v == 0.0 else v / self.chassis.wheelbase * np.tan(self._prev_delta)
        )
        self._prev_delta = delta
        cmd = ChassisCommand(v=v, omega=omega, vy=0.0)
        self._x, self._y, self._theta = self.chassis.integrate(
            self._x, self._y, self._theta, cmd, self.dt
        )
        self._theta = (self._theta + np.pi) % (2 * np.pi) - np.pi
        self._v = float(cmd.v)
        self._omega = float(cmd.omega)

    @property
    def pose(self) -> np.ndarray:
        """float64 位姿 [x, y, theta]（get_odometry 按 obs 契约是 float32）。"""
        return np.array([self._x, self._y, self._theta], dtype=np.float64)

    def get_odometry(self) -> np.ndarray:
        return np.array([self._x, self._y, self._theta, self._v, self._omega])

    def get_lidar_front(self) -> np.ndarray:
        return np.full(180, np.inf)

    def get_lidar_rear(self) -> np.ndarray:
        return np.full(180, np.inf)

    def get_path(self) -> np.ndarray:
        return np.zeros((0, 3))

    def get_path_global(self) -> np.ndarray:
        return np.zeros((0, 3))

    def check_collision(self) -> bool:
        return False

    def check_goal_reached(self, threshold: float = 0.3) -> bool:
        return False

    def close(self) -> None:
        return None

    @property
    def goal_heading(self) -> float:
        return 0.0


def _curve_path() -> np.ndarray:
    """一条固定的弧线路径（世界系，含朝向列）。"""
    t = np.linspace(0.0, 1.0, 12)
    return np.column_stack(
        [
            1.0 + 3.0 * t,
            1.0 + 2.0 * np.sin(np.pi * t),
            np.zeros_like(t),
        ]
    )


def _random_actions(n: int = N_STEPS, seed: int = SEED, dim: int = 2) -> np.ndarray:
    """归一化动作序列（盒 [-1, 1] 内均匀，长度按车型的 action_dim）。"""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, dim))


def _run_equivalence() -> tuple[float, float]:
    """跑一遍 200 步对照，返回 (位姿 max|dxy|, max|dtheta|)。

    SimpleBridge 与适配器共享**同一个** AckermannChassis 实例（env 注入桥
    时就是这么要求的），内层替身自己有滞后状态。
    """
    chassis = AckermannChassis()
    path = _curve_path()
    heading = 0.25

    reference = SimpleBridge(chassis)
    reference.reset(path_waypoints=path, initial_heading=heading)

    inner = _PhysicalAckermannBridge(chassis, dt=reference.dt)
    bridge = PhysicalCommandBridge(inner=inner, chassis=chassis)
    bridge.reset(path_waypoints=path, initial_heading=heading)

    max_dxy = 0.0
    max_dtheta = 0.0
    for action in _random_actions():
        reference.step(action)
        bridge.step(action)
        # 用 float64 内部状态对拍：get_odometry() 按 obs 契约返回 float32，
        # 精度只到 ~1e-7，看不见 1e-12 量级的差异。
        p_ref = np.array([reference._x, reference._y, reference._theta])
        p_new = inner.pose
        max_dxy = max(max_dxy, abs(p_ref[0] - p_new[0]), abs(p_ref[1] - p_new[1]))
        dtheta = (p_ref[2] - p_new[2] + np.pi) % (2 * np.pi) - np.pi
        max_dtheta = max(max_dtheta, abs(dtheta))
    return float(max_dxy), float(max_dtheta)


# ── 1. 映射逐位：to_physical == chassis.physical_action（三车型 200 例）─────
@pytest.mark.parametrize("name", CHASSIS_NAMES)
def test_to_physical_matches_chassis_contract_bitwise(name: str) -> None:
    """to_physical 就是 chassis.physical_action 的 float64 视图，逐位相等。"""
    chassis = make_chassis(name)
    bridge = PhysicalCommandBridge(inner=_RecordingBridge(), chassis=chassis)
    actions = _random_actions(dim=chassis.action_dim)
    assert actions.shape == (N_STEPS, chassis.action_dim)

    worst = 0.0
    for action in actions:
        expected = np.asarray(chassis.physical_action(action), dtype=np.float64)
        got = bridge.to_physical(action)
        assert got.dtype == np.float64
        assert got.shape == expected.shape == (chassis.action_dim,)
        assert np.array_equal(got, expected)
        worst = max(worst, float(np.max(np.abs(got - expected))))
    assert worst == 0.0, f"{name}: max|d| = {worst:.3e}"

    if name != "ackermann":
        return
    # 阿克曼：delta 直接取 _map_action 的第一个返回值，0 ulp（W23 反解曾差 64/200 例 1 ulp）
    ulps = [abs(bridge.to_physical(a)[0] - chassis._map_action(a)[0]) for a in actions]
    assert max(ulps) == 0.0, f"delta max|d| = {max(ulps):.3e}"
    # 向量顺序 [delta, v]：v 在第二位
    delta, v = chassis._map_action(actions[0])
    assert np.array_equal(
        bridge.to_physical(actions[0]), np.array([delta, v], dtype=np.float64)
    )


# ── 2. 纯函数：与底盘执行器状态（跨步 _prev_delta）无关 ─────────────────────
@pytest.mark.parametrize("name", CHASSIS_NAMES)
def test_to_physical_is_pure_function(name: str) -> None:
    """同一 action 连续两次逐位相同；穿插 chassis.step() 之后仍然相同。"""
    chassis = make_chassis(name)
    bridge = PhysicalCommandBridge(inner=_RecordingBridge(), chassis=chassis)
    actions = _random_actions(n=32, seed=SEED + 1, dim=chassis.action_dim)

    first = [bridge.to_physical(a).copy() for a in actions]
    second = [bridge.to_physical(a).copy() for a in actions]
    for a, b in zip(first, second):
        assert np.array_equal(a, b)

    # 推进底盘状态：阿克曼会改写私有 _prev_delta，映射必须不受影响
    for action in actions:
        chassis.step(action)
    third = [bridge.to_physical(a).copy() for a in actions]
    for a, b in zip(first, third):
        assert np.array_equal(a, b)


# ── 3. 等价性保持：与 SimpleBridge 200 步位姿一致 ───────────────────────────
def test_equivalence_with_simple_bridge() -> None:
    """200 步随机动作后位姿与 SimpleBridge 一致到 1e-12（W23 实测 1.55e-15）。"""
    max_dxy, max_dtheta = _run_equivalence()
    assert max_dxy <= TOL_POSE, f"max|dxy| = {max_dxy:.3e}"
    assert max_dtheta <= TOL_POSE, f"max|dtheta| = {max_dtheta:.3e}"


# ── 4. 差速/麦轮：内层收到的就是 [v, omega] / [v, vy, omega] ────────────────
@pytest.mark.parametrize("name", ("diff_drive", "mecanum"))
def test_parameterized_vector_order_and_command_fields(name: str) -> None:
    """差速 [v, omega]、麦轮 [v, vy, omega]，与 command() 字段逐位相等。"""
    chassis = make_chassis(name)
    inner = _RecordingBridge()
    bridge = PhysicalCommandBridge(inner=inner, chassis=chassis)
    actions = _random_actions(dim=chassis.action_dim)

    worst = [0.0, 0.0, 0.0]
    for action in actions:
        bridge.step(action)
        cmd = chassis.command(action)
        got = inner.steps[-1]
        assert got.dtype == np.float64
        if name == "diff_drive":
            assert got.shape == (2,)
            assert got[0] == cmd.v
            assert got[1] == cmd.omega
            assert np.array_equal(got, np.array([cmd.v, cmd.omega], dtype=np.float64))
            worst[0] = max(worst[0], abs(got[0] - cmd.v))
            worst[1] = max(worst[1], abs(got[1] - cmd.omega))
        else:
            assert got.shape == (3,)
            assert got[0] == cmd.v
            assert got[1] == cmd.vy
            assert got[2] == cmd.omega
            assert np.array_equal(
                got, np.array([cmd.v, cmd.vy, cmd.omega], dtype=np.float64)
            )
            worst[0] = max(worst[0], abs(got[0] - cmd.v))
            worst[1] = max(worst[1], abs(got[1] - cmd.vy))
            worst[2] = max(worst[2], abs(got[2] - cmd.omega))
    assert len(inner.steps) == N_STEPS
    assert worst == [0.0, 0.0, 0.0], f"{name}: max|d| = {worst}"


# ── 5. env 关键回归：外部桥路径 info["mapped_action"][0] 不再恒 0 ─────────────
def test_env_mapped_action_uses_chassis_contract() -> None:
    """ChassisSimEnv + PhysicalCommandBridge 30 步：转向维等于 physical_action()[0]。

    改动前 _physical_action 读 self.chassis._prev_delta，而外部桥只调 command()、
    从不推进底盘状态，于是 mapped_action[0] 恒 0（training/logger.py 的 TB
    action/* 与 TUI 一直在消费它）。
    """
    chassis = AckermannChassis()
    inner = _PhysicalAckermannBridge(chassis, dt=DT)
    bridge = PhysicalCommandBridge(inner=inner, chassis=chassis)
    env = ChassisSimEnv(chassis, bridge=bridge)
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)

    nonzero = 0
    worst = 0.0
    for _ in range(30):
        action = rng.uniform(-0.9, 0.9, size=2)
        obs, reward, terminated, truncated, info = env.step(action)
        mapped = info["mapped_action"]
        assert len(mapped) == 2
        # 内层收到的就是适配器映射出的 [delta, v]；env 的转向维必须与之一致
        assert mapped[0] == inner.received[-1][0]
        assert mapped[1] == inner.received[-1][1]
        expected = float(chassis.physical_action(action)[0])
        assert mapped[0] == expected
        worst = max(worst, abs(mapped[0] - expected))
        nonzero += int(mapped[0] != 0.0)
        assert obs is not None and reward is not None
        if terminated or truncated:
            break
    env.close()

    assert nonzero > 0, "外部桥路径的信息转向维仍恒 0（W23 缺陷回归）"
    assert worst == 0.0, f"max|d| = {worst:.3e}"


def test_env_simple_bridge_mapped_action_unchanged() -> None:
    """SimpleBridge 路径：mapped_action[0] 仍等于当步物理 delta（行为保值）。

    改动前它读 _prev_delta，而 SimpleBridge.step() 会先用同一 action 推进
    chassis.step()，把 _prev_delta 写成同一个 delta —— 两种写法逐位同值。
    """
    chassis = AckermannChassis()
    env = ChassisSimEnv(chassis)
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)

    worst = 0.0
    for _ in range(30):
        action = rng.uniform(-0.9, 0.9, size=2)
        _obs, _reward, terminated, truncated, info = env.step(action)
        mapped = info["mapped_action"]
        expected = float(chassis.physical_action(action)[0])
        assert mapped[0] == expected
        assert mapped[1] == float(chassis.command(action).v)
        worst = max(worst, abs(mapped[0] - expected))
        if terminated or truncated:
            break
    env.close()
    assert worst == 0.0, f"max|d| = {worst:.3e}"


# ── 成员契约与透传 ──────────────────────────────────────────────────────────
def test_all_members_implemented_explicitly_and_forwarded() -> None:
    """每个成员都在本类里显式实现，并真的转发到内层桥（各一次）。"""
    # 1. 没有未实现的抽象成员
    assert not inspect.isabstract(PhysicalCommandBridge)
    assert PhysicalCommandBridge.__abstractmethods__ == frozenset()
    # 2. 成员集合与基类一致，且都在本类 __dict__ 里（不是继承基类的默认实现）
    assert {n for n in dir(SensorProvider) if not n.startswith("_")} == (
        REQUIRED_MEMBERS
    )
    for name in REQUIRED_MEMBERS:
        assert name in PhysicalCommandBridge.__dict__, f"{name} 没有显式实现"

    # 3. 逐个调用一次，看记录型替身里都到了
    scan = {"front": {"ranges": np.zeros(2)}, "rear": {"ranges": np.ones(2)}}
    inner = _DynamicRecordingBridge(scan=scan)
    bridge = PhysicalCommandBridge(inner=inner, chassis=AckermannChassis())

    obstacles = [{"shape": {"name": "circle"}, "state": [1.0, 2.0, 0.0]}]
    waypoints = _curve_path()[:, :2]
    bridge.reset(obstacles=obstacles, path_waypoints=waypoints, initial_heading=0.5)
    assert inner.reset_args is not None
    assert inner.reset_args[0] is obstacles
    assert inner.reset_args[1] is waypoints
    assert inner.reset_args[2] == 0.5

    action = np.array([0.2, 0.4])
    bridge.step(action)
    # 转发的是 physical_action 的结果（换一个全新的映射器重算，证明不依赖状态）
    expected = PhysicalCommandBridge(
        inner=_RecordingBridge(), chassis=AckermannChassis()
    ).to_physical(action)
    assert np.array_equal(inner.steps[0], expected)

    bridge.get_lidar_front()
    bridge.get_lidar_rear()
    bridge.get_odometry()
    bridge.get_path()
    bridge.get_path_global()
    bridge.check_collision()
    bridge.check_goal_reached(0.7)
    bridge.close()
    assert bridge.goal_heading == 1.25
    specs = [{"shape": "circle"}]
    bridge.set_dynamic_obstacles(specs)
    assert inner.dynamic_specs is specs
    bridge.get_scan("rear")
    bridge.get_sensor_reading()

    assert set(inner.calls) == REQUIRED_MEMBERS
    assert all(count == 1 for count in inner.calls.values())

    # 4. 内层不支持动态障碍 -> NotImplementedError（点名内层类型）
    plain = PhysicalCommandBridge(inner=_RecordingBridge(), chassis=AckermannChassis())
    with pytest.raises(NotImplementedError, match="_RecordingBridge"):
        plain.set_dynamic_obstacles([])


def test_accepts_non_ackermann_chassis() -> None:
    """差速/麦轮现在有明确的物理向量定义，构造期不再报 TypeError。"""
    for name in CHASSIS_NAMES:
        chassis = make_chassis(name)
        bridge = PhysicalCommandBridge(inner=_RecordingBridge(), chassis=chassis)
        assert bridge.chassis is chassis
        assert bridge.to_physical(np.zeros(chassis.action_dim)).shape == (
            chassis.action_dim,
        )


def test_get_scan_and_sensor_reading_passthrough() -> None:
    """get_scan / get_sensor_reading 内层返回什么就透传什么（含 None）。"""
    front = {"angle_min": -np.pi, "ranges": np.array([1.0, np.inf], np.float32)}
    rear = {"angle_min": 0.0, "ranges": np.array([2.0], np.float32)}
    reading = {"imu": {"ax": 0.1, "yaw_rate": 0.0}}
    inner = _RecordingBridge(
        scan={"front": front, "rear": rear}, sensor_reading=reading
    )
    bridge = PhysicalCommandBridge(inner=inner, chassis=AckermannChassis())

    assert bridge.get_scan("front") is front
    assert bridge.get_scan() is front
    assert bridge.get_scan("rear") is rear
    assert inner.scan_sides == ["front", "front", "rear"]

    assert bridge.get_sensor_reading() is reading
    inner.sensor_reading = None
    assert bridge.get_sensor_reading() is None


# ── 6. 可选传感器：内层没有该能力时返回 None，而不是 AttributeError（0.1.4）──
class _NoSensorBridge(_RecordingBridge):
    """内层桥**没有**可选传感器能力：撤掉覆盖，只剩基类的非抽象默认实现。"""

    get_sensor_reading = SensorProvider.get_sensor_reading


class _DuckNoSensorBridge:
    """主仓 NumpySimBridge 的形态：duck-typed、不继承 SensorProvider、无该方法。"""

    _ABSENT = "get_sensor_reading"

    def __init__(self, delegate: SensorProvider) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        if name == self._ABSENT:
            raise AttributeError(name)
        return getattr(self._delegate, name)


def test_sensor_reading_none_when_inner_lacks_capability() -> None:
    """内层只有基类默认实现 / 连该属性都没有 ⇒ None（0.1.4 前是 AttributeError）。"""
    chassis = AckermannChassis()
    base_only = PhysicalCommandBridge(_NoSensorBridge(), chassis)
    assert base_only.get_sensor_reading() is None

    duck = PhysicalCommandBridge(
        _DuckNoSensorBridge(_PhysicalAckermannBridge(chassis)), chassis
    )
    assert not hasattr(duck.inner, "get_sensor_reading")
    assert duck.get_sensor_reading() is None


def test_sensor_reading_forwarded_from_capable_inner() -> None:
    """内层有该方法且返回 dict ⇒ 原样转发（同一个对象，不拷贝）。"""
    reading = {"imu": {"ax": 0.1, "yaw_rate": 0.0}, "scan": None}
    inner = _RecordingBridge(sensor_reading=reading)
    bridge = PhysicalCommandBridge(inner, AckermannChassis())
    assert bridge.get_sensor_reading() is reading
    assert inner.calls == {"get_sensor_reading": 1}


def test_env_steps_with_inner_without_sensor_capability() -> None:
    """原始症状回归：无传感器能力的内层 + 适配器 + env，10 步不得 AttributeError。"""
    chassis = AckermannChassis()
    inner = _DuckNoSensorBridge(_PhysicalAckermannBridge(chassis))
    env = ChassisSimEnv(chassis, bridge=PhysicalCommandBridge(inner, chassis))
    env.reset(seed=SEED)
    steps = 0
    try:
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(
                np.array([0.1 * (i % 3), 0.5])
            )
            steps += 1
            assert obs is not None and reward is not None
            for sensor_key in ("imu", "wheel", "scan", "scan_meta"):
                assert sensor_key not in info
            if terminated or truncated:
                break
    finally:
        env.close()
    assert steps >= 5
