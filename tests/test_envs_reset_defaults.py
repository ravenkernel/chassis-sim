"""reset() 的 options 优先级（1.1.0 E）+ 内建默认场景（3.0.0 W1）。

一条规则：options 显式给出（含空 dict `{}`）一律以调用方为准；options=None 且
`env.default_reset_options` 非空时才回落到它。于是 `cs.make(scenario=...)` 造出来
的 env 直接 `reset()` 场景就生效，而显式 options 仍然覆盖、绝不掺入默认值。

3.0.0 起 options=None 且没有默认 options 时不再退化成固定直线：env 接
EnvConfig.default_scenario（默认 dense_static，纯 core、不需要 numba），并且
**每次 reset 重掷一张地图**（显式 seed 可复现）。

3.0.x 起再加一条前提：**桥必须是 env 自己造的**。注入的桥（主仓
PhysicalCommandBridge(NumpySimBridge) 形态）自带世界，默认场景不得把
dense_static 的 obstacles/path 塞给它——那是静默偏航，不是响亮失败。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import chassis_sim as cs
from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

#: 默认场景保底障碍数（与 env._OBSTACLE_MIN_KEEP 同值：丢障碍不会把场景掏空）
MIN_DEFAULT_OBSTACLES = 6

_PATH = np.column_stack([np.linspace(1.0, 6.0, 20), np.full(20, 5.0), np.zeros(20)])
_ONE_CIRCLE = [
    {"shape": {"name": "circle", "radius": 0.30}, "state": [3.0, 5.5, 0.0]},
]


def _scene_env() -> ChassisSimEnv:
    """corridor 是 core 场景（不需要 planning 扩展）。"""
    return cs.make("ackermann", scenario="corridor", max_steps=50)


class TestDefaultResetOptions:
    def test_reset_without_options_consumes_the_defaults(self):
        env = _scene_env()
        defaults = dict(env.default_reset_options)
        assert defaults  # 场景真的带出了 reset 键

        auto, _ = env.reset(seed=0)
        obstacles_auto = len(env._obstacles)
        explicit, _ = env.reset(seed=0, options=defaults)
        assert np.array_equal(auto, explicit)
        assert obstacles_auto == len(env._obstacles) == 2  # corridor 的两堵墙
        # reset 不就地改那份默认值
        assert env.default_reset_options == defaults

    def test_explicit_empty_dict_does_not_fall_back(self):
        env = _scene_env()
        auto, _ = env.reset(seed=0)
        empty, _ = env.reset(seed=0, options={})
        assert not np.array_equal(auto, empty)
        # 空 options ⇒ 内建直线兜底场景（无静态障碍），与 corridor 的两堵墙不同
        assert env._obstacles == []

    def test_explicit_options_override_the_defaults(self):
        env = _scene_env()
        options = {"obstacles": [dict(o) for o in _ONE_CIRCLE], "path_waypoints": _PATH}
        obs, _ = env.reset(seed=0, options=options)
        assert obs.shape == (128,)
        assert len(env._obstacles) == 1  # 只用了调用方那份，没有混进 corridor
        assert env.bridge.get_path_global().shape[0] == len(_PATH)

    def test_options_none_is_the_same_as_omitted(self):
        env = _scene_env()
        omitted, _ = env.reset(seed=0)
        passed_none, _ = env.reset(seed=0, options=None)
        assert np.array_equal(omitted, passed_none)

    def test_env_without_scenario_uses_the_builtin_scene(self):
        env = cs.make("ackermann", max_steps=50)
        assert env.default_reset_options == {}  # make() 没挂场景 ⇒ 默认值仍是空
        auto, _ = env.reset(seed=0)
        assert len(env._obstacles) >= MIN_DEFAULT_OBSTACLES
        empty, _ = env.reset(seed=0, options={})
        assert not np.array_equal(auto, empty)  # 空 options 仍是 2.x 的直线
        assert env._obstacles == []

    def test_env_without_the_attribute_still_resets(self):
        # 直接构造的 env 没有 make() 挂上的 default_reset_options
        env = ChassisSimEnv()
        assert not hasattr(env, "default_reset_options")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (128,)
        assert len(env._obstacles) >= MIN_DEFAULT_OBSTACLES

    def test_scenario_drives_the_episode_after_reset(self):
        env = _scene_env()
        env.reset(seed=0)
        obs, *_ = env.step(np.array([0.2, 0.8], dtype=np.float32))
        assert obs.shape == (128,)
        assert len(env._obstacles) == 2


def _fingerprint(obstacles) -> tuple:
    """地图指纹：[x, y] 取到 9 位小数 + 障碍数（丢障碍也是换图）。"""
    return (
        len(obstacles),
        tuple(tuple(round(float(v), 9) for v in o["state"][:2]) for o in obstacles),
    )


class TestBuiltinDefaultScene:
    """3.0.0 W1：不给 options 时的默认场景 = 自带障碍 + 可重掷 + 可复现。"""

    def test_default_scene_is_a_real_obstacle_map(self):
        env = cs.make("ackermann", max_steps=50)
        try:
            env.reset(seed=0)
            assert len(env._obstacles) >= MIN_DEFAULT_OBSTACLES
            assert env.bridge.get_path_global().shape[0] > 2
            assert not env.bridge.check_collision()  # 起点不能落在障碍里
        finally:
            env.close()

    def test_two_consecutive_resets_reroll_the_map(self):
        env = cs.make("ackermann", max_steps=50)
        try:
            env.reset()
            first = _fingerprint(env._obstacles)
            env.reset()
            assert _fingerprint(env._obstacles) != first
        finally:
            env.close()

    def test_same_seed_reproduces_the_same_map_and_obs(self):
        env = cs.make("ackermann", max_steps=50)
        try:
            obs_a, _ = env.reset(seed=7)
            fp_a = _fingerprint(env._obstacles)
            obs_b, _ = env.reset(seed=7)
            assert _fingerprint(env._obstacles) == fp_a
            assert np.array_equal(obs_a, obs_b)
        finally:
            env.close()

    def test_same_sensor_seed_rolls_the_same_sequence(self):
        a = ChassisSimEnv("ackermann", sensor_seed=3)
        b = ChassisSimEnv("ackermann", sensor_seed=3)
        try:
            a.reset()
            b.reset()
            assert _fingerprint(a._obstacles) == _fingerprint(b._obstacles)
            a.reset()
            b.reset()
            assert _fingerprint(a._obstacles) == _fingerprint(b._obstacles)
        finally:
            a.close()
            b.close()

    def test_default_scenario_none_restores_the_straight_line(self):
        env = cs.make("ackermann", default_scenario=None, max_steps=50)
        try:
            env.reset(seed=0)
            assert env._obstacles == []
        finally:
            env.close()

    def test_explicit_scenario_still_wins_over_the_default(self):
        env = cs.make("ackermann", scenario="corridor", max_steps=50)
        try:
            env.reset(seed=0)
            assert len(env._obstacles) == 2  # corridor 的两堵墙，而不是 dense_static
        finally:
            env.close()

    def test_default_scene_matches_the_scenario_geometry(self):
        env = cs.make("ackermann", max_steps=50)
        try:
            env.reset(seed=11)
            path = np.asarray(env.bridge.get_path_global())
            ref = np.asarray(
                scenario_to_reset_options(load_scenario("dense_static"))[
                    "path_waypoints"
                ]
            )
            assert path.shape == ref.shape
            assert np.allclose(path[:, :2], ref[:, :2])  # 只掷障碍，不掷路径
        finally:
            env.close()

    def test_reset_without_numba(self):
        """默认场景必须能纯 core 跑：numba 被挡住时 reset 仍然成功。"""
        src = str(Path(cs.__file__).resolve().parent)
        script = "\n".join(
            [
                "import sys",
                "class _Block:",
                "    def find_spec(self, name, path=None, target=None):",
                "        if name == 'numba' or name.startswith('numba.'):",
                "            raise ImportError('numba blocked by test')",
                "        return None",
                "sys.meta_path.insert(0, _Block())",
                "import chassis_sim as cs",
                "env = cs.make('ackermann', max_steps=20)",
                "obs, _ = env.reset()",
                "assert obs.shape == (128,), obs.shape",
                "assert len(env._obstacles) >= 6",
                "assert 'numba' not in sys.modules",
                "print('OK')",
            ]
        )
        env_vars = {**os.environ, "PYTHONPATH": str(Path(src).parent)}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env_vars,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout


# ══════════════════════════════════════════════════════════════════════
# 3.0.x：默认场景只在“世界完全由 env 负责”时接管
# ══════════════════════════════════════════════════════════════════════
#: 内建直线兜底的点数（_straight_scene：1,5 → 9,5，0.05 m 间隔）
_STRAIGHT_N = 161

_MAPGEN_PATH = np.column_stack([np.linspace(1.0, 4.0, 7), np.full(7, 3.0), np.zeros(7)])
_MAPGEN_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.25}, "state": [2.5, 3.4, 0.0]},
]


class _RecordingDuckBridge:
    """主仓 NumpySimBridge 的形状：duck-typed、不继承 SensorProvider。

    内部仍是一台真 SimpleBridge（物理与传感器照旧），只把 reset 的实参录下来：
    “桥到底收到了什么”是这条回归唯一能证明的观测面。
    """

    def __init__(self, chassis: AckermannChassis) -> None:
        self.chassis = chassis
        self._inner = SimpleBridge(chassis, dt=0.1, domain_rand=False, seed=0)
        self.reset_calls: list[dict] = []

    def reset(self, **kwargs):
        self.reset_calls.append(dict(kwargs))
        return self._inner.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FixedScene:
    """确定性地图（不依赖 MapGenerator 的随机性）。"""

    stage_index = -1

    def __init__(self, obstacles, path) -> None:
        self.obstacles = obstacles
        self.path = path
        self.goal_heading = 0.0
        self.initial_heading = 0.0

    def generate(self, rng):
        return (self.obstacles, self.path, self.goal_heading, self.initial_heading)


def _injected_env(**kwargs) -> tuple[ChassisSimEnv, _RecordingDuckBridge]:
    """注入桥的 env；底盘必须是同一个实例（构造期一致性校验）。"""
    chassis = AckermannChassis()
    bridge = _RecordingDuckBridge(chassis)
    env = ChassisSimEnv(chassis, bridge=bridge, max_steps=50, **kwargs)
    return env, bridge


class TestDefaultSceneGating:
    """判据 = 无 options + 无 map_generator + **桥由 env 自造**，三者缺一不接管。"""

    def test_injected_bridge_never_receives_the_builtin_scene(self):
        env, bridge = _injected_env()
        try:
            assert env._bridge_owned is False
            assert env._default_scenario_active(None) is False
            obs, _ = env.reset(seed=0, options=None)
            kwargs = bridge.reset_calls[-1]
            # 桥收到的仍是 2.x 的内建直线兜底，不是 dense_static（后者 ≥6 个障碍）
            assert kwargs["obstacles"] == []
            path = np.asarray(kwargs["path_waypoints"])
            assert path.shape == (_STRAIGHT_N, 3)
            assert np.allclose(path[:, 0], np.linspace(1.0, 9.0, _STRAIGHT_N))
            assert np.allclose(path[:, 1], 5.0)
            assert env._obstacles == []
            assert obs.shape == (128,)
        finally:
            env.close()

    def test_self_made_bridge_still_rolls_the_builtin_scene(self):
        env = cs.make("ackermann", max_steps=50)
        try:
            assert env._bridge_owned is True
            assert env._default_scenario_active(None) is True
            env.reset(seed=0)
            assert len(env._obstacles) >= MIN_DEFAULT_OBSTACLES
            path = np.asarray(env.bridge.get_path_global())
            ref = np.asarray(
                scenario_to_reset_options(load_scenario("dense_static"))[
                    "path_waypoints"
                ]
            )
            assert path.shape == ref.shape
            assert np.allclose(path[:, :2], ref[:, :2])
        finally:
            env.close()

    def test_injected_bridge_with_map_generator_keeps_the_caller_world(self):
        scene = _FixedScene(_MAPGEN_OBSTACLES, _MAPGEN_PATH)
        env, bridge = _injected_env(map_generator=scene)
        try:
            assert env._default_scenario_active(None) is False
            env.reset(seed=0, options=None)
            kwargs = bridge.reset_calls[-1]
            assert len(kwargs["obstacles"]) == len(_MAPGEN_OBSTACLES)
            assert np.asarray(kwargs["path_waypoints"]).shape == _MAPGEN_PATH.shape
            assert len(env._obstacles) == len(_MAPGEN_OBSTACLES)
        finally:
            env.close()

    def test_self_made_bridge_with_explicit_empty_options_keeps_the_straight_line(self):
        env = cs.make("ackermann", max_steps=50)
        try:
            assert env._default_scenario_active({}) is False
            env.reset(seed=0, options={})
            assert env._obstacles == []
            path = np.asarray(env.bridge.get_path_global())
            assert path.shape == (_STRAIGHT_N, 3)
            assert np.allclose(path[:, 0], np.linspace(1.0, 9.0, _STRAIGHT_N))
            assert np.allclose(path[:, 1], 5.0)
            assert np.allclose(path[:, 2], 0.0)
        finally:
            env.close()
