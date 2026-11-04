"""reset() 的 options 优先级（1.1.0 E）。

一条规则：options 显式给出（含空 dict `{}`）一律以调用方为准；options=None 且
`env.default_reset_options` 非空时才回落到它。于是 `cs.make(scenario=...)` 造出来
的 env 直接 `reset()` 场景就生效，而显式 options 仍然覆盖、绝不掺入默认值。
"""

from __future__ import annotations

import numpy as np

import chassis_sim as cs
from chassis_sim.envs import ChassisSimEnv

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

    def test_env_without_scenario_is_unchanged(self):
        env = cs.make("ackermann", max_steps=50)
        assert env.default_reset_options == {}
        auto, _ = env.reset(seed=0)
        empty, _ = env.reset(seed=0, options={})
        assert np.array_equal(auto, empty)

    def test_env_without_the_attribute_still_resets(self):
        # 直接构造的 env 没有 make() 挂上的 default_reset_options
        env = ChassisSimEnv()
        assert not hasattr(env, "default_reset_options")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (128,)

    def test_scenario_drives_the_episode_after_reset(self):
        env = _scene_env()
        env.reset(seed=0)
        obs, *_ = env.step(np.array([0.2, 0.8], dtype=np.float32))
        assert obs.shape == (128,)
        assert len(env._obstacles) == 2
