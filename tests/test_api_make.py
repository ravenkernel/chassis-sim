"""统一入口 make() 与 emit_reward_components 双家修复的行为锁定。"""

import os
import subprocess
import sys

import numpy as np
import pytest

import chassis_sim as cs
from chassis_sim import (
    Act,
    ChassisSimEnv,
    EnvConfig,
    Err,
    Goal,
    GoalSig,
    Lidar,
    LidarSegment,
    ObsSpec,
    Odom,
    PathProgress,
    PathSegment,
    PathValid,
)
from chassis_sim.scenarios import ScenarioNotFoundError

OBS_DIMS = {"ackermann": 128, "diff_drive": 128, "mecanum": 129}


def _spec_494(act_dim: int = 2) -> ObsSpec:
    """v0.1.4 的 9 段 494 维观测：1.0.0 起只能显式写，库不再内置 494。"""
    return ObsSpec(
        (
            LidarSegment(
                name="lidar",
                sensors=("front", "rear"),
                ray_counts={"front": 180, "rear": 180},
            ),
            Odom(),
            PathSegment(points=30),
            Goal(),
            Act(action_dim=act_dim),
            Err(),
            GoalSig(),
            PathValid(),
            PathProgress(),
        )
    )


def _lidars() -> list:
    """前 180 + 后 180 两支 Lidar：spec 里引用的传感器名必须真的装配。"""
    return [Lidar("front", n_rays=180), Lidar("rear", n_rays=180)]


def _step_info_len(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    return len(info)


class TestConstructionPaths:
    def test_str_kwargs_equals_cfg(self):
        kwargs = {"v_max": 0.9, "dt": 0.05, "max_steps": 123, "path_lookahead": 7.0}
        a = cs.make("ackermann", **kwargs)
        b = cs.make(EnvConfig(**kwargs))
        assert (a.v_max, a.dt, a.max_steps) == (b.v_max, b.dt, b.max_steps)
        assert a.obs_spec.resolved().describe() == b.obs_spec.resolved().describe()

    def test_kwargs_override_cfg(self):
        cfg = EnvConfig(v_max=0.5, max_steps=50)
        env = cs.make(cfg, v_max=0.8, max_steps=200)
        assert env.v_max == 0.8
        assert env.max_steps == 200
        # 不就地改用户传进来的 cfg
        assert cfg.v_max == 0.5 and cfg.max_steps == 50

    def test_cfg_field_wins_over_library_default(self):
        env = cs.make(EnvConfig(max_path_points=12))
        assert env.obs_spec.resolved().total_dim == 4 + 48 + 2 + 2

    def test_chassis_str_is_stored_in_cfg(self):
        env = cs.make("diff_drive")
        assert env.chassis_name == "diff_drive"

    def test_bad_first_arg_raises_type_error(self):
        with pytest.raises(TypeError, match="EnvConfig"):
            cs.make(42)


class TestDefaultObservations:
    @pytest.mark.parametrize("name", sorted(OBS_DIMS))
    def test_default_dim_and_reset_shape(self, name):
        env = cs.make(name)
        assert env.obs_spec.resolved().total_dim == OBS_DIMS[name]
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_DIMS[name],)

    def test_explicit_494_still_works(self):
        env = cs.make("ackermann", obs=_spec_494(), sensors=_lidars())
        assert env.obs_spec.resolved().total_dim == 494
        obs, _ = env.reset(seed=0)
        assert obs.shape == (494,)

    def test_obs_alias_matches_obs_spec(self):
        spec = ObsSpec.default(action_dim=2, path_points=30)
        a = cs.make("ackermann", obs=spec)
        b = cs.make("ackermann", obs_spec=spec)
        assert a.obs_spec.resolved().describe() == b.obs_spec.resolved().describe()

    def test_sensors_kwarg_reaches_bridge(self):
        env = cs.make("ackermann", sensors=[cs.Lidar(name="front", n_rays=32)])
        assert [s.name for s in env._sensor_specs] == ["front"]


class TestOverrideValidation:
    def test_unknown_key_names_itself_and_counts(self):
        with pytest.raises(ValueError) as excinfo:
            cs.make("ackermann", turbo=1)
        message = str(excinfo.value)
        assert "turbo" in message
        assert "可用键共 57 个" in message

    def test_device_is_unknown_key(self):
        """2.0.0：env 的 device 形参已删，make() 不能再把它当合法键收下。"""
        with pytest.raises(ValueError) as excinfo:
            cs.make("ackermann", device="cpu")
        message = str(excinfo.value)
        assert "device" in message
        assert "可用键" in message

    def test_alias_conflict_raises(self):
        spec = ObsSpec.default()
        with pytest.raises(ValueError, match="别名"):
            cs.make("ackermann", obs=spec, obs_spec=spec)

    def test_seed_and_sensor_seed_are_independent_inputs(self):
        """1.1.0 D：seed 不再是 sensor_seed 的别名，两者各管各的、不冲突。

        seed=1 是运行种子（域随机化噪声用种），sensor_seed=2 是传感器随机流种子；
        同时给出是合法组合（旧断言要求抛别名冲突，那正是本波修掉的语义）。
        """
        env = cs.make("ackermann", seed=1, sensor_seed=2)
        assert (env.seed, env.sensor_seed) == (1, 2)

    def test_seed_feeds_the_sensor_rng_when_sensor_seed_is_absent(self):
        """只给 seed ⇒ 传感器流等价于 1.0.x 的 sensor_seed=同值（逐位相同）。"""
        spec = ObsSpec(
            (
                Odom(),
                cs.LidarSegment(
                    name="front", sensors=("front",), ray_counts={"front": 16}
                ),
            )
        )
        sensors = [cs.Lidar(name="front", n_rays=16, noise_std=0.05)]
        same = cs.make("ackermann", seed=7, sensors=sensors, obs=spec)
        explicit = cs.make("ackermann", sensor_seed=7, sensors=sensors, obs=spec)
        other = cs.make("ackermann", seed=8, sensors=sensors, obs=spec)
        obs_same = same.reset(seed=0)[0]
        obs_explicit = explicit.reset(seed=0)[0]
        obs_other = other.reset(seed=0)[0]
        assert np.array_equal(obs_same, obs_explicit)
        assert not np.array_equal(obs_same, obs_other)
        # 两者同时给出 ⇒ 传感器流由 sensor_seed 决定，seed 只管域随机化噪声
        both = cs.make("ackermann", seed=8, sensor_seed=7, sensors=sensors, obs=spec)
        assert np.array_equal(both.reset(seed=0)[0], obs_explicit)


class TestEmitRewardComponents:
    """双家缺陷：cfg 字段曾被形参默认 True 永久盖住（cfg=False 无效）。"""

    def test_default_is_eighteen_keys(self):
        assert _step_info_len(ChassisSimEnv()) == 18

    def test_cfg_false_gives_seventeen_keys(self):
        env = ChassisSimEnv(cfg=EnvConfig(emit_reward_components=False))
        assert _step_info_len(env) == 17

    def test_kwarg_false_beats_cfg_true(self):
        env = ChassisSimEnv(
            cfg=EnvConfig(emit_reward_components=True), emit_reward_components=False
        )
        assert _step_info_len(env) == 17

    def test_kwarg_true_beats_cfg_false(self):
        env = ChassisSimEnv(
            cfg=EnvConfig(emit_reward_components=False), emit_reward_components=True
        )
        assert _step_info_len(env) == 18

    def test_make_path_honours_cfg_field(self):
        assert _step_info_len(cs.make(EnvConfig(emit_reward_components=False))) == 17
        assert _step_info_len(cs.make("ackermann")) == 18


class TestTopLevelExports:
    def test_all_matches_the_export_table(self):
        # __all__ 是字面量（排序由 ruff RUF022 守着），模块表是 dict —— 两者必须一致
        assert set(cs.__all__) == set(cs._EXPORTS) | {"__version__"}

    def test_every_exported_name_resolves(self):
        missing = [name for name in cs.__all__ if not hasattr(cs, name)]
        assert missing == []
        assert set(dir(cs)) == set(cs.__all__)

    def test_import_does_not_pull_envs_or_bridges(self):
        # 导出按需加载：import chassis_sim 不能连带 import chassis_sim.envs
        # （它模块级 import bridges，会破坏 tests/test_sensors_core.py 的纯度契约）
        code = (
            "import sys, chassis_sim\n"
            "bad = [m for m in sys.modules\n"
            "       if m == 'chassis_sim.envs' or m.startswith('chassis_sim.bridges')]\n"
            "print(bad)\n"
            "assert not bad, bad\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"


class TestScenario:
    def test_scenario_attaches_reset_options(self):
        env = cs.make("ackermann", scenario="corridor")
        assert env.scenario is not None
        assert set(env.default_reset_options) >= {"obstacles", "path_waypoints"}
        env.reset(seed=0, options=env.default_reset_options)

    def test_no_scenario_means_empty_options(self):
        env = cs.make("ackermann")
        assert env.scenario is None
        assert env.default_reset_options == {}

    def test_unknown_scenario_is_loud(self):
        with pytest.raises(ScenarioNotFoundError):
            cs.make("ackermann", scenario="no-such-scenario-xyz")
