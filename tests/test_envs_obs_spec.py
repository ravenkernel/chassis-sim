"""env ↔ ObsSpec 接线（S3）：默认最小集、ObsLayout.as_spec() 的 494 回归、
Custom 段端到端、以及 lidar 段引用传感器套件读数。

口径（0.2.0）：env 默认观测是**最小集** odom+path+goal+act（128/128/129 维）。
v0.1.4 的 494 维（360 雷达 + err/goal_sig/path_valid/path_progress）在 0.2.0 里
是**显式选择**：obs_spec=ObsLayout(2, 30).as_spec()。库不内置这个 RL 预设 ——
默认值要能被任何底盘直接用，而 494 固定假设 180+180 线雷达。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.envs.chassis_env import ChassisSimEnv
from chassis_sim.envs.observation import ObservationBuilder, ObsLayout
from chassis_sim.obs import Custom, LidarSegment, ObsSpec, Odom
from chassis_sim.sensors import Lidar

ENV_DIMS = {"ackermann": 128, "diff_drive": 128, "mecanum": 129}
ACTION_DIMS = {"ackermann": 2, "diff_drive": 2, "mecanum": 3}

DEFAULT_SLICES = {
    "ackermann": {
        "odom": slice(0, 4),
        "path": slice(4, 124),
        "goal": slice(124, 126),
        "act": slice(126, 128),
    },
    "diff_drive": {
        "odom": slice(0, 4),
        "path": slice(4, 124),
        "goal": slice(124, 126),
        "act": slice(126, 128),
    },
    "mecanum": {
        "odom": slice(0, 4),
        "path": slice(4, 124),
        "goal": slice(124, 126),
        "act": slice(126, 129),
    },
}

LEGACY_DIM = 494  # ObsLayout(action_dim=2, max_path_points=30).total_dim


class TestDefaultSpecIsMinimal:
    @pytest.mark.parametrize("name", sorted(ENV_DIMS))
    def test_default_obs_dims(self, name):
        env = ChassisSimEnv(name)
        try:
            assert env.observation_space.shape == (ENV_DIMS[name],)
            assert env.observation_space.dtype == np.float32
            assert env.obs_spec_describe() == DEFAULT_SLICES[name]
            obs, _ = env.reset(seed=0)
            assert obs.shape == (ENV_DIMS[name],) and obs.dtype == np.float32
            obs, _, _, _, _ = env.step(np.zeros(ACTION_DIMS[name], dtype=np.float32))
            assert obs.shape == (ENV_DIMS[name],)
        finally:
            env.close()

    def test_default_spec_has_no_lidar_and_no_extra_tail(self):
        """最小集只有 4 段：lidar/err/goal_sig/path_valid/path_progress 都不在。"""
        env = ChassisSimEnv("ackermann")
        try:
            names = list(env.obs_spec_describe())
            assert names == ["odom", "path", "goal", "act"]
            assert env.obs_spec.resolved().total_dim == 128
            assert env.observation_space.shape == (128,)
        finally:
            env.close()

    def test_obs_spec_is_readonly_view(self):
        """obs_spec_describe() 返回副本：改它不能污染 env 的切片表。"""
        env = ChassisSimEnv("ackermann")
        try:
            env.obs_spec_describe()["odom"] = slice(99, 100)
            assert env.obs_spec_describe()["odom"] == slice(0, 4)
        finally:
            env.close()


class TestLayoutAsSpecIsByteIdentical:
    @staticmethod
    def _raws(steps: int = 12):
        """同一 env 的一串 raw obs（同 seed ⇒ 与训练侧取到的输入同分布）。"""
        env = ChassisSimEnv("ackermann")
        try:
            env.reset(seed=7)
            raws = [env._get_raw_obs()]
            for i in range(steps):
                env.step(np.array([0.3 * np.sin(i), 0.5 * np.cos(0.5 * i)], np.float32))
                raws.append(env._get_raw_obs())
        finally:
            env.close()
        return raws

    def test_as_spec_matches_layout_build(self):
        """ObsLayout.build(...) 与 as_spec().resolved().build(ctx) 逐字节相同。"""
        layout = ObsLayout(action_dim=2, max_path_points=30)
        resolved = layout.as_spec().resolved()
        assert resolved.total_dim == layout.total_dim == LEGACY_DIM
        assert resolved.describe() == {
            name: slice(s, e) for name, (s, e) in layout._slices.items()
        }

        legacy = ObservationBuilder(max_path_points=30)
        spec_builder = ObservationBuilder(max_path_points=30)
        err = np.array([0.3, -0.2], dtype=np.float32)
        goal_sig = np.array([0.1, 0.99], dtype=np.float32)
        worst = 0.0
        for raw in self._raws():
            old = legacy.build(
                raw,
                prev_delta=0.1,
                prev_v=0.4,
                err_signals=err,
                goal_signals=goal_sig,
                path_progress=0.42,
            )
            ctx = spec_builder.build_context(
                raw,
                prev_delta=0.1,
                prev_v=0.4,
                err_signals=err,
                goal_signals=goal_sig,
                path_progress=0.42,
            )
            assert set(ctx.lidar) == {"legacy_front", "legacy_rear"}
            new = resolved.build(ctx)
            assert old.tobytes() == new.tobytes()
            worst = max(worst, float(np.max(np.abs(old - new))))
        assert worst == 0.0

    def test_env_accepts_legacy_layout_as_explicit_spec(self):
        """env 侧显式选择 494：维度与切片都和 ObsLayout 完全对齐。"""
        env = ChassisSimEnv("ackermann", obs_spec=ObsLayout(2, 30).as_spec())
        try:
            assert env.observation_space.shape == (LEGACY_DIM,)
            layout = ObsLayout(action_dim=2, max_path_points=30)
            assert env.obs_spec_describe() == {
                name: slice(s, e) for name, (s, e) in layout._slices.items()
            }
            obs, _ = env.reset(seed=0)
            assert obs.shape == (LEGACY_DIM,)
            # lidar 段用的是 legacy 双雷达（前 180 + 后 180），与旧 build 一致
            lidar = np.asarray(obs[:360], dtype=np.float32)
            front = np.asarray(env.bridge.get_lidar_front(), dtype=np.float32)
            rear = np.asarray(env.bridge.get_lidar_rear(), dtype=np.float32)
            assert np.array_equal(lidar, np.concatenate([front, rear]))
        finally:
            env.close()

    @pytest.mark.parametrize("name", sorted(ENV_DIMS))
    def test_legacy_spec_dims(self, name):
        """494 不是默认值，但显式要就该拿到 494（act 段宽度随底盘动作维）。"""
        env = ChassisSimEnv(name, obs_spec=ObsLayout(2, 30).as_spec())
        try:
            assert env.observation_space.shape == (LEGACY_DIM,)
        finally:
            env.close()


class TestCustomSegment:
    def test_custom_segment_end_to_end(self):
        """Custom("bias", 1, fn)：obs 多 1 维，值 == fn(同一份 ctx)。"""
        seen: dict[str, object] = {}

        def bias(ctx):
            value = float(np.sum(np.abs(np.asarray(ctx.odom_rel)))) + 0.25
            seen["ctx"] = ctx
            return np.array([value], dtype=np.float32)

        base = tuple(ObsSpec.default(action_dim=2).segments)
        spec = ObsSpec(list(base) + [Custom("bias", 1, bias)])
        env = ChassisSimEnv("ackermann", obs_spec=spec)
        try:
            assert env.observation_space.shape == (129,)
            assert env.obs_spec_describe()["bias"] == slice(128, 129)
            obs, _ = env.reset(seed=0)
            assert obs.shape == (129,)
            sl = env.obs_spec_describe()["bias"]
            assert np.array_equal(obs[sl], bias(seen["ctx"]))
            reset_value = float(obs[sl][0])

            obs, _, _, _, _ = env.step(np.array([0.4, 0.6], dtype=np.float32))
            assert np.array_equal(obs[sl], bias(seen["ctx"]))
            assert obs[sl].shape == (1,) and obs.dtype == np.float32
            # 走一步之后 odom_rel 非零 ⇒ 该维必须真的变了（不是常量占位）
            assert float(obs[sl][0]) != reset_value
            assert float(obs[sl][0]) == pytest.approx(
                float(np.sum(np.abs(seen["ctx"].odom_rel))) + 0.25
            )
        finally:
            env.close()

    def test_custom_segment_rejects_wrong_width_at_run_time(self):
        """fn 少给一维 ⇒ 抛 ValueError（不静默 pad）。"""
        env = ChassisSimEnv(
            "ackermann",
            obs_spec=ObsSpec(
                list(ObsSpec.default(action_dim=2).segments)
                + [Custom("bad", 2, lambda ctx: np.zeros(1, dtype=np.float32))]
            ),
        )
        try:
            with pytest.raises(ValueError, match="期望宽度 2"):
                env.reset(seed=0)
        finally:
            env.close()


class TestSensorReadingsInObs:
    N_RAYS = 32

    def _obs_spec(self, ray_counts):
        return ObsSpec(
            [
                Odom(),
                LidarSegment(name="scan_seg", sensors=("scan",), ray_counts=ray_counts),
            ]
        )

    def test_lidar_segment_reads_suite_scan(self):
        """spec 的 lidar 段显式引用套件读数 ⇒ 32 个射线值逐字节进 obs。"""
        spec = self._obs_spec({"scan": self.N_RAYS})
        env = ChassisSimEnv(
            "ackermann", sensors=[Lidar("scan", n_rays=self.N_RAYS)], obs_spec=spec
        )
        try:
            assert env.observation_space.shape == (4 + self.N_RAYS,)
            env.reset(seed=0)
            obs, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
            sl = env.obs_spec_describe()["scan_seg"]
            assert sl == slice(4, 4 + self.N_RAYS)
            reading = np.asarray(
                env.bridge.get_sensor_reading()["scan"]["scan"], dtype=np.float32
            )
            assert reading.shape == (self.N_RAYS,)
            assert obs[sl].tobytes() == reading.tobytes()
            # info 里的是同一帧（延迟 0 / 丢帧 0）
            assert (
                obs[sl].tobytes()
                == np.asarray(info["scan"]["scan"], dtype=np.float32).tobytes()
            )
        finally:
            env.close()

    def test_lidar_ray_counts_filled_from_sensor_specs(self):
        """不写 ray_counts 时由 env 的 sensors= 补宽度（不必两处重复声明）。"""
        spec = self._obs_spec(None)
        resolved = spec.resolved({"scan": {"n_rays": self.N_RAYS}})
        assert resolved.total_dim == 4 + self.N_RAYS
        env = ChassisSimEnv(
            "ackermann", sensors=[Lidar("scan", n_rays=self.N_RAYS)], obs_spec=spec
        )
        try:
            assert env.observation_space.shape == (4 + self.N_RAYS,)
            obs, _ = env.reset(seed=0)
            assert obs.shape == (4 + self.N_RAYS,)
            assert env.obs_spec_describe()["scan_seg"] == slice(4, 4 + self.N_RAYS)
        finally:
            env.close()

    def test_lidar_segment_without_ray_counts_and_without_sensors_raises(self):
        """既没 ray_counts、env 也没装同名传感器 ⇒ 构造期报错（不猜宽度）。"""
        with pytest.raises(ValueError, match="scan"):
            ChassisSimEnv("ackermann", obs_spec=self._obs_spec(None))
