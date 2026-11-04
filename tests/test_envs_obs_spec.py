"""env ↔ ObsSpec 接线（S3）：默认最小集、ObsLayout.as_spec() 的 494 回归、
Custom 段端到端、以及 lidar 段引用传感器套件读数。

口径（1.0.0）：env 默认观测是**最小集** odom+path+goal+act（128/128/129 维）。
v0.1.4 的 494 维（360 雷达 + err/goal_sig/path_valid/path_progress）只能由**显式
9 段 ObsSpec** 表达（ObsLayout 已删）：库里没有 494 预设 —— 494 固定假设
180+180 线雷达，而这两支雷达现在是真实装配的传感器（Lidar("front") / Lidar("rear")）。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.chassis import AckermannChassis
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.obs import (
    Act,
    Custom,
    Err,
    Goal,
    GoalSig,
    LidarSegment,
    ObsSpec,
    Odom,
    PathProgress,
    PathSegment,
    PathValid,
)
from chassis_sim.obs.context import ContextBuilder
from chassis_sim.sensors import Lidar

ENV_DIMS = {"ackermann": 128, "diff_drive": 128, "mecanum": 129}
ACTION_DIMS = {"ackermann": 2, "diff_drive": 2, "mecanum": 3}


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
    """前 180 + 后 180 两支 Lidar（mount 取底盘的真实安装位）。"""
    front, rear = AckermannChassis().lidar_mounts()
    return [
        Lidar("front", mount=front, n_rays=180),
        Lidar("rear", mount=rear, n_rays=180),
    ]


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

LEGACY_DIM = 494  # 9 段显式 spec（lidar 360 + odom 4 + path 120 + goal 2 + act 2 + err 2 + goal_sig 2 + 1 + 1）


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


class TestExplicit494Spec:
    """494 = 显式 9 段 spec（库不再内置）：段序/切片/逐位都对齐主仓实测金标准。"""

    GOLDEN_SLICES = {
        "lidar": slice(0, 360),
        "odom": slice(360, 364),
        "path": slice(364, 484),
        "goal": slice(484, 486),
        "act": slice(486, 488),
        "err": slice(488, 490),
        "goal_sig": slice(490, 492),
        "path_valid": slice(492, 493),
        "path_progress": slice(493, 494),
    }

    @staticmethod
    def _frames() -> dict:
        """写死的 360 维雷达帧（前 180 + 后 180），不用 RNG。"""
        return {
            "front": np.linspace(1.0, 2.0, 180, dtype=np.float32),
            "rear": np.linspace(2.0, 1.0, 180, dtype=np.float32),
        }

    @staticmethod
    def _raws(steps: int = 12):
        """同一 env 的一串 raw 读数（同 seed ⇒ 与训练侧取到的输入同分布）。"""
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

    def test_spec_is_494_and_matches_golden_slices(self):
        resolved = _spec_494().resolved()
        assert resolved.total_dim == LEGACY_DIM
        assert resolved.describe() == self.GOLDEN_SLICES

    def test_context_builder_is_deterministic_across_instances(self):
        """同一串 raw 喂两个独立 ContextBuilder ⇒ 拼出的观测逐字节相同。"""
        err = np.array([0.3, -0.2], dtype=np.float32)
        goal_sig = np.array([0.1, 0.99], dtype=np.float32)
        a, b = ContextBuilder(max_path_points=30), ContextBuilder(max_path_points=30)
        resolved = _spec_494().resolved()
        worst = 0.0
        for raw in self._raws():
            out = []
            for builder in (a, b):
                ctx = builder.build_context(
                    raw,
                    prev_delta=0.1,
                    prev_v=0.4,
                    err_signals=err,
                    goal_signals=goal_sig,
                    path_progress=0.42,
                    lidar=self._frames(),
                )
                out.append(resolved.build(ctx))
            assert out[0].tobytes() == out[1].tobytes()
            worst = max(worst, float(np.max(np.abs(out[0] - out[1]))))
        assert worst == 0.0

    def test_env_accepts_explicit_494_spec(self):
        """env 侧显式选择 494：维度/切片都对金标准，lidar 段取自两支 Lidar。"""
        env = ChassisSimEnv("ackermann", obs_spec=_spec_494(), sensors=_lidars())
        try:
            assert env.observation_space.shape == (LEGACY_DIM,)
            assert env.obs_spec_describe() == self.GOLDEN_SLICES
            obs, _ = env.reset(seed=0)
            assert obs.shape == (LEGACY_DIM,)
            reading = env.bridge.get_sensor_reading()
            front = np.asarray(reading["front"]["scan"], dtype=np.float32)
            rear = np.asarray(reading["rear"]["scan"], dtype=np.float32)
            assert np.array_equal(obs[:360], np.concatenate([front, rear]))
        finally:
            env.close()

    @pytest.mark.parametrize("name", sorted(ENV_DIMS))
    def test_explicit_spec_dims(self, name):
        """494 不是默认值，但显式要就该拿到 494（act 段宽度不超过 action_dim）。"""
        env = ChassisSimEnv(name, obs_spec=_spec_494(), sensors=_lidars())
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
