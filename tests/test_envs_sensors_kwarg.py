"""ChassisSimEnv(..., sensors=[...]) 构造器透传（0.2.0 的列表式传感器 spec）。

三种情形的契约（见 chassis_env.py __init__ 的 sensors 分支）：

  1. 只给 sensors=、不注入 bridge ⇒ env 用**自己那台 chassis** 造 SimpleBridge
     并挂上这些 spec（用户不必再手写 chassis/bridge/env 三行）。
  2. 只给 bridge=               ⇒ 与加入 sensors 形参前逐位相同（桥自带的套件照旧走）。
  3. 两者都给                   ⇒ ValueError。选抛错而不是"事后转交"的理由：
     SimpleBridge 只在 __init__ 里装配 SensorSuite、也没有 set_sensors，
     "悄悄忽略"会让用户以为传感器开着。错误文案给出两条修法。

不变量（0.2.0 起）：观测维度由 obs_spec 决定，与开不开传感器**无关**（默认
128 / 128 / 129）；传感器读数默认只进 info（键 = 传感器名，值是该传感器的读数
字典），除非 obs_spec 的 lidar 段显式引用它 —— 后者见 tests/test_envs_obs_spec.py。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import make_chassis
from chassis_sim.envs.chassis_env import ChassisSimEnv
from chassis_sim.sensors import Imu, Lidar, WheelOdom

SENSOR_NAMES = {"imu", "wheel", "scan"}
OBS_DIM = {"ackermann": 128, "diff_drive": 128, "mecanum": 129}
ACTION_DIM = {"ackermann": 2, "diff_drive": 2, "mecanum": 3}


def _specs(noise_std: float = 0.0) -> list:
    """全开的一套传感器：IMU + 轮速计 + 前向 180 线雷达。"""
    return [
        Imu("imu"),
        WheelOdom("wheel"),
        Lidar("scan", n_rays=180, noise_std=noise_std),
    ]


def _actions(n: int, dim: int = 2) -> np.ndarray:
    return np.linspace(-0.5, 0.5, n * dim).reshape(n, dim).astype(np.float32)


class TestSensorsOnly:
    def test_env_builds_its_own_bridge_with_sensors(self):
        env = ChassisSimEnv(sensors=_specs())
        try:
            assert isinstance(env.bridge, SimpleBridge)
            # 桥真的挂了套件，并且读数直接取得到（不是只存了个列表）
            assert env.bridge._sensors is not None
            env.reset(seed=0)
            obs, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
            reading = env.bridge.get_sensor_reading()
            assert reading is not None and set(reading) == SENSOR_NAMES
            assert obs.shape == (OBS_DIM["ackermann"],)
            assert SENSOR_NAMES <= set(info)
            scan = np.asarray(info["scan"]["scan"])
            assert scan.shape == (180,) and scan.dtype == np.float32
            assert set(info["imu"]) == {"ax", "ay", "yaw_rate", "yaw", "t"}
            assert set(info["wheel"]) == {"v_left", "v_right", "v", "t"}
            assert info["scan"]["scan_meta"]["n_rays"] == 180
            assert info["scan"]["scan_meta"]["name"] == "scan"
        finally:
            env.close()

    def test_sensors_add_keys_and_change_nothing_else(self):
        """同 seed 同动作 30 步：开传感器后 obs 逐位相同，info 只多 3 个键。"""
        acts = _actions(30)
        on = ChassisSimEnv(sensors=_specs())
        off = ChassisSimEnv()
        try:
            obs_on, _ = on.reset(seed=3)
            obs_off, _ = off.reset(seed=3)
            seq_on, seq_off = [obs_on.copy()], [obs_off.copy()]
            for a in acts:
                obs_on, _, _, _, info_on = on.step(a)
                obs_off, _, _, _, info_off = off.step(a)
                seq_on.append(obs_on.copy())
                seq_off.append(obs_off.copy())
        finally:
            on.close()
            off.close()
        worst = max(float(np.max(np.abs(x - y))) for x, y in zip(seq_on, seq_off))
        assert worst == 0.0, worst
        assert set(info_on) - set(info_off) == SENSOR_NAMES
        assert set(info_off) & SENSOR_NAMES == set()

    def test_sensor_seed_makes_noise_reproducible(self):
        """同一个 sensor_seed ⇒ 两次构造的 scan 逐位相同；换个 seed ⇒ 不同。"""
        scans = []
        for _ in range(2):
            env = ChassisSimEnv(sensors=_specs(noise_std=0.05), sensor_seed=13)
            try:
                env.reset(seed=0)
                seq = []
                for a in _actions(6):
                    _, _, _, _, info = env.step(a)
                    seq.append(np.asarray(info["scan"]["scan"]).copy())
                scans.append(np.array(seq))
            finally:
                env.close()
        np.testing.assert_array_equal(scans[0], scans[1])

        other = ChassisSimEnv(sensors=_specs(noise_std=0.05), sensor_seed=99)
        try:
            other.reset(seed=0)
            other.step(np.zeros(2, dtype=np.float32))
            other_scan = np.asarray(other.bridge.get_sensor_reading()["scan"]["scan"])
        finally:
            other.close()
        # 反向对照：若噪声没接线（noise_std 被忽略），两组 scan 会完全一样
        assert not np.array_equal(scans[0][0], other_scan)


class TestBridgeOnly:
    def test_injected_bridge_is_used_verbatim(self):
        """只给 bridge=（老写法）：桥实例原样使用，行为与加入 sensors 形参前一致。"""
        chassis = make_chassis("ackermann")
        bridge = SimpleBridge(chassis)
        env = ChassisSimEnv(chassis, bridge=bridge)
        try:
            assert env.bridge is bridge
            assert env.bridge.get_sensor_reading() is None
            obs, _ = env.reset(seed=0)
            assert obs.shape == (OBS_DIM["ackermann"],)
            _, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
            assert set(info) & SENSOR_NAMES == set()
        finally:
            env.close()

    def test_injected_bridge_with_its_own_sensors_still_works(self):
        """自己构造桥时传 sensors（等价于旧的三行写法）——本改动不能破坏它。"""
        chassis = make_chassis("ackermann")
        env = ChassisSimEnv(chassis, bridge=SimpleBridge(chassis, sensors=_specs()))
        try:
            env.reset(seed=0)
            obs, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
            assert obs.shape == (OBS_DIM["ackermann"],)
            assert SENSOR_NAMES <= set(info)
        finally:
            env.close()


class TestBothGiven:
    def test_sensors_plus_bridge_raises_value_error(self):
        chassis = make_chassis("ackermann")
        with pytest.raises(ValueError, match="sensors= 只在 env 自己造桥") as exc:
            ChassisSimEnv(chassis, bridge=SimpleBridge(chassis), sensors=_specs())
        message = str(exc.value)
        # 文案必须给出两条出路，否则用户只知道"不行"
        assert "去掉 bridge=" in message and "去掉 sensors=" in message

    def test_raises_before_touching_the_bridge(self):
        """构造即失败：注入的桥没有被 reset/step 过，env 也没留下半成品。"""
        chassis = make_chassis("ackermann")
        bridge = SimpleBridge(chassis)
        with pytest.raises(ValueError):
            ChassisSimEnv(chassis, bridge=bridge, sensors=_specs())
        assert bridge.get_sensor_reading() is None


class TestObsDimIndependentOfSensors:
    @pytest.mark.parametrize("name", sorted(OBS_DIM))
    def test_obs_dim_independent_of_sensors(self, name):
        """三车型：开/关传感器 obs 维度相同，且等于 128/128/129。"""
        dim = ACTION_DIM[name]
        envs = [ChassisSimEnv(name, sensors=_specs()), ChassisSimEnv(name)]
        try:
            for env in envs:
                obs, _ = env.reset(seed=1)
                assert obs.shape == (OBS_DIM[name],), (name, obs.shape)
                obs, _, _, _, info = env.step(np.zeros(dim, dtype=np.float32))
                assert obs.shape == (OBS_DIM[name],)
                if env.bridge.get_sensor_reading() is None:
                    assert set(info) & SENSOR_NAMES == set()
                else:
                    assert SENSOR_NAMES <= set(info)
        finally:
            for env in envs:
                env.close()
