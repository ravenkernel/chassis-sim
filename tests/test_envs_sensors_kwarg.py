"""ChassisSimEnv(..., sensors=SensorConfig(...)) 构造器透传。

三种情形的契约（见 chassis_env.py __init__ 的 sensors 分支）：

  1. 只给 sensors=、不注入 bridge  ⇒ env 用**自己那台 chassis** 造 SimpleBridge 并挂传感器
     （用户不必再手写 chassis/bridge/env 三行）。
  2. 只给 bridge=               ⇒ 与加入 sensors 形参前逐位相同（桥自带的传感器照旧走）。
  3. 两者都给                   ⇒ ValueError。选抛错而不是"事后转交"的理由：SimpleBridge
     只在 __init__ 里装配 SensorSuite、也没有 set_sensors，而 bridges/** 不在本次可改范围；
     "悄悄忽略"会让用户以为传感器开着。错误文案给出两条修法。

不变量：传感器仍然**只进 info 不进 obs**（obs 494 / 494 / 495，与开传感器前逐位相同）。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.bridges.sensors import LidarParams, SensorConfig
from chassis_sim.chassis import make_chassis
from chassis_sim.envs.chassis_env import ChassisSimEnv

SENSOR_KEYS = {"imu", "wheel", "scan", "scan_meta"}
OBS_DIM = {"ackermann": 494, "diff_drive": 494, "mecanum": 495}
ACTION_DIM = {"ackermann": 2, "diff_drive": 2, "mecanum": 3}


def _cfg(**kw) -> SensorConfig:
    """全开的传感器配置（imu + wheel + 前向 lidar）。"""
    kw.setdefault("imu", True)
    kw.setdefault("wheel", True)
    kw.setdefault("lidar", LidarParams())
    return SensorConfig(**kw)


def _actions(n: int, dim: int = 2) -> np.ndarray:
    return np.linspace(-0.5, 0.5, n * dim).reshape(n, dim).astype(np.float32)


class TestSensorsOnly:
    def test_env_builds_its_own_bridge_with_sensors(self):
        env = ChassisSimEnv(sensors=_cfg())
        try:
            assert isinstance(env.bridge, SimpleBridge)
            # 桥真的挂了传感器套件，并且传感器直接读得到（不是只存了个 cfg）
            assert env.bridge._sensors is not None
            # 传感器读数在 step 之后才生成（_sensor_reading 由 _update_sensors 填）
            env.reset(seed=0)
            obs, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
            reading = env.bridge.get_sensor_reading()
            assert reading is not None and set(reading) == SENSOR_KEYS
            assert obs.shape == (OBS_DIM["ackermann"],)
            assert SENSOR_KEYS <= set(info)
            scan = np.asarray(info["scan"])
            assert scan.shape == (180,) and scan.dtype == np.float32
            assert set(info["imu"]) == {"ax", "ay", "yaw_rate", "yaw", "t"}
            assert set(info["wheel"]) == {"v_left", "v_right", "v", "t"}
            assert info["scan_meta"]["n_rays"] == 180
            assert info["scan_meta"]["side"] == "front"
        finally:
            env.close()

    def test_sensors_add_keys_and_change_nothing_else(self):
        """同 seed 同动作 30 步：开传感器后 obs 逐位相同，info 只多 4 个键。"""
        acts = _actions(30)
        on = ChassisSimEnv(sensors=_cfg())
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
        assert set(info_on) - set(info_off) == SENSOR_KEYS
        assert set(info_off) & SENSOR_KEYS == set()

    def test_sensor_seed_makes_noise_reproducible(self):
        """sensor_seed 固定 ⇒ 两个进程内连跑的 scan 逐位相同（不碰全局 np.random）。"""
        cfg = _cfg(lidar=LidarParams(noise_std=0.05), sensor_seed=13)
        runs = []
        for _ in range(2):
            env = ChassisSimEnv(sensors=cfg)
            try:
                env.reset(seed=0)
                scans = []
                for a in _actions(6):
                    _, _, _, _, info = env.step(a)
                    scans.append(np.asarray(info["scan"]).copy())
                runs.append(np.array(scans))
            finally:
                env.close()
        np.testing.assert_array_equal(runs[0], runs[1])
        # 反向对照：若噪声没接线（noise_std 被忽略），两个 seed 的 scan 会完全一样
        other = ChassisSimEnv(
            sensors=_cfg(lidar=LidarParams(noise_std=0.05), sensor_seed=99)
        )
        try:
            other.reset(seed=0)
            other.step(np.zeros(2, dtype=np.float32))
            other_scan = np.asarray(other.bridge.get_sensor_reading()["scan"])
        finally:
            other.close()
        assert not np.array_equal(runs[0][0], other_scan)


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
            assert set(info) & SENSOR_KEYS == set()
        finally:
            env.close()

    def test_injected_bridge_with_its_own_sensors_still_works(self):
        """自己构造桥时传 sensors（等价于旧的三行写法）——本改动不能破坏它。"""
        chassis = make_chassis("ackermann")
        env = ChassisSimEnv(chassis, bridge=SimpleBridge(chassis, sensors=_cfg()))
        try:
            env.reset(seed=0)
            obs, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
            assert obs.shape == (OBS_DIM["ackermann"],)
            assert SENSOR_KEYS <= set(info)
        finally:
            env.close()


class TestBothGiven:
    def test_sensors_plus_bridge_raises_value_error(self):
        chassis = make_chassis("ackermann")
        with pytest.raises(ValueError, match="sensors= 只在 env 自己造桥") as exc:
            ChassisSimEnv(chassis, bridge=SimpleBridge(chassis), sensors=_cfg())
        message = str(exc.value)
        # 文案必须给出两条出路，否则用户只知道"不行"
        assert "去掉 bridge=" in message and "去掉 sensors=" in message

    def test_raises_before_touching_the_bridge(self):
        """构造即失败：注入的桥没有被 reset/step 过，env 也没留下半成品。"""
        chassis = make_chassis("ackermann")
        bridge = SimpleBridge(chassis)
        with pytest.raises(ValueError):
            ChassisSimEnv(chassis, bridge=bridge, sensors=_cfg())
        assert bridge.get_sensor_reading() is None


class TestObsDimsUnchanged:
    @pytest.mark.parametrize("name", sorted(OBS_DIM))
    def test_obs_dim_independent_of_sensors(self, name):
        """三车型：开/关传感器 obs 维度相同，且等于 494/494/495。"""
        dim = ACTION_DIM[name]
        envs = [ChassisSimEnv(name, sensors=_cfg()), ChassisSimEnv(name)]
        try:
            for env in envs:
                obs, _ = env.reset(seed=1)
                assert obs.shape == (OBS_DIM[name],), (name, obs.shape)
                obs, _, _, _, info = env.step(np.zeros(dim, dtype=np.float32))
                assert obs.shape == (OBS_DIM[name],)
                if env.bridge.get_sensor_reading() is None:
                    assert set(info) & SENSOR_KEYS == set()
                else:
                    assert SENSOR_KEYS <= set(info)
        finally:
            for env in envs:
                env.close()
