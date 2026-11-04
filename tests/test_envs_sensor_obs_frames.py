"""env 观测里的 lidar 段：reset 首帧 + 缺帧零阶保持（S3b）。

口径（0.2.0 → S3b 修复）：

* 桥在 reset() 末尾就产出一帧真实读数（位姿/朝向/时间都已复位之后），因此
  reset 返回的观测里 lidar 段不再是一律 max_range 的假数据；
* 缺帧（丢帧 / latency 预热期）时 env 按传感器名保持"最后一次有效扫描"，
  绝不因为一帧缺失就谎报"四周空旷"；只有**从未**有过有效帧才用 max_range 兜底；
* env.reset() 清空这份缓存，跨回合不残留上一回合的扫描。
"""

from __future__ import annotations

import numpy as np

from chassis_sim.envs.chassis_env import ChassisSimEnv
from chassis_sim.obs import LidarSegment, ObsSpec, Odom
from chassis_sim.sensors import Imu, Lidar, WheelOdom

N_RAYS = 32
MAX_RANGE = 10.0
# 无碰撞直线：起点 (1, 1)、朝向 0；两个圆形障碍放在车头前方，保证扫描不全是量程外
PATH = np.column_stack([np.linspace(1.0, 8.0, 40), np.ones(40), np.zeros(40)])
OBSTACLES_A = [
    {"shape": {"name": "circle", "radius": 0.40}, "state": [3.0, 1.4, 0.0]},
    {"shape": {"name": "circle", "radius": 0.30}, "state": [2.5, 0.6, 0.0]},
]
OBSTACLES_B = [
    {"shape": {"name": "circle", "radius": 0.40}, "state": [4.0, 0.8, 0.0]},
]
ACTION = np.array([0.2, 0.4], dtype=np.float32)


def _obs_spec() -> ObsSpec:
    """最小集 + 一个引用套件读数 "scan" 的 lidar 段（宽度由 env 的 sensors= 补）。"""
    return ObsSpec(
        [Odom(), LidarSegment(name="scan_seg", sensors=("scan",), ray_counts=None)]
    )


def _scene(options_obstacles) -> dict:
    return {"obstacles": options_obstacles, "path_waypoints": PATH}


def _scan_slice(env: ChassisSimEnv) -> slice:
    return env.obs_spec_describe()["scan_seg"]


class TestResetInfo:
    def test_reset_info_has_the_same_sensor_keys_as_step(self):
        """reset() 的 info 里传感器部分与 step() 同键、同结构（S3c）。"""
        env = ChassisSimEnv(
            "ackermann",
            max_steps=5,
            sensors=[Lidar("scan", n_rays=N_RAYS), Imu("imu"), WheelOdom("wheel")],
            sensor_seed=3,
        )
        try:
            _, info = env.reset(seed=0)
            _, _, _, _, step_info = env.step(ACTION)
            assert sorted(info) == ["imu", "scan", "wheel"]
            assert set(info) <= set(step_info)
            assert sorted(info) == sorted(k for k in step_info if k in info)
            assert info["scan"]["scan"].shape == (N_RAYS,)
            assert info["scan"]["scan_meta"]["valid"] is True
            for key in ("scan", "imu", "wheel"):
                assert sorted(info[key]) == sorted(step_info[key])
        finally:
            env.close()


class TestResetFirstFrame:
    def test_reset_obs_lidar_is_real_first_frame(self):
        """reset 的观测 == 桥在 reset 末尾产出的那一帧（不再是一律 max_range）。"""
        env = ChassisSimEnv(
            "ackermann", sensors=[Lidar("scan", n_rays=N_RAYS)], obs_spec=_obs_spec()
        )
        try:
            obs, _ = env.reset(seed=0, options=_scene(OBSTACLES_A))
            sl = _scan_slice(env)
            scan_obs = obs[sl]
            reading = env.bridge.get_sensor_reading()["scan"]
            scan_bridge = np.asarray(reading["scan"], dtype=np.float32)
            assert scan_obs.shape == (N_RAYS,)
            assert scan_obs.tobytes() == scan_bridge.tobytes()
            assert scan_obs[0] != np.float32(MAX_RANGE)
            assert not np.all(scan_obs == MAX_RANGE)
            assert reading["scan_meta"]["range_max"] == MAX_RANGE
        finally:
            env.close()

    def test_bridge_reset_produces_frame_at_initial_pose(self):
        """桥单独 reset 后就有读数，且静姿帧与"位姿复位"一致（theta=初始朝向）。"""
        env = ChassisSimEnv("ackermann", sensors=[Lidar("scan", n_rays=N_RAYS)])
        try:
            bridge = env.bridge
            bridge.reset(
                obstacles=OBSTACLES_A, path_waypoints=PATH, initial_heading=0.0
            )
            reading = bridge.get_sensor_reading()
        finally:
            env.close()
        assert reading is not None and reading["scan"]["scan"] is not None
        meta = reading["scan"]["scan_meta"]
        assert meta["t"] == 0.0 and meta["frame_t"] == 0.0
        assert meta["valid"] is True


class TestMissingFrameHold:
    def test_latency_warmup_fills_max_range_then_real_frame(self):
        """延迟 2 步：reset 与第一步无历史 ⇒ 全 max_range；第二步交付 t=0 的真帧。"""
        env = ChassisSimEnv(
            "ackermann",
            sensors=[Lidar("scan", n_rays=N_RAYS, latency_steps=2)],
            obs_spec=_obs_spec(),
        )
        try:
            obs, _ = env.reset(seed=0, options=_scene(OBSTACLES_A))
            sl = _scan_slice(env)
            assert np.all(obs[sl] == MAX_RANGE)
            assert env.bridge.get_sensor_reading()["scan"]["scan"] is None

            obs, _, _, _, _ = env.step(ACTION)
            assert np.all(obs[sl] == MAX_RANGE)

            obs, _, _, _, _ = env.step(ACTION)
            meta = env.bridge.get_sensor_reading()["scan"]["scan_meta"]
            delivered = np.asarray(
                env.bridge.get_sensor_reading()["scan"]["scan"], dtype=np.float32
            )
            assert meta["frame_t"] == 0.0
            assert obs[sl].tobytes() == delivered.tobytes()
            assert not np.all(obs[sl] == MAX_RANGE)
        finally:
            env.close()

    def test_dropout_holds_last_valid_scan(self):
        """已有历史时整帧丢失 ⇒ 观测该段 == 上一帧（零阶保持），不是全 10。"""
        drop_spec = Lidar("scan", n_rays=N_RAYS)
        env = ChassisSimEnv("ackermann", sensors=[drop_spec], obs_spec=_obs_spec())
        try:
            obs, _ = env.reset(seed=0, options=_scene(OBSTACLES_A))
            sl = _scan_slice(env)
            for _ in range(2):
                obs, _, _, _, _ = env.step(ACTION)
            prev_scan = obs[sl].copy()
            # Lidar 是 frozen dataclass：测试里直接把丢帧率翻到 1.0，制造"从这一步起
            # 全部丢帧"，从而确定性地走进缺帧分支（不依赖随机种子撞运气）。
            object.__setattr__(drop_spec, "dropout_prob", 1.0)
            obs, _, _, _, _ = env.step(ACTION)
            reading = env.bridge.get_sensor_reading()["scan"]
            assert reading["scan"] is None
            assert reading["scan_meta"]["dropped"] is True
            assert obs[sl].tobytes() == prev_scan.tobytes()
            assert not np.all(obs[sl] == MAX_RANGE)
            # 丢帧不推进缓存：连续两帧丢失仍保持同一帧
            obs2, _, _, _, _ = env.step(ACTION)
            assert obs2[sl].tobytes() == prev_scan.tobytes()
        finally:
            env.close()


class TestCacheClearedOnReset:
    def test_seed_cache_does_not_leak_across_episodes(self):
        """第二回合 reset 时延迟尚未出帧 ⇒ 全 max_range（不保持上一回合末帧）。"""
        env = ChassisSimEnv(
            "ackermann",
            sensors=[Lidar("scan", n_rays=N_RAYS, latency_steps=1)],
            obs_spec=_obs_spec(),
        )
        try:
            obs, _ = env.reset(seed=0, options=_scene(OBSTACLES_A))
            sl = _scan_slice(env)
            last = obs[sl].copy()
            for _ in range(5):
                obs, _, _, _, _ = env.step(ACTION)
                last = obs[sl].copy()
            assert not np.all(last == MAX_RANGE)

            obs2, _ = env.reset(seed=1, options=_scene(OBSTACLES_B))
            assert env.bridge.get_sensor_reading()["scan"]["scan"] is None
            assert np.all(obs2[sl] == MAX_RANGE)
            assert obs2[sl].tobytes() != last.tobytes()

            # 新回合的第一步交付新回合 t=0 的真帧，不是上一回合的残留
            obs3, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
            meta = env.bridge.get_sensor_reading()["scan"]["scan_meta"]
            assert meta["frame_t"] == 0.0
            assert not np.all(obs3[sl] == MAX_RANGE)
        finally:
            env.close()
