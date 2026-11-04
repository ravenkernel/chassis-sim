"""env 级动态障碍端到端：reset options 第四键 → 桥 → 碰撞/LiDAR。

覆盖三件事（父代理追加的验收 h）：
(i) 同一 seed 同一动作序列，含动态障碍时 LiDAR 出现明显更近的回波；
(ii) 含动态障碍时某个 tick 真的碰撞，而静态基线在同一时间轴上不碰撞；
(iii) 不支持动态障碍的桥在收到该键时必须**响亮报错**（NotImplementedError），不得静默跳过。
"""

from __future__ import annotations

import numpy as np
import pytest

from chassis_sim.bridges import SensorProvider
from chassis_sim.envs import ChassisSimEnv, ObsLayout
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

#: 前 360 维是 LiDAR（ObsLayout 的第一段；默认观测已不含 LiDAR，见 _run）
N_LIDAR = 360
STEPS = 14


def _options(with_dynamic: bool) -> dict:
    """dynamic_crossing 的 reset options；with_dynamic=False 时剔除第四键（回到静态语义）。"""
    scenario = load_scenario("dynamic_crossing")
    options = scenario_to_reset_options(scenario, seed=scenario.seed)
    if not with_dynamic:
        options = {k: v for k, v in options.items() if k != "dynamic_obstacles"}
    return options


def _run(with_dynamic: bool, steps: int = STEPS) -> tuple[np.ndarray, list[dict]]:
    """跑同一个开环动作（a0=0 ⇒ 直行，a1=1 ⇒ 1 m/s），返回末步 obs 与逐步 info。"""
    env = ChassisSimEnv(
        "ackermann",
        max_steps=steps,
        # 本用例要读前 360 维 LiDAR ⇒ 显式点名 v0.1.4 的 494 布局
        # （0.2.0 的默认观测是最小集，不含 lidar 段）
        obs_spec=ObsLayout(action_dim=2, max_path_points=30).as_spec(),
    )
    try:
        obs, _ = env.reset(seed=0, options=_options(with_dynamic))
        infos: list[dict] = []
        for _ in range(steps):
            obs, _reward, terminated, truncated, info = env.step(
                np.array([0.0, 1.0], dtype=np.float32)
            )
            infos.append(
                {
                    "collision": bool(info["collision"]),
                    "reason": info["termination_reason"],
                    "d_min": float(info["d_min"]),
                }
            )
            if terminated or truncated:
                break
        return np.array(obs[:N_LIDAR], dtype=np.float64), infos
    finally:
        env.close()


def test_env_lidar_sees_dynamic_obstacle() -> None:
    """(i) 第 12 步：有动态障碍时某条射线 0.23 m，静态基线同一射线 8.91 m。"""
    base, _ = _run(with_dynamic=False, steps=12)
    dyn, _ = _run(with_dynamic=True, steps=12)
    idx = int(np.argmin(dyn))
    assert dyn[idx] < 0.5, (idx, float(dyn[idx]))
    assert base[idx] > 8.0, (idx, float(base[idx]))
    # 静态基线的最小回波来自墙面（约 2.29 m），动态障碍把它压到 0.5 m 以内
    assert float(base.min()) > 2.0
    assert float(dyn.min()) < 0.5


def test_env_collides_with_dynamic_obstacle_only() -> None:
    """(ii) 静态基线 30 步 timeout 无碰撞；含动态障碍时第 13 步 collision。"""
    _base_obs, base_infos = _run(with_dynamic=False, steps=30)
    assert not any(info["collision"] for info in base_infos)
    assert base_infos[-1]["reason"] == "timeout"

    _dyn_obs, dyn_infos = _run(with_dynamic=True, steps=30)
    hits = [k + 1 for k, info in enumerate(dyn_infos) if info["collision"]]
    assert hits and hits[0] == 13, hits
    assert dyn_infos[hits[0] - 1]["reason"] == "collision"


def test_env_reset_switches_dynamic_obstacles_and_clears_them() -> None:
    """运行中换一组：静态 → 动态生效且相位归零；动态 → 静态必须显式清空。"""
    env = ChassisSimEnv("ackermann", max_steps=50)
    try:
        env.reset(seed=0, options=_options(with_dynamic=False))
        assert env.bridge.dynamic_positions().shape == (0, 3)

        env.reset(seed=0, options=_options(with_dynamic=True))
        positions = env.bridge.dynamic_positions()
        assert positions.shape == (1, 3)
        assert np.array_equal(positions[0, :2], np.array([3.0, 4.4]))  # t = 0 的初态
        assert float(env.bridge.sim_time) == 0.0

        for _ in range(5):
            env.step(np.array([0.0, 1.0], dtype=np.float32))
        assert float(env.bridge.sim_time) == pytest.approx(0.5)

        env.reset(seed=0, options=_options(with_dynamic=False))
        assert env.bridge.dynamic_positions().shape == (0, 3)
    finally:
        env.close()


class _StaticOnlyBridge(SensorProvider):
    """只实现抽象方法的桩：不覆写 set_dynamic_obstacles ⇒ 必须落回基类的响亮报错。"""

    def __init__(self) -> None:
        # 注入桥与 env 自建底盘不是同一实例 ⇒ 这里不放 chassis，让 env 走自己的那台
        self.chassis = None
        self._world_xmin, self._world_xmax = 0.0, 10.0
        self._world_ymin, self._world_ymax = 0.0, 10.0

    def reset(self, obstacles=None, path_waypoints=None, initial_heading=None) -> None:
        raise AssertionError("不该走到 bridge.reset：动态障碍的转发必须先失败")

    def step(self, action):  # pragma: no cover - 到不了
        raise AssertionError

    def get_lidar_front(self):  # pragma: no cover - 到不了
        raise AssertionError

    def get_lidar_rear(self):  # pragma: no cover - 到不了
        raise AssertionError

    def get_odometry(self):  # pragma: no cover - 到不了
        raise AssertionError

    def get_path(self):  # pragma: no cover - 到不了
        raise AssertionError

    def get_path_global(self):  # pragma: no cover - 到不了
        raise AssertionError

    def check_collision(self) -> bool:  # pragma: no cover - 到不了
        raise AssertionError

    def check_goal_reached(self) -> bool:  # pragma: no cover - 到不了
        raise AssertionError

    def close(self) -> None:
        return None

    @property
    def goal_heading(self) -> float:
        return 0.0


def test_bridge_without_dynamic_support_raises() -> None:
    """(iii) 不支持动态障碍的桥：env 转发该键时必须抛 NotImplementedError。"""
    bridge = _StaticOnlyBridge()
    with pytest.raises(NotImplementedError, match="不支持动态障碍"):
        bridge.set_dynamic_obstacles([{"type": "linear"}])

    env = ChassisSimEnv("ackermann", bridge=bridge, max_steps=10)
    try:
        with pytest.raises(NotImplementedError, match="不支持动态障碍"):
            env.reset(seed=0, options=_options(with_dynamic=True))
    finally:
        env.close()
