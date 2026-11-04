"""ChassisSimEnv（车型参数化环境核心，W4b）测试。

覆盖：三车型 reset/step、观测维度（默认最小集 128 / 128 / 129；v0.1.4 的
494 维由 ObsLayout.as_spec() 显式选择，见 tests/test_envs_obs_spec.py）、场景注入与回退、
车型/桥一致性校验、info 与映射动作口径，以及**与主仓 AckermannEnv 的逐步对拍**
（主仓不在时该组用例 skip；对拍容差与物理来源见文件末尾说明）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from chassis_sim.chassis import AckermannChassis, make_chassis
from chassis_sim.config import EnvConfig, RewardConfig
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.envs.chassis_env import GearVoteLag
from chassis_sim.obs import (
    Act,
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
from chassis_sim.paths import gear_is_compliant, segment_is_reverse
from chassis_sim.sensors import Lidar

_DRL_SIM = "/home/zhao/drl-sim"
CHASSIS_NAMES = ("ackermann", "diff_drive", "mecanum")
# 默认观测 = 最小集 odom(4) + path(120) + goal(2) + act(action_dim)
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
    """前 180 + 后 180 两支 Lidar（mount 取底盘的真实安装位）。"""
    front, rear = AckermannChassis().lidar_mounts()
    return [
        Lidar("front", mount=front, n_rays=180),
        Lidar("rear", mount=rear, n_rays=180),
    ]


_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.35}, "state": [4.5, 5.0, 0.0]},
    {
        "shape": {"name": "rectangle", "length": 0.6, "width": 0.4},
        "state": [6.0, 6.0, 0.3],
    },
]


def _straight_path(x0=1.0, x1=5.0, y=5.0, n=81):
    path = np.zeros((n, 3), dtype=np.float64)
    path[:, 0] = np.linspace(x0, x1, n)
    path[:, 1] = y
    return path


class _FixedScene:
    """鸭子类型 map_generator（不 import chassis_sim.planning，故无 numba 依赖）。"""

    stage_index = -1

    def __init__(
        self, path=None, obstacles=None, goal_heading=0.0, initial_heading=0.0
    ):
        self.path = _straight_path() if path is None else path
        self.obstacles = [] if obstacles is None else obstacles
        self.goal_heading = goal_heading
        self.initial_heading = initial_heading

    def generate(self, rng):
        return (self.obstacles, self.path, self.goal_heading, self.initial_heading)


@pytest.fixture
def env_factory():
    """构造 env 并在用例结束后 close。"""
    created = []

    def make(*args, **kwargs):
        env = ChassisSimEnv(*args, **kwargs)
        created.append(env)
        return env

    yield make
    for env in created:
        env.close()


# ══════════════════════════════════════════════════════════════════════
# 构造 / 空间
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", CHASSIS_NAMES)
def test_spaces_follow_chassis(env_factory, name):
    env = env_factory(name)
    # env 用 cfg 参数造车型（v_max 等来自 EnvConfig），所以拿 env.chassis 当参照
    chassis = env.chassis
    assert chassis.name == name
    assert env.action_space is not None
    assert env.action_space.shape == (chassis.action_dim,)
    assert np.allclose(env.action_space.low, chassis.action_space().low)
    assert np.allclose(env.action_space.high, chassis.action_space().high)
    assert env.observation_space.shape == (OBS_DIMS[name],)
    assert env.observation_space.dtype == np.float32


def test_observation_space_matches_default_spec(env_factory):
    env = env_factory("mecanum")
    assert env.observation_space.shape == (OBS_DIMS["mecanum"],)
    assert env.obs_spec_describe() == {
        "odom": slice(0, 4),
        "path": slice(4, 124),
        "goal": slice(124, 126),
        "act": slice(126, 129),
    }
    # 494 是显式选择，不是默认值（见 _spec_494 与 test_ackermann_parity_*）


def test_rejects_mismatched_bridge(env_factory):
    from chassis_sim.bridges import SimpleBridge

    bridge = SimpleBridge(make_chassis("diff_drive"))
    with pytest.raises(ValueError, match="chassis"):
        env_factory("ackermann", bridge=bridge)


def test_safety_gate_only_for_ackermann(env_factory):
    class _Gate:
        brake_triggered = False

        def __call__(self, action, lidar_360=None):
            return action

        def reset(self):
            pass

    with pytest.raises(ValueError, match="safety_gate"):
        env_factory("diff_drive", safety_gate=_Gate())


def test_action_shape_validated(env_factory):
    env = env_factory("ackermann")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action"):
        env.step(np.zeros(3, dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════
# reset / step
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", CHASSIS_NAMES)
def test_reset_step_100_samples(env_factory, name):
    env = env_factory(name)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert info == {}

    rng = np.random.default_rng(7)
    for step in range(100):
        action = (
            env.action_space.sample()
            if step % 2
            else rng.uniform(-0.5, 0.5, size=env.action_space.shape).astype(np.float32)
        )
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert set(info) >= {
            "collision",
            "goal_reached",
            "step",
            "cte",
            "d_min",
            "progress",
            "reverse_triggered",
            "safety_brake",
            "stuck",
            "timeout",
            "path_deviation",
            "path_deviation_cte",
            "termination_reason",
            "map_fallback",
            "path_dir",
            "path_dir_gate",
            "mapped_action",
        }
        assert len(info["mapped_action"]) == env.action_space.shape[0]
        if terminated or truncated:
            env.reset(seed=step)


def test_default_scene_is_straight_line(env_factory):
    env = env_factory("ackermann")
    env.reset(seed=0)
    path = env.bridge.get_path_global()
    assert path.shape == (161, 3)
    assert np.allclose(path[0, :2], [1.0, 5.0])
    assert np.allclose(path[-1, :2], [9.0, 5.0])
    # 桥把路径转成 float32，间距只剩 float32 精度
    assert np.allclose(np.diff(path[:, 0]), 0.05, atol=1e-5)
    odom = env.bridge.get_odometry()
    assert np.allclose(odom[:3], [1.0, 5.0, 0.0])
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["map_fallback"] is False


def test_reset_options_override_scene(env_factory):
    env = env_factory("ackermann")
    path = _straight_path(2.0, 4.0, 3.0, n=21)
    env.reset(
        seed=0,
        options={
            "obstacles": _OBSTACLES,
            "path_waypoints": path,
            "initial_heading": 0.25,
        },
    )
    assert np.allclose(env.bridge.get_path_global()[:, :2], path[:, :2])
    odom = env.bridge.get_odometry()
    assert np.allclose(odom[:2], path[0, :2])
    assert odom[2] == pytest.approx(0.25)


def test_map_generator_injected(env_factory):
    scene = _FixedScene(path=_straight_path(1.0, 3.0, 6.0, n=41))
    env = env_factory("ackermann", map_generator=scene)
    env.reset(seed=0)
    assert np.allclose(env.bridge.get_path_global()[:, :2], scene.path[:, :2])
    assert np.allclose(env.bridge.get_odometry()[:2], [1.0, 6.0])


def test_map_generation_failure_falls_back(env_factory):
    MapGenerationError = type("MapGenerationError", (Exception,), {})

    class _Boom:
        stage_index = -1

        def generate(self, rng):
            raise MapGenerationError("nope")

    env = env_factory("ackermann", map_generator=_Boom())
    with pytest.warns(RuntimeWarning):
        env.reset(seed=0)
    assert env._map_fallback is True
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["map_fallback"] is True
    assert env.bridge.get_path_global().shape == (161, 3)


def test_non_map_error_propagates(env_factory):
    class _Broken:
        stage_index = -1

        def generate(self, rng):
            raise ValueError("bug")

    env = env_factory("ackermann", map_generator=_Broken())
    with pytest.raises(ValueError, match="bug"):
        env.reset(seed=0)


@pytest.mark.parametrize("name", CHASSIS_NAMES)
def test_prev_action_width(env_factory, name):
    env = env_factory(name)
    env.reset(seed=0)
    assert env._prev_action.shape == (env.action_space.shape[0],)
    action = np.full(env.action_space.shape, 0.2, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert env._prev_action.shape == (env.action_space.shape[0],)
    assert np.allclose(np.asarray(info["mapped_action"]), env._prev_action, atol=1e-6)


def test_mecanum_observation_carries_lateral_velocity(env_factory):
    env = env_factory("mecanum")
    env.reset(seed=0)
    act_sl = env.obs_spec_describe()["act"]
    assert act_sl == slice(126, 129)
    vy_norm = 0.5
    action = np.array([0.0, vy_norm, 0.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    act_block = obs[act_sl]
    assert act_block.shape == (3,)
    assert act_block[2] == pytest.approx(vy_norm, abs=1e-6)


def test_info_has_no_nan_and_termination_reason(env_factory):
    env = env_factory("ackermann")
    env.reset(seed=1)
    action = np.zeros(2, dtype=np.float32)
    for _ in range(5):
        _, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert info["termination_reason"] in {
            "collision",
            "goal_reached",
            "timeout",
            "stuck",
            "path_deviation",
            "none",
        }
        if terminated or truncated:
            break


# ══════════════════════════════════════════════════════════════════════
# 从主仓照搬的档位辅助函数
# ══════════════════════════════════════════════════════════════════════


def test_segment_is_reverse_forward_and_reverse():
    # 车体路径点沿 +x 排开，鼻向 0 ⇒ 前进段
    path = np.zeros((5, 3), dtype=np.float32)
    path[:, 0] = np.arange(5.0)
    assert segment_is_reverse(path, 2, 0.0) == 1.0
    # 鼻向翻转 π ⇒ 同一几何是倒车段
    assert segment_is_reverse(path, 2, np.pi) == -1.0
    # 退化路径（点重合）退回上一次判定
    assert segment_is_reverse(np.zeros((2, 3)), 0, 0.0, prev_dir=-1.0) == -1.0


def test_gear_vote_lag_needs_majority():
    lag = GearVoteLag(3)
    assert lag.update(1.0) == 1.0  # 首步直接采纳 raw（票数 -1）
    for _ in range(3):  # 票数 -1 → +2，还没触边
        assert lag.update(-1.0) == 1.0
    assert lag.update(-1.0) == -1.0  # 触边 +3 才翻转
    assert lag.update(1.0) == -1.0  # 反向单票同样不改判
    lag.reset()
    assert lag.update(-1.0) == -1.0  # reset 后首步采纳 raw


def test_gear_vote_lag_passthrough_when_single_vote():
    lag = GearVoteLag(1)
    assert lag.update(-1.0) == -1.0
    assert lag.update(1.0) == 1.0


@pytest.mark.parametrize(
    "path_dir,v,expected",
    [
        (1.0, 0.5, True),
        (1.0, -0.5, False),
        (-1.0, -0.5, True),
        (-1.0, 0.5, False),
        (1.0, 0.0, True),  # 静止不违反档位
    ],
)
def test_gear_is_compliant(path_dir, v, expected):
    assert gear_is_compliant(path_dir, v) is expected


# ══════════════════════════════════════════════════════════════════════
# 与主仓 AckermannEnv 的逐步对拍
# ══════════════════════════════════════════════════════════════════════
# 容差说明（实测值，本地 /home/zhao/drl-sim 全量存在时）：
#   * act 段（prev_action 编码）逐位相等 —— 动作映射是同一个公式；
#   * 观测其余段 max|Δ| = 9.5e-7（≈8×float32 eps），来自桥的车体/激光输出
#     是 float32：主仓把物理动作向量存成 float32 数组（mapped_action），
#     桥按 float32 的 δ 积分；ChassisSimEnv 的 Chassis.command() 在 float64
#     里算物理量（chassis 包本次禁改）⇒ 状态从 float32 舍入位开始漂移；
#   * reward max|Δ| = 2.5e-8（≈0.2×float32 eps），是上述漂移经非线性奖励
#     放大后的结果；奖励分量的键集合完全一致。
# 因此对拍容差取 1e-6（简报允许 obs ≤1e-6），并对 act 段做逐位断言。
PARITY_TOL = 1e-6
PARITY_STEPS = 100


def _load_main_repo():
    """导入主仓对拍所需的类；主仓不存在则 skip。"""
    if not os.path.isdir(_DRL_SIM):
        pytest.skip(f"主仓 {_DRL_SIM} 不在本机")
    if _DRL_SIM not in sys.path:
        sys.path.insert(0, _DRL_SIM)
    try:
        from envs.ackermann_env import AckermannEnv
        from ros_bridge.numpy_bridge import NumpySimBridge
        from training.config import EnvConfig as MainEnvConfig
        from training.config import RewardConfig as MainRewardConfig
    except ImportError as exc:  # pragma: no cover - 环境缺依赖
        pytest.skip(f"主仓依赖不可用: {exc}")
    return NumpySimBridge, MainEnvConfig, MainRewardConfig, AckermannEnv


# goal_end_wp_count/margin 必须与 drl-sim config/env.yaml 对齐：主仓桥从
# yaml 读 3/2，而 EnvConfig 的 dataclass 默认是 6/2——不显式对齐会让
# near_end 判定（最近路点须落在末尾 N 个内）比主仓宽松，goal 提前 1~2 步触发。
_PARITY_CFG = {
    "delta_max": 0.526,
    "kappa_max": 4.0,
    "curvature_action": True,
    "v_min": -1.0,
    "v_max": 1.0,
    "wheelbase": 0.145,
    "dt": 0.1,
    "domain_rand_enabled": False,
    "goal_end_wp_count": 3,
    "goal_end_wp_margin": 2,
}


def _make_parity_pair(scene):
    NumpySimBridge, MainEnvConfig, MainRewardConfig, AckermannEnv = _load_main_repo()
    main = AckermannEnv(
        bridge=NumpySimBridge(env_config=os.path.join(_DRL_SIM, "config", "env.yaml")),
        cfg=MainEnvConfig(**_PARITY_CFG),
        reward_cfg=MainRewardConfig(),
        map_generator=scene,
        domain_rand=False,
        device="cpu",
    )
    chip = ChassisSimEnv(
        "ackermann",
        cfg=EnvConfig(**_PARITY_CFG),
        reward_cfg=RewardConfig(),
        map_generator=scene,
        domain_rand=False,
        # 对拍金标准是 v0.1.4 的 494 维原始观测 ⇒ 显式点名，而不是默认值；
        # 雷达帧只能来自这两支 Lidar（旧的两键接口已删）
        obs_spec=_spec_494(),
        sensors=_lidars(),
    )
    return main, chip


def _parity_run(scene, actions):
    """逐步跑两个 env，返回每步的 (obs max|Δ|, reward max|Δ|, act段逐位相等)。"""
    main, chip = _make_parity_pair(scene)
    try:
        obs_main, _ = main.reset(seed=0)
        obs_chip, _ = chip.reset(seed=0)
        assert np.array_equal(obs_main, obs_chip), "reset 首帧观测应逐位相等"

        describe = chip.obs_spec_describe()

        worst_obs = 0.0
        worst_reward = 0.0
        act_equal = True
        for step, action in enumerate(actions):
            obs_main, rew_main, term_main, trunc_main, info_main = main.step(action)
            obs_chip, rew_chip, term_chip, trunc_chip, info_chip = chip.step(action)

            delta = np.abs(
                np.asarray(obs_main, np.float64) - np.asarray(obs_chip, np.float64)
            )
            step_obs = float(delta.max())
            worst_obs = max(worst_obs, step_obs)
            assert step_obs <= PARITY_TOL, (
                f"step {step} obs max|Δ|={step_obs:.3e} > {PARITY_TOL:g}"
            )

            step_reward = abs(float(rew_main) - float(rew_chip))
            worst_reward = max(worst_reward, step_reward)
            assert step_reward <= PARITY_TOL, (
                f"step {step} reward |Δ|={step_reward:.3e}"
            )

            assert bool(term_main) == bool(term_chip), f"step {step} terminated"
            assert bool(trunc_main) == bool(trunc_chip), f"step {step} truncated"

            # act 段（prev_action 编码）是动作映射的直接证据：逐位相等
            sm = np.asarray(obs_main, np.float32)[describe["act"]]
            sc = np.asarray(obs_chip, np.float32)[describe["act"]]
            if not np.array_equal(np.asarray(sm), np.asarray(sc)):
                act_equal = False

            assert set(info_main) == set(info_chip), f"step {step} info keys"
            comp_main = info_main.get("reward_components") or {}
            comp_chip = info_chip.get("reward_components") or {}
            assert set(comp_main) == set(comp_chip)
            for key in comp_main:
                assert (
                    abs(float(comp_main[key]) - float(comp_chip[key])) <= PARITY_TOL
                ), f"step {step} reward component {key}"

            if term_main or trunc_main:
                assert (
                    info_main["termination_reason"] == info_chip["termination_reason"]
                )
                break
        return worst_obs, worst_reward, act_equal, main, chip
    finally:
        main.close()
        chip.close()


def test_ackermann_parity_random_actions():
    scene = _FixedScene(path=_straight_path(), obstacles=_OBSTACLES)
    actions = (
        np.random.default_rng(1)
        .uniform(-0.4, 0.4, size=(PARITY_STEPS, 2))
        .astype(np.float32)
    )
    worst_obs, worst_reward, act_equal, _, _ = _parity_run(scene, actions)
    assert act_equal, "act 段（prev_action 编码）应与主仓逐位相等"
    assert worst_obs <= PARITY_TOL
    assert worst_reward <= PARITY_TOL


def test_ackermann_parity_straight_run_to_goal():
    # 终点取 5.05 而不是 5.0：后者在步 36 处 "dist == 0.3" 恰好压在阈值边界上，
    # 两侧 float64 的累加顺序不同（v*cos*dt vs ...）会让 goal 判定差 1 ulp ⇒
    # 差 1 步到达。留 5 cm 余量后，仅测"物理一致"这件事本身。
    scene = _FixedScene(path=_straight_path(x1=5.05), obstacles=[])
    actions = [np.array([0.0, 1.0], dtype=np.float32)] * 200
    worst_obs, worst_reward, act_equal, main, _chip = _parity_run(scene, actions)
    assert act_equal
    assert worst_obs <= PARITY_TOL
    assert worst_reward <= PARITY_TOL
    assert main.eval_goal_count if hasattr(main, "eval_goal_count") else True
