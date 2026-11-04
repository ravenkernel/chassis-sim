"""评测器核心：Wilson 区间、回合汇总、种子、max_steps 截断、序列化与表格。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gymnasium import spaces

from chassis_sim.eval import (
    EpisodeMetrics,
    EvalResult,
    controller_observation,
    evaluate,
    to_markdown,
    wilson_interval,
)

OBS = np.zeros(4, dtype=np.float32)


class _StubBridge:
    """只要 get_odometry / get_path_global —— 控制器与 set_path 的唯一数据源。"""

    def __init__(self, env: _StubEnv) -> None:
        self._env = env

    def get_odometry(self) -> np.ndarray:
        return np.array([0.1 * self._env.steps, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_path_global(self) -> np.ndarray:
        path = getattr(self._env, "_path", None)
        if path is not None:
            return np.asarray(path, dtype=np.float32)
        return np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)


class _StubEnv:
    """最小环境：plan[episode] = (结局, 步数)，结局取 goal/collision/timeout/truncated。"""

    def __init__(self, plan: list[tuple[str, int]]) -> None:
        self._plan = list(plan)
        self.episode = -1
        self.steps = 0
        self.seeds: list[int | None] = []
        self.closed = False
        self.unwrapped = self
        self.bridge = _StubBridge(self)
        self.dt = 0.1
        self._obstacles = [
            {"shape": {"name": "circle", "radius": 0.2}, "state": [1.0, 1.0, 0.0]}
        ]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.episode += 1
        self.seeds.append(seed)
        self.steps = 0
        return OBS, {}

    def step(self, action):
        assert self.action_space.contains(np.asarray(action, dtype=np.float32))
        self.steps += 1
        kind, n = self._plan[self.episode]
        done = self.steps >= n
        terminated = bool(done and kind in ("goal", "collision"))
        truncated = bool(done and kind in ("timeout", "truncated"))
        info = {
            "goal_reached": bool(done and kind == "goal"),
            "collision": bool(done and kind == "collision"),
            "timeout": bool(done and kind == "timeout"),
            "d_min": 0.5 + 0.1 * self.steps,
            "termination_reason": kind if done else "none",
        }
        return OBS, 1.0, terminated, truncated, info

    def close(self) -> None:
        self.closed = True


class _StubController:
    def __init__(self) -> None:
        self.resets = 0
        self.paths: list[np.ndarray] = []

    def reset(self) -> None:
        self.resets += 1

    def set_path(self, path_world: np.ndarray) -> None:
        self.paths.append(np.asarray(path_world))

    def act(self, observation: dict) -> np.ndarray:
        assert observation["pose"].shape == (3,)
        assert observation["dt"] == pytest.approx(0.1)
        assert observation["obstacles"]
        return np.zeros(2, dtype=np.float32)


def _metrics(success: bool, steps: int = 10, ret: float = 2.0) -> EpisodeMetrics:
    return EpisodeMetrics(
        success=success,
        collision=not success,
        timeout=False,
        steps=steps,
        episode_return=ret,
        min_obstacle_clearance=0.35,
    )


def _result(ctrl: str, chassis: str, success: bool) -> EvalResult:
    return EvalResult(episodes=(_metrics(success),), controller=ctrl, chassis=chassis)


def test_wilson_interval_known_values_and_edges():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=5e-4)
    assert hi == pytest.approx(0.7634, abs=5e-4)
    # 全成功/全失败：区间单侧贴边（Wilson 不减到 0 宽度）
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0)
    assert wilson_interval(10, 10)[0] > 0.0
    # n=0 不能抛 ZeroDivisionError
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_evaluate_aggregates_outcomes():
    env = _StubEnv([("goal", 3), ("collision", 2), ("timeout", 4), ("truncated", 5)])
    result = evaluate(env, _StubController(), n_episodes=4, seed=7)
    assert result.n == 4
    assert (result.successes, result.collisions, result.timeouts) == (1, 1, 2)
    assert result.success_rate == pytest.approx(0.25)
    assert result.collision_rate == pytest.approx(0.25)
    assert result.timeout_rate == pytest.approx(0.5)  # truncated 计为超时
    assert result.mean_steps == pytest.approx(3.5)
    assert result.mean_return == pytest.approx(3.5)
    assert result.seed == 7
    lo, hi = result.success_interval
    assert lo < 0.25 < hi


def test_evaluate_seeds_paths_and_lifecycle():
    env = _StubEnv([("goal", 2)] * 3)
    ctrl = _StubController()
    evaluate(env, ctrl, n_episodes=3, seed=10)
    assert env.seeds == [10, 11, 12]
    assert ctrl.resets == 3
    assert len(ctrl.paths) == 3
    assert ctrl.paths[0].shape == (2, 3)
    assert env.closed is False  # 传实例时评测器不负责 close


def test_evaluate_closes_env_created_by_factory():
    created: list[_StubEnv] = []

    def factory() -> _StubEnv:
        env = _StubEnv([("goal", 2)])
        created.append(env)
        return env

    result = evaluate(factory, _StubController(), n_episodes=1, seed=0)
    assert result.n == 1
    assert len(created) == 1
    assert created[0].closed is True


def test_max_steps_truncates_and_counts_as_timeout():
    env = _StubEnv([("timeout", 999)])
    result = evaluate(env, _StubController(), n_episodes=1, seed=0, max_steps=5)
    ep = result.episodes[0]
    assert ep.steps == 5
    assert ep.timeout is True
    assert ep.success is False and ep.collision is False


def test_n_episodes_must_be_positive():
    with pytest.raises(ValueError):
        evaluate(_StubEnv([]), _StubController(), n_episodes=0)


def test_controller_observation_exposes_pose_and_dt():
    env = _StubEnv([("goal", 2)])
    obs = controller_observation(env)
    assert obs["pose"].shape == (3,)
    assert obs["dt"] == pytest.approx(0.1)
    assert obs["obstacles"] is env._obstacles


def test_result_to_dict_is_json_serialisable():
    env = _StubEnv([("goal", 2), ("collision", 3)])
    result = evaluate(
        env, _StubController(), n_episodes=2, seed=0, labels=("ctrl", "ackermann")
    )
    payload = result.to_dict()
    assert payload["controller"] == "ctrl"
    assert payload["chassis"] == "ackermann"
    assert payload["n"] == 2
    assert len(payload["episodes"]) == 2
    text = json.dumps(payload, ensure_ascii=False)  # 不得抛 TypeError
    assert "success_rate" in text


def test_to_markdown_matrix_and_detail_table():
    rows = [
        ("pp", "ackermann", _result("pp", "ackermann", True)),
        ("pp", "mecanum", _result("pp", "mecanum", False)),
    ]
    md = to_markdown(rows)
    assert "| controller | ackermann | mecanum |" in md
    assert "100.0% ±" in md and "(n=1)" in md
    assert "| pp | ackermann | 0.0% | 0.0% | 10.0 | 2.00 | 0.350 m |" in md
    assert "碰撞率" in md and "超时率" in md and "平均回报" in md


def test_to_markdown_accepts_mapping_and_single_row():
    md = to_markdown({("pp", "ackermann"): _result("pp", "ackermann", True)})
    assert "| pp | ackermann | 0.0% |" in md
    single = _result("pp", "ackermann", True).to_markdown()
    assert "| pp |" in single and "ackermann" in single


# ── 控制器/底盘一致性护栏 ─────────────────────────────────────────────
# 真 env 的底盘来自 EnvConfig（v_max=1.2 / v_min=-0.6），而 make_chassis(kind) 的默认
# 参数是 v_max=1.0 / v_min=-1.0 ⇒ 用 make_chassis 造的控制器按 1.0 归一化、Chassis.command()
# 按 1.2 反解，请求 0.6 实跑 0.72（+20%）。下面用真 env 取证，不用 _StubEnv（它没有底盘）。


class _NoLimitsController(_StubController):
    """不按物理量归一化的控制器（random 基线的形状）：没有 limits，护栏无从比对。"""

    limits = None

    def act(self, observation: dict) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


@pytest.mark.parametrize("kind", ["ackermann", "diff_drive", "mecanum"])
def test_evaluate_rejects_controller_calibrated_for_another_chassis(kind):
    from chassis_sim.chassis import make_chassis
    from chassis_sim.controllers import PurePursuitController
    from chassis_sim.envs import ChassisSimEnv

    env = ChassisSimEnv(kind, max_steps=5)
    try:
        controller = PurePursuitController(chassis=make_chassis(kind))
        with pytest.raises(ValueError) as excinfo:
            evaluate(env, controller, n_episodes=1, seed=0, max_steps=5)
    finally:
        env.close()

    msg = str(excinfo.value)
    assert "控制器是按另一台底盘标定的" in msg
    assert "controller:" in msg and "env.chassis:" in msg
    # 两侧的尺子都要打出来（v_max 1.0 vs 1.2 就是那 20%；v_min 随车型 -0.6/-1.2）
    assert "v_min: 控制器 -1.0 vs env -" in msg
    assert "v_max: 控制器 1.0 vs env 1.2" in msg
    assert "不一致项:" in msg


def test_evaluate_allow_mismatch_warns_without_raising(capsys):
    from chassis_sim.chassis import make_chassis
    from chassis_sim.controllers import PurePursuitController
    from chassis_sim.envs import ChassisSimEnv

    env = ChassisSimEnv("ackermann", max_steps=5)
    try:
        controller = PurePursuitController(chassis=make_chassis("ackermann"))
        result = evaluate(
            env, controller, n_episodes=1, seed=0, max_steps=5, allow_mismatch=True
        )
    finally:
        env.close()

    err = capsys.readouterr().err
    assert "[WARN][eval]" in err
    assert "控制器是按另一台底盘标定的" in err
    assert result.n == 1  # 确实跑了，而不仅是没抛


def test_evaluate_accepts_controller_built_from_env_chassis():
    from chassis_sim.controllers import PurePursuitController
    from chassis_sim.envs import ChassisSimEnv

    env = ChassisSimEnv("ackermann", max_steps=5)
    try:
        controller = PurePursuitController(chassis=env.unwrapped.chassis)
        result = evaluate(env, controller, n_episodes=1, seed=0, max_steps=5)
    finally:
        env.close()

    assert result.n == 1


def test_evaluate_skips_guard_for_controller_without_limits():
    from chassis_sim.envs import ChassisSimEnv

    env = ChassisSimEnv("ackermann", max_steps=5)
    try:
        result = evaluate(env, _NoLimitsController(), n_episodes=1, seed=0, max_steps=5)
    finally:
        env.close()

    assert result.n == 1


@pytest.mark.parametrize("kind", ["ackermann", "diff_drive", "mecanum"])
def test_roundtrip_requested_speed_is_preserved_by_env_chassis(kind):
    """控制器请求 ROUNDTRIP_V=0.6 → env 底盘实际 0.6（≤1e-9）；对着错底盘则是 0.72。"""
    from chassis_sim.chassis import make_chassis
    from chassis_sim.controllers import PurePursuitController
    from chassis_sim.envs import ChassisSimEnv

    request = 0.6
    env = ChassisSimEnv(kind, max_steps=5)
    try:
        chassis = env.unwrapped.chassis
        action = PurePursuitController(chassis=chassis).to_chassis_action(request, 0.0)
        v_actual = float(chassis.command(np.asarray(action, dtype=np.float64)).v)
        wrong_action = PurePursuitController(
            chassis=make_chassis(kind)
        ).to_chassis_action(request, 0.0)
        v_wrong = float(chassis.command(np.asarray(wrong_action, dtype=np.float64)).v)
    finally:
        env.close()

    assert v_actual == pytest.approx(request, abs=1e-9)
    assert v_wrong == pytest.approx(request * 1.2, rel=1e-12)  # 1.0 归一化 / 1.2 反解


# ── 独立地图 / 固定地图：reset_options 的两种形态 ─────────────────────
def _map_for(index: int) -> dict:
    """第 index 张假地图：只换路径（障碍固定），指纹随之变化。"""
    return {
        "path_waypoints": np.array(
            [[0.0, float(index), 0.0], [5.0, float(index), 0.0]], dtype=np.float64
        )
    }


class _MapStubEnv(_StubEnv):
    """把 options["path_waypoints"] 当地图 —— 用来验证"每回合是不是同一张图"。"""

    def __init__(self, plan: list[tuple[str, int]]) -> None:
        super().__init__(plan)
        self.options_seen: list[dict | None] = []
        self._path: np.ndarray | None = None

    def reset(self, seed=None, options=None):
        self.options_seen.append(None if options is None else dict(options))
        self._path = None if options is None else options.get("path_waypoints")
        return super().reset(seed=seed, options=options)


def test_evaluate_callable_reset_options_gives_one_map_per_episode():
    """callable 每回合调一次：索引 0..n-1，且每回合留下不同的地图指纹。"""
    env = _MapStubEnv([("goal", 2), ("goal", 3), ("goal", 4)])
    seen: list[int] = []

    def factory(index: int) -> dict:
        seen.append(index)
        return _map_for(index)

    result = evaluate(
        env, _StubController(), n_episodes=3, max_steps=50, reset_options=factory
    )
    assert seen == [0, 1, 2]
    assert len(env.options_seen) == 3
    for index, options in enumerate(env.options_seen):
        np.testing.assert_allclose(
            options["path_waypoints"], _map_for(index)["path_waypoints"]
        )
    assert result.reset_options_kind == "per-episode"
    assert result.distinct_maps == 3
    assert result.maps_independent is True
    assert result.same_map_repeated is False
    assert len(set(result.map_fingerprints)) == 3


def test_fixed_mapping_keeps_legacy_same_map_semantics(capsys):
    """旧写法（固定 dict）不被破坏：n 个回合仍是同一张图，只是现在会告警并标注。"""
    env = _MapStubEnv([("goal", 3), ("goal", 3), ("goal", 3)])
    result = evaluate(
        env, _StubController(), n_episodes=3, max_steps=50, reset_options=_map_for(4)
    )
    err = capsys.readouterr().err
    assert "[WARN][eval]" in err
    assert "同一张地图" in err and "不代表独立样本" in err

    rows = [(e.steps, e.episode_return, e.success) for e in result.episodes]
    assert rows[0] == rows[1] == rows[2]
    assert result.reset_options_kind == "fixed"
    assert result.distinct_maps == 1 and result.same_map_repeated is True
    assert result.maps_independent is False

    md = to_markdown([("pp", "ackermann", result)])
    assert "⚠固定地图" in md
    assert "固定地图行" in md and "不可引用" in md


def test_evaluate_callable_returning_same_map_still_warns(capsys):
    """callable 也可能每回合返回同一张图 —— 指纹自证，照样告警。"""
    env = _MapStubEnv([("goal", 2), ("goal", 2), ("goal", 2)])
    result = evaluate(
        env,
        _StubController(),
        n_episodes=3,
        max_steps=50,
        reset_options=lambda i: _map_for(0),
    )
    err = capsys.readouterr().err
    assert "[WARN][eval]" in err and "指纹全部相同" in err
    assert result.reset_options_kind == "per-episode"
    assert result.same_map_repeated is True and result.maps_independent is False


def test_evaluate_rejects_non_mapping_from_reset_options_callable():
    env = _StubEnv([("goal", 2)])
    with pytest.raises(TypeError, match="必须返回 Mapping"):
        evaluate(
            env,
            _StubController(),
            n_episodes=1,
            max_steps=50,
            reset_options=lambda i: [1, 2, 3],
        )


def test_eval_result_exposes_map_independence_fields():
    """EvalResult 自证接口：to_dict 里也带着，下游不必回读 env。"""
    metrics = (
        EpisodeMetrics(True, False, False, 5, 1.0, 0.4, "goal", "aaaa"),
        EpisodeMetrics(True, False, False, 6, 1.1, 0.4, "goal", "bbbb"),
    )
    result = EvalResult(episodes=metrics, reset_options_kind="per-episode")
    assert result.map_fingerprints == ("aaaa", "bbbb")
    assert result.distinct_maps == 2 and result.maps_independent is True
    payload = result.to_dict()
    assert payload["reset_options_kind"] == "per-episode"
    assert payload["distinct_maps"] == 2 and payload["maps_independent"] is True
    assert payload["map_fingerprints"] == ["aaaa", "bbbb"]
    assert payload["episodes"][0]["map_fingerprint"] == "aaaa"
