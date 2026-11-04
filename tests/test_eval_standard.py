"""1.0 的 evaluate 契约：episodes=、env 实例优先、EvalResult.markdown()。

这里的 env 全是本地替身（只需要 reset/step/bridge），不碰 chassis_sim.envs ——
本文件因此不依赖环境本体，跑得快；真 env 的招牌用法在文件末尾单独一条。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from chassis_sim.chassis import make_chassis
from chassis_sim.controllers import make_controller
from chassis_sim.eval import EpisodeMetrics, EvalResult, evaluate, to_markdown


class _StubBridge:
    """只要 get_odometry / get_path_global：控制器观测与地图指纹的唯一来源。"""

    def __init__(self, env: _StubEnv) -> None:
        self._env = env

    def get_odometry(self) -> np.ndarray:
        return np.array([0.0, float(self._env.seed or 0), 0.0, 0.0, 0.0])

    def get_path_global(self) -> np.ndarray:
        return np.asarray(self._env.path, dtype=np.float64)


class _StubEnv:
    """最小 env：固定走 n_steps 步后报 goal；每回合按 seed 换一条路径（地图指纹随之变）。"""

    def __init__(self, steps: int = 3) -> None:
        self.n_steps = int(steps)
        self.steps = 0
        self.seed: int | None = None
        self.seeds: list[int | None] = []
        self.closed = False
        self.unwrapped = self
        self.bridge = _StubBridge(self)
        self.dt = 0.1
        self._obstacles = [
            {"shape": {"name": "circle", "radius": 0.2}, "state": [1.0, 1.0, 0.0]}
        ]
        self.path = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.seed = seed
        self.seeds.append(seed)
        self.steps = 0
        offset = float(seed or 0)
        self.path = np.array([[0.0, offset, 0.0], [5.0, offset, 0.0]], dtype=np.float64)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.n_steps
        info = {
            "goal_reached": done,
            "collision": False,
            "timeout": False,
            "d_min": 1.0,
            "termination_reason": "goal" if done else "",
        }
        return np.zeros(2, dtype=np.float32), 1.0, done, False, info

    def close(self) -> None:
        self.closed = True


class _StubController:
    """没有 limits（不做底盘标定比对）、act 恒零。"""

    def __init__(self) -> None:
        self.resets = 0
        self.paths: list[np.ndarray] = []

    def reset(self) -> None:
        self.resets += 1

    def set_path(self, path_world: np.ndarray) -> None:
        self.paths.append(np.asarray(path_world))

    def act(self, observation: dict) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


def test_signature_is_the_1_0_shape() -> None:
    params = inspect.signature(evaluate).parameters
    assert list(params) == [
        "env",
        "controller",
        "episodes",
        "seed",
        "max_steps",
        "reset_options",
        "labels",
        "allow_mismatch",
    ]
    assert params["episodes"].kind is inspect.Parameter.KEYWORD_ONLY


def test_episodes_keyword_controls_episode_count() -> None:
    env = _StubEnv(steps=2)
    result = evaluate(env, _StubController(), episodes=3, seed=0)
    assert result.n == 3
    assert len(env.seeds) == 3


def test_old_n_episodes_keyword_is_gone() -> None:
    """1.0 无兼容层：旧关键字直接 TypeError，而不是静默当成 episodes。"""
    with pytest.raises(TypeError):
        evaluate(_StubEnv(), _StubController(), n_episodes=2)  # type: ignore[call-arg]


def test_episodes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="episodes 必须 > 0"):
        evaluate(_StubEnv(), _StubController(), episodes=0)


def test_env_instance_and_factory_paths_agree() -> None:
    """招牌用法（传实例）与工厂路径必须逐回合一致，且只有工厂路径会 close。"""
    instance_env = _StubEnv(steps=2)
    from_instance = evaluate(
        instance_env, _StubController(), episodes=3, seed=5, labels=("pp", "ackermann")
    )

    factory_env = _StubEnv(steps=2)
    from_factory = evaluate(
        lambda: factory_env,
        _StubController(),
        episodes=3,
        seed=5,
        labels=("pp", "ackermann"),
    )

    assert from_instance.success_rate == from_factory.success_rate
    assert from_instance.success_interval == from_factory.success_interval
    assert from_instance.map_fingerprints == from_factory.map_fingerprints
    assert from_instance.to_dict()["episodes"] == from_factory.to_dict()["episodes"]
    # 传实例：评测器不负责 close；传工厂：评测器负责 close
    assert instance_env.closed is False
    assert factory_env.closed is True


def test_callable_env_instance_is_treated_as_instance() -> None:
    """判定规则：有 reset/step 就是实例 —— 可调用的 env 也绝不被当成工厂调用。"""

    class _CallableEnv(_StubEnv):
        def __call__(self) -> _StubEnv:
            raise AssertionError("可调用的 env 实例必须按实例处理，不能调用它")

    env = _CallableEnv(steps=2)
    result = evaluate(env, _StubController(), episodes=2, seed=0)
    assert result.n == 2
    assert env.closed is False


def test_first_argument_must_be_env_or_factory() -> None:
    with pytest.raises(TypeError, match="env 实例"):
        evaluate(42, _StubController(), episodes=1)  # type: ignore[arg-type]


def test_markdown_method_matches_to_markdown() -> None:
    env = _StubEnv(steps=2)
    result = evaluate(
        env, _StubController(), episodes=2, seed=0, labels=("pp", "ackermann")
    )
    rendered = result.markdown()
    assert rendered == result.to_markdown()
    assert rendered == to_markdown([("pp", "ackermann", result)])
    assert "| pp |" in rendered
    assert "ackermann" in rendered
    assert "100.0% ±" in rendered and "(n=2)" in rendered


def test_markdown_signature_is_a_passthrough() -> None:
    params = list(inspect.signature(EvalResult.markdown).parameters.values())
    assert params[0].name == "self"  # 未绑定方法，第一个参数是 self
    assert [p.kind for p in params[1:]] == [
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ]


def test_markdown_forwards_rows_and_title() -> None:
    result = EvalResult(
        episodes=(
            EpisodeMetrics(
                success=True,
                collision=False,
                timeout=False,
                steps=10,
                episode_return=2.0,
                min_obstacle_clearance=0.35,
            ),
        ),
        controller="pp",
        chassis="ackermann",
    )
    other = EvalResult(
        episodes=(
            EpisodeMetrics(
                success=False,
                collision=True,
                timeout=False,
                steps=20,
                episode_return=-1.0,
            ),
        ),
        controller="dwa",
        chassis="mecanum",
    )
    both = result.markdown(
        [("pp", "ackermann", result), ("dwa", "mecanum", other)], title="拼表标题"
    )
    assert "拼表标题" in both
    assert "| pp |" in both and "| dwa |" in both
    assert "| controller | ackermann | mecanum |" in both


class _StubEnvWithChassis(_StubEnv):
    """带真底盘的最小 env：给"省略 controller"的默认绑定用（不碰 chassis_sim.envs）。"""

    def __init__(self, steps: int = 3) -> None:
        super().__init__(steps=steps)
        self.chassis = make_chassis("ackermann")


def test_controller_can_be_omitted_and_binds_to_the_env_chassis() -> None:
    """招牌用法：evaluate(env, episodes=N) 的默认控制器 = 这台 env 的 pure pursuit。"""
    assert inspect.signature(evaluate).parameters["controller"].default is None

    implicit = evaluate(_StubEnvWithChassis(steps=2), episodes=3, seed=7)
    explicit_env = _StubEnvWithChassis(steps=2)
    explicit = evaluate(
        explicit_env,
        make_controller("pp", chassis=explicit_env.unwrapped.chassis),
        episodes=3,
        seed=7,
    )

    assert implicit.n == explicit.n == 3
    assert implicit.controller == "PurePursuitController"
    # 同 seed 同回合数 ⇒ 逐回合完全一致（默认绑定不许改变任何语义）
    assert implicit.to_dict()["episodes"] == explicit.to_dict()["episodes"]
    assert implicit.to_dict() == explicit.to_dict()


def test_omitted_controller_needs_a_usable_chassis_on_the_env() -> None:
    with pytest.raises(TypeError) as excinfo:
        evaluate(_StubEnv(steps=2), episodes=1)  # 这个替身没有 chassis

    message = str(excinfo.value)
    assert "evaluate(env, controller)" in message
    assert 'make_controller("pp", chassis=env.unwrapped.chassis)' in message

    # 工厂路径同样要在报错时 close 掉自己造出来的 env（默认控制器也在 try 内）
    created = _StubEnv(steps=2)
    with pytest.raises(TypeError):
        evaluate(lambda: created, episodes=1)
    assert created.closed is True


def test_headline_usage_with_real_env() -> None:
    """cs.evaluate(env, cs.make_controller("pp"), episodes=...)：真 env + 真控制器。"""
    from chassis_sim.api import make
    from chassis_sim.controllers import make_controller

    env = make("ackermann", max_steps=20)
    try:
        controller = make_controller("pp", chassis=env.unwrapped.chassis)
        result = evaluate(env, controller, episodes=2, seed=0, max_steps=20)
    finally:
        env.close()

    assert result.n == 2
    assert result.seed == 0
    assert 0.0 <= result.success_rate <= 1.0
    assert result.successes == sum(1 for episode in result.episodes if episode.success)
