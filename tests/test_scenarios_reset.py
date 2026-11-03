"""reset options、env 接线、planning 生成与 numba 守卫测试。

覆盖验收口径 b（四个 core 场景真建 env 跑 50 步）、d（numba 缺失时 core 仍可用、
planning 抛带安装提示的 ImportError）与 g（六个 planning 场景真的调通）。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.scenarios import (
    ScenarioResetOptionsFactory,
    load_scenario,
    scenario_reset_options_factory,
    scenario_to_reset_options,
)
from chassis_sim.scenarios import reset_options as reset_mod

CORE_SCENES = ("straight", "circle", "corridor", "dense_static")
PLANNING_SCENES = (
    "obstacles_sparse",
    "obstacles_medium",
    "obstacles_dense",
    "obstacles_dense_reverse",
    "foldback_reverse",
    "maze_grid",
)
OBS_DIM = 494


@pytest.mark.parametrize("name", CORE_SCENES)
def test_core_reset_options_are_env_ready(name: str) -> None:
    scenario = load_scenario(name)
    options = scenario_to_reset_options(scenario, seed=scenario.seed)
    assert set(options) == {"obstacles", "path_waypoints", "initial_heading"}

    path = options["path_waypoints"]
    assert path.ndim == 2
    assert path.shape[1] == 3
    assert path.shape[0] >= 2
    assert path.dtype == np.float64
    assert np.isfinite(path).all()

    # 鼻向列存在（env 侧 CTE / 档位判定读 path[:, 2]）
    assert options["initial_heading"] == pytest.approx(float(path[0, 2]))
    # 重采样间距不超过 YAML 的 spacing
    steps = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    assert steps.max() <= scenario.path.params["spacing"] + 1e-9
    # 全部落在 world_bounds 内
    xmin, xmax, ymin, ymax = scenario.world_bounds
    assert path[:, 0].min() >= xmin and path[:, 0].max() <= xmax
    assert path[:, 1].min() >= ymin and path[:, 1].max() <= ymax
    # 障碍转成了 bridge 格式
    assert len(options["obstacles"]) == len(scenario.obstacles)
    for obstacle, payload in zip(scenario.obstacles, options["obstacles"]):
        assert payload["shape"]["name"] == obstacle.kind
        assert len(payload["state"]) == 3


@pytest.mark.parametrize("name", CORE_SCENES)
def test_core_scene_env_runs_fifty_steps(name: str) -> None:
    scenario = load_scenario(name)
    options = scenario_to_reset_options(scenario, seed=scenario.seed)
    env = ChassisSimEnv("ackermann", max_steps=100)
    rng = np.random.default_rng(7)
    try:
        obs, _info = env.reset(seed=scenario.seed, options=options)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        for step in range(50):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            obs, _reward, terminated, truncated, info = env.step(action)
            if step == 0:
                # 走的是场景注入分支，没退回空地直线
                assert info["map_fallback"] is False
            assert obs.shape == (OBS_DIM,)
            assert obs.dtype == np.float32
            assert np.isfinite(obs).all()
            if terminated or truncated:
                obs, info = env.reset(seed=scenario.seed, options=options)
                assert obs.shape == (OBS_DIM,)
    finally:
        env.close()


def test_planning_scenes_generate_maps_with_map_generator() -> None:
    pytest.importorskip("numba")
    for name in PLANNING_SCENES:
        scenario = load_scenario(name)
        options = scenario_to_reset_options(scenario, seed=scenario.seed)

        assert options["obstacles"], f"{name} 没生成障碍"
        for payload in options["obstacles"]:
            assert payload["shape"]["name"] in ("circle", "rectangle")
            assert len(payload["state"]) == 3

        path = options["path_waypoints"]
        assert path.ndim == 2 and path.shape[1] == 3
        assert path.shape[0] >= 2
        assert np.isfinite(path).all()
        xmin, xmax, ymin, ymax = scenario.world_bounds
        assert path[:, 0].min() >= xmin - 1e-9 and path[:, 0].max() <= xmax + 1e-9
        assert path[:, 1].min() >= ymin - 1e-9 and path[:, 1].max() <= ymax + 1e-9
        assert np.isfinite(options["initial_heading"])


def test_planning_generation_is_seeded() -> None:
    pytest.importorskip("numba")
    scenario = load_scenario("obstacles_sparse")
    first = scenario_to_reset_options(scenario, seed=123)
    second = scenario_to_reset_options(scenario, seed=123)
    np.testing.assert_allclose(first["path_waypoints"], second["path_waypoints"])
    assert len(first["obstacles"]) == len(second["obstacles"])


def test_missing_numba_makes_planning_scene_raise_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "numba", None)
    scenario = load_scenario("maze_grid")
    with pytest.raises(ImportError) as excinfo:
        scenario_to_reset_options(scenario)
    message = str(excinfo.value)
    assert "pip install chassis-sim[planning]" in message
    assert "maze_grid" in message


def test_missing_numba_does_not_affect_core_scenes(monkeypatch) -> None:
    env = ChassisSimEnv("ackermann", max_steps=20)
    try:
        monkeypatch.setitem(sys.modules, "numba", None)
        for name in CORE_SCENES:
            scenario = load_scenario(name)
            options = scenario_to_reset_options(scenario, seed=scenario.seed)
            obs, _info = env.reset(seed=scenario.seed, options=options)
            assert obs.shape == (OBS_DIM,)
    finally:
        env.close()


# ── scenario_reset_options_factory：每回合重掷一张地图 ───────────────
def _map_key(options: dict) -> str:
    """把一张地图压成可比较的字符串（路径 + 每个障碍的名字/位姿）。"""
    path = np.round(np.asarray(options["path_waypoints"], dtype=float), 6)
    parts = [path.tobytes().hex()]
    for obstacle in options["obstacles"]:
        shape = getattr(obstacle, "shape", obstacle)
        state = np.round(np.asarray(getattr(shape, "state", ()), dtype=float), 6).tolist()
        parts.append(f"{getattr(shape, 'name', type(shape).__name__)}{state}")
    return "|".join(parts)


@pytest.mark.parametrize("name", CORE_SCENES)
def test_factory_on_core_scene_repeats_the_same_map(name: str) -> None:
    """core 场景的地图写死在 YAML 里 ⇒ 重掷也是同一张图，factory 不掩盖这点。"""
    factory = scenario_reset_options_factory(name)
    keys = [_map_key(factory(i)) for i in range(3)]
    assert keys[0] == keys[1] == keys[2]
    assert factory.stats == {"attempts": 3, "retries": 0, "failures": 0, "episodes": 3}


def test_factory_on_planning_scene_gives_distinct_maps_and_is_seeded() -> None:
    pytest.importorskip("numba")
    first = scenario_reset_options_factory("obstacles_dense", base_seed=11)
    keys = [_map_key(first(i)) for i in range(3)]
    assert len(set(keys)) == 3  # 每回合一张不同的图
    assert first.stats == {"attempts": 3, "retries": 0, "failures": 0, "episodes": 3}

    again = scenario_reset_options_factory("obstacles_dense", base_seed=11)
    assert [_map_key(again(i)) for i in range(3)] == keys  # 同 base_seed ⇒ 逐位可复现

    other = scenario_reset_options_factory("obstacles_dense", base_seed=12)
    assert [_map_key(other(i)) for i in range(3)] != keys  # 换 seed ⇒ 换地图


def test_factory_retries_map_generation_errors_with_next_seed(monkeypatch) -> None:
    calls: list[int | None] = []

    class MapGenerationError(RuntimeError):
        pass

    def flaky(scenario, *, seed=None):
        calls.append(seed)
        if len(calls) == 1:
            raise MapGenerationError("boom")
        return {
            "obstacles": [],
            "path_waypoints": np.zeros((2, 3), dtype=np.float64),
            "initial_heading": 0.0,
        }

    monkeypatch.setattr(reset_mod, "scenario_to_reset_options", flaky)
    factory = ScenarioResetOptionsFactory(
        load_scenario("straight"), base_seed=5, max_retries=2
    )
    options = factory(0)
    assert set(options) == {"obstacles", "path_waypoints", "initial_heading"}
    assert calls == [5, 6]  # 失败后换 seed=base_seed+1 重试
    assert factory.stats == {"attempts": 2, "retries": 1, "failures": 0, "episodes": 1}


def test_factory_raises_with_retry_rate_when_retries_exhausted(monkeypatch) -> None:
    class MapGenerationError(RuntimeError):
        pass

    def always_fail(scenario, *, seed=None):
        raise MapGenerationError(f"no map for seed {seed}")

    monkeypatch.setattr(reset_mod, "scenario_to_reset_options", always_fail)
    factory = ScenarioResetOptionsFactory(
        load_scenario("straight"), base_seed=100, max_retries=1
    )
    with pytest.raises(MapGenerationError) as excinfo:
        factory(2)
    message = str(excinfo.value)
    assert "straight" in message and "第 2 回合" in message
    assert "[102, 103]" in message and "max_retries=1" in message
    # 失败被计数、且抛出去（不会变成一个"少掉的回合"）
    assert factory.stats == {"attempts": 2, "retries": 1, "failures": 1, "episodes": 0}


def test_factory_does_not_retry_non_map_errors(monkeypatch) -> None:
    calls: list[int | None] = []

    def broken(scenario, *, seed=None):
        calls.append(seed)
        raise ValueError("配置坏了")

    monkeypatch.setattr(reset_mod, "scenario_to_reset_options", broken)
    factory = ScenarioResetOptionsFactory(load_scenario("straight"), base_seed=0)
    with pytest.raises(ValueError, match="配置坏了"):
        factory(0)
    assert calls == [0]  # 不是 MapGenerationError ⇒ 一次就抛，不重试
    assert factory.stats["attempts"] == 1 and factory.stats["failures"] == 0

