"""benchmark 层（策略 × 场景 × seed）的单元覆盖。

三条验收对应关系：

* 独立样本闸门：固定地图（straight）⇒ maps_independent=False 且
  success_ci 是 None（不是"照样算个数"）；可重掷地图
  （obstacles_sparse + scenario_reset_options_factory）⇒
  distinct_maps == n、maps_independent=True，success_ci 与
  wilson_interval(successes, n) 手算一致。
* 四个基线可直接调用：默认真跑 zero/random/pp/dwa，且默认场景必须是
  core 的（缺 numba 也跑得动）。
* 协议：seeds 少于 3 个 ⇒ protocol_ok=False + [WARN][eval]，表格照出。

这里刻意不覆盖 evaluate() 的内部行为（tests/test_eval_core.py 已守着），
只守 benchmark 自己新增的那一层。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium.spaces import Box

from chassis_sim.api import make
from chassis_sim.eval import (
    BASELINES,
    MIN_SEEDS,
    POLICY_KINDS,
    BenchmarkCell,
    BenchmarkResult,
    RandomPolicy,
    ZeroPolicy,
    benchmark,
    benchmark_cell,
    make_policy,
    wilson_interval,
)


def _space() -> Box:
    low, high = np.array([-1.0, -1.0], np.float32), np.array([1.0, 1.0], np.float32)
    return Box(low=low, high=high, dtype=np.float32)


def _env(**overrides: object):
    """一辆便宜的 ackermann（不渲染、步数上限压到几十步）。"""
    kwargs = {"max_steps": 40}
    kwargs.update(overrides)
    return make("ackermann", **kwargs)


def _requires_numba() -> None:
    """planning 场景要 Hybrid-A*（numba）；core-only 装法下这些用例应 skip。"""
    pytest.importorskip("numba", reason="planning 场景需要 chassis-sim[planning]")


# ── 策略：zero / random ────────────────────────────────────────────
def test_zero_policy_returns_zeros_in_action_space_shape_and_dtype() -> None:
    policy = ZeroPolicy(_space())

    action = policy.act()

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert not action.any()


def test_random_policy_is_seeded_and_stays_in_bounds() -> None:
    space = _space()
    left, right = RandomPolicy(space, seed=7), RandomPolicy(space, seed=7)

    left_actions = np.stack([left.act() for _ in range(8)])
    right_actions = np.stack([right.act() for _ in range(8)])

    assert np.array_equal(left_actions, right_actions)  # 同 seed ⇒ 同动作序列
    assert left_actions.dtype == np.float32
    assert (left_actions >= -1.0).all() and (left_actions <= 1.0).all()
    assert np.unique(left_actions).size > 1  # 真在采样，不是常数


def test_random_policy_reset_advances_the_stream() -> None:
    policy = RandomPolicy(_space(), seed=3)
    first = policy.act()

    policy.reset()
    after_reset = policy.act()

    assert not np.array_equal(first, after_reset)


# ── 策略名解析 ────────────────────────────────────────────────────
def test_make_policy_resolves_builtin_names() -> None:
    env = _env(scenario="straight")
    try:
        zero = make_policy("zero", env)
        pp = make_policy("pp", env)
    finally:
        env.close()

    assert isinstance(zero, ZeroPolicy)
    assert pp.__class__.__name__ == "PurePursuitController"


def test_make_policy_rejects_unknown_name() -> None:
    env = _env(scenario="straight")
    try:
        with pytest.raises(ValueError, match="未知策略"):
            make_policy("teleop", env)
    finally:
        env.close()


def test_baselines_cover_the_four_documented_policies() -> None:
    assert POLICY_KINDS == ("zero", "random", "pp", "dwa")
    assert tuple(spec.name for spec in BASELINES) == POLICY_KINDS


# ── 验收 1：固定地图不给 success_ci 数字 ──────────────────────────
def test_fixed_map_cell_reports_not_independent_and_no_ci() -> None:
    cell = benchmark_cell("pp", "straight", episodes=2, max_steps=60)

    assert isinstance(cell, BenchmarkCell)
    assert cell.n == 2
    assert cell.distinct_maps == 1
    assert cell.maps_independent is False
    assert cell.success_interval is None
    assert cell.success_ci is None


def test_default_config_is_flagged_not_independent_and_prints_no_ci() -> None:
    """默认（退化固定直线）+ 协议要求的 3 个 seed：表照出，CI 一律不给数字。"""
    result = benchmark(episodes=1, max_steps=60)

    assert result.protocol_ok is True
    assert {cell.policy for cell in result.cells} == set(POLICY_KINDS)
    for summary in result.summaries:
        assert summary.maps_independent is False
        assert summary.success_ci is None

    table = result.to_markdown()
    assert "非独立样本行" in table
    assert "—" in table


# ── 验收 2：可重掷地图 ⇒ 独立样本 + Wilson ───────────────────────
def test_relocatable_maps_give_independent_samples_and_wilson_ci() -> None:
    _requires_numba()
    episodes = 20
    result = benchmark(
        ["zero"],
        ["obstacles_sparse"],
        seeds=(0,),
        episodes=episodes,
        max_steps=60,
        out=None,
    )
    summary = result.summaries[0]

    assert summary.n == episodes
    assert summary.distinct_maps == episodes
    assert summary.maps_independent is True
    assert len(set(summary.map_fingerprints)) == episodes
    assert summary.success_interval is not None and summary.success_ci is not None

    low, high = wilson_interval(summary.successes, summary.n)
    expected = max(summary.success_rate - low, high - summary.success_rate)
    assert summary.success_ci == pytest.approx(expected, abs=1e-12)
    assert summary.success_interval == (pytest.approx(low), pytest.approx(high))


def test_maps_are_shared_between_policies_for_pairing() -> None:
    _requires_numba()
    result = benchmark(
        ["zero", "random"],
        ["obstacles_sparse"],
        seeds=(0,),
        episodes=2,
        max_steps=30,
    )
    zero, random = result.cells

    assert zero.map_fingerprints == random.map_fingerprints
    assert zero.map_seeds == random.map_seeds == (0, 1)


def test_core_scenario_rerolls_maps_without_planning_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有 numba 也能给独立样本：逐回合换一张图不一定要 Hybrid-A*。"""
    from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

    monkeypatch.setitem(sys.modules, "numba", None)
    base = dict(scenario_to_reset_options(load_scenario("dense_static")))

    def options_factory(scenario: object, seed: int):
        def provider(index: int) -> dict[str, object]:
            options = dict(base)
            obstacles = list(options.get("obstacles") or [])
            obstacles.append(
                {
                    "shape": {"name": "circle", "radius": 0.25},
                    "state": [3.0 + 0.4 * index + 0.4 * seed, 3.0, 0.0],
                }
            )
            options["obstacles"] = obstacles
            return options

        return provider

    episodes = 5
    result = benchmark(
        ["zero"],
        ["dense_static"],
        seeds=(0,),
        episodes=episodes,
        max_steps=20,
        reset_options_factory=options_factory,
    )
    summary = result.summaries[0]

    assert summary.distinct_maps == episodes
    assert summary.maps_independent is True
    assert summary.success_ci is not None
    low, high = wilson_interval(summary.successes, summary.n)
    expected = max(summary.success_rate - low, high - summary.success_rate)
    assert summary.success_ci == pytest.approx(expected, abs=1e-12)


# ── 协议闸门 ──────────────────────────────────────────────────────
def test_fewer_than_three_seeds_flags_protocol_but_still_renders(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = benchmark(["zero"], ["straight"], seeds=(0,), episodes=1, max_steps=20)

    assert MIN_SEEDS == 3
    assert result.protocol_ok is False
    assert "协议要求 >= 3" in capsys.readouterr().err
    assert "协议未满" in result.to_markdown()


def test_summaries_merge_every_seed() -> None:
    result = benchmark(["zero"], ["straight"], seeds=(0, 1), episodes=2, max_steps=20)

    assert len(result.cells) == 2
    assert result.summaries[0].n == 4
    assert result.summaries[0].seeds == (0, 1)


# ── 落盘 ──────────────────────────────────────────────────────────
def test_write_creates_parent_directory_and_ends_with_newline(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "bench.md"
    result = benchmark(
        ["zero"], ["straight"], seeds=(0,), episodes=1, max_steps=20, out=out
    )

    assert isinstance(result, BenchmarkResult)
    assert Path(result.markdown_path) == out
    text = out.read_text(encoding="utf-8")
    assert text.startswith("**chassis-sim · benchmark")
    assert text.endswith("\n")
    assert "| 策略 |" in text


# ── 输入校验 ──────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"policies": []}, "policies 不能为空"),
        ({"seeds": ()}, "seeds 不能为空"),
        ({"seeds": (0, 0)}, "seeds 不能重复"),
        ({"seeds": (-1,)}, "seeds 必须是非负"),
        ({"episodes": 0}, "episodes 必须是正整数"),
        ({"policies": ["zero", "zero"]}, "策略名重复"),
        ({"scenarios": []}, "scenarios 不能为空"),
    ],
)
def test_bad_input_is_rejected_before_any_episode(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        benchmark(**{"scenarios": ["straight"], "max_steps": 10, **kwargs})


def test_benchmark_runs_headless_without_planning_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认场景必须是 core 的：缺 numba 时 benchmark 默认配置照样跑得动。"""
    monkeypatch.setitem(sys.modules, "numba", None)

    result = benchmark(["random"], ["straight"], seeds=(0,), episodes=1, max_steps=20)

    assert result.cells[0].n == 1


# ── 被控对象旋钮：dynamics / 传感器退化 ──────────────────────────────
def _plant_env(
    *,
    dynamics: str | None = None,
    sensor_noise_std: float | None = None,
    sensor_latency_steps: int | None = None,
):
    """用 benchmark 自己的工厂造一台 corridor 上的车（core 场景，不需要 numba）。"""
    from chassis_sim.eval.benchmark import _default_env_factory

    return _default_env_factory(
        "ackermann",
        "corridor",
        dynamics=dynamics,
        sensor_noise_std=sensor_noise_std,
        sensor_latency_steps=sensor_latency_steps,
    )(0)


def _front_scans(env: object, steps: int) -> list[np.ndarray | None]:
    env.reset(seed=0)
    scans: list[np.ndarray | None] = []
    for _ in range(steps):
        _, _, _, _, info = env.step(np.array([0.0, 0.3], np.float32))
        reading = info.get("front")
        scans.append(
            None if reading is None else np.asarray(reading["scan"], np.float64)
        )
    return scans


def test_plant_knobs_reach_the_chassis_and_the_sensors() -> None:
    env = _plant_env(dynamics="wheel", sensor_noise_std=0.02, sensor_latency_steps=1)
    try:
        specs = {spec.name: spec for spec in env._sensor_specs}
        assert env._dynamics_level == "wheel"
        assert set(specs) == {"front", "rear"}
        assert specs["front"].n_rays == 60
        assert specs["front"].noise_std == pytest.approx(0.02)
        assert specs["front"].latency_steps == 1
        assert specs["rear"].latency_steps == 1
    finally:
        env.close()


def test_plant_defaults_leave_the_environment_vanilla() -> None:
    env = _plant_env()
    try:
        assert env._dynamics_level == "none"
        assert tuple(env._sensor_specs) == ()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0, 0.3], np.float32))
        assert "front" not in info  # 没有配传感器 ⇒ 连键都不出现
    finally:
        env.close()


def test_sensor_noise_actually_perturbs_the_scan() -> None:
    clean_env = _plant_env(sensor_noise_std=0.0)
    noisy_env = _plant_env(sensor_noise_std=0.05)
    try:
        clean, noisy = _front_scans(clean_env, 4), _front_scans(noisy_env, 4)
    finally:
        clean_env.close()
        noisy_env.close()

    assert clean[-1] is not None and noisy[-1] is not None
    delta = noisy[-1] - clean[-1]
    assert np.max(np.abs(delta)) > 0.0  # 不是逐位相同
    assert 0.02 < float(np.std(delta)) < 0.12  # 量级就是那 0.05 m


def test_sensor_latency_delivers_the_previous_frame_verbatim() -> None:
    live_env = _plant_env(sensor_latency_steps=0)
    delayed_env = _plant_env(sensor_latency_steps=1)
    try:
        live, delayed = _front_scans(live_env, 4), _front_scans(delayed_env, 4)
    finally:
        live_env.close()
        delayed_env.close()

    assert live[0] is not None and delayed[0] is not None
    pairs = [(live[i], delayed[i + 1]) for i in range(len(live) - 1)]
    for earlier, later in pairs:
        assert later is not None
        assert np.array_equal(earlier, later)  # 零阶保持：晚一步交付的正是上一帧


def test_plant_is_recorded_in_the_result_dict_and_markdown() -> None:
    result = benchmark(
        ["zero"],
        ["corridor"],
        seeds=(0, 1, 2),
        episodes=1,
        max_steps=20,
        dynamics="limits",
        sensor_noise_std=0.02,
        sensor_latency_steps=1,
    )

    assert result.plant == {
        "dynamics": "limits",
        "sensor_noise_std": 0.02,
        "sensor_latency_steps": 1,
    }
    assert result.to_dict()["plant"] == result.plant
    assert "被控对象" in result.to_markdown()
    assert len(result.cells) == 3  # 每个 seed 一个格
    assert all(cell.n == 1 for cell in result.cells)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dynamics": "tyres"}, "dynamics"),
        ({"sensor_noise_std": -0.01}, "sensor_noise_std"),
        ({"sensor_latency_steps": -1}, "sensor_latency_steps"),
    ],
)
def test_bad_plant_values_are_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        benchmark(
            ["zero"], ["corridor"], seeds=(0,), episodes=1, max_steps=10, **kwargs
        )
