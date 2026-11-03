"""动态障碍的数据模型与 YAML 契约：解析、严格校验、reset options 透传。

校验用例全部走公开 API load_scenario(路径)：写一个最小场景 YAML 到 tmp_path，
只替换 dynamic_obstacles 块，因此每条报错都是"用户真的那样写 YAML"时会看到的原文。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chassis_sim.scenarios import (
    CircularDynamicObstacle,
    LinearDynamicObstacle,
    ScenarioValidationError,
    load_scenario,
    scenario_to_reset_options,
)

_BASE = """\
name: probe
description: 校验用最小场景
world_bounds: [0.0, 10.0, 0.0, 10.0]
obstacles: []
path:
  type: straight
  params:
    start: [1.000, 5.000]
    goal: [5.000, 5.000]
    spacing: 0.05
dynamic_obstacles:
{block}
seed: 0
requires: null
"""

_LINEAR = """\
  - type: linear
    start: [3.000, 4.400]
    velocity: [0.000, 0.300]
    radius: 0.250
    period: 20.000
    pingpong: true"""


def _load(tmp_path: Path, block: str):
    """把 dynamic_obstacles 块拼进最小场景并加载（校验失败时向上传播）。"""
    path = tmp_path / "probe.yaml"
    path.write_text(_BASE.format(block=block), encoding="utf-8")
    return load_scenario(path)


def _error(tmp_path: Path, block: str) -> str:
    """返回严格校验的报错原文。"""
    with pytest.raises(ScenarioValidationError) as excinfo:
        _load(tmp_path, block)
    return str(excinfo.value)


# ═════════════════════════ 正向：dynamic_crossing 的解析结果 ═════════════════════════
def test_dynamic_crossing_parses_into_typed_object() -> None:
    scenario = load_scenario("dynamic_crossing")
    assert repr(scenario.dynamic_obstacles) == (
        "(LinearDynamicObstacle(start=(3.0, 4.4), velocity=(0.0, 0.3), "
        "radius=0.25, period=20.0, pingpong=True),)"
    )
    (obstacle,) = scenario.dynamic_obstacles
    assert isinstance(obstacle, LinearDynamicObstacle)
    assert obstacle.kind == "linear"
    assert obstacle.start == (3.0, 4.4)
    assert obstacle.velocity == (0.0, 0.3)
    assert obstacle.radius == 0.25
    assert obstacle.period == 20.0
    assert obstacle.pingpong is True
    assert scenario.summary() == (
        "dynamic_crossing: path=straight obstacles=0 dynamic=1 requires=None"
    )


def test_to_env_dict_roundtrips_every_field() -> None:
    scenario = load_scenario("dynamic_crossing")
    assert scenario.dynamic_obstacles[0].to_env_dict() == {
        "type": "linear",
        "start": [3.0, 4.4],
        "velocity": [0.0, 0.3],
        "radius": 0.25,
        "period": 20.0,
        "pingpong": True,
    }
    circular = CircularDynamicObstacle(
        center=(1.0, 2.0), radius_orbit=0.5, omega=1.0, phase=0.0, radius=0.2
    )
    assert circular.kind == "circular"
    assert circular.to_env_dict() == {
        "type": "circular",
        "center": [1.0, 2.0],
        "radius_orbit": 0.5,
        "omega": 1.0,
        "phase": 0.0,
        "radius": 0.2,
    }


def test_linear_defaults_and_optional_phase(tmp_path: Path) -> None:
    linear = _load(
        tmp_path,
        "  - type: linear\n    start: [0.0, 0.0]\n    velocity: [1.0, 0.0]\n    radius: 0.1",
    ).dynamic_obstacles[0]
    assert (linear.period, linear.pingpong) == (0.0, False)  # period 缺省 = 不循环

    circular = _load(
        tmp_path,
        "  - type: circular\n    center: [0.0, 0.0]\n    radius_orbit: 1.0\n"
        "    omega: 1.0\n    radius: 0.2",
    ).dynamic_obstacles[0]
    assert circular.phase == 0.0  # phase 可选，缺省 0


# ═════════════════════════ reset options 透传 ═════════════════════════
def test_reset_options_carry_dynamic_key_only_when_non_empty() -> None:
    scenario = load_scenario("dynamic_crossing")
    dynamic = scenario_to_reset_options(scenario)
    assert sorted(dynamic) == [
        "dynamic_obstacles",
        "initial_heading",
        "obstacles",
        "path_waypoints",
    ]
    assert dynamic["dynamic_obstacles"] == [scenario.dynamic_obstacles[0].to_env_dict()]

    static = scenario_to_reset_options(load_scenario("straight"))
    assert sorted(static) == ["initial_heading", "obstacles", "path_waypoints"]


# ═════════════════════════ 反向：严格校验 ═════════════════════════
def test_unknown_key_suggests_the_closest_name(tmp_path: Path) -> None:
    message = _error(tmp_path, _LINEAR.replace("velocity:", "velocty:"))
    assert "未知键 'velocty'" in message
    assert "did you mean 'velocity'?" in message
    assert "允许的键: type, start, velocity, radius, period, pingpong" in message


def test_unknown_key_from_another_schema_is_rejected(tmp_path: Path) -> None:
    message = _error(tmp_path, _LINEAR + "\n    speed: 0.3")
    assert "未知键 'speed'" in message
    assert "did you mean" in message  # 建议的是最近的合法键，不保证语义正确


def test_unknown_type_lists_the_supported_kinds(tmp_path: Path) -> None:
    message = _error(tmp_path, "  - type: spiral\n    radius: 0.2")
    assert "未知动态障碍类型 'spiral' 支持: linear, circular" in message


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (_LINEAR.replace("radius: 0.250", "radius: -0.250"), "必须 > 0，实际是 -0.25"),
        (_LINEAR.replace("radius: 0.250", "radius: 0.000"), "必须 > 0，实际是 0.0"),
        (
            _LINEAR.replace("velocity: [0.000, 0.300]", "velocity: [.nan, 0.300]"),
            "必须是有限数值",
        ),
        (
            _LINEAR.replace("velocity: [0.000, 0.300]", "velocity: [0.000, .inf]"),
            "必须是有限数值",
        ),
        (
            _LINEAR.replace("start: [3.000, 4.400]", "start: [3.000, 4.400, 5.0]"),
            "必须是 [x, y] 两个数",
        ),
        (
            _LINEAR.replace("period: 20.000", "period: -1.000"),
            "必须 >= 0（0 = 不循环）",
        ),
        (_LINEAR.replace("pingpong: true", 'pingpong: "yes"'), "期望布尔值"),
        (_LINEAR.replace("period: 20.000\n    ", ""), "pingpong 需要 period > 0"),
        (
            (
                "  - type: circular\n    center: [0.0, 0.0]\n    radius_orbit: 1.0\n"
                "    omega: .nan\n    radius: 0.2"
            ),
            "必须是有限数值",
        ),
        (
            "  - type: circular\n    center: [0.0, 0.0]\n    omega: 1.0\n    radius: 0.2",
            "缺少必填键 'radius_orbit'",
        ),
    ],
)
def test_invalid_dynamic_obstacle_is_rejected(
    tmp_path: Path, block: str, expected: str
) -> None:
    assert expected in _error(tmp_path, block)
