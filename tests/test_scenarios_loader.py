"""chassis_sim.scenarios 的 YAML schema、严格校验与数据文件契约测试。

覆盖：list/load 的基本行为、未知键 did-you-mean、缺必填键、类型/顺序/半径错误、
动态障碍的解析与严格校验、requires ⇔ path.type 的跨字段规则，以及 11 个
内置 YAML 的几何自洽性（corridor 净宽、circle 半径、dense_static 可通行缝）。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from chassis_sim.scenarios import (
    DATA_DIR,
    ScenarioNotFoundError,
    ScenarioValidationError,
    curriculum_stages,
    list_scenarios,
    load_curriculum,
    load_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_SCENES = (
    "straight",
    "circle",
    "corridor",
    "corridor_medium",
    "dense_static",
    "dynamic_crossing",
)
PLANNING_SCENES = (
    "obstacles_sparse",
    "obstacles_medium",
    "obstacles_dense",
    "obstacles_dense_reverse",
    "foldback_reverse",
    "maze_grid",
)


# ═════════════════════════ 夹具 ═════════════════════════
def _baseline() -> dict:
    """一份合法的最小场景（各项检查的对照基准）。"""
    return {
        "name": "fixture",
        "description": "测试夹具",
        "world_bounds": [0.0, 10.0, 0.0, 10.0],
        "obstacles": [],
        "path": {
            "type": "straight",
            "params": {"start": [1.0, 5.0], "goal": [5.0, 5.0], "spacing": 0.05},
        },
        "seed": 0,
        "requires": None,
    }


def _dump(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "fixture.yaml"
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


# ═════════════════════════ 列表与加载 ═════════════════════════
def test_list_scenarios_returns_twelve_sorted_names() -> None:
    names = list_scenarios()
    assert len(names) == 12
    assert names == sorted(names)
    assert set(names) == set(CORE_SCENES) | set(PLANNING_SCENES)


@pytest.mark.parametrize("name", list_scenarios())
def test_every_scenario_loads_with_expected_metadata(name: str) -> None:
    scenario = load_scenario(name)
    assert scenario.name == name
    assert scenario.description
    assert scenario.source == DATA_DIR / f"{name}.yaml"
    assert scenario.world_bounds == (0.0, 10.0, 0.0, 10.0)
    if name == "dynamic_crossing":
        assert scenario.dynamic_obstacles  # 唯一带动态障碍的内置场景
    else:
        assert scenario.dynamic_obstacles == ()


def test_load_accepts_path_and_bare_filename(tmp_path: Path) -> None:
    path = _dump(tmp_path, _baseline())
    assert load_scenario(path).name == "fixture"
    assert load_scenario(str(path)).name == "fixture"
    # data/ 目录里的场景也可以用裸文件名加载
    assert load_scenario("corridor.yaml").name == "corridor"


def test_unknown_scenario_name_lists_available() -> None:
    with pytest.raises(ScenarioNotFoundError) as excinfo:
        load_scenario("no_such_scene")
    message = str(excinfo.value)
    assert "no_such_scene" in message
    for name in list_scenarios():
        assert name in message


def test_missing_scenario_file_lists_available(tmp_path: Path) -> None:
    with pytest.raises(ScenarioNotFoundError) as excinfo:
        load_scenario(tmp_path / "nope.yaml")
    assert "nope.yaml" in str(excinfo.value)
    assert "corridor" in str(excinfo.value)


# ═════════════════════════ 严格键校验 ═════════════════════════
def test_unknown_key_reports_path_and_did_you_mean(tmp_path: Path) -> None:
    data = _baseline()
    data["descrition"] = "typo"
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "未知键 'descrition'" in message
    assert "did you mean 'description'" in message


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    data = _baseline()
    data["path"]["params"]["spacings"] = 0.05
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "path.params" in message
    assert "未知键 'spacings'" in message
    assert "did you mean 'spacing'" in message


def test_missing_required_key_names_key_and_file(tmp_path: Path) -> None:
    data = _baseline()
    del data["seed"]
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "缺少必填键 'seed'" in message


def test_wrong_type_for_seed_is_rejected(tmp_path: Path) -> None:
    data = _baseline()
    data["seed"] = "zero"
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "seed" in message
    assert "整数" in message


def test_world_bounds_wrong_length_is_rejected(tmp_path: Path) -> None:
    data = _baseline()
    data["world_bounds"] = [0.0, 10.0, 0.0]
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "world_bounds" in message
    assert "[xmin, xmax, ymin, ymax]" in message


def test_world_bounds_wrong_order_is_rejected(tmp_path: Path) -> None:
    data = _baseline()
    data["world_bounds"] = [0.0, 0.0, 10.0, 10.0]
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "world_bounds" in message
    assert "xmin < xmax" in message


@pytest.mark.parametrize("radius", [0.0, -0.5])
def test_non_positive_circle_radius_is_rejected(tmp_path: Path, radius: float) -> None:
    data = _baseline()
    data["obstacles"] = [{"type": "circle", "x": 5.0, "y": 5.0, "radius": radius}]
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "obstacles[0].radius" in message
    assert "必须 > 0" in message


def test_non_positive_rectangle_size_is_rejected(tmp_path: Path) -> None:
    data = _baseline()
    data["obstacles"] = [
        {"type": "rectangle", "x": 5.0, "y": 5.0, "length": 1.0, "width": 0.0}
    ]
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    assert "obstacles[0].width" in str(excinfo.value)


def test_unknown_obstacle_kind_suggests_close_match(tmp_path: Path) -> None:
    data = _baseline()
    data["obstacles"] = [{"type": "circl", "x": 5.0, "y": 5.0, "radius": 0.3}]
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "未知障碍类型 'circl'" in message
    assert "did you mean 'circle'" in message


def test_waypoints_with_mixed_columns_are_rejected(tmp_path: Path) -> None:
    data = _baseline()
    data["path"] = {
        "type": "waypoints",
        "params": {"points": [[1.0, 5.0], [3.0, 5.0, 0.0]], "spacing": 0.05},
    }
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    assert "混合列数" in str(excinfo.value)


def test_unknown_stage_name_suggests_close_match(tmp_path: Path) -> None:
    data = _baseline()
    data["path"] = {"type": "stage", "params": {"stage": "obstacles_sparce"}}
    data["requires"] = "planning"
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "未知课程阶段 'obstacles_sparce'" in message
    assert "did you mean 'obstacles_sparse'" in message


def test_dynamic_obstacles_are_parsed_into_typed_objects(tmp_path: Path) -> None:
    data = _baseline()
    data["dynamic_obstacles"] = [
        {"type": "linear", "start": [2.0, 2.0], "velocity": [0.0, 0.3], "radius": 0.2}
    ]
    path = _dump(tmp_path, data)
    scenario = load_scenario(path)
    assert len(scenario.dynamic_obstacles) == 1
    dyn = scenario.dynamic_obstacles[0]
    assert dyn.kind == "linear"
    assert dyn.start == (2.0, 2.0)
    assert dyn.velocity == (0.0, 0.3)
    assert dyn.radius == 0.2
    assert dyn.period == 0.0
    assert dyn.pingpong is False


def test_requires_planning_needs_stage_path(tmp_path: Path) -> None:
    data = _baseline()
    data["requires"] = "planning"
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "requires" in message
    assert "path.type == 'stage'" in message


def test_stage_path_needs_requires_planning(tmp_path: Path) -> None:
    data = _baseline()
    data["path"] = {"type": "stage", "params": {"stage": "obstacles_sparse"}}
    path = _dump(tmp_path, data)
    with pytest.raises(ScenarioValidationError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "path.type" in message
    assert "requires: planning" in message


def test_name_must_match_file_stem_in_data_dir(tmp_path: Path) -> None:
    data = _baseline()
    data["name"] = "renamed"
    path = tmp_path / "fixture.yaml"
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    # 数据目录外的文件不要求同名，data/ 内的文件才要求
    assert load_scenario(path).name == "renamed"
    original = (DATA_DIR / "straight.yaml").read_text(encoding="utf-8")
    try:
        (DATA_DIR / "straight.yaml").write_text(
            original.replace("name: straight", "name: renamed", 1), encoding="utf-8"
        )
        with pytest.raises(ScenarioValidationError) as excinfo:
            load_scenario("straight")
        assert "文件名 stem 'straight'" in str(excinfo.value)
    finally:
        (DATA_DIR / "straight.yaml").write_text(original, encoding="utf-8")
    assert load_scenario("straight").name == "straight"


# ═════════════════════════ 数据文件自洽性 ═════════════════════════
def _segments(
    points: list[list[float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        ((points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]))
        for i in range(len(points) - 1)
    ]


def _segment_distance(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def test_straight_scene_spans_1_to_5_metres() -> None:
    scenario = load_scenario("straight")
    assert scenario.obstacles == ()
    assert scenario.path.type == "straight"
    assert scenario.path.params["start"] == [1.0, 5.0]
    assert scenario.path.params["goal"] == [5.0, 5.0]


def test_circle_scene_lies_on_radius_1p5_circle() -> None:
    scenario = load_scenario("circle")
    points = scenario.path.params["points"]
    assert scenario.obstacles == ()
    assert scenario.path.type == "waypoints"
    assert len(points) >= 20
    for x, y in points:
        assert math.hypot(x - 5.0, y - 5.0) == pytest.approx(1.5, abs=2e-3)
    # 留缺口：首末点不能重合（否则 reset 当步就判到达）
    assert math.dist(points[0], points[-1]) > 0.5


def test_corridor_scene_net_width_is_0p6_metres() -> None:
    scenario = load_scenario("corridor")
    assert len(scenario.obstacles) == 2
    assert {obstacle.kind for obstacle in scenario.obstacles} == {"rectangle"}
    lower, upper = sorted(scenario.obstacles, key=lambda obstacle: obstacle.y)
    gap = (upper.y - upper.width / 2.0) - (lower.y + lower.width / 2.0)
    assert gap == pytest.approx(0.6, abs=1e-9)
    # 两条矩形足够长（覆盖 x ∈ [0.5, 6.5]）
    for obstacle in scenario.obstacles:
        assert obstacle.length >= 6.0


def test_corridor_medium_scene_net_width_is_0p8_metres() -> None:
    """中间档只搬墙位：同样是两堵长墙，净宽 0.8 m。"""
    scenario = load_scenario("corridor_medium")
    assert len(scenario.obstacles) == 2
    lower, upper = sorted(scenario.obstacles, key=lambda obstacle: obstacle.y)
    gap = (upper.y - upper.width / 2.0) - (lower.y + lower.width / 2.0)
    assert gap == pytest.approx(0.8, abs=1e-9)
    for obstacle in scenario.obstacles:
        assert obstacle.length >= 6.0
    assert scenario.path.params["start"] == [1.0, 5.0]
    assert scenario.path.params["goal"] == [8.0, 5.0]


def test_dense_static_scene_is_dense_but_passable() -> None:
    scenario = load_scenario("dense_static")
    assert len(scenario.obstacles) >= 12
    points = scenario.path.params["points"]
    segments = _segments(points)
    for obstacle in scenario.obstacles:
        if obstacle.kind == "circle":
            half_extent = obstacle.radius
        else:
            half_extent = 0.5 * math.hypot(obstacle.length, obstacle.width)
        clearance = (
            min(_segment_distance((obstacle.x, obstacle.y), a, b) for a, b in segments)
            - half_extent
        )
        assert clearance >= 0.30, f"{obstacle} 距参考路径只有 {clearance:.3f} m"


def test_planning_scenes_match_curriculum_stages() -> None:
    stages = [str(stage["name"]) for stage in curriculum_stages(load_curriculum())]
    assert stages == list(PLANNING_SCENES)
    for name in PLANNING_SCENES:
        scenario = load_scenario(name)
        assert scenario.requires == "planning"
        assert scenario.path.type == "stage"
        assert scenario.path.stage_name == name
        assert scenario.obstacles == ()
        assert name in stages


def test_package_data_declares_scenario_yaml() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "scenarios/data/*.yaml" in pyproject
    assert len(list(DATA_DIR.glob("*.yaml"))) == 12
