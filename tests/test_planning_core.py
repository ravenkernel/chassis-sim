"""core 安装路径与 cusp 过滤闸门（W5 验收点 2/3）。

三件事，都不需要主仓存在：

1. **无 numba 的 core 安装** —— numba 只挂在 `planning` extra 上，
   `pip install chassis-sim` 出来的核心包里 `import chassis_sim.planning` 必须
   照常工作（慢，但语义不变）。这里用子进程把 `numba` 设成不可 import，
   断言恒等装饰器接手、并且 stage 0 / seed 0 的产物与 golden（有 numba 环境
   下由主仓录的那份）逐位相同 —— 是"退化但等价"，不是"退化且悄悄变了"。
2. **cusp 过滤生效** —— 主仓口径 6 stage × 10 seed = 60 条产出路径，违规 0 条，
   同时 `cusp_rejections` 计数 > 0。后半句是防空转的：过滤若从未触发，
   前半句自动成立却毫无意义。
3. **配置走 package data** —— 换一个 CWD 也能读出包内 curriculum.yaml；
   `MapGenerator` 的构造参数与主仓同名同默认值。

另有一组合成路径：判据必须既能拒绝（短折返 / 窄发卡各自被点名）也能放行，
否则"违规 0 条"可能只是这个函数永远返回 None。
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import chassis_sim
from chassis_sim.planning import (
    MapGenerationError,
    MapGenerator,
    load_curriculum,
    make_map_generator,
)

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "planning_golden.json"
_SRC_ROOT = Path(chassis_sim.__file__).resolve().parent.parent
N_STAGES = 6
N_SEEDS_CUSP = 10

_NO_NUMBA_SCRIPT = """
import hashlib
import json
import sys

sys.modules["numba"] = None          # 之后 import numba 会抛 ImportError
import numpy as np                   # noqa: E402

import chassis_sim.planning as planning          # noqa: E402
from chassis_sim.planning import hybrid_astar    # noqa: E402

numba_loaded = sys.modules.get("numba") is not None
mg = planning.make_map_generator(stage_index=0)
obstacles, path, goal_heading, initial_heading = mg.generate(
    np.random.default_rng(0),
)
print(json.dumps({
    "numba_loaded": numba_loaded,
    "jit_module": hybrid_astar.jit.__module__,
    "n_obstacles": len(obstacles),
    "path_shape": list(path.shape),
    "path_sha256": hashlib.sha256(
        np.ascontiguousarray(path).tobytes()).hexdigest(),
    "goal_heading": repr(float(goal_heading)),
    "initial_heading": repr(float(initial_heading)),
}))
"""


def _golden_case(stage_index: int, seed: int) -> dict:
    data = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if int(case["stage"]) == stage_index and int(case["seed"]) == seed:
            return case
    raise AssertionError(f"golden 里没有 stage {stage_index} seed {seed}")


def _synthetic_path(points) -> np.ndarray:
    """把 (x, y) 序列变成一列 yaw=0 的路径（yaw 保持 0 = 车头始终朝 +x）。"""
    return np.array([[x, y, 0.0] for x, y in points], dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════
# 1. 无 numba 的 core 安装路径
# ═════════════════════════════════════════════════════════════════════


def test_core_install_without_numba(tmp_path: Path) -> None:
    """屏蔽 numba 后仍能 import + 生成，且产物与 golden 逐位相同。"""
    proc = subprocess.run(
        [sys.executable, "-c", _NO_NUMBA_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_SRC_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["numba_loaded"] is False
    assert payload["jit_module"] == "chassis_sim.planning.hybrid_astar", (
        "jit 不是本模块里的恒等装饰器 —— numba 没有被屏蔽掉，这条测试没有意义"
    )
    case = _golden_case(0, 0)
    assert payload["n_obstacles"] == case["n_obstacles"]
    assert payload["path_shape"] == case["path_shape"]
    assert payload["path_sha256"] == case["path_sha256"]
    assert payload["goal_heading"] == case["goal_heading"]
    assert payload["initial_heading"] == case["initial_heading"]


# ═════════════════════════════════════════════════════════════════════
# 2. 配置走 package data + 构造参数与主仓一致
# ═════════════════════════════════════════════════════════════════════


def test_curriculum_read_from_package_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """换 CWD 也能读包内 curriculum.yaml：不依赖仓库根，不 open 主仓路径。"""
    monkeypatch.chdir(tmp_path)
    curriculum = load_curriculum()
    assert [s["name"] for s in curriculum["stages"]] == [
        "obstacles_sparse",
        "obstacles_medium",
        "obstacles_dense",
        "obstacles_dense_reverse",
        "foldback_reverse",
        "maze_grid",
    ]
    assert curriculum["active_stage"] == 0
    ha = curriculum["defaults"]["hybrid_astar"]
    assert ha["min_cusp_fold_arc"] == 0.8
    assert ha["min_cusp_leg_separation"] == 0.35
    assert ha["cusp_min_arc_gap"] == 1.0
    assert make_map_generator().stage_index == curriculum["active_stage"]


def test_map_generator_api_matches_upstream() -> None:
    """构造参数与主仓同名同默认值（既有调用方可以原样搬过来）。"""
    params = inspect.signature(MapGenerator.__init__).parameters
    assert list(params) == [
        "self",
        "stage_config",
        "defaults",
        "stage_index",
        "env_cfg",
    ]
    assert params["defaults"].default is None
    assert params["stage_index"].default == -1
    assert params["env_cfg"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["stage_config"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert (MapGenerator.ROBOT_HALF_L, MapGenerator.ROBOT_HALF_W) == (0.19, 0.10)
    assert MapGenerator.ROBOT_COLLISION_MARGIN == 0.03
    assert MapGenerator.GOAL_HEADING_AVG_WINDOW == 5
    with pytest.raises(IndexError, match="越界"):
        make_map_generator(stage_index=N_STAGES)
    with pytest.raises(ValueError, match="stages"):
        make_map_generator(curriculum={"active_stage": 0, "stages": []})


# ═════════════════════════════════════════════════════════════════════
# 3. cusp 过滤：判据可正可负 + 在真实产出上生效
# ═════════════════════════════════════════════════════════════════════


def test_cusp_criteria_accept_and_reject() -> None:
    """直行放行；短折返（0.4 m）与窄发卡（0.2 m 间距）各自被点名。"""
    mg = make_map_generator(stage_index=0)
    xs = np.arange(0.0, 2.0001, 0.05)
    straight = _synthetic_path([(x, 0.0) for x in xs])
    short_fold = _synthetic_path(
        [(x, 0.0) for x in xs] + [(x, 0.0) for x in np.arange(1.95, 1.55, -0.05)],
    )
    hairpin = _synthetic_path(
        [(x, 0.0) for x in np.arange(0.0, 3.0001, 0.05)]
        + [(3.0, 0.2)]
        + [(x, 0.2) for x in np.arange(2.95, 0.4999, -0.05)],
    )
    assert mg._cusp_violation(straight) is None
    assert "短折返" in (mg._cusp_violation(short_fold) or "")
    assert "窄发卡" in (mg._cusp_violation(hairpin) or "")


def test_cusp_filter_effective_on_generated_paths() -> None:
    """6 stage × 10 seed：产出路径违规 0 条，且过滤确实被触发过。"""
    curriculum = load_curriculum()
    checked = 0
    rejections = 0
    errors = 0
    violating: list[tuple[int, int, str]] = []
    stages_hit: set[int] = set()
    for stage_index in range(N_STAGES):
        for seed in range(N_SEEDS_CUSP):
            mg = make_map_generator(curriculum=curriculum, stage_index=stage_index)
            try:
                _, path, _, _ = mg.generate(np.random.default_rng(seed))
            except MapGenerationError:
                errors += 1
                continue
            checked += 1
            stages_hit.add(stage_index)
            rejections += mg.cusp_rejections
            reason = mg._cusp_violation(path)
            if reason is not None:
                violating.append((stage_index, seed, reason))
    assert stages_hit == set(range(N_STAGES))
    assert checked >= 55, f"只产出 {checked} 条路径（errors={errors}），样本太少"
    assert violating == []
    assert rejections > 0, "cusp 过滤一次都没触发 —— '违规 0 条' 就成了空转"
