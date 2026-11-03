"""规划器对拍 golden：chassis_sim.planning ≡ 主仓 drl-sim/envs（W5 验收点 1）。

两件事：

(i) **生成阶段**（`/home/zhao/drl-sim` 存在时）—— 用**本仓**
    `chassis_sim.config` package data 里的 curriculum.yaml（同一份 stage_config /
    defaults），把主仓 `envs.map_generator.MapGenerator` 在 6 stage × 20 seed 上的
    `generate()` 观测逐位录进 `tests/data/planning_golden.json`：
    obstacles（结构与其中全部 float）、path（float64 全部字节）、goal_heading /
    initial_heading（repr 精确回读）、cusp_rejections。
    重录命令：`env PYTHONPATH=src python tests/test_planning_parity.py`

(ii) **断言阶段** —— 只依赖那份 JSON，不需要主仓存在（CI 就没有）：
    同 stage 同 seed 重放本仓 `MapGenerator`，逐字段全等。

为什么 golden 由主仓录：拿本仓自己录的 golden 自证一致是同义反复；这份 JSON 是
外部参照物，本仓实现只是被比较的一方。obstacles / path 这种大数组按
`dtype + shape + sha256(tobytes())` 比，比逐元素 assert 更早失败、更可读。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.planning import MapGenerator, load_curriculum

_DRL_SIM = "/home/zhao/drl-sim"
_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "planning_golden.json"
N_STAGES = 6
N_SEEDS = 20


def _f(value: float) -> str:
    """float 的精确文本形式（repr 可无损回读，不用 %.17g 之类近似格式）。"""
    return repr(float(value))


def _canon(obj):
    """把观测值折成可 JSON 比较的规范形式：float 一律走 repr，不比较格式化。"""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return [
            "ndarray",
            str(arr.dtype),
            list(arr.shape),
            hashlib.sha256(arr.tobytes()).hexdigest(),
        ]
    if isinstance(obj, np.generic):
        return ["np.generic", str(obj.dtype), _f(obj)]
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, float):
        return ["float", _f(obj)]
    if isinstance(obj, (str, int)) or obj is None:
        return obj
    return ["repr", repr(obj)]


def _digest(obj) -> str:
    """规范形式的 sha256：结构与每个 float 都被纳入，改一位就换一个值。"""
    payload = json.dumps(
        _canon(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observe(gen_cls, curriculum: dict, stage_index: int, seed: int) -> dict:
    """跑一次 generate()，返回可 JSON 化的观测摘要（与 golden 的 case 同构）。"""
    mg = gen_cls(
        curriculum["stages"][stage_index],
        curriculum["defaults"],
        stage_index=stage_index,
    )
    try:
        obstacles, path, goal_heading, initial_heading = mg.generate(
            np.random.default_rng(seed),
        )
    except Exception as exc:  # noqa: BLE001 - 该 stage/seed 上该抛错就原样记录
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    canon_path = _canon(path)
    return {
        "status": "ok",
        "n_obstacles": len(obstacles),
        "obstacles_sha256": _digest(obstacles),
        "path_dtype": canon_path[1],
        "path_shape": canon_path[2],
        "path_sha256": canon_path[3],
        "goal_heading": _f(goal_heading),
        "initial_heading": _f(initial_heading),
        "cusp_rejections": int(mg.cusp_rejections),
    }


# ═════════════════════════════════════════════════════════════════════
# 生成阶段（只在有主仓的机器上跑；pytest 不走这里）
# ═════════════════════════════════════════════════════════════════════


def _record() -> dict:
    sys.path.insert(0, _DRL_SIM)
    from envs.map_generator import MapGenerator as UpstreamMapGenerator

    curriculum = load_curriculum()
    cases = []
    for stage_index in range(N_STAGES):
        for seed in range(N_SEEDS):
            rec = _observe(UpstreamMapGenerator, curriculum, stage_index, seed)
            cases.append({"stage": stage_index, "seed": seed, **rec})
    return {
        "meta": {
            "title": "MapGenerator 全量对拍（W5-1）：主仓 envs/map_generator.py → 本仓 planning",
            "reference": "envs/map_generator.py + envs/hybrid_astar.py + envs/rs_curve.py",
            "reference_root": _DRL_SIM,
            "curriculum": "chassis_sim.config:curriculum.yaml（importlib.resources 读）",
            "stages": N_STAGES,
            "seeds": N_SEEDS,
            "float_encoding": "python repr(float) 字符串，float() 精确回读",
            "array_encoding": "dtype + shape + sha256(ascontiguousarray(a).tobytes())",
            "obstacles_encoding": "规范 JSON（键排序，float 走 repr）的 sha256",
            "recorder": "PYTHONPATH=src python tests/test_planning_parity.py",
        },
        "cases": cases,
    }


def main() -> int:
    if not os.path.isdir(_DRL_SIM):
        print(f"无法录制：{_DRL_SIM} 不存在（golden 只在有主仓的机器上生成）")
        return 1
    data = _record()
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    n_ok = sum(1 for c in data["cases"] if c["status"] == "ok")
    print(
        f"已写入 {_GOLDEN_PATH}（{_GOLDEN_PATH.stat().st_size} 字节，"
        f"{len(data['cases'])} cases / ok {n_ok} / error {len(data['cases']) - n_ok}）",
    )
    return 0


# ═════════════════════════════════════════════════════════════════════
# 断言阶段（只依赖 JSON）
# ═════════════════════════════════════════════════════════════════════


def _load_golden() -> dict:
    """读 golden；缺失时给一份空的（录制模式下文件正是本进程写出来的）。"""
    if not _GOLDEN_PATH.exists():
        return {"meta": {}, "cases": []}
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


_GOLDEN = _load_golden()
_CURRICULUM = load_curriculum()


def _explain(case: dict, got: dict, want: dict) -> str:
    lines = [f"stage {case['stage']} seed {case['seed']} 与主仓 golden 不一致"]
    for key in sorted(set(got) | set(want)):
        if got.get(key) != want.get(key):
            lines.append(f"  {key}: 主仓 {want.get(key)!r} vs 本仓 {got.get(key)!r}")
    return "\n".join(lines)


@pytest.mark.parametrize(
    "case",
    _GOLDEN["cases"],
    ids=[f"s{c['stage']}-seed{c['seed']}" for c in _GOLDEN["cases"]],
)
def test_generate_matches_upstream_golden(case: dict) -> None:
    """同 stage 同 seed：obstacles / path / 两个 heading / cusp_rejections 全等。"""
    got = _observe(MapGenerator, _CURRICULUM, int(case["stage"]), int(case["seed"]))
    want = {k: v for k, v in case.items() if k not in ("stage", "seed")}
    assert got == want, _explain(case, got, want)


def test_golden_covers_every_stage_and_seed() -> None:
    """golden 本身得是立体的：6 stage × 20 seed 一格不缺。"""
    assert _GOLDEN["cases"], (
        f"golden 缺失：{_GOLDEN_PATH}（跑 tests/test_planning_parity.py 重录）"
    )
    pairs = {(int(c["stage"]), int(c["seed"])) for c in _GOLDEN["cases"]}
    assert pairs == {(s, seed) for s in range(N_STAGES) for seed in range(N_SEEDS)}
    assert _GOLDEN["meta"]["stages"] == N_STAGES
    assert _GOLDEN["meta"]["seeds"] == N_SEEDS


def test_golden_is_not_a_vacuous_fixture() -> None:
    """防空转闸门：golden 里 ok / error 两种状态、以及 cusp 真的触发过，都要出现。

    否则"全等"可能只是两边都什么都不产：
    * 全是 error ⇒ 对拍只比了异常文本；
    * cusp_rejections 全 0 ⇒ 折返过滤在 golden 的 seed 上根本没被激活，
      验收点 2 的"过滤生效"就没有参照。
    """
    statuses = {c["status"] for c in _GOLDEN["cases"]}
    assert statuses == {"ok", "error"}
    rejections = [int(c.get("cusp_rejections", 0)) for c in _GOLDEN["cases"]]
    assert sum(rejections) > 0
    assert any(r > 0 for r in rejections)


if __name__ == "__main__":
    raise SystemExit(main())
