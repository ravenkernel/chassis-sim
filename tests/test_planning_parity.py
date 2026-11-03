"""规划器对拍 golden：chassis_sim.planning ≡ 主仓 drl-sim/envs（W5 验收点 1）。

两件事：

(i) **生成阶段**（`/home/zhao/drl-sim` 存在时）—— 用**本仓**
    `chassis_sim.config` package data 里的 curriculum.yaml（同一份 stage_config /
    defaults），把主仓 `envs.map_generator.MapGenerator` 在 6 stage × 20 seed 上的
    `generate()` 观测录进 `tests/data/planning_golden.json`：
    obstacles（结构与其中全部 float）、path（float64 全部字节 + 首尾/三分点 ≤6 行
    抽样）、goal_heading / initial_heading（repr 精确回读）、cusp_rejections。
    重录命令：`env PYTHONPATH=src python tests/test_planning_parity.py`

(ii) **断言阶段** —— 只依赖那份 JSON，不需要主仓存在（CI 就没有）：
    同 stage 同 seed 重放本仓 `MapGenerator`：结构类字段全等，float 按容差比。

为什么 golden 由主仓录：拿本仓自己录的 golden 自证一致是同义反复；这份 JSON 是
外部参照物，本仓实现只是被比较的一方。obstacles / path 这种大数组按
`dtype + shape + sha256(tobytes())` 比，比逐元素 assert 更早失败、更可读。

**默认容差 + 可选严格**：sha256 只能整块相等或不等 —— golden 是主仓在本机
（numpy 1.x + 特定 CPU）录的，换 numpy 版本 / SIMD 实现后字节不再逐位复现，
但数学等价，那不该红。所以默认档比 `path_samples`（可做容差的最小参照物），
位级 sha256 只在 `CHASSIS_STRICT_GOLDEN=1` 的严格档里断言。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.planning import MapGenerator, load_curriculum

_DRL_SIM = "/home/zhao/drl-sim"
_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "planning_golden.json"
N_STAGES = 6
N_SEEDS = 20


STRICT_ENV = "CHASSIS_STRICT_GOLDEN"
_FLOAT_TOL = 1e-9


def _strict_mode() -> bool:
    """严格档：位级只在录制环境（同一 numpy / CPU）内保证，CI 默认走容差档。"""
    return os.environ.get(STRICT_ENV) == "1"


def _f(value: float) -> str:
    """float 的精确文本形式（repr 可无损回读，不用 %.17g 之类近似格式）。"""
    return repr(float(value))


def _sample_rows(path: np.ndarray) -> list[list[str]]:
    """路径首/尾与三分点共 ≤6 行（索引去重升序），每行 3 个 float 走 repr 编码。

    sha256 只有整块相等与不等两种结局，ulp 级差异会让它整块变红；抽样行是可做
    容差的最小参照物，还能把偏差量化成数值（`path_sha256` 仍留给严格档）。
    """
    arr = np.ascontiguousarray(path)
    n = int(arr.shape[0])
    idxs = sorted(
        {i for i in (0, 1, n // 3, 2 * n // 3, n - 2, n - 1) if 0 <= i < n},
    )
    return [[_f(v) for v in arr[i]] for i in idxs]


def _rows(samples: list[list[str]]) -> np.ndarray:
    """golden 里的 repr 字符串行 → float 数组（容差比较用）。"""
    return np.array([[float(v) for v in row] for row in samples], dtype=np.float64)


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
        "path_samples": _sample_rows(path),
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
            "recorded_env": {
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "tolerance": (
                "默认档（CI）：path_samples 用 np.allclose(atol=1e-9, rtol=0)，"
                "goal_heading / initial_heading 用 abs(a-b)<=1e-9；"
                f"严格档（{STRICT_ENV}=1）才额外断言 path_sha256 / "
                "obstacles_sha256 精确相等（位级只在录制环境内保证）"
            ),
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


# 结构类字段：与 numpy / CPU 无关，必须精确相等。
_EXACT_FIELDS = (
    "status",
    "error_type",
    "error_message",
    "n_obstacles",
    "path_dtype",
    "path_shape",
    "cusp_rejections",
)


def _mismatches(want: dict, got: dict) -> list[str]:
    """默认档的逐字段比对；返回人读的不一致清单（空 = 通过）。

    float 只承诺在录制环境内逐位，跨 numpy / SIMD 实现是 ulp 级差异，
    因此 heading 与 path_samples 走 1e-9 容差，sha256 只留给严格档。
    """
    bad: list[str] = []
    for key in _EXACT_FIELDS:
        if want.get(key) != got.get(key):
            bad.append(f"  {key}: 主仓 {want.get(key)!r} vs 本仓 {got.get(key)!r}")
    if want.get("status") != "ok":
        return bad

    for key in ("goal_heading", "initial_heading"):
        if abs(float(got[key]) - float(want[key])) > _FLOAT_TOL:
            bad.append(
                f"  {key}: 主仓 {want[key]} vs 本仓 {got[key]}"
                f"（容差 abs<={_FLOAT_TOL:g}）",
            )

    got_rows = _rows(got.get("path_samples") or [])
    want_rows = _rows(want.get("path_samples") or [])
    if got_rows.shape != want_rows.shape:
        bad.append(f"  path_samples: 形状 {got_rows.shape} vs {want_rows.shape}")
    elif not np.allclose(got_rows, want_rows, atol=_FLOAT_TOL, rtol=0.0):
        worst = float(np.max(np.abs(got_rows - want_rows)))
        bad.append(
            f"  path_samples: 最大偏差 {worst:.3e} 超过 atol={_FLOAT_TOL:g}"
            f"（主仓 {want['path_samples']} vs 本仓 {got['path_samples']}）",
        )

    if _strict_mode():
        for key in ("path_sha256", "obstacles_sha256"):
            if want.get(key) != got.get(key):
                bad.append(
                    f"  {key}: 主仓 {want.get(key)!r} vs 本仓 {got.get(key)!r}"
                    f"（{STRICT_ENV}=1 的位级断言只在本录制环境内保证，"
                    "跨 numpy / CPU 不保证）",
                )
    return bad


@pytest.mark.parametrize(
    "case",
    _GOLDEN["cases"],
    ids=[f"s{c['stage']}-seed{c['seed']}" for c in _GOLDEN["cases"]],
)
def test_generate_matches_upstream_golden(case: dict) -> None:
    """同 stage 同 seed：结构精确，heading / path_samples 按 1e-9 容差。

    位级 sha256 只在 CHASSIS_STRICT_GOLDEN=1 时断言：golden 的字节是主仓在本机
    （numpy 1.x + 特定 CPU）录的，跨 numpy 版本 / SIMD 实现不再逐位复现 ——
    数学等价却 ulp 不同，不是移植错了。
    """
    got = _observe(MapGenerator, _CURRICULUM, int(case["stage"]), int(case["seed"]))
    want = {k: v for k, v in case.items() if k not in ("stage", "seed")}
    bad = _mismatches(want, got)
    assert not bad, "\n".join(
        [f"stage {case['stage']} seed {case['seed']} 与主仓 golden 不一致", *bad],
    )


def test_golden_covers_every_stage_and_seed() -> None:
    """golden 本身得是立体的：6 stage × 20 seed 一格不缺。"""
    assert _GOLDEN["cases"], (
        f"golden 缺失：{_GOLDEN_PATH}（跑 tests/test_planning_parity.py 重录）"
    )
    pairs = {(int(c["stage"]), int(c["seed"])) for c in _GOLDEN["cases"]}
    assert pairs == {(s, seed) for s in range(N_STAGES) for seed in range(N_SEEDS)}
    assert _GOLDEN["meta"]["stages"] == N_STAGES
    assert _GOLDEN["meta"]["seeds"] == N_SEEDS


def test_golden_carries_tolerance_samples() -> None:
    """每个 ok case 都要带可比容差的 path_samples（3..6 行）。

    缺了它默认档就只能退回比 sha256，跨 numpy / CPU 必红。
    """
    bad = [
        (c["stage"], c["seed"])
        for c in _GOLDEN["cases"]
        if c["status"] == "ok" and not 3 <= len(c.get("path_samples") or []) <= 6
    ]
    assert not bad, f"这些 case 缺 path_samples 或行数不在 3..6：{bad}"
    assert set(_GOLDEN["meta"]["recorded_env"]) == {"numpy", "python", "platform"}
    assert "path_samples" in _GOLDEN["meta"]["tolerance"]


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
