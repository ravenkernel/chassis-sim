"""四个基线跑同一批 held-out 场景，把基准表写成一张 Markdown。

    env -u PYTHONPATH python examples/benchmark_baselines.py [out.md]

要点：

* scenarios 与 seeds 都是"批次"参数：同一 (场景, seed) 的地图在四个策略间
  共享（配对比较），地图种子 = seed * episodes + 回合号，可复算。
* 有 planning extra（numba）时用 obstacles_sparse（每回合现场生成地图）；
  core-only 装法则退到 dense_static，并按全局回合号平移整组障碍（同族不同款，
  每个回合真换一张图）—— 两条路都能让 distinct_maps 等于回合数、
  maps_independent=True，success_ci 才有数字。
  换成 straight 这类固定地图场景，表里会明确写 maps_independent=False 并把
  success_ci 置空 —— 这不是 bug，是独立样本闸门。
* 协议要求 >= 3 个 seed；少于 3 个仍出表，但 protocol_ok=False。
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any

from chassis_sim.eval import benchmark
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

SEEDS = (0, 1, 2)
EPISODES = 5
MAX_STEPS = 300
PLANNING_SCENARIO = "obstacles_sparse"
CORE_SCENARIO = "dense_static"


def _planning_available() -> bool:
    try:
        importlib.import_module("numba")
    except ImportError:
        return False
    return True


def _core_only_options(scenario: Any, seed: int) -> Callable[[int], dict[str, Any]]:
    """core-only 退路：按全局回合号（seed * EPISODES + index）平移整组障碍。"""
    base = dict(scenario_to_reset_options(load_scenario(scenario)))
    obstacles = list(base.get("obstacles") or [])
    if not obstacles:  # 没有障碍就换不出图，只能靠调用方换场景
        raise SystemExit(f"[error] 场景 {scenario!r} 没有静态障碍，换不出独立样本")

    def provider(index: int) -> dict[str, Any]:
        shift = 0.15 * (seed * EPISODES + index)
        options = dict(base)
        options["obstacles"] = [
            {
                **item,
                "state": [item["state"][0] + shift, item["state"][1], item["state"][2]],
            }
            for item in obstacles
        ]
        return options

    return provider


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else "benchmark.md"
    if _planning_available():
        scenarios: tuple[str, ...] = (PLANNING_SCENARIO,)
        factory = None
    else:
        scenarios = (CORE_SCENARIO,)
        factory = _core_only_options
        print(f"[info] 没有 numba：改用 core 场景 {CORE_SCENARIO}（逐回合平移障碍）")
    result = benchmark(
        None,  # None = 四个内置基线 zero / random / pp / dwa
        scenarios,
        seeds=SEEDS,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        reset_options_factory=factory,
        out=out,
    )
    print(result.to_markdown())
    print(f"\nprotocol_ok={result.protocol_ok}  markdown={result.markdown_path}")
    for summary in result.summaries:
        ci = "—" if summary.success_ci is None else f"±{summary.success_ci:.3f}"
        print(
            f"  {summary.policy:<7} {summary.scenario:<16} "
            f"n={summary.n:<3} success={summary.success_rate:.1%} {ci} "
            f"distinct_maps={summary.distinct_maps} "
            f"independent={summary.maps_independent}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
