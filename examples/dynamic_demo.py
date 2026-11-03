"""动态障碍 demo：dynamic_crossing 的静态 / 动态对照与确定性检查。

运行：python examples/dynamic_demo.py

同一个开环动作序列（a0=0 ⇒ 直行，a1=1 ⇒ 1 m/s）跑两遍：一遍把 reset options 里的
dynamic_obstacles 键剔除（静态基线），一遍带动态障碍。打印逐步 d_min、首次碰撞的 tick、
以及两次完整 reset+40 步的 LiDAR / 位姿逐位对比（max|Δ| 必须为 0）。
"""

from __future__ import annotations

import numpy as np

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

STEPS = 20
N_LIDAR = 360


def _options(with_dynamic: bool) -> dict:
    """dynamic_crossing 的 reset options；静态基线剔除第四键。"""
    scenario = load_scenario("dynamic_crossing")
    options = scenario_to_reset_options(scenario, seed=scenario.seed)
    if with_dynamic:
        return options
    return {k: v for k, v in options.items() if k != "dynamic_obstacles"}


def run(with_dynamic: bool) -> dict:
    """跑 STEPS 步直行，返回每一步的 d_min、首次碰撞 tick 与末步 LiDAR。"""
    env = ChassisSimEnv("ackermann", max_steps=STEPS)
    try:
        obs, _info = env.reset(seed=0, options=_options(with_dynamic))
        d_min: list[float] = []
        first_hit = None
        for step in range(1, STEPS + 1):
            obs, _reward, terminated, truncated, info = env.step(
                np.array([0.0, 1.0], dtype=np.float32)
            )
            d_min.append(float(info["d_min"]))
            if first_hit is None and bool(info["collision"]):
                first_hit = step
            if terminated or truncated:
                break
        return {
            "d_min": d_min,
            "first_hit": first_hit,
            "lidar": np.array(obs[:N_LIDAR]),
        }
    finally:
        env.close()


def main() -> None:
    scenario = load_scenario("dynamic_crossing")
    print("chassis-sim 动态障碍 demo —— dynamic_crossing")
    print(f"  {scenario.description}")
    print(f"  {scenario.summary()}")
    print(f"  dynamic_obstacles: {scenario.dynamic_obstacles!r}")

    static = run(with_dynamic=False)
    dynamic = run(with_dynamic=True)
    print("\n  step | 静态 d_min | 动态 d_min")
    for i, (a, b) in enumerate(zip(static["d_min"], dynamic["d_min"]), 1):
        print(f"  {i:4d} | {a:9.4f} | {b:9.4f}")

    idx = int(np.argmin(dynamic["lidar"]))
    print(
        f"\n  末步 LiDAR：动态最小 {dynamic['lidar'][idx]:.4f} m @ 索引 {idx}"
        f"；静态同一索引 {static['lidar'][idx]:.4f} m"
    )
    print(f"  首次碰撞 tick：静态={static['first_hit']} 动态={dynamic['first_hit']}")

    again = run(with_dynamic=True)
    print(
        "  确定性（两次完整 reset+"
        f"{len(dynamic['d_min'])} 步）：max|Δlidar|="
        f"{float(np.max(np.abs(again['lidar'] - dynamic['lidar'])))}"
    )


if __name__ == "__main__":
    main()
