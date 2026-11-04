"""动态障碍 demo：dynamic_crossing 的静态 / 动态对照与确定性检查。

运行（仓库根目录）：python examples/dynamic_demo.py

1.0.0 的规矩：默认环境**不装配任何雷达**。不挂传感器时 info["d_min"] 恒等于
bridge.max_range，动态/静态对照会退化成两条常数线。所以这里显式挂一支
Lidar("lidar")，读数从 info["lidar"] 取 —— info 的传感器键就是传感器名，
键下是 {"scan", "scan_meta"}。

同一个开环动作序列（a0=0 ⇒ 直行，a1=1 ⇒ 1 m/s）跑两遍：一遍把 reset options
里的 dynamic_obstacles 键剔除（静态基线），一遍带动态障碍。打印逐步 d_min、首次
碰撞的 tick，以及两次完整 reset+STEPS 步的扫描 / 观测逐位对比。
"""

from __future__ import annotations

import numpy as np

import chassis_sim as cs
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

STEPS = 20
SENSOR = "lidar"
SENSORS = (cs.Lidar(SENSOR, mount=(0.145, 0, 0), n_rays=180, fov=np.pi, max_range=5.0),)


def _options(with_dynamic: bool) -> dict:
    """dynamic_crossing 的 reset options；静态基线剔除第四键。"""
    scenario = load_scenario("dynamic_crossing")
    options = scenario_to_reset_options(scenario, seed=scenario.seed)
    if with_dynamic:
        return options
    return {k: v for k, v in options.items() if k != "dynamic_obstacles"}


def run(with_dynamic: bool) -> dict:
    """跑 STEPS 步直行，返回逐步 d_min、首次碰撞 tick、末步 obs 与扫描。"""
    env = cs.make("ackermann", max_steps=STEPS, sensors=SENSORS, seed=0)
    try:
        env.reset(seed=0, options=_options(with_dynamic))
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
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "scan": np.asarray(info[SENSOR]["scan"], dtype=np.float32).copy(),
        }
    finally:
        env.close()


def main() -> None:
    scenario = load_scenario("dynamic_crossing")
    print("chassis-sim 动态障碍 demo —— dynamic_crossing")
    print(f"  {scenario.description}")
    print(f"  {scenario.summary()}")
    print(f"  传感器: {[spec.name for spec in SENSORS]}（默认环境一支都不装）")
    print(f"  dynamic_obstacles: {scenario.dynamic_obstacles!r}")

    static = run(with_dynamic=False)
    dynamic = run(with_dynamic=True)
    print("\n  step | 静态 d_min | 动态 d_min")
    for i, (a, b) in enumerate(zip(static["d_min"], dynamic["d_min"]), 1):
        print(f"  {i:4d} | {a:9.4f} | {b:9.4f}")

    idx = int(np.argmin(dynamic["scan"]))
    print(
        f"\n  末步 {SENSOR} 扫描：动态最小 {dynamic['scan'][idx]:.4f} m @ 索引 {idx}"
        f"；静态同一索引 {static['scan'][idx]:.4f} m"
    )
    print(f"  首次碰撞 tick：静态={static['first_hit']} 动态={dynamic['first_hit']}")

    again = run(with_dynamic=True)
    diff_scan = float(np.max(np.abs(again["scan"] - dynamic["scan"])))
    diff_obs = float(np.max(np.abs(again["obs"] - dynamic["obs"])))
    print(
        f"  确定性（两次完整 reset+{len(dynamic['d_min'])} 步）："
        f"max|Δscan|={diff_scan} max|Δobs|={diff_obs}"
    )


if __name__ == "__main__":
    main()
