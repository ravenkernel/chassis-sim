"""场景集 demo：用 chassis_sim.scenarios 直接驱动 ChassisSimEnv。

运行：python examples/scenario_demo.py

对 straight / corridor / dense_static 三个 core 场景（不需要 numba）各建一个 env、
各跑 50 步随机动作；回合被碰撞/超时终止时立刻 reset 并继续，保证每个场景都跑满
50 步，从而能稳定观察到 obs 形状与 info 关键项。
"""

from __future__ import annotations

import numpy as np

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

SCENARIOS = ("straight", "corridor", "dense_static")
STEPS = 50
INFO_KEYS = (
    "step",
    "collision",
    "goal_reached",
    "cte",
    "d_min",
    "progress",
    "termination_reason",
)


def _fmt_info(info: dict) -> str:
    """只挑关键项：数值保留 3 位小数，其余原样打印。"""
    parts = []
    for key in INFO_KEYS:
        if key not in info:
            continue
        value = info[key]
        parts.append(
            f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
        )
    return "  ".join(parts)


def run(name: str, rng: np.random.Generator) -> None:
    """加载一个 core 场景，跑 STEPS 步随机动作并打印统计。"""
    scenario = load_scenario(name)
    options = scenario_to_reset_options(scenario, seed=scenario.seed)
    env = ChassisSimEnv("ackermann", max_steps=STEPS * 3)
    try:
        obs, info = env.reset(seed=scenario.seed, options=options)
        print(f"\n=== {name} ===")
        print(f"  {scenario.description}")
        print(f"  {scenario.summary()}")
        print(
            f"  reset: path_waypoints={options['path_waypoints'].shape}"
            f" initial_heading={options['initial_heading']:+.3f}"
            f" obstacles={len(options['obstacles'])}"
            f" obs.shape={obs.shape} obs.dtype={obs.dtype}"
        )
        total_reward, collisions, episodes = 0.0, 0, 1
        d_min = float("inf")
        for _ in range(STEPS):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            collisions += int(bool(info["collision"]))
            d_min = min(d_min, float(info["d_min"]))
            if terminated or truncated:
                print(
                    f"  episode {episodes} 于 step {info['step']} 结束"
                    f"（{info['termination_reason']}）→ reset 后继续跑满 {STEPS} 步"
                )
                episodes += 1
                obs, info = env.reset(seed=scenario.seed + episodes, options=options)
        print(
            f"  {STEPS} 步随机动作: 总回报={total_reward:+.2f}"
            f" 碰撞={collisions} episodes={episodes} min_d_min={d_min:.3f} m"
        )
        print(f"  obs.shape={obs.shape} 末步 info: {_fmt_info(info)}")
    finally:
        env.close()


def main() -> None:
    """按固定 RNG 依次跑三个 core 场景（输出可复现）。"""
    print("chassis-sim 场景集 demo —— 三个 core 场景 × 50 步随机动作")
    rng = np.random.default_rng(20261104)
    for name in SCENARIOS:
        run(name, rng)


if __name__ == "__main__":
    main()
