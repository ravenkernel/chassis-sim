"""控制器 × 车型评测矩阵：{pure_pursuit, dwa, random} × 3 车型 × n=10 张独立地图。

用法（仓库根目录）：:

    python examples/eval_matrix.py

产物：stdout 的 Markdown 两块表 + /tmp/chassis_sim_eval_matrix.json。

**每个回合是一张不同的地图**：`reset_options` 传的是
`scenario_reset_options_factory(SCENARIO, base_seed=SEED)`，它在**每个回合**用
`base_seed + 回合号` 现场生成一张 stage 地图（`scenario_to_reset_options` 是
一次性生成、同一条 dict 喂 n 次就是同一张图重放 n 次 —— 那样的 Wilson 区间
分母不是 n 个独立样本）。所有格子共用同一组 seed（配对比较：列间差异不是地图
差异），每格打印 `distinct_maps` 自证 n 个回合确实不同图。

场景取 `obstacles_sparse`：障碍少而稀，参考路径对 pure_pursuit / dwa 有解，
random 基线会真的撞上（否则碰撞率与 min_obstacle_clearance 两列永远是 0/None）。
`obstacles_dense` 上三个控制器几乎全灭，表格分不出高下。

动作界一律从 env.action_space / env.chassis 读：random 控制器直接在该空间上
采样，其余控制器经 Controller.to_chassis_action 做车型逆映射，脚本里不出现
±1.2 / ±omega_max 这类物理界（diff_drive 的动作空间正在被并发改动）。

控制器**必须**用评测 env 的那个底盘构造：这里统一走 `cs.make_controller(..., env=env)`
（下面用探针 env 取），不许用 `make_chassis(kind)`：它的默认 v_max=1.0 /
v_min=-1.0 与 EnvConfig 的 1.2 / -0.6 不同源，to_chassis_action 按 1.0 归一化、
Chassis.command() 按 1.2 反解 ⇒ 请求 0.5 m/s 实跑 0.6 m/s（+20%）。
`evaluate()` 会比对两侧 limits 并在不一致时抛 ValueError；每行还会打印
`ROUNDTRIP_V` 的速度往返（请求 vs env 实际），作为「同标定」的实证。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import chassis_sim as cs
from chassis_sim.eval import EvalResult, evaluate, to_markdown
from chassis_sim.scenarios import scenario_reset_options_factory

CHASSIS = ("ackermann", "diff_drive", "mecanum")
CONTROLLERS = ("pure_pursuit", "dwa", "random")
#: planning 场景：每回合用 base_seed + 回合号 现场生成一张地图
SCENARIO = "obstacles_sparse"
N_EPISODES = 10
MAX_STEPS = 300
SEED = 0
#: 速度往返自检的请求值 [m/s]：控制器请求它，env 底盘必须还原出同一个数
ROUNDTRIP_V = 0.6
JSON_PATH = Path("/tmp/chassis_sim_eval_matrix.json")


class RandomController:
    """基线：在 env.action_space 上均匀采样（不需要观测）。"""

    def __init__(self, action_space, seed: int = 0) -> None:
        self.action_space = action_space
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def set_path(self, path_world) -> None:
        pass

    def act(self, observation: dict) -> np.ndarray:
        low = self.action_space.low.astype(np.float64)
        high = self.action_space.high.astype(np.float64)
        sample = low + (high - low) * self._rng.random(low.shape)
        return sample.astype(self.action_space.dtype)


def _make_env(chassis: str) -> cs.ChassisSimEnv:
    return cs.make(chassis, max_steps=MAX_STEPS)


def _make_controller(name: str, env: cs.ChassisSimEnv):
    if name == "pure_pursuit":
        return cs.make_controller("pp", env=env)
    if name == "dwa":
        return cs.make_controller("dwa", env=env)
    if name == "random":
        return RandomController(env.action_space, seed=SEED)
    raise ValueError(f"未知控制器 {name!r}；支持 {list(CONTROLLERS)}")


def _speed_roundtrip(controller, chassis) -> float | None:
    """控制器请求 ROUNDTRIP_V 时，env 底盘 command() 还原出的实际 v [m/s]。

    这是「控制器 limits 与 env 底盘 limits 同标定」的直接证据，也是 evaluate()
    护栏存在的理由：两侧一致时 to_chassis_action → Chassis.command() 必须回到
    同一个数（≤1e-9）；若控制器按 make_chassis(kind) 的默认参数标定
    （v_max=1.0）而 env 用 EnvConfig 的 v_max=1.2，同一个 action 会被 env 解成
    0.72（+20%）。不按物理量归一化的控制器（random 基线）返回 None。
    """
    if not hasattr(controller, "to_chassis_action"):
        return None
    action = controller.to_chassis_action(ROUNDTRIP_V, 0.0)
    return float(chassis.command(np.asarray(action, dtype=np.float64)).v)


def main() -> int:
    t_start = time.perf_counter()
    rows: list[tuple[str, str, EvalResult]] = []
    attempts = retries = failures = 0
    for name in CONTROLLERS:
        for chassis in CHASSIS:
            # 控制器需要底盘的物理参数（轴距/限值）才能做逆映射，所以先借一个
            # 探针 env 构造控制器，评测本身另起全新的 env。
            probe = _make_env(chassis)
            try:
                controller = _make_controller(name, probe)
                v_roundtrip = _speed_roundtrip(controller, probe.chassis)
            finally:
                probe.close()

            if v_roundtrip is not None:
                print(
                    f"[check] {name:13s} × {chassis:10s} 速度往返: 控制器请求 "
                    f"v={ROUNDTRIP_V:.2f} -> env 底盘实际 v={v_roundtrip:.12f} "
                    f"(|Δ|={abs(v_roundtrip - ROUNDTRIP_V):.1e})",
                    file=sys.stderr,
                )

            # 每格一只新 factory：同一 base_seed ⇒ 每个格子面对同一组 10 张地图
            factory = scenario_reset_options_factory(SCENARIO, base_seed=SEED)
            t0 = time.perf_counter()
            result = evaluate(
                lambda chassis=chassis: _make_env(chassis),
                controller,
                episodes=N_EPISODES,
                seed=SEED,
                max_steps=MAX_STEPS,
                reset_options=factory,
                labels=(name, chassis),
            )
            rows.append((name, chassis, result))
            stats = factory.stats
            attempts += stats["attempts"]
            retries += stats["retries"]
            failures += stats["failures"]
            if not result.maps_independent:
                print(
                    f"[WARN][matrix] {name} × {chassis} 的 {result.n} 个回合只跑出 "
                    f"{result.distinct_maps} 张不同地图 ⇒ 这一格不能按独立样本引用",
                    file=sys.stderr,
                )
            print(
                f"[eval] {name:13s} × {chassis:10s} n={result.n:2d} "
                f"maps={result.distinct_maps:2d} success={result.success_rate:6.1%} "
                f"collision={result.collision_rate:6.1%} "
                f"timeout={result.timeout_rate:6.1%} "
                f"steps={result.mean_steps:5.1f} "
                f"gen_retries={stats['retries']} "
                f"t={time.perf_counter() - t0:5.2f}s",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - t_start
    print(
        to_markdown(
            rows,
            title=f"chassis-sim · 控制器 × 车型（{SCENARIO}，"
            f"n={N_EPISODES} 张独立地图，Wilson 95% CI）",
        )
    )
    print(
        f"\n地图生成：attempts={attempts} retries={retries} failures={failures}"
        f"（MapGenerationError 重试 = 换 base_seed+回合号+k 再生成；"
        f"failures>0 会让本脚本非零退出，不会静默丢样本）",
        file=sys.stderr,
    )
    payload = {
        "scenario": SCENARIO,
        "episodes": N_EPISODES,
        "max_steps": MAX_STEPS,
        "seed": SEED,
        "elapsed_s": round(elapsed, 3),
        "map_generation": {
            "attempts": attempts,
            "retries": retries,
            "failures": failures,
        },
        "results": [result.to_dict() for _, _, result in rows],
    }
    JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n耗时 {elapsed:.1f}s；JSON -> {JSON_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
