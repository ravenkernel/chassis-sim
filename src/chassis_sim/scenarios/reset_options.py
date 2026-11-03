"""把 :class:`Scenario` 翻译成 `env.reset(options=...)` 的入参。

`ChassisSimEnv._reset_impl` 的 override 分支只认三个键（见
`envs/chassis_env.py`）::

    {"obstacles": [桥格式 dict...],
     "path_waypoints": ndarray (N, 3) [x, y, 鼻向 theta],
     "initial_heading": float}

外加**可选**的第 4 个键（只在场景确实带动态障碍时出现，因此静态场景的 payload
与加入动态障碍前**逐键相同**）::

    {"dynamic_obstacles": [桥格式 dict...]}   # 见 scenario.DynamicObstacle.to_env_dict()

两条路径：

* **core**（`straight` / `waypoints`）：障碍与路径全部来自 YAML，只做 numpy
  重采样（按 `spacing` 加密，并按切线补出鼻向列）；
* **planning**（`path.type == "stage"`）：现场调 `MapGenerator` 生成地图。
  import 落在**函数内**，因此 `import chassis_sim.scenarios` 不会拉起 numba；
  numba 缺失时抛带 `pip install chassis-sim[planning]` 的 ImportError。

两个入口：

* `scenario_to_reset_options(scenario, seed=...)` —— **一次**生成，返回 dict。
  同一个 dict 喂 n 次 `env.reset` 就是同一张图重放 n 次（见
  `ScenarioResetOptionsFactory` 的说明）；
* `scenario_reset_options_factory(scenario, base_seed=0)` —— 返回
  `callable(episode_index) -> dict`，**每回合**按 `base_seed + episode_index`
  重掷一张地图，可直接喂 `evaluate(..., reset_options=factory)`。
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np

from chassis_sim.scenarios.loader import (
    curriculum_stages,
    load_curriculum,
    load_scenario,
)
from chassis_sim.scenarios.scenario import (
    PathSpec,
    Scenario,
    ScenarioError,
    ScenarioValidationError,
)

__all__ = ["ScenarioResetOptionsFactory", "scenario_reset_options_factory",
           "scenario_to_reset_options"]

_PLANNING_IMPORT_HINT = (
    "场景 {name!r} 需要 Hybrid-A* 地图生成器（chassis_sim.planning），它依赖 numba。"
    "请先安装 planning extra：pip install chassis-sim[planning]"
)


def scenario_to_reset_options(
    scenario: Scenario, *, seed: int | None = None
) -> dict[str, Any]:
    """产出 `env.reset(options=...)` 的键值。

    Args:
        scenario: 已校验的场景。
        seed: 覆盖场景自带种子（只影响 planning 生成的地图；core 场景只记录）。

    Returns:
        `{"obstacles": [...], "path_waypoints": ndarray (N, 3), "initial_heading": float}`，
        场景带动态障碍时额外含 `"dynamic_obstacles": [...]`（静态场景不出现该键）。

    Raises:
        ImportError: planning 场景但 numba 不可用（消息含 pip 安装提示）。
        ScenarioError: 生成器返回非法路径等运行期错误。
    """
    if scenario.path.type == "stage":
        options = _planning_options(scenario, scenario.seed if seed is None else seed)
    else:
        path = _materialise_path(scenario.path)
        options = {
            "obstacles": [obstacle.to_env_dict() for obstacle in scenario.obstacles],
            "path_waypoints": path,
            "initial_heading": float(path[0, 2]),
        }
    if scenario.dynamic_obstacles:
        options["dynamic_obstacles"] = [
            obstacle.to_env_dict() for obstacle in scenario.dynamic_obstacles
        ]
    return options


# ═════════════════════════════════════════════════════════════════════
# core：YAML → (N, 3) 路径
# ═════════════════════════════════════════════════════════════════════
def _materialise_path(spec: PathSpec) -> np.ndarray:
    """把路径规格重采样成 `(N, 3)` 的 `[x, y, 鼻向]` 数组。"""
    if spec.type == "straight":
        points = _densify(
            [spec.params["start"], spec.params["goal"]], spec.params["spacing"]
        )
    elif spec.type == "waypoints":
        points = _densify(spec.params["points"], spec.params["spacing"])
    else:  # stage 走 planning 分支，这里只防御程序内构造的非法 spec
        raise ScenarioValidationError(
            f"路径类型 {spec.type!r} 无法在本地重采样（stage 需要 MapGenerator）"
        )
    return _with_headings(points)


def _densify(points: list[list[float]], spacing: float) -> np.ndarray:
    """沿折线按 `spacing` 加密（保留全部顶点，不去重）。"""
    arr = np.asarray(points, dtype=np.float64)
    columns = 3 if arr.shape[1] >= 3 else 2
    chunks = [arr[0:1]]
    for start, end in itertools.pairwise(arr):
        dist = float(math.hypot(end[0] - start[0], end[1] - start[1]))
        if dist <= 1e-12:
            continue
        steps = max(1, math.ceil(dist / spacing))
        t = np.arange(1, steps + 1, dtype=np.float64) / steps
        segment = np.empty((steps, columns), dtype=np.float64)
        segment[:, 0] = start[0] + t * (end[0] - start[0])
        segment[:, 1] = start[1] + t * (end[1] - start[1])
        if columns == 3:
            # theta 列按**最短弧**插值，避免 +pi/-pi 附近线性插值绕远路
            delta = (end[2] - start[2] + math.pi) % (2.0 * math.pi) - math.pi
            segment[:, 2] = start[2] + t * delta
        chunks.append(segment)
    return np.vstack(chunks)


def _with_headings(points: np.ndarray) -> np.ndarray:
    """2 列路径按切线补出鼻向列；已是 3 列的路径原样截取前三列。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[1] >= 3:
        return np.ascontiguousarray(arr[:, :3])
    xy = arr[:, :2]
    delta = np.diff(xy, axis=0)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    out = np.empty((len(xy), 3), dtype=np.float64)
    out[:, :2] = xy
    out[:-1, 2] = headings
    out[-1, 2] = headings[-1]
    return out


# ═════════════════════════════════════════════════════════════════════
# planning：延迟 import + numba 守卫
# ═════════════════════════════════════════════════════════════════════
def _require_numba(scenario_name: str) -> None:
    """numba 不可用 ⇒ 抛带安装提示的 ImportError。

    `chassis_sim.planning` 自己对 numba 缺失是 try/except + 恒等装饰器降级
    （core 安装下 import 不失败），所以"能不能 import planning"不能当作可用性
    判据 —— 必须在这里显式探测。
    """
    try:
        import numba  # noqa: F401
    except Exception as exc:  # 缺失 / 半装 / sys.modules 置 None 走同一提示
        raise ImportError(_PLANNING_IMPORT_HINT.format(name=scenario_name)) from exc


def _planning_options(scenario: Scenario, seed: int) -> dict[str, Any]:
    """调 MapGenerator 生成障碍与参考路径（世界边界取自场景 YAML）。"""
    _require_numba(scenario.name)
    from chassis_sim.planning import make_map_generator

    curriculum = load_curriculum()
    names = [str(stage.get("name")) for stage in curriculum_stages(curriculum)]
    stage_name = scenario.path.stage_name
    if stage_name not in names:
        raise ScenarioValidationError(
            f"{scenario.name}: 课程阶段 {stage_name!r} 不在 curriculum.yaml 中"
            f"（可用: {', '.join(names)}）"
        )
    generator = make_map_generator(
        curriculum,
        stage_index=names.index(stage_name),
        env_cfg={"world": {"bounds": list(scenario.world_bounds)}},
    )
    obstacles, path, _goal_heading, initial_heading = generator.generate(
        np.random.default_rng(seed)
    )
    path_array = np.asarray(path, dtype=np.float64)
    if path_array.ndim != 2 or path_array.shape[0] < 2:
        raise ScenarioError(
            f"{scenario.name}: MapGenerator 返回了非法路径 shape={path_array.shape}"
        )
    return {
        "obstacles": list(obstacles),
        "path_waypoints": path_array,
        "initial_heading": float(initial_heading),
    }


# ═════════════════════════════════════════════════════════════════════
# 每回合重掷地图：evaluate() 的 reset_options 回调
# ═════════════════════════════════════════════════════════════════════
def _is_map_generation_error(exc: BaseException) -> bool:
    """鸭子类型判定 `MapGenerationError`：不 import planning（core 无需 numba）。

    与 `envs/chassis_env.py` 同一判据（按类名走 MRO），因此 core 安装下也能
    重试；planning 缺失时那个异常根本不会出现（先抛 ImportError）。
    """
    return any(cls.__name__ == "MapGenerationError" for cls in type(exc).__mro__)


class ScenarioResetOptionsFactory:
    """「每回合重掷一张地图」的 `reset_options` 回调（喂给 `evaluate()`）。

    为什么需要它
    ------------
    `scenario_to_reset_options()` 在**调用那一刻**就把障碍与参考路径生成好并存进
    dict；同一个 dict 再喂 n 次 `env.reset(options=...)` 时，回合 i 的
    `seed + i` 只推进 env 内部 RNG，不会重掷地图 ⇒ n 个"样本"其实是 1 张图
    重放 n 次，Wilson 区间（`EvalResult.success_ci`）的分母不是 n 个独立样本。

    本回调把生成推迟到**每个回合**：

        factory = scenario_reset_options_factory("obstacles_dense", base_seed=0)
        result = evaluate(env_factory, controller, n_episodes=10, reset_options=factory)
        # 回合 i 的地图 = MapGenerator(default_rng(base_seed + i))

    core 场景（`straight` / `waypoints`，障碍与路径写死在 YAML 里）**没有**随机
    成分：指纹恒定，n 个回合仍是同一张图（`EvalResult.maps_independent` 为
    False，evaluate 会打 `[WARN][eval]`）。要独立地图只能用 planning 场景
    （`requires: planning`，即 `path.type == "stage"`）。

    退化策略（`MapGenerationError`，例如某个种子下 Hybrid-A* 无解）
    ---------------------------------------------------------------
    按 **seed 偏移**重试：episode i 的第 k 次尝试用 `seed_i + k`（k = 0..max_retries）。
    仍是 `(base_seed, episode_index, k)` 的纯函数 ⇒ 同一 base_seed 重跑逐位可复现。
    重试**不静默**：次数记在 `attempts`（总尝试）/ `retries`（因此多花的尝试）/
    `failures`（重试用尽的回合数）上；重试用尽直接抛最后一次 MapGenerationError，
    消息里带场景名、回合号与试过的种子 —— 评测不会把它当成一个"跳过的回合"
    （静默跳过会让 n 悄悄变小，比失败更坏）。
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        base_seed: int | None = 0,
        max_retries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries 必须 >= 0，实得 {max_retries}")
        self.scenario = scenario
        # None ⇒ 用场景自带种子（scenario.seed）；否则以它为准（可复现的入口）
        self.base_seed = int(scenario.seed if base_seed is None else base_seed)
        self.max_retries = int(max_retries)
        #: 真正调过 scenario_to_reset_options 的次数
        self.attempts = 0
        #: 其中因 MapGenerationError 而多花的尝试次数
        self.retries = 0
        #: 重试用尽、最终失败的回合数（失败会抛异常，不会变成"少一个样本"）
        self.failures = 0
        #: 成功产出 options 的回合数
        self.episodes = 0

    def __call__(self, episode_index: int) -> dict[str, Any]:
        """第 `episode_index` 回合的 options（地图种子 = base_seed + episode_index + k）。"""
        index = int(episode_index)
        first_seed = self.base_seed + index
        seeds: list[int] = []
        last_error: BaseException | None = None
        for k in range(self.max_retries + 1):
            seed = first_seed + k
            seeds.append(seed)
            self.attempts += 1
            if k:
                self.retries += 1
            try:
                options = scenario_to_reset_options(self.scenario, seed=seed)
            except Exception as exc:  # 只重试地图生成失败，其余异常照抛
                if not _is_map_generation_error(exc):
                    raise
                last_error = exc
                continue
            self.episodes += 1
            return options

        self.failures += 1
        assert last_error is not None  # 只有走过 except 分支才会到这里
        detail = (
            f"场景 {self.scenario.name!r} 第 {index} 回合的地图连续 "
            f"{len(seeds)} 次生成失败（试过种子 {seeds}，"
            f"max_retries={self.max_retries}）。这里**不**跳过该回合："
            "静默跳过会让样本数悄悄变小，比失败更坏。"
        )
        raise _with_detail(last_error, detail) from last_error

    @property
    def stats(self) -> dict[str, int]:
        """`attempts/retries/failures/episodes` 的拷贝（报告生成失败率用）。"""
        return {
            "attempts": self.attempts,
            "retries": self.retries,
            "failures": self.failures,
            "episodes": self.episodes,
        }


def _with_detail(exc: BaseException, detail: str) -> BaseException:
    """保留原异常类型，把定位信息拼进消息（MapGenerationError 是单消息签名）。"""
    try:
        return type(exc)(f"{exc}\n{detail}")
    except Exception:  # noqa: BLE001 - 构造不出来就退回原异常，定位信息降级为丢失
        return exc


def scenario_reset_options_factory(
    scenario_or_name: str | Path | Scenario,
    *,
    base_seed: int | None = 0,
    max_retries: int = 3,
) -> ScenarioResetOptionsFactory:
    """把场景编译成「每回合重掷一张地图」的 `reset_options` 回调。

    Args:
        scenario_or_name: 已校验的 :class:`Scenario`，或场景名 / YAML 路径
            （交给 :func:`load_scenario`，与 `chassis_sim.scenarios` 同口径）。
        base_seed: 回合 i 的地图种子 = `base_seed + i`。None ⇒ 用场景自带
            `scenario.seed`。同一 base_seed 的重跑逐位可复现。
        max_retries: `MapGenerationError` 时额外重试几次（每次种子 +1）。

    Returns:
        `callable(episode_index) -> dict`，带 `attempts/retries/failures/episodes`
        计数（`MapGenerationError` 不静默吞掉，见
        :class:`ScenarioResetOptionsFactory`）。

    Raises:
        ValueError: `max_retries < 0`。
    """
    scenario = (
        scenario_or_name
        if isinstance(scenario_or_name, Scenario)
        else load_scenario(scenario_or_name)
    )
    return ScenarioResetOptionsFactory(
        scenario, base_seed=base_seed, max_retries=max_retries
    )

