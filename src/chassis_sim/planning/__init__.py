"""Hybrid-A* 全局规划 + Reeds-Shepp 曲线 + 课程地图生成。

三个模块（hybrid_astar / map_generator / rs_curve）共用两处 import 约定：

1. 一律用绝对导入 `chassis_sim.envs.xxx` / `chassis_sim.planning.xxx`；
2. `from numba import jit` 走 try/except + 恒等装饰器 —— numba 是 **可选**
   加速器（extra: planning），core 安装（无 numba）下 `import chassis_sim.planning`
   不得失败，只是规划变慢。

课程配置从包内 curriculum.yaml 读（importlib.resources，与 CWD 无关）::

    mg = make_map_generator()                 # curriculum.yaml 的 active_stage
    mg = make_map_generator(stage_index=4)    # 指定阶段
    obstacles, path, goal_heading, initial_heading = mg.generate(
        np.random.default_rng(0))

构造参数保持兼容（`(stage_config, defaults=None, stage_index=-1, *,
env_cfg=None)`），因此既有调用方可以原样搬过来。
"""

from __future__ import annotations

import importlib.resources

import yaml

from chassis_sim.planning.hybrid_astar import (
    DEFAULT_ROBOT_INFLATION,
    hybrid_astar_search,
)
from chassis_sim.planning.map_generator import (
    MapGenerationError,
    MapGenerator,
    robot_half_dims_from_env_cfg,
    world_bounds_from_env_cfg,
)
from chassis_sim.planning.rs_curve import (
    reeds_shepp_length,
    reeds_shepp_path,
    reeds_shepp_path_pose,
)

__all__ = [
    "DEFAULT_ROBOT_INFLATION",
    "MapGenerationError",
    "MapGenerator",
    "hybrid_astar_search",
    "load_curriculum",
    "make_map_generator",
    "reeds_shepp_length",
    "reeds_shepp_path",
    "reeds_shepp_path_pose",
    "robot_half_dims_from_env_cfg",
    "world_bounds_from_env_cfg",
]

#: 包内课程配置文件（chassis_sim.config 的 package data，随 wheel 发布）。
CURRICULUM_RESOURCE = "curriculum.yaml"


def load_curriculum() -> dict:
    """读包内 curriculum.yaml，返回 `{active_stage, defaults, stages}`。

    用 importlib.resources 定位 —— 与 CWD 无关，装成 wheel / zip 后同样可用
    （与 `chassis_sim.envs.config.load_default_config` 同一模式）。
    """
    resource = importlib.resources.files("chassis_sim.config").joinpath(
        CURRICULUM_RESOURCE
    )
    return yaml.safe_load(resource.read_text(encoding="utf-8"))


def make_map_generator(
    curriculum: dict | None = None,
    stage_index: int | None = None,
    *,
    env_cfg=None,
) -> MapGenerator:
    """按 curriculum.yaml 装配 `MapGenerator`。

    装配方式：stage 配置 + defaults 一起传给 `MapGenerator`，并把阶段索引
    交给 reward 的 stage_scale。

    Args:
        curriculum: 已解析的课程 dict；None 时读包内 curriculum.yaml。
        stage_index: 目标阶段；None 时取 `curriculum['active_stage']`。
        env_cfg: 车体尺寸 / 世界边界来源（见 `MapGenerator.__init__`）。
    """
    if curriculum is None:
        curriculum = load_curriculum()
    stages = curriculum.get("stages") or []
    if not stages:
        raise ValueError("curriculum 缺少 stages（无法装配 MapGenerator）")
    idx = curriculum.get("active_stage", 0) if stage_index is None else stage_index
    idx = int(idx)
    if not 0 <= idx < len(stages):
        raise IndexError(
            f"stage_index 越界: {idx}（curriculum 共 {len(stages)} 个阶段）"
        )
    return MapGenerator(
        stages[idx], curriculum.get("defaults"), stage_index=idx, env_cfg=env_cfg
    )
