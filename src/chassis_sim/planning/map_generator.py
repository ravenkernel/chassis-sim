# ruff: noqa: F401, F841, I001, PLR1730, RUF059, RUF100
# fmt: off
"""Programmatic training map generator with random obstacles and paths."""
from __future__ import annotations
from chassis_sim.paths import np_angle_wrap
import numpy as np
from chassis_sim.planning.rs_curve import reeds_shepp_path, reeds_shepp_path_pose  # noqa: E402


class MapGenerationError(RuntimeError):
    """Raised when no valid path can be found — caller should retry reset()."""
from chassis_sim.planning.hybrid_astar import (
    _check_collision_batch,
    _check_collision_point_arr,
    _obstacles_to_arrays,
)
from chassis_sim.planning.hybrid_astar import hybrid_astar_search
from chassis_sim.planning.hybrid_astar import DEFAULT_ROBOT_INFLATION
import warnings


# 世界边界兜底常量（米）。默认 10×10 —— 与 config/env.yaml 的
# world.width/height 一致，因此默认配置下解析结果逐位等于修复前的硬编码值。
# 解析优先级（见 `MapGenerator.world_bounds`）：
#   1. `_path_cfg['world_bounds']` —— curriculum.yaml 显式覆盖，最高优先级；
#   2. 传入的 `env_cfg` 推导（`world_bounds_from_env_cfg`）；
#   3. 本常量。
DEFAULT_WORLD_BOUNDS = (0.0, 10.0, 0.0, 10.0)


def world_bounds_from_env_cfg(env_cfg) -> tuple[float, float, float, float] | None:
    """从 `EnvConfig` / env.yaml 风格配置推导 `(xmin, xmax, ymin, ymax)`。

    P2-15（世界边界单一真相源）: 规划器、起终点采样、障碍采样与复验四处原先各自
    兜底 `[0.0, 10.0, 0.0, 10.0]`，而 `world_bounds` 从来不进任何 YAML
    ⇒ 把 config/env.yaml 的 `world.width/height` 改成 4×4 时，bridge 的 LiDAR
    截断与出界判定按 4×4 生效，地图却依然生成到 x≈9.6（参考路径跑到世界外）。
    现与 `robot_half_dims_from_env_cfg` 同一模式从 env_cfg 推导。

    支持的输入形态（按优先级）:
      1. 带 `world_width` / `world_height` 属性的对象（EnvConfig 若新增）；
      2. `TrainingConfig`（读其 `env_raw`）或 env.yaml 解析后的 dict
         ——显式 `world.bounds` 优先（与 `NumpySimBridge` 同源），否则由
         `world.width`/`world.height` 推导 `[0, w, 0, h]`（原点固定 0,0，
         与 sim_bridge 的四面边界墙一致）。

    拿不到或数值非法（非有限/非正/上下界反了）时返回 `None`，由调用方回退常量。
    """
    if env_cfg is None:
        return None

    # ── 1. 属性直读 ──
    wv = getattr(env_cfg, "world_width", None)
    hv = getattr(env_cfg, "world_height", None)
    if wv is not None and hv is not None:
        try:
            wf, hf = float(wv), float(hv)
        except (TypeError, ValueError):
            return None
        if np.isfinite(wf) and np.isfinite(hf) and wf > 0.0 and hf > 0.0:
            return (0.0, wf, 0.0, hf)
        return None

    # ── 2. env.yaml 风格 dict（TrainingConfig.env_raw 或直接传 dict）──
    raw = getattr(env_cfg, "env_raw", env_cfg)
    if not isinstance(raw, dict):
        return None
    world = raw.get("world")
    if not isinstance(world, dict):
        return None
    if "bounds" in world:
        try:
            vals = tuple(float(v) for v in world["bounds"])
        except (TypeError, ValueError):
            return None
        if (len(vals) == 4 and all(np.isfinite(vals))
                and vals[1] > vals[0] and vals[3] > vals[2]):
            return vals
        return None
    try:
        wf = float(world.get("width", 10.0))
        hf = float(world.get("height", 10.0))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(wf) and np.isfinite(hf)) or wf <= 0.0 or hf <= 0.0:
        return None
    return (0.0, wf, 0.0, hf)


def robot_half_dims_from_env_cfg(env_cfg) -> tuple[float, float] | None:
    """从 ``EnvConfig`` / env.yaml 风格配置推导 ``(half_length, half_width)``。

    任务 3（车体几何单一真相源，2026-10-05）: 规划/复验用的车体足迹必须与
    env 侧（``NumpySimBridge`` 读 ``config/env.yaml`` 的 ``robot.shape``）同源，
    否则改了车体尺寸而规划器不知道，就会生成"照着走必撞"的参考路径
    （本项目历史上正中过这个坑：``robot_inflation`` 未加时 76.7% 的参考路径必撞）。

    支持的输入形态（按优先级）:
      1. 带 ``robot_length`` / ``robot_width`` 的对象（EnvConfig 若新增该字段）；
      2. 带 ``robot_half_length`` / ``robot_half_width``（或 ``half_length`` /
         ``half_width``）的对象——已是半长/半宽，直接使用；
      3. ``TrainingConfig``（读其 ``env_raw``）或直接传 env.yaml 解析后的 dict
         ——取 ``robot.shape.length`` / ``robot.shape.width`` 的一半。

    拿不到或数值非法（非正/非数）时返回 ``None``，由调用方回退到常量。
    """
    if env_cfg is None:
        return None

    # ── 1/2. 属性直读 ──
    for l_name, w_name, already_half in (
        ("robot_length", "robot_width", False),
        ("robot_half_length", "robot_half_width", True),
        ("half_length", "half_width", True),
    ):
        lv = getattr(env_cfg, l_name, None)
        wv = getattr(env_cfg, w_name, None)
        if lv is None or wv is None:
            continue
        try:
            lf, wf = float(lv), float(wv)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(lf) and np.isfinite(wf)) or lf <= 0.0 or wf <= 0.0:
            return None
        return (lf, wf) if already_half else (lf / 2.0, wf / 2.0)

    # ── 3. env.yaml 风格 dict（TrainingConfig.env_raw 或直接传 dict）──
    raw = getattr(env_cfg, "env_raw", env_cfg)
    if isinstance(raw, dict):
        robot = raw.get("robot")
        shape = robot.get("shape") if isinstance(robot, dict) else None
        if not isinstance(shape, dict):
            return None
        try:
            lf = float(shape["length"])
            wf = float(shape["width"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (np.isfinite(lf) and np.isfinite(wf)) or lf <= 0.0 or wf <= 0.0:
            return None
        return lf / 2.0, wf / 2.0
    return None


def _obstacle_sample_points(obs: dict) -> np.ndarray:
    """Generate sample points covering an obstacle's shape for collision checking."""
    shape = obs["shape"]
    x, y, theta = obs["state"][0], obs["state"][1], obs["state"][2]

    if shape["name"] == "circle":
        r = shape["radius"]
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pts = np.column_stack([x + r * np.cos(angles), y + r * np.sin(angles)])
        return np.vstack([np.array([[x, y]]), pts])  # center + boundary ring
    else:
        half_l = shape["length"] / 2
        half_w = shape["width"] / 2
        corners = np.array([
            [-half_l, -half_w], [half_l, -half_w],
            [half_l, half_w], [-half_l, half_w],
            [0, 0],  # center
        ])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = corners @ rot.T
        return rotated + np.array([x, y])


class MapGenerator:
    """Generates random obstacles and paths from curriculum stage config.

    Usage:
        stage_config = curriculum["stages"][active_stage]
        mg = MapGenerator(stage_config)
        obstacles, path, goal_heading, initial_heading = mg.generate(np.random.default_rng(seed))
    """

    # ── 机器人足迹（与 config/env.yaml robot.shape 0.38×0.20 同源）──
    # 用于 generate() 末尾的 env 级 SAT 复验；显式 path_cfg 覆盖优先级最高，
    # 其次由传入的 env_cfg 推导（单一真相源），最后回退到这里的常量。
    # 常量值 = 默认 env.yaml 的一半（0.38/2、0.20/2），由
    # tests/test_direction_semantics.py 锁死"推导值 == 常量"，默认配置下逐位不变。
    ROBOT_HALF_L = 0.19
    ROBOT_HALF_W = 0.10
    ROBOT_COLLISION_MARGIN = 0.03   # 与 NumpySimBridge._check_collision 的 margin 一致

    # 末点角度非有限时，圆均值的取样窗口（见 _compute_goal_heading）
    GOAL_HEADING_AVG_WINDOW = 5

    def __init__(self, stage_config: dict, defaults: dict | None = None,
                 stage_index: int = -1, *, env_cfg=None):
        """Parse parameter ranges from a curriculum stage dict.

        If defaults is provided, its 'hybrid_astar' values are used as
        fallbacks for any key not present in the stage's hybrid_astar block.

        stage_index: 课程阶段索引（供 reward 的 stage_scale 使用；未知时为 -1）。

        env_cfg: 关键字参数（默认 None，保持既有调用方不变）。传入 ``EnvConfig`` /
            ``TrainingConfig`` / env.yaml 解析后的 dict 时，车体半长/半宽由它推导
            ——车体尺寸的单一真相源（任务 3，2026-10-05）。拿不到时回退到类常量
            （值与默认 env.yaml 一致）。``_path_cfg`` 的显式
            ``robot_half_length``/``robot_half_width`` 覆盖优先级仍然最高。
        """
        self._stage_index = stage_index
        self._env_cfg = env_cfg
        # 折返（cusp）合法性拒绝计数（诊断用；不参与任何判定）
        self._cusp_rejections = 0
        self._apply_config(stage_config, defaults)

    @property
    def stage_index(self) -> int:
        """Current curriculum stage index (0-based), or -1 if not set."""
        return self._stage_index

    def _apply_config(self, stage_config: dict,
                      defaults: dict | None = None) -> None:
        """合并 defaults + stage 的 hybrid_astar 配置，并注入规划器默认值。"""
        self._obstacle_cfg = stage_config.get("obstacles", {})
        path_cfg: dict = {}
        if defaults and "hybrid_astar" in defaults:
            path_cfg.update(defaults["hybrid_astar"])
        path_cfg.update(stage_config.get("hybrid_astar", {}))
        # 规划器足迹膨胀默认值（config 未声明时用 hybrid_astar 的默认常量）
        path_cfg.setdefault("robot_inflation", DEFAULT_ROBOT_INFLATION)
        # HA* 组装路径输出间距：0.1 m 密集输出，避免 20 点弦线切角（config 可覆盖）
        path_cfg.setdefault("output_point_spacing", 0.10)
        # 世界边界（P2-15）：显式 path_cfg 覆盖 > env_cfg 推导 > 常量 10×10。
        # 解析结果写回 path_cfg ⇒ _search_segment 把它原样传给 Hybrid-A*，
        # 规划 / 起终点采样 / 障碍采样 / 路径复验四处同源（此前各自兜底 10×10）。
        path_cfg["world_bounds"] = list(self._resolve_world_bounds(path_cfg))
        # 参考路径横向抖动幅度（2026-10-23）：0.0 = 关闭 ⇒ 逐位不变且不消耗 rng。
        # 装配期校验（与 reward 参数同一模式）：负值/非有限值要么把点甩出世界、
        # 要么让路径语义失效，必须吵闹失败而不是静默生成怪路径。
        path_cfg["path_perturb_sigma"] = self._validate_path_perturb_sigma(
            path_cfg.get("path_perturb_sigma", 0.0))
        # 抖动方式（2026-10-23 追加）：point = 逐点 iid（默认，行为与首版逐位相同）；
        # control = 控制点抽样 + 沿线弧长线性插值（平滑，见 _sample_control_offsets）。
        path_cfg["path_perturb_mode"] = self._validate_path_perturb_mode(
            path_cfg.get("path_perturb_mode", "point"))
        path_cfg["path_perturb_control_spacing"] = (
            self._validate_path_perturb_control_spacing(
                path_cfg.get("path_perturb_control_spacing", 0.5)))
        # 折返（cusp）合法性阈值（2026-10-28）。0.0 = 关闭该判据（代码默认全关，
        # 与加入本功能前逐位相同）；config/curriculum.yaml 的 defaults 里显式开启。
        path_cfg["min_cusp_fold_arc"] = self._validate_cusp_threshold(
            path_cfg.get("min_cusp_fold_arc", 0.0), "min_cusp_fold_arc")
        path_cfg["min_cusp_leg_separation"] = self._validate_cusp_threshold(
            path_cfg.get("min_cusp_leg_separation", 0.0),
            "min_cusp_leg_separation")
        path_cfg["cusp_min_arc_gap"] = self._validate_cusp_threshold(
            path_cfg.get("cusp_min_arc_gap", 1.0), "cusp_min_arc_gap")
        self._path_cfg = path_cfg

    @property
    def _robot_inflation(self) -> float:
        """规划器足迹膨胀半径 (m)。"""
        try:
            return float(self._path_cfg.get("robot_inflation",
                                            DEFAULT_ROBOT_INFLATION))
        except (TypeError, ValueError):
            return float(DEFAULT_ROBOT_INFLATION)

    @staticmethod
    def _validate_path_perturb_sigma(raw) -> float:
        """校验 hybrid_astar.path_perturb_sigma（米，0 = 关闭），返回 float。"""
        try:
            sigma = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"hybrid_astar.path_perturb_sigma 必须是数值（米，0 = 关闭），"
                f"收到 {raw!r}")
        if not np.isfinite(sigma):
            raise ValueError(
                f"hybrid_astar.path_perturb_sigma 必须是有限数，收到 {sigma}")
        if sigma < 0.0:
            raise ValueError(
                f"hybrid_astar.path_perturb_sigma 必须 >= 0（0 = 关闭），收到 {sigma}")
        return sigma

    @staticmethod
    def _validate_path_perturb_mode(raw) -> str:
        """校验 hybrid_astar.path_perturb_mode 只能是 point / control。"""
        mode = str(raw).strip().lower()
        if mode not in ("point", "control"):
            raise ValueError(
                "hybrid_astar.path_perturb_mode 只能是 'point' 或 'control'，"
                f"收到 {raw!r}")
        return mode

    @staticmethod
    def _validate_path_perturb_control_spacing(raw) -> float:
        """校验 hybrid_astar.path_perturb_control_spacing（米，须 > 0）。"""
        try:
            spacing = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                "hybrid_astar.path_perturb_control_spacing 必须是数值（米），"
                f"收到 {raw!r}")
        if not np.isfinite(spacing):
            raise ValueError(
                "hybrid_astar.path_perturb_control_spacing 必须是有限数，"
                f"收到 {spacing}")
        if spacing <= 0.0:
            raise ValueError(
                "hybrid_astar.path_perturb_control_spacing 必须 > 0（米），"
                f"收到 {spacing}")
        return spacing

    @property
    def path_perturb_sigma(self) -> float:
        """参考路径横向抖动标准差 (m)；0.0 = 关闭（默认，逐位不变）。"""
        try:
            return float(self._path_cfg.get("path_perturb_sigma", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def path_perturb_mode(self) -> str:
        """抖动方式："point"（逐点 iid，默认）或 "control"（控制点 + 弧长插值）。"""
        return str(self._path_cfg.get("path_perturb_mode", "point")).strip().lower()

    @property
    def path_perturb_control_spacing(self) -> float:
        """control 模式的控制点弧长间距 (m)；默认 0.5。"""
        try:
            return float(self._path_cfg.get("path_perturb_control_spacing", 0.5))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _validate_cusp_threshold(value, name: str) -> float:
        """折返阈值装配期校验：必须是有限数且 >= 0（0 = 关闭该判据）。"""
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"hybrid_astar.{name} 必须是数字，收到 {value!r}") from None
        if not np.isfinite(val):
            raise ValueError(
                f"hybrid_astar.{name} 必须是有限数，收到 {value!r}")
        if val < 0.0:
            raise ValueError(
                f"hybrid_astar.{name} 必须 >= 0（米；0 = 关闭），收到 {val}")
        return val

    @property
    def min_cusp_fold_arc(self) -> float:
        """极大倒车段的最小允许弧长 (m)；0.0 = 关闭该判据。"""
        return float(self._path_cfg.get("min_cusp_fold_arc", 0.0))

    @property
    def min_cusp_leg_separation(self) -> float:
        """倒车腿与前进腿的最小允许间距 (m)；0.0 = 关闭该判据。"""
        return float(self._path_cfg.get("min_cusp_leg_separation", 0.0))

    @property
    def cusp_min_arc_gap(self) -> float:
        """比较两腿间距时要求的最小弧长间隔 (m)，默认 1.0。"""
        return float(self._path_cfg.get("cusp_min_arc_gap", 1.0))

    @property
    def cusp_rejections(self) -> int:
        """因折返不合规被丢弃的候选路径数（本次 MapGenerator 生命周期内累计）。"""
        return int(self._cusp_rejections)

    @property
    def robot_half_length(self) -> float:
        """有效车体半长 (m)，不含碰撞 margin。

        解析优先级（任务 3，单一真相源）:
          1. ``_path_cfg['robot_half_length']`` —— config 显式覆盖，最高优先级；
          2. 传入的 ``env_cfg`` 推导（``robot_half_dims_from_env_cfg``）；
          3. 类常量 ``ROBOT_HALF_L``（= 默认 env.yaml 0.38 的一半）。
        """
        try:
            explicit = self._path_cfg.get("robot_half_length")
            if explicit is not None:
                return float(explicit)
        except (TypeError, ValueError):
            pass
        derived = robot_half_dims_from_env_cfg(getattr(self, "_env_cfg", None))
        return float(derived[0]) if derived is not None else float(self.ROBOT_HALF_L)

    @property
    def robot_half_width(self) -> float:
        """有效车体半宽 (m)，不含碰撞 margin。优先级同 ``robot_half_length``。"""
        try:
            explicit = self._path_cfg.get("robot_half_width")
            if explicit is not None:
                return float(explicit)
        except (TypeError, ValueError):
            pass
        derived = robot_half_dims_from_env_cfg(getattr(self, "_env_cfg", None))
        return float(derived[1]) if derived is not None else float(self.ROBOT_HALF_W)

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        """世界边界 `(xmin, xmax, ymin, ymax)`（米）。

        解析优先级（P2-15，单一真相源）:
          1. `_path_cfg['world_bounds']` —— config 显式覆盖，最高优先级；
          2. 传入的 `env_cfg` 推导（`world_bounds_from_env_cfg`）；
          3. 常量 `DEFAULT_WORLD_BOUNDS`（= 默认 env.yaml 的 10×10）。
        与 `robot_half_length` 同一模式：默认配置下三条解析给出同一个值
        ⇒ 行为逐位不变（tests/test_world_bounds_x5.py 锁死）。
        """
        return tuple(float(v) for v in self._resolve_world_bounds(self._path_cfg))

    def _resolve_world_bounds(self, path_cfg: dict) -> tuple[float, float, float, float]:
        """把世界边界的解析优先级落成一个 4 元组（非法显式值告警后继续推导）。"""
        explicit = path_cfg.get("world_bounds")
        if explicit is not None:
            try:
                vals = tuple(float(v) for v in explicit)
            except (TypeError, ValueError):
                vals = ()
            if (len(vals) == 4 and np.isfinite(vals).all()
                    and vals[1] > vals[0] and vals[3] > vals[2]):
                return vals
            warnings.warn(
                f"MapGenerator: 非法的 world_bounds={explicit!r}（需 4 个有限数且 "
                f"xmax>xmin, ymax>ymin），改用 env_cfg / 常量推导出的世界边界"
            )
        derived = world_bounds_from_env_cfg(getattr(self, "_env_cfg", None))
        return derived if derived is not None else DEFAULT_WORLD_BOUNDS

    @staticmethod
    def _clamp_region(region, world_bounds, boundary_margin: float) -> list[float]:
        """把 `[x0, x1, y0, y1]` 起/终点采样区域夹进世界边界内。

        `boundary_margin` 内缩后若某个轴被夹成空区间（区域完全落在世界外，
        例如把 10×10 调好的「goal_region=[7,9]」套到 4×4 世界），退回该轴的
        整条可用带并告警——否则会采出世界外的起点/终点（P2-15 的现场）。
        世界比 `boundary_margin` 还小时同样退回整条世界带。
        """
        x_lo = float(world_bounds[0]) + float(boundary_margin)
        x_hi = float(world_bounds[1]) - float(boundary_margin)
        y_lo = float(world_bounds[2]) + float(boundary_margin)
        y_hi = float(world_bounds[3]) - float(boundary_margin)
        if x_hi < x_lo:
            x_lo, x_hi = float(world_bounds[0]), float(world_bounds[1])
        if y_hi < y_lo:
            y_lo, y_hi = float(world_bounds[2]), float(world_bounds[3])

        out = [
            min(max(float(region[0]), x_lo), x_hi),
            min(max(float(region[1]), x_lo), x_hi),
            min(max(float(region[2]), y_lo), y_hi),
            min(max(float(region[3]), y_lo), y_hi),
        ]
        if out[1] < out[0]:
            out[0], out[1] = x_lo, x_hi
            warnings.warn(
                f"MapGenerator: start/goal 区域 x 区间 {region[0]}~{region[1]} 落在"
                f"世界 x∈[{world_bounds[0]}, {world_bounds[1]}] 之外，改用整条可用带"
            )
        if out[3] < out[2]:
            out[2], out[3] = y_lo, y_hi
            warnings.warn(
                f"MapGenerator: start/goal 区域 y 区间 {region[2]}~{region[3]} 落在"
                f"世界 y∈[{world_bounds[2]}, {world_bounds[3]}] 之外，改用整条可用带"
            )
        # 区域被整个世界夹成一个点（世界太小 / 区域完全在世界外）——不静默：
        # 起点与终点会挤在同一点，min_start_goal_dist 必然无解。
        for axis, lo, hi, r_lo, r_hi in (("x", out[0], out[1], region[0], region[1]),
                                         ("y", out[2], out[3], region[2], region[3])):
            if hi - lo < 1e-9 and float(r_hi) - float(r_lo) > 1e-9:
                warnings.warn(
                    f"MapGenerator: start/goal 区域的 {axis} 区间 {r_lo}~{r_hi} 被世界"
                    f"尺寸夹成一个点 {lo:.3f}（世界太小，检查 world.width/height）"
                )
        return out

    def update_config(self, stage_config: dict, defaults: dict | None = None,
                      stage_index: int | None = None,
                      *, env_cfg=None) -> None:
        """热更新配置（课程阶段/难度变化时由 trainer 调用）。

        无需重建 MapGenerator——仅替换内部配置引用。

        stage_index: 传入（非 None）时同步更新 ``self._stage_index``，
        使 reward 的 stage_scale 跟随课程阶段；None = 保持原值，
        兼容历史的 ``update_config(stage_config, defaults)`` 两参数调用。

        env_cfg: 传入（非 None）时替换车体几何真相源；None = 保持原引用，
        不会把已注入的 env_cfg 清空。
        """
        # env_cfg 必须先于 _apply_config 生效：世界边界在 _apply_config 里解析
        # 并写回 path_cfg，先解析后换 env_cfg 会沿用旧世界尺寸（P2-15）。
        if env_cfg is not None:
            self._env_cfg = env_cfg
        self._apply_config(stage_config, defaults)
        if stage_index is not None:
            self._stage_index = int(stage_index)

    def generate(self, rng: np.random.Generator) -> tuple[list[dict], np.ndarray, float, float]:
        """Generate one random map.

        Returns:
            (obstacles, path_waypoints, goal_heading, initial_heading)
            - obstacles: list of dicts for ir-sim create_obstacle(**kwargs)
            - path_waypoints: np.ndarray of shape (num_points, 2) in world coords
            - goal_heading: float in [-pi, pi], **终点处的鼻向**（路径末点自带的
              车头朝向，非运动方向；倒车段上二者相差 π，见 _compute_goal_heading）
            - initial_heading: float in [-pi, pi], robot's starting heading
        """
        max_retries = self._path_cfg.get("max_generate_retries", 10)
        last_cusp_reason: str | None = None
        for _ in range(max_retries):
            obstacles = self._generate_obstacles(rng)
            path, rs_start_theta = self._generate_path(rng, obstacles=obstacles)
            if path is None:
                continue
            # ── env 级足迹复验 (2026-08-17, B2/B3 兜底) ──
            # 规划器用"点 + 膨胀"近似车体；这里用与 NumpySimBridge._check_collision
            # 等价的 SAT 足迹检查复验最终（重采样后）路径，任何位姿擦障即丢弃该地图
            # 并重试。避免"参考路径照着走必撞"。
            if self._check_path_footprint(path, obstacles):
                continue
            # 折返合法性（2026-10-28）：不合规就整张图重采（含障碍）。
            # 默认阈值在 config/curriculum.yaml 的 defaults.hybrid_astar；代码默认
            # 0.0 = 关闭。不逐次 warn（实测拒绝率 ~10%，训练时会把 stderr 刷满），
            # 只计数；全部重试用尽时把末次原因写进异常消息。
            cusp_reason = self._cusp_violation(path)
            if cusp_reason is not None:
                self._cusp_rejections += 1
                last_cusp_reason = cusp_reason
                continue
            goal_heading = self._compute_goal_heading(path)
            initial_heading = self._compute_initial_heading(path, start_theta=rs_start_theta)
            # 最终路径确定之后、返回之前注入横向误差（默认 sigma=0 ⇒ 逐位不变，
            # 且不消耗 rng）。必须在足迹复验与两个 heading 之后：
            #   1. 复验针对规划器输出，抖动让"以为安全、实际更近"成立；
            #   2. 首末点不动 + theta 列不改，两个 heading 与抖动前逐位相同。
            path = self._perturb_path_lateral(path, rng)
            return obstacles, path, goal_heading, initial_heading
        raise MapGenerationError(
            f"Failed to generate solvable map after {max_retries} retries — "
            f"all RS paths collide with obstacles"
            + (f"; 末次因折返不合规被拒：{last_cusp_reason}"
               if last_cusp_reason else "")
        )

    # ------------------------------------------------------------------
    # Obstacle generation
    # ------------------------------------------------------------------

    def _generate_obstacles(self, rng: np.random.Generator) -> list[dict]:
        if not self._obstacle_cfg.get("enabled", False):
            return []

        generation_mode = self._obstacle_cfg.get("generation", "random")

        if generation_mode == "grid":
            return self._generate_obstacles_grid(rng)

        if generation_mode != "random":
            return []

        count_min, count_max = self._obstacle_cfg["count"]
        shapes = self._obstacle_cfg["shapes"]
        circle_cfg = self._obstacle_cfg.get("circle", {})
        rect_cfg = self._obstacle_cfg.get("rectangle", {})
        margin = self._obstacle_cfg.get("position_margin", 1.0)

        n = int(rng.integers(count_min, count_max + 1))
        if n == 0:
            return []

        shape_sequence = [shapes[int(rng.integers(0, len(shapes)))] for _ in range(n)]
        # 若有多种形状可用，确保每种形状至少出现一次
        for shape_name in set(shapes):
            if shape_name not in shape_sequence:
                shape_sequence[rng.integers(0, n)] = shape_name

        spacing = self._obstacle_cfg.get("obstacle_spacing", 0.3)
        max_pos_retries = self._obstacle_cfg.get("position_retries", 100)
        # 世界尺寸来自同一真相源（默认 10×10 ⇒ 与 uniform(margin, 10-margin) 逐位相同）
        world_bounds = self.world_bounds

        obstacles = []
        # Pre-convert obstacle arrays for fast JIT collision checks.
        # Rebuilt incrementally after each successful placement.
        circles_arr = np.zeros((0, 3), dtype=np.float64)
        rects_arr = np.zeros((0, 6), dtype=np.float64)
        # Cached sample points of all placed obstacles (updated incrementally)
        all_pts = np.empty((0, 2), dtype=np.float64)
        dropped_count = 0
        for i, shape_name in enumerate(shape_sequence):
            shape_def = {"name": shape_name}
            if shape_name == "circle":
                r_min, r_max = circle_cfg.get("radius", [0.2, 0.6])
                shape_def["radius"] = float(rng.uniform(r_min, r_max))
            elif shape_name == "rectangle":
                l_min, l_max = rect_cfg.get("length", [0.5, 1.5])
                w_min, w_max = rect_cfg.get("width", [0.2, 0.5])
                shape_def["length"] = float(rng.uniform(l_min, l_max))
                shape_def["width"] = float(rng.uniform(w_min, w_max))

            placed = False
            for _ in range(max_pos_retries):
                x = float(rng.uniform(world_bounds[0] + margin, world_bounds[1] - margin))
                y = float(rng.uniform(world_bounds[2] + margin, world_bounds[3] - margin))
                theta = float(rng.uniform(0, 2 * np.pi)) if shape_name == "rectangle" else 0.0

                candidate = {
                    "shape": shape_def,
                    "state": [x, y, theta],
                    "kinematics": {"name": "static"},
                    "name": f"obs_{i}",
                }

                # Fast JIT collision: convert single candidate to arrays + check all points
                cand_circles, cand_rects = _obstacles_to_arrays([candidate])
                cand_pts = _obstacle_sample_points(candidate)
                # Check candidate vs existing obstacles
                collides = False
                for row in cand_pts:
                    if _check_collision_point_arr(float(row[0]), float(row[1]),
                                                  circles_arr, rects_arr, spacing):
                        collides = True
                        break
                if collides:
                    continue
                # Check existing pts vs candidate obstacle
                for row in all_pts:
                    if _check_collision_point_arr(float(row[0]), float(row[1]),
                                                  cand_circles, cand_rects, spacing):
                        collides = True
                        break
                if collides:
                    continue
                obstacles.append(candidate)
                # Incrementally update pre-converted arrays
                circles_arr = np.vstack([circles_arr, cand_circles]) if len(cand_circles) > 0 else circles_arr
                rects_arr = np.vstack([rects_arr, cand_rects]) if len(cand_rects) > 0 else rects_arr
                all_pts = np.vstack([all_pts, cand_pts]) if len(all_pts) > 0 else cand_pts
                placed = True
                break

            if not placed:
                dropped_count += 1

        if dropped_count > 0:
            warnings.warn(
                f"Could not place {dropped_count}/{len(shape_sequence)} obstacles "
                f"after {max_pos_retries} retries each -- map may be sparse"
            )

        return obstacles

    def _generate_obstacles_grid(self, rng: np.random.Generator) -> list[dict]:
        """Generate grid-based obstacles for maze-like environments (stage 5)."""
        grid_size = float(self._obstacle_cfg.get("grid_size", 1.0))
        fill_ratio = self._obstacle_cfg.get("fill_ratio", [0.4, 0.7])
        margin = self._obstacle_cfg.get("position_margin", 1.0)
        world_bounds = self.world_bounds
        start_region = self._path_cfg.get("start_region", [1.0, 3.0, 1.0, 3.0])
        goal_region = self._path_cfg.get("goal_region", [7.0, 9.0, 7.0, 9.0])

        n_cols = int((world_bounds[1] - world_bounds[0]) / grid_size)
        n_rows = int((world_bounds[3] - world_bounds[2]) / grid_size)

        available = []
        for i in range(n_cols):
            for j in range(n_rows):
                cx = world_bounds[0] + (i + 0.5) * grid_size
                cy = world_bounds[2] + (j + 0.5) * grid_size
                if (start_region[0] - grid_size < cx < start_region[1] + grid_size and
                    start_region[2] - grid_size < cy < start_region[3] + grid_size):
                    continue
                if (goal_region[0] - grid_size < cx < goal_region[1] + grid_size and
                    goal_region[2] - grid_size < cy < goal_region[3] + grid_size):
                    continue
                if margin < cx < world_bounds[1] - margin and margin < cy < world_bounds[3] - margin:
                    available.append((cx, cy))

        target_fill = float(rng.uniform(fill_ratio[0], fill_ratio[1]))
        n_fill = max(1, min(int(len(available) * target_fill), len(available)))
        chosen_indices = rng.choice(len(available), size=n_fill, replace=False)

        obstacle_size = grid_size * 0.4
        obstacles = []
        for idx in chosen_indices:
            x, y = available[idx]
            theta = float(rng.uniform(0, 2 * np.pi))
            obstacles.append({
                "shape": {"name": "rectangle", "length": obstacle_size, "width": obstacle_size},
                "state": [x, y, theta],
                "kinematics": {"name": "static"},
                "name": f"obs_g{idx}",
            })
        return obstacles

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def _sample_start_goal(self, rng, obstacles):
        """Sample start/goal positions with clearance from obstacles.

        Retries with progressively relaxed clearance if the region is
        densely occupied. Never returns positions that collide with
        obstacles (margin=0)."""
        start_region = self._path_cfg.get("start_region", [1.0, 3.0, 1.0, 3.0])
        goal_region = self._path_cfg.get("goal_region", [7.0, 9.0, 7.0, 9.0])
        clearance = self._path_cfg.get("start_goal_clearance", 1.0)
        min_dist = self._path_cfg.get("min_start_goal_dist", 3.0)
        boundary_margin = self._path_cfg.get("boundary_margin", 0.0)
        world_bounds = self.world_bounds

        # 起终点区域夹进世界边界（含 boundary_margin）——与 HA* / 障碍采样同一组边界
        start_region = self._clamp_region(start_region, world_bounds, boundary_margin)
        goal_region = self._clamp_region(goal_region, world_bounds, boundary_margin)

        # 障碍数组只转换一次（旧实现每次采样点都重新转换 + 1×N 广播检查）
        _circles, _rects = _obstacles_to_arrays(obstacles or []) if obstacles is not None \
            else (np.zeros((0, 3)), np.zeros((0, 6)))

        for clearance_relax in [clearance, clearance * 0.5, 0.1]:
            for _ in range(200):
                start = np.array([
                    float(rng.uniform(start_region[0], start_region[1])),
                    float(rng.uniform(start_region[2], start_region[3])),
                ])
                goal = np.array([
                    float(rng.uniform(goal_region[0], goal_region[1])),
                    float(rng.uniform(goal_region[2], goal_region[3])),
                ])
                # 起终点距离不够 → 重试
                if np.hypot(goal[0] - start[0], goal[1] - start[1]) < min_dist:
                    continue
                if obstacles is not None:
                    if not (_check_collision_point_arr(float(start[0]), float(start[1]),
                                                       _circles, _rects, clearance_relax)
                            or _check_collision_point_arr(float(goal[0]), float(goal[1]),
                                                          _circles, _rects, clearance_relax)):
                        if clearance_relax < clearance:
                            warnings.warn(
                                f"Start/goal sampling relaxed clearance from "
                                f"{clearance}m to {clearance_relax}m due to dense obstacles"
                            )
                        return start, goal
                else:
                    return start, goal

        # Absolute last resort: positions without any clearance guarantee
        warnings.warn("Start/goal sampling exhausted all retries — "
                      "positions may be near obstacles")
        return start, goal

    def _generate_path(self, rng: np.random.Generator, obstacles: list | None = None
                       ) -> tuple[np.ndarray, float | None]:
        """Generate path using Hybrid-A* search with optional via-points."""
        point_spacing = self._path_cfg.get("point_spacing", 0.05)
        start, goal = self._sample_start_goal(rng, obstacles)
        theta_range = self._path_cfg.get("start_theta_range", [-np.pi, np.pi])
        goal_theta_range = self._path_cfg.get("goal_theta_range", [-np.pi, np.pi])

        # Try via-points multi-segment path first
        via_cfg = self._path_cfg.get("via_points", {})
        if via_cfg.get("enabled", False):
            result = self._generate_path_with_via(
                start, goal, obstacles, via_cfg, theta_range, goal_theta_range,
                rng, point_spacing,
            )
            if result is not None:
                return result

        # Direct start→goal search
        return self._generate_path_direct(
            start, goal, obstacles, rng, point_spacing,
            theta_range, goal_theta_range,
        )

    def _search_segment(self, from_pt, to_pt, start_theta, goal_theta,
                        obstacles, rng):
        """Run single Hybrid-A* search. Returns (N,3) array or None."""
        path, info = hybrid_astar_search(
            (from_pt[0], from_pt[1], start_theta),
            (to_pt[0], to_pt[1], goal_theta),
            obstacles, self._path_cfg, rng,
        )
        if path is not None and info["success"]:
            return path
        return None

    def _generate_path_with_via(self, start, goal, obstacles, via_cfg,
                                 theta_range, goal_theta_range, rng, point_spacing):
        """Multi-sample via-point path generation with quality scoring.

        Samples N_ATTEMPTS via-point configurations, scores each, picks the best.
        Applies junction blending, heading continuity check, and curvature
        validation. Falls back to direct path if all attempts fail.
        """
        N_ATTEMPTS = 5
        EARLY_STOP_SCORE = 0.3  # junction gap small, length balanced, curvature OK

        reverse_prob = self._path_cfg.get("reverse_probability", 0.3)
        path_reverse_prob = self._path_cfg.get("path_reverse_probability", 0.3)

        best_score = float("inf")
        best_path = None
        best_start_theta = None
        best_reversed_flag = False

        for _ in range(N_ATTEMPTS):
            via_pts = self._sample_via_points(start, goal, obstacles, via_cfg, rng)
            if not via_pts:
                continue

            waypoints = [start] + via_pts + [goal]
            segments = []

            # Build heading for first segment
            first_target = waypoints[1]
            base_dir = float(np.arctan2(first_target[1] - start[1],
                                        first_target[0] - start[0]))
            offset = float(rng.uniform(theta_range[0], theta_range[1]))
            prev_theta = base_dir + offset
            if rng.random() < reverse_prob:
                prev_theta = np_angle_wrap(prev_theta + np.pi)

            # Search each segment
            segment_failed = False
            for i in range(len(waypoints) - 1):
                seg_from = waypoints[i]
                seg_to = waypoints[i + 1]
                seg_dir = float(np.arctan2(seg_to[1] - seg_from[1],
                                           seg_to[0] - seg_from[0]))
                seg_to_theta = seg_dir + float(rng.uniform(goal_theta_range[0],
                                                            goal_theta_range[1]))
                seg_path = self._search_segment(
                    seg_from, seg_to, prev_theta, seg_to_theta,
                    obstacles, rng,
                )
                if seg_path is None:
                    segment_failed = True
                    break
                segments.append(seg_path)
                prev_theta = float(seg_path[-1, 2])

            if segment_failed:
                continue

            # Score before expensive operations
            score = self._score_via_configuration(segments, via_pts, obstacles)
            if score is None or score >= best_score:
                continue

            # Blend junctions between adjacent segments
            full_path = segments[0]
            for i in range(1, len(segments)):
                full_path = MapGenerator._blend_junction(full_path, segments[i])

            # Heading continuity pre-check (before smooth — saves compute)
            if not MapGenerator._check_heading_continuity(full_path):
                continue

            # Reverse if needed, then force endpoints
            reversed_flag = rng.random() < path_reverse_prob
            if reversed_flag:
                full_path = full_path[::-1]
                full_path[:, 2] = np_angle_wrap(full_path[:, 2] + np.pi)

            if reversed_flag:
                full_path[0, :2] = goal
                full_path[-1, :2] = start
            else:
                full_path[0, :2] = start
                full_path[-1, :2] = goal

            best_score = score
            best_path = full_path
            best_reversed_flag = reversed_flag

            if score < EARLY_STOP_SCORE:
                break  # P1: early stop — path quality excellent, skip remaining attempts

        if best_path is None:
            return None

        # Full post-processing on best candidate only
        path = self._resample_path(best_path, point_spacing)
        path = self._smooth_path(path, obstacles, margin=self._robot_inflation)

        # Curvature validation (tightened: 2× tolerance, 2% violation ratio)
        # Use vehicle physical turning radius (0.25m), not search param
        if not self._validate_path_curvature(path, point_spacing,
                                              min_turning_radius=0.25):
            return None

        start_theta = float(path[0, 2])
        return path, start_theta

    def _sample_via_points(self, start, goal, obstacles, via_cfg, rng):
        """Sample 1-2 via-points on grid, avoiding start/goal cells."""
        grid = via_cfg.get("grid", 3)
        count_range = via_cfg.get("count", [1, 2])
        clearance = via_cfg.get("clearance", 0.5)
        bounds = self.world_bounds

        cell_w = (bounds[1] - bounds[0]) / grid
        cell_h = (bounds[3] - bounds[2]) / grid

        def _cell(pt):
            return (int((pt[0] - bounds[0]) / cell_w),
                    int((pt[1] - bounds[2]) / cell_h))

        start_cell = _cell(start)
        goal_cell = _cell(goal)
        all_cells = [(i, j) for i in range(grid) for j in range(grid)]
        available = [c for c in all_cells
                     if c != start_cell and c != goal_cell]
        if not available:
            return []

        n = min(int(rng.integers(count_range[0], count_range[1] + 1)),
                len(available))
        chosen = list(rng.choice(available, size=n, replace=False))

        via_pts = []
        for ci, cj in chosen:
            x_min = bounds[0] + ci * cell_w + 0.5
            x_max = bounds[0] + (ci + 1) * cell_w - 0.5
            y_min = bounds[2] + cj * cell_h + 0.5
            y_max = bounds[2] + (cj + 1) * cell_h - 0.5
            for _ in range(50):
                vx = float(rng.uniform(x_min, x_max))
                vy = float(rng.uniform(y_min, y_max))
                if obstacles is None or not MapGenerator._check_collision(
                    np.array([[vx, vy]]), obstacles, clearance,
                ):
                    via_pts.append(np.array([vx, vy]))
                    break
        return via_pts

    def _resample_path(self, path, point_spacing):
        """按固定间距重新采样路径（长路径点多、短路径点少）。"""
        if len(path) < 2:
            return path
        xy = path[:, :2]
        diffs = np.diff(xy, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cumsum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cumsum[-1]
        if total < 1e-9:
            return path
        # 按间距生成采样距离，首尾点始终保留
        num_points = max(2, int(np.ceil(total / point_spacing)) + 1)
        sample_dists = np.linspace(0, total, num_points)
        result = np.empty((num_points, path.shape[1]))
        for i, d in enumerate(sample_dists):
            idx = np.searchsorted(cumsum, d, side="right") - 1
            idx = max(0, min(idx, len(path) - 2))
            t = (d - cumsum[idx]) / max(seg_lens[idx], 1e-9)
            t = max(0.0, min(1.0, t))
            result[i, 0] = path[idx, 0] + t * diffs[idx, 0]
            result[i, 1] = path[idx, 1] + t * diffs[idx, 1]
            t0, t1 = path[idx, 2], path[idx + 1, 2]
            dtheta = np_angle_wrap(t1 - t0)
            result[i, 2] = np_angle_wrap(t0 + t * dtheta)
        return result

    @staticmethod
    def _smooth_path(path: np.ndarray, obstacles: list[dict],
                     num_iters: int = 80, alpha: float = 0.4, beta: float = 0.08,
                     margin: float = 0.0):
        """Laplacian 平滑 + 弹性约束，保持避碰。

        保留 Hybrid-A* 倒车段朝向：smooth 前检测 backward 点，smooth 后还原翻转。

        Args:
            path: (N, 3) waypoints (x, y, theta)。
            obstacles: 障碍物列表。
            num_iters: 平滑迭代次数。
            alpha: 平滑强度（0=不平滑, 1=完全拉向邻居均值）。
            beta: 弹性强度（0=可自由移动, 1=固定原位）。
            margin: 碰撞检查的足迹膨胀半径 (m)，保持平滑后净空不被吃掉。
        Returns:
            Smoothed (N, 3) path.
        """
        if len(path) < 4:
            return path.copy()

        n = len(path)
        xy = path[:, :2].copy()
        original = xy.copy()

        # 障碍数组只转换一次，逐点用 jitted 标量检查。
        # （旧实现每个候选点都走 _check_collision → _obstacles_to_arrays + 1 点
        #  numpy 广播，80×N 次调用占总规划耗时 >60%）
        circles, rects = _obstacles_to_arrays(obstacles or [])

        for _ in range(num_iters):
            max_disp = 0.0  # P2: track convergence
            for i in range(1, len(xy) - 1):
                # Laplacian smooth: pull toward neighbors' average
                smooth_xy = (xy[i - 1] + xy[i + 1]) / 2.0
                # Elastic: pull back toward original position
                elastic_xy = original[i]
                # Combined update
                new_xy = (1 - alpha - beta) * xy[i] + alpha * smooth_xy + beta * elastic_xy
                # Check collision at new position (带足迹膨胀, jitted 标量检查)
                if not _check_collision_point_arr(
                    float(new_xy[0]), float(new_xy[1]), circles, rects, margin,
                ):
                    disp = np.hypot(new_xy[0] - xy[i, 0], new_xy[1] - xy[i, 1])
                    if disp > max_disp:
                        max_disp = disp
                    xy[i] = new_xy

            if max_disp < 1e-3:  # converged — <1mm movement
                break

        # Detect backward segments from ORIGINAL path BEFORE smoothing.
        # Smoothing changes XY but does NOT change which segments are backward —
        # the vehicle still faces opposite to motion direction on those segments.
        # We detect from original path (heading vs motion_dir) and reapply after
        # recomputing headings from smoothed XY.
        orig_is_backward = np.zeros(n, dtype=bool)
        for i in range(n - 1):
            dx = path[i + 1, 0] - path[i, 0]
            dy = path[i + 1, 1] - path[i, 1]
            motion_dir = np.arctan2(dy, dx)
            diff = abs(np_angle_wrap(path[i, 2] - motion_dir))
            orig_is_backward[i] = diff > np.pi / 2
        if n >= 2:
            dx = path[-1, 0] - path[-2, 0]
            dy = path[-1, 1] - path[-2, 1]
            motion_dir = np.arctan2(dy, dx)
            diff = abs(np_angle_wrap(path[-1, 2] - motion_dir))
            orig_is_backward[-1] = diff > np.pi / 2

        # Recompute headings from smoothed XY (all motion_dir).
        # Then reapply backward flip to segments that were backward pre-smoothing.
        headings = MapGenerator._compute_path_headings(
            np.column_stack([xy, np.zeros(n)]),
        )
        headings[orig_is_backward] = np_angle_wrap(headings[orig_is_backward] + np.pi)
        return np.column_stack([xy, headings])

    @staticmethod
    def _blend_junction(seg_a: np.ndarray, seg_b: np.ndarray,
                        n_blend: int = 5) -> np.ndarray:
        """Blend two path segments at their junction with interpolated transition points.

        Takes last 2 points of seg_a and first 2 points of seg_b, generates
        n_blend interpolated points that smoothly bridge the junction. Removes
        the original 4 junction-adjacent points, replacing them with the blend.

        XY: linear interpolation from seg_a[-2] to seg_b[1].
        Heading: shortest-angle interpolation (slerp-like).

        Args:
            seg_a: (N, 3) first segment waypoints (x, y, theta).
            seg_b: (M, 3) second segment waypoints (x, y, theta).
            n_blend: number of blended points to insert.

        Returns:
            Concatenated array: seg_a[:-2] + blend + seg_b[2:].
            If either segment is too short, returns simple concatenation.
        """
        if len(seg_a) < 3 or len(seg_b) < 3:
            return np.concatenate([seg_a, seg_b], axis=0)

        # Anchors: point before junction in seg_a, point after junction in seg_b
        anchor_a = seg_a[-2].astype(np.float64)  # (x, y, theta)
        anchor_b = seg_b[1].astype(np.float64)   # (x, y, theta)

        # Linear interpolation between anchors
        t = np.linspace(0, 1, n_blend)
        blend_xy = np.column_stack([
            anchor_a[0] + t * (anchor_b[0] - anchor_a[0]),
            anchor_a[1] + t * (anchor_b[1] - anchor_a[1]),
        ])

        # Shortest-angle heading interpolation
        dtheta = np_angle_wrap(anchor_b[2] - anchor_a[2])
        blend_theta = anchor_a[2] + t * dtheta
        blend_theta = np_angle_wrap(blend_theta)

        blend = np.column_stack([blend_xy, blend_theta])
        return np.concatenate([seg_a[:-2], blend, seg_b[2:]], axis=0)

    @staticmethod
    def _check_heading_continuity(path: np.ndarray, max_consecutive: int = 3,
                                  max_gap_rad: float = np.deg2rad(45)) -> bool:
        """Reject paths with clusters of sharp heading changes.

        Scans adjacent waypoints for heading differences. If max_consecutive
        or more consecutive points exceed max_gap_rad heading gap, the path
        has an unsmoothable junction and should be rejected early.

        Args:
            path: (N, 3) concatenated waypoints (x, y, theta).
            max_consecutive: number of consecutive excessive gaps to trigger rejection.
            max_gap_rad: heading gap threshold in radians.

        Returns:
            True if the path passes the continuity check.
        """
        if len(path) < max_consecutive + 1:
            return True
        headings = path[:, 2]
        dtheta = np.abs(np.diff(headings))
        dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)  # shortest angle
        # Count consecutive points exceeding the gap threshold
        exceeded = dtheta > max_gap_rad
        run_length = 0
        for flag in exceeded:
            if flag:
                run_length += 1
                if run_length >= max_consecutive:
                    return False
            else:
                run_length = 0
        return True

    def _score_via_configuration(self, segments: list, via_pts: list,
                                 obstacles: list) -> float | None:
        """Score a via-point path configuration. Lower is better.

        Returns None if the configuration fails any hard-reject threshold.

        Dimensions (all computed per-segment):
          - Junction gap: mean(|θ_last - θ_first|) across junctions (weight 0.4)
          - Length balance: max(len) / min(len) (weight 0.3)
          - Segment curvature: max(|Δθ|) / max_allowed (weight 0.3)

        Hard fails: max junction gap > 60°, length ratio > 5×, via_pt too close
        to obstacles.
        """
        point_spacing = self._path_cfg.get("point_spacing", 0.05)
        turning_radius = self._path_cfg.get("turning_radius", 0.25)
        junction_gap_max_deg = self._path_cfg.get("junction_gap_max_deg", 60)
        len_ratio_max = self._path_cfg.get("via_len_ratio_max", 5.0)
        max_allowed = point_spacing / max(turning_radius, 0.01)  # 1× theoretical limit

        # --- Junction gaps ---
        junction_gaps = []
        for i in range(len(segments) - 1):
            gap = abs(np_angle_wrap(segments[i + 1][0, 2] - segments[i][-1, 2]))
            junction_gaps.append(gap)
            if gap > np.deg2rad(junction_gap_max_deg):  # hard reject: >N° gap
                return None
        mean_junction_gap = float(np.mean(junction_gaps)) if junction_gaps else 0.0

        # --- Length balance ---
        seg_lens = [float(np.sum(np.hypot(
            np.diff(seg[:, 0]), np.diff(seg[:, 1]),
        ))) for seg in segments]
        if min(seg_lens) < 1e-9:
            return None  # degenerate segment
        len_ratio = max(seg_lens) / min(seg_lens)
        if len_ratio > len_ratio_max:  # hard reject: >N× imbalance
            return None

        # --- Segment curvature ---
        max_curv_ratio = 0.0
        for seg in segments:
            if len(seg) < 2:
                continue
            dtheta = np.abs(np.diff(seg[:, 2]))
            dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
            seg_max = float(np.max(dtheta)) if len(dtheta) > 0 else 0.0
            max_curv_ratio = max(max_curv_ratio, seg_max / max(max_allowed, 1e-9))

        # --- Obstacle clearance (hard reject only) ---
        clearance = self._path_cfg.get("start_goal_clearance", 1.0)
        for vp in via_pts:
            if obstacles and MapGenerator._check_collision(
                np.array([[vp[0], vp[1]]]), obstacles, clearance,
            ):
                return None

        # Weighted score (lower = better)
        # Normalize junction_gap against configured max
        junction_score = mean_junction_gap / np.deg2rad(junction_gap_max_deg)
        # Normalize length_ratio: 1.0 = 0, max = 1.0
        balance_score = (len_ratio - 1.0) / max(len_ratio_max - 1.0, 1.0)
        # Curvature ratio: 0 = 0, 1.0 = 1.0 (at theoretical max)
        curv_score = min(max_curv_ratio, 1.0)

        return 0.4 * junction_score + 0.3 * balance_score + 0.3 * curv_score

    @staticmethod
    def _validate_path_curvature(path: np.ndarray, point_spacing: float,
                                  min_turning_radius: float,
                                  max_violation_ratio: float = 0.02) -> bool:
        """校验路径是否满足车辆最小转弯半径约束。

        Args:
            path: (N, 3) waypoints (x, y, theta)。
            point_spacing: 路径点间距（米）。
            min_turning_radius: 车辆最小转弯半径（米）。
            max_violation_ratio: 允许的超限点比例（超此比例则拒绝）。

        Returns:
            True 如果路径可被车辆跟随。
        """
        if len(path) < 3:
            return True
        headings = path[:, 2]
        # 相邻点间的角度差，归一化到 [-π, π]
        dtheta = np.diff(headings)
        dtheta = np_angle_wrap(dtheta)
        # 最大允许角度差 = 点间距 / 最小转弯半径
        max_dtheta = point_spacing / max(min_turning_radius, 0.01)
        # 允许 2x 理论最大角度差（离散采样噪声，was 3x）
        violations = np.sum(np.abs(dtheta) > max_dtheta * 2.0)
        return violations / max(len(dtheta), 1) < max_violation_ratio

    # ------------------------------------------------------------------
    # 折返（cusp）合法性 —— 2026-10-28
    # ------------------------------------------------------------------
    # 参考路径里的「倒车腿」= 一段连续的反向段。运行时 envs/ackermann_env.py 的
    # segment_is_reverse 用同一判据（段运动方向 vs 鼻向夹角 > π/2），因此这里的
    # 静态判定与训练时的动态判定同源。两条判据（均可由 config 单独关掉）：
    #
    #   1. 短折返：某个极大倒车段的弧长 < min_cusp_fold_arc。这种「倒 0.5 m 再往前
    #      开」的路径会让 path_dir / heading_error / 档位闸门在几个点内翻两次，
    #      而车长 0.38 m —— 生理上就是无意义的抖一下。
    #   2. 窄发卡：倒车腿与前进腿（弧长间隔 >= cusp_min_arc_gap）最近距离
    #      < min_cusp_leg_separation。两条腿在车身半宽量级内并行时，车在一条腿上
    #      做横向修正就会贴到另一条腿的参考线；观测侧 ±25 点（≈±1.25 m）的最近点
    #      窗口在那一段同时含两条腿。量法见 cusp_leg_separation 的 docstring
    #      （它是"存在一对弧长相隔 >= gap 却贴得极近的点"，不是平行偏移本身）。
    #
    # 实测（2026-10-28，真 planner，每 stage 60 条候选路径，min_arc_gap=1.0）：
    #   含倒车段 55.8%；折返弧长 < 0.8 m 的候选图 33%（stage 50/42/44/57/66/74%），
    #   腿间距 < 0.35 m 的 14%（40/21/10/16/29/26%），两者任一 41%。
    #   开启后产出的路径违规 0/60；reset 墙钟 stage5 p50 480→569 ms。
    #   用户裁决（2026-10-28）取两条都开 —— 接受 stage3-5 实际难度变轻的代价。
    @staticmethod
    def _path_segment_dirs(path: np.ndarray):
        """逐段返回 (段长, 段中点弧长, 是否倒车)，长度均为 N-1。"""
        arr = np.asarray(path, dtype=np.float64)
        d = np.diff(arr[:, :2], axis=0)
        seg_len = np.hypot(d[:, 0], d[:, 1])
        move = np.arctan2(d[:, 1], d[:, 0])
        is_rev = np.abs(np_angle_wrap(move - arr[:-1, 2])) > np.pi / 2
        arc_end = np.cumsum(seg_len)
        arc_start = arc_end - seg_len
        return seg_len, 0.5 * (arc_start + arc_end), is_rev

    @staticmethod
    def cusp_fold_arc_lengths(path: np.ndarray) -> list[float]:
        """所有**极大倒车段**的弧长（米）；无倒车段 ⇒ 空列表。"""
        if path is None or len(path) < 3:
            return []
        seg_len, _, is_rev = MapGenerator._path_segment_dirs(path)
        out: list[float] = []
        i = 0
        while i < len(is_rev):
            if not is_rev[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(is_rev) and is_rev[j + 1]:
                j += 1
            out.append(float(seg_len[i:j + 1].sum()))
            i = j + 1
        return out

    @staticmethod
    def cusp_leg_separation(path: np.ndarray,
                            min_arc_gap: float = 1.0) -> float | None:
        """倒车腿与前进腿的最小欧氏距离（米，按段中点量）。

        只比较弧长间隔 >= min_arc_gap 的段对 —— 折返顶点附近两条腿按构造相交，
        把那里的 0 距离算进来会让这个量失去意义（实测 median 随 min_arc_gap
        从 0.4 m 的 0.45 涨到 1.5 m 的 1.70，全靠截断位置决定）。

        None = 没有可比较的对（无倒车段，或所有倒车段都在 min_arc_gap 之内）。

        ⚠️ 这个量**不等于两腿的平行偏移**：宽发卡（>=1 m）时最小的一对落在连接
        段与反向段之间，量值可以小于几何偏移（合成 1.20 m 发卡上量到 0.73）。
        阈值语义因此是"存在一对弧长相隔 >= gap、却近到 X 以内的点" —— 正是
        PathExtractor 的 ±25 点最近点窗口会串腿的那个条件。
        """
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim != 2 or len(arr) < 3:
            return None
        seg_len, mid_s, is_rev = MapGenerator._path_segment_dirs(arr)
        i_rev = np.where(is_rev)[0]
        i_fwd = np.where(~is_rev)[0]
        if len(i_rev) == 0 or len(i_fwd) == 0:
            return None
        mid_p = np.column_stack([0.5 * (arr[:-1, 0] + arr[1:, 0]),
                                 0.5 * (arr[:-1, 1] + arr[1:, 1])])
        dist = np.hypot(mid_p[i_rev][:, None, 0] - mid_p[i_fwd][None, :, 0],
                        mid_p[i_rev][:, None, 1] - mid_p[i_fwd][None, :, 1])
        gap_ok = np.abs(mid_s[i_rev][:, None] - mid_s[i_fwd][None, :]) >= min_arc_gap
        if not gap_ok.any():
            return None
        return float(np.where(gap_ok, dist, np.inf).min())

    def _cusp_violation(self, path: np.ndarray) -> str | None:
        """折返合法性：返回违规原因（None = 合法）。

        两个阈值都为 0 ⇒ 关闭检查，恒合法（= 加入本功能前的行为）。
        """
        min_fold = self.min_cusp_fold_arc
        min_sep = self.min_cusp_leg_separation
        if min_fold <= 0.0 and min_sep <= 0.0:
            return None
        if path is None or len(path) < 3:
            return None
        if min_fold > 0.0:
            folds = MapGenerator.cusp_fold_arc_lengths(path)
            short = [f for f in folds if f < min_fold]
            if short:
                return (f"短折返：倒车段弧长 {min(short):.2f} m "
                        f"< min_cusp_fold_arc {min_fold:.2f} m")
        if min_sep > 0.0:
            sep = MapGenerator.cusp_leg_separation(
                path, min_arc_gap=self.cusp_min_arc_gap)
            if sep is not None and sep < min_sep:
                return (f"窄发卡：倒车腿与前进腿间距 {sep:.2f} m "
                        f"< min_cusp_leg_separation {min_sep:.2f} m")
        return None

    def _generate_path_direct(self, start, goal, obstacles, rng, point_spacing,
                               theta_range, goal_theta_range):
        """直接 start→goal Hybrid-A* 搜索，带回退链。"""
        reverse_prob = self._path_cfg.get("reverse_probability", 0.3)
        path_reverse_prob = self._path_cfg.get("path_reverse_probability", 0.3)

        # 以 goal 方向为基准，start_theta_range 控制偏移量（非绝对角度）
        goal_dir = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
        offset = float(rng.uniform(theta_range[0], theta_range[1]))
        start_theta = goal_dir + offset
        goal_theta = goal_dir + float(rng.uniform(goal_theta_range[0], goal_theta_range[1]))

        if rng.random() < reverse_prob:
            start_theta = np_angle_wrap(goal_dir + np.pi)

        path, info = hybrid_astar_search(
            (start[0], start[1], start_theta),
            (goal[0], goal[1], goal_theta),
            obstacles, self._path_cfg, rng,
        )

        if path is not None and info["success"]:
            reversed_flag = rng.random() < path_reverse_prob
            if reversed_flag:
                path = path[::-1]
                # 翻转朝向角: 路径点序反向 → 运动方向相反 → 朝向需 +π
                path[:, 2] = np_angle_wrap(path[:, 2] + np.pi)
                path[0, :2] = goal
                path[-1, :2] = start
            else:
                path[0, :2] = start
                path[-1, :2] = goal
            path = self._resample_path(path, point_spacing)
            # Validate against vehicle's physical minimum turning radius (0.25m),
            # NOT the search conservative parameter (turning_radius in config may be
            # larger for planning margin — using it for validation is too strict).
            if self._validate_path_curvature(path, point_spacing,
                                              min_turning_radius=0.25):
                path = self._smooth_path(path, obstacles, margin=self._robot_inflation)
                # start_theta 取自平滑后的路径首点朝向（_smooth_path 从 XY 重算朝向）
                if reversed_flag:
                    start_theta = float(path[0, 2])
                return path, start_theta
            # 路径曲率超限，继续 fallback

        # Fallback: RS curve with deterministic 8-direction grid search.
        # 8×8=64 combos sorted by closeness to goal_dir — faster and no
        # straggler tail vs random 500-try loop.
        warnings.warn("Hybrid-A* failed, falling back to RS curve")
        radius = self._path_cfg.get("turning_radius", 0.25)
        dist = np.hypot(goal[0] - start[0], goal[1] - start[1])
        rs_num_points = max(2, int(np.ceil(dist / point_spacing)) + 1)
        use_reverse = self._path_cfg.get("reverse_primitives", False)

        _angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 8 directions
        _valid_rs: list[tuple[float, np.ndarray, list, float, float]] = []

        for _a_st in _angles:
            for _a_gt in _angles:
                rs_pose, rs_segs = reeds_shepp_path_pose(
                    (start[0], start[1], float(_a_st)), (goal[0], goal[1], float(_a_gt)),
                    radius, rs_num_points,
                )
                if rs_pose is None:
                    continue
                if not use_reverse and any(s['type'] == 'backward' for s in rs_segs):
                    continue
                rs_xy = rs_pose[:, :2]
                if rs_xy.min() < 0.0 or rs_xy.max() > 10.0:
                    continue
                if obstacles is not None and MapGenerator._check_collision(
                        rs_xy, obstacles, self._robot_inflation):
                    continue
                _len = float(sum(abs(s['length']) for s in rs_segs))
                _valid_rs.append((_len, rs_pose, rs_segs, float(_a_st), float(_a_gt)))

        _valid_rs.sort(key=lambda x: x[0])  # shortest path first

        for _len, rs_pose, rs_segs, _st, _gt in _valid_rs:
            # rs_pose[:, 2] is NOSE heading from vehicle integration.
            # Keep as-is — downstream consumers (PurePursuit, env heading_err)
            # expect nose direction. Motion direction derived via path_dir at runtime.
            # XY 必须取自**本条**候选（rs_pose[:, :2]），不能用枚举循环遗留的
            # rs_xy —— 那个变量出内层循环后等于「最后一条枚举到的候选」，
            # 会把 A 候选的 XY 配 B 候选的朝向，且这条 XY 从未做过碰撞检查。
            rs_path = np.column_stack([rs_pose[:, :2], rs_pose[:, 2]])
            reversed_flag = rng.random() < path_reverse_prob
            if reversed_flag:
                rs_path = rs_path[::-1]
                rs_path[:, 2] = np_angle_wrap(rs_path[:, 2] + np.pi)
                _st = float(rs_path[0, 2])
            if reversed_flag:
                rs_path[0, :2] = goal
                rs_path[-1, :2] = start
            else:
                rs_path[0, :2] = start
                rs_path[-1, :2] = goal
            rs_path = self._resample_path(rs_path, point_spacing)
            if self._validate_path_curvature(rs_path, point_spacing,
                                              min_turning_radius=0.25):
                return rs_path, _st

        # 最终回退：直线路径
        warnings.warn("RS fallback failed, using straight line")
        dist = np.hypot(goal[0] - start[0], goal[1] - start[1])
        n_pts = max(2, int(np.ceil(dist / point_spacing)) + 1)
        t = np.linspace(0, 1, n_pts)
        heading = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
        xy = np.column_stack([
            start[0] + t * (goal[0] - start[0]),
            start[1] + t * (goal[1] - start[1]),
        ])
        if obstacles and MapGenerator._check_collision(
                xy, obstacles, self._robot_inflation):
            warnings.warn("Straight-line fallback passes through obstacles — regenerating map")
            return None, None
        path = np.column_stack([xy, np.full(n_pts, heading)])
        return path, heading

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_path_headings(path: np.ndarray) -> np.ndarray:
        """Compute heading at each waypoint from path direction.

        Uses central differences for interior points, forward/backward
        differences for endpoints.
        """
        n = len(path)
        headings = np.empty(n, dtype=np.float64)
        for i in range(n):
            if i == 0:
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
            elif i == n - 1:
                dx = path[-1][0] - path[-2][0]
                dy = path[-1][1] - path[-2][1]
            else:
                dx = path[i + 1][0] - path[i - 1][0]
                dy = path[i + 1][1] - path[i - 1][1]
            headings[i] = np.arctan2(dy, dx)
        return headings

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_collision(path: np.ndarray, obstacles: list[dict],
                         safety_margin: float = 0.0) -> bool:
        """Return True if any path point collides with any obstacle.

        Vectorized: O(N+M) via broadcasting, replaces O(N×M) double loop.

        旋转约定与 env SAT / LiDAR 一致（局部系 R(−θ)）。2026-08-17 修复：
        本函数原先对 sinθ 取负以补偿 ``_check_collision_batch`` 的镜像 bug；
        该 bug 已在 hybrid_astar.py 内修正，故此补偿一并删除（否则双重翻转）。
        """
        circles, rects = _obstacles_to_arrays(obstacles)
        return bool(_check_collision_batch(
            np.asarray(path[:, :2], dtype=np.float64), circles, rects, safety_margin,
        ).any())

    def _check_path_footprint(self, path: np.ndarray, obstacles: list[dict]) -> bool:
        """env 级 SAT 足迹复验：路径上任一位姿(x,y,θ)车体是否与障碍/墙体重叠。

        与 ``NumpySimBridge._check_collision`` 语义等价（逐位姿复刻，不做近似）：
        - 矩形障碍：车体 OBB vs 障碍 OBB 的 4 轴 SAT（机器人自身 2 轴 + 障碍 2 轴）
        - 圆形障碍：车体四角落入"圆 + margin" ∪ 圆心落入"车体矩形 ± (half + r)"
          （bridge 用后者做保守判定，角落处比精确圆-矩形距离更严格，两处必须保持一致）
        - 世界边界：与 bridge 相同的**按朝向**轴对齐包围盒判定（0.05 m margin，
          旋转后的半宽/半高，P2-16）
        障碍在 margin 上外扩 ``ROBOT_COLLISION_MARGIN``（与 bridge 的 0.03 一致）。

        Args:
            path: (N, 3) [x, y, θ]（θ 为车头朝向）。
            obstacles: 障碍物 dict 列表（可为空）。

        Returns:
            True 表示存在碰撞位姿（该地图应被丢弃）。
        """
        if path is None or len(path) == 0:
            return True
        p = np.asarray(path, dtype=np.float64)
        px = p[:, 0]
        py = p[:, 1]
        th = p[:, 2] if p.shape[1] >= 3 else np.zeros(len(p))
        ct, st = np.cos(th), np.sin(th)
        m = float(self._path_cfg.get("robot_collision_margin", self.ROBOT_COLLISION_MARGIN))
        # 车体半长/半宽统一走 property（path_cfg 显式覆盖 > env_cfg 推导 > 常量），
        # 保证规划/复验足迹与 env.yaml 的 robot.shape 同源（任务 3）。
        hl = self.robot_half_length + m
        hw = self.robot_half_width + m

        if obstacles:
            circles, rects = _obstacles_to_arrays(obstacles)

            # ── 圆形障碍：复刻 NumpySimBridge._check_collision 的两项判定 ──
            # (a) 车体四角落入圆内 (半径 + margin)
            # (b) 圆心落入"车体矩形外扩 r"的 AABB（bridge 的保守判定，角落处比
            #     精确距离更严格）—— 必须与 bridge 保持一致，否则复验放行的位姿
            #     env 仍判碰撞。
            for i in range(circles.shape[0]):
                cx, cy, cr = circles[i, 0], circles[i, 1], circles[i, 2]
                r_eff = cr + m
                dx = cx - px
                dy = cy - py
                # (a) 四角
                for _sx, _sy in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
                    wx = px + ct * _sx - st * _sy
                    wy = py + st * _sx + ct * _sy
                    if np.any((wx - cx) ** 2 + (wy - cy) ** 2 < r_eff * r_eff):
                        return True
                # (b) 圆心在车体局部系内（±(hl+r), ±(hw+r)）
                lx = dx * ct + dy * st          # R(−θ)
                ly = -dx * st + dy * ct
                if np.any((np.abs(lx) < hl + cr) & (np.abs(ly) < hw + cr)):
                    return True

            # ── 矩形障碍：OBB-OBB 4 轴 SAT ──
            for i in range(rects.shape[0]):
                cx, cy = rects[i, 0], rects[i, 1]
                rhl, rhw = rects[i, 2] + m, rects[i, 3] + m
                cr_, sr_ = rects[i, 4], rects[i, 5]
                dx = cx - px
                dy = cy - py
                # 轴1/2：车体自身轴 (ct,st) / (−st,ct)
                d1 = np.abs(ct * dx + st * dy)
                r1 = rhl * np.abs(ct * cr_ + st * sr_) + rhw * np.abs(-ct * sr_ + st * cr_)
                sep1 = d1 > hl + r1
                d2 = np.abs(-st * dx + ct * dy)
                r2 = rhl * np.abs(-st * cr_ + ct * sr_) + rhw * np.abs(st * sr_ + ct * cr_)
                sep2 = d2 > hw + r2
                # 轴3/4：障碍自身轴 (cr_,sr_) / (−sr_,cr_)
                d3 = np.abs(cr_ * dx + sr_ * dy)
                r3 = hl * np.abs(cr_ * ct + sr_ * st) + hw * np.abs(-cr_ * st + sr_ * ct)
                sep3 = d3 > r3 + rhl
                d4 = np.abs(-sr_ * dx + cr_ * dy)
                r4 = hl * np.abs(-sr_ * ct + cr_ * st) + hw * np.abs(sr_ * st + cr_ * ct)
                sep4 = d4 > r4 + rhw
                if np.any(~(sep1 | sep2 | sep3 | sep4)):
                    return True

        # ── 世界边界：按朝向的轴对齐包围盒（P2-16，与 bridge 逐位一致）──
        # 旧实现用 hl 判 x、hw 判 y，与 θ 无关：θ=90° 时车身沿 y 的半宽其实是
        # hl（0.22 含 margin）却按 hw（0.13）判 ⇒ 车尾可侵入墙 0.09 m 仍判无碰撞。
        # ros_bridge/numpy_bridge.py::_check_collision 是同一份逻辑，两处必须同改。
        wb = self.world_bounds
        wall_margin = 0.05
        ext_x = np.abs(ct) * hl + np.abs(st) * hw
        ext_y = np.abs(st) * hl + np.abs(ct) * hw
        outside = ((px - ext_x < wb[0] + wall_margin) | (px + ext_x > wb[1] - wall_margin)
                   | (py - ext_y < wb[2] + wall_margin) | (py + ext_y > wb[3] - wall_margin))
        return bool(np.any(outside))

    # ------------------------------------------------------------------
    # 参考路径横向抖动（path_perturb_sigma，2026-10-23）
    # ------------------------------------------------------------------

    def _perturb_path_lateral(self, path: np.ndarray, rng) -> np.ndarray:
        """沿路径法向给参考路径注入 N(0, sigma) 横向误差（sigma=0 时原样返回）。

        目的：模拟"定位误差 / 地图过期 / 规划精度不足"，让参考路径可能贴得比车体
        还近，从而制造"盲跟路径会撞、必须主动偏离"的实验条件。抖动发生在
        _check_path_footprint 复验**之后** —— 复验只看规划器输出的路径 ⇒
        "规划器以为是安全的、实际贴得更近"这一语义成立。

        保持的性质（tests/test_path_perturbation.py 逐条锁住）:
          a) 首点与末点位置不动（env 的 reset/goal 判定用起终点位姿）；
          b) 抖动后的点仍在 world_bounds 内 —— 逐点把标量偏移夹进可行区间
             [d_lo, d_hi]（clip 关于 d 连续 ⇒ 不引入跳变）；
          c) theta 列（鼻向）一个数都不改 —— 方向语义由 env 独立判定；
          d) 只由传入的 rng 驱动 ⇒ 同 seed 可复现。

        sigma == 0 时**不抽任何样本**：env 跨 reset 复用同一个 Generator，
        抽一次就会改变后续地图的随机流 ⇒ "默认 0 逐位不变"会失效。
        """
        sigma = self.path_perturb_sigma
        if path is None or sigma <= 0.0:
            return path
        p = np.asarray(path)
        if p.ndim != 2 or p.shape[1] < 2 or p.shape[0] < 3:
            return path
        xy = np.asarray(p[:, :2], dtype=np.float64)
        n = xy.shape[0]

        # 切线：只用路径点自身的 (x, y) 中心差分 —— **不用 theta 列**：
        # 鼻向在倒车段与运动方向差 pi，拿它算法向会把抖动方向弄反。
        tang = np.empty_like(xy)
        tang[1:-1] = xy[2:] - xy[:-2]
        tang[0] = xy[1] - xy[0]
        tang[-1] = xy[-1] - xy[-2]
        norm = np.hypot(tang[:, 0], tang[:, 1])
        ok = norm > 1e-12
        tangent = np.zeros_like(xy)
        tangent[ok] = tang[ok] / norm[ok, None]
        # 法向 = 切线旋转 90 度。顺/逆时针都可以：N(0, sigma) 关于 0 对称，
        # 抖动方向的正负号不影响分布。
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

        if self.path_perturb_mode == "control":
            # 位移场必须沿路径连续：交叉点（前进 <-> 倒车档位切换）上相邻点的切线
            # 反转 ~180 度、法向随之反转，若沿用逐点法向，同一条平滑插值出的偏置会
            # 在那里变成方向相反的两个位移（实测单步 0.55 m，正是本模式最大的粗糙源）。
            # 横向偏置的符号本身任意（N(0, sigma) 关于 0 对称），故取"沿路径连续"的
            # 那一支。point 模式**不改**（首版行为逐位冻结）。
            normal = self._continuous_normal(normal)
            d = self._sample_control_offsets(xy, sigma, rng)
        else:
            # point（默认）：每个点独立抽 N(0, sigma) —— 2026-10-23 首版行为，
            # sigma=0 时下面的早退分支保证逐位不变。
            d = rng.normal(0.0, sigma, size=n)
        d[0] = 0.0
        d[-1] = 0.0
        d[~ok] = 0.0          # 零长度切线处法向无定义 ⇒ 该点不抖
        d = self._clip_offset_to_bounds(xy, normal, d)

        out = np.array(path, copy=True)   # 保留 dtype / 列数 / theta 列原值
        out[:, 0] = xy[:, 0] + d * normal[:, 0]
        out[:, 1] = xy[:, 1] + d * normal[:, 1]
        return out

    @staticmethod
    def _continuous_normal(normal: np.ndarray) -> np.ndarray:
        """把逐点法向按"相邻点内积 >= 0"做符号延拓（消除交叉点处的 180 度反转）。

        参考路径第 2 列存的是鼻向，前进/倒车切换处 XY 切线会反转 ~180 度：法向也
        跟着反转，于是"同一条连续的横向偏置"在反转点两侧变成方向相反的位移。
        把符号取成沿路径连续后，位移场连续，单步位移回到 |Δd| 量级。
        零长度切线的点法向为 0（内积 0），不触发翻转。
        """
        n = normal.shape[0]
        if n < 2:
            return normal
        dot = np.einsum("ij,ij->i", normal[1:], normal[:-1])
        signs = np.cumprod(np.where(dot < 0.0, -1.0, 1.0))
        out = normal.copy()
        out[1:] *= signs[:, None]
        return out

    def _sample_control_offsets(self, xy: np.ndarray, sigma: float,
                                rng) -> np.ndarray:
        """mode="control"：在控制点抽 N(0, sigma)，沿弧长线性插值到每个路径点。

        为什么需要（实测，见 tests/test_path_perturbation.py 与对比表）：逐点 iid
        噪声在 0.05 m 点距上等价于"每 5 cm 换一次方向"，sigma=0.2/0.3 时平均点距从
        0.049 m 涨到 0.24/0.35 m、单步最大位移 1.0/1.5 m、每图 6~18 个自交 ——
        那不是"定位漂移导致的平滑横向误差"，而是一条锯齿折线。控制点插值把噪声的
        空间相关长度钉成 path_perturb_control_spacing（默认 0.5 m），相邻点的偏移
        差被压到 sigma*点距/间距 量级 ⇒ 路径仍是一条合理曲线。

        控制点按弧长 0, s, 2s, ..., L 取（末点强制落在终点上），首末两个控制点的
        抽样强制为 0 ⇒ 首末点位置不动。退化情形（路径总长 < s ⇒ 只有两个控制点）
        插值恒 0，即**根本不抖**（要抖就把 spacing 调小，这是有意的保守行为）。
        """
        n = xy.shape[0]
        seg = np.hypot(*np.diff(xy, axis=0).T) if n > 1 else np.zeros(0)
        arc = np.concatenate([[0.0], np.cumsum(seg)]) if n > 0 else np.zeros(0)
        total = float(arc[-1]) if arc.size else 0.0
        spacing = self.path_perturb_control_spacing
        if not (total > 1e-9) or not (spacing > 0.0):
            return np.zeros(n)
        n_ctrl = int(np.floor(total / spacing))
        ctrl_arc = np.arange(n_ctrl + 1, dtype=np.float64) * spacing
        if total - ctrl_arc[-1] > 1e-9:
            ctrl_arc = np.append(ctrl_arc, total)
        draws = rng.normal(0.0, sigma, size=ctrl_arc.size)
        draws[0] = 0.0
        draws[-1] = 0.0
        return np.interp(arc, ctrl_arc, draws)

    def _clip_offset_to_bounds(self, xy: np.ndarray, normal: np.ndarray,
                               d: np.ndarray) -> np.ndarray:
        """把标量横向偏移 d 逐点夹进"点仍在世界边界内"的可行区间。

        对每个轴单独求可行区间：p_axis + d * n_axis in [lo, hi]
          n_axis > 0 -> d in [(lo - p) / n, (hi - p) / n]
          n_axis < 0 -> d in [(hi - p) / n, (lo - p) / n]
          n_axis = 0 -> 该轴不构成约束
        区间端点关于 d 连续，clip(d, lo, hi) 因而连续 ⇒ 夹边界不会产生跳变。
        原路径点已在世界内（_check_path_footprint 保证），故 lo <= 0 <= hi；
        退化情形（点本就在界外 / lo > hi）取 0，即该点不抖。
        """
        wb = self.world_bounds
        n = d.shape[0]
        lo = np.full(n, -np.inf)
        hi = np.full(n, np.inf)
        for axis, (b_lo, b_hi) in enumerate(((wb[0], wb[1]), (wb[2], wb[3]))):
            p_c = xy[:, axis]
            n_c = normal[:, axis]
            pos = n_c > 0.0
            neg = n_c < 0.0
            hi[pos] = np.minimum(hi[pos], (b_hi - p_c[pos]) / n_c[pos])
            lo[pos] = np.maximum(lo[pos], (b_lo - p_c[pos]) / n_c[pos])
            hi[neg] = np.minimum(hi[neg], (b_lo - p_c[neg]) / n_c[neg])
            lo[neg] = np.maximum(lo[neg], (b_hi - p_c[neg]) / n_c[neg])
        feasible = lo <= hi
        return np.where(feasible, np.clip(d, lo, hi), 0.0)
    # ------------------------------------------------------------------
    # Goal heading
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_goal_heading(path: np.ndarray) -> float:
        """返回**终点处的鼻向（nose heading）**，float ∈ [-π, π]。

        B5 修复 (2026-10-05)。原实现取最后两点 XY 差 ``atan2(dy, dx)``——那是
        **运动方向**，不是车头朝向。参考路径第 2 列存的是鼻向（RS 车辆积分所得，
        见 ``_generate_path`` 的注释 "rs_pose[:, 2] is NOSE heading from vehicle
        integration. Keep as-is — downstream consumers expect nose direction"），
        倒车段上 ``运动方向 = 鼻向 ± π``。

        为什么必须改（已量化）: stage3/stage4 分别约 20% 的地图末段是倒车段
        （实测 20.0% / 21.5%）。这些地图上策略"正确地倒车到达"终点时车头朝向
        与旧 ``goal_heading`` 相差 π，于是
          1. 终点奖励 ``envs/reward.py`` 的 ``10×cos(goal_heading_error)`` 给 −10
             而非 +10（±20 分差）；
          2. 观测 ``goal_sig``（``envs/observation.py`` 的 sin/cos(ghe)）把目标
             朝向指反，等价于给策略一个错误的终点姿态目标。
        统一为"终点鼻向"后，两处语义都与"车到达时应让车头对齐终点鼻向"一致。

        抗 ±π 环绕: 优先直接使用 ``path[-1, 2]``（积分所得，无噪声，不需要平滑）。
        仅当末点角度非有限时才回退到最后若干点的**圆均值**
        （``arctan2(mean sin, mean cos)``）——绝不能对角度做算术平均：
        ``(+3.0 与 −3.0)`` 的算术平均是 0.0，方向恰好相反。

        2 列路径（无 θ）没有鼻向信息，退回旧的 XY 差分（运动方向），
        保持对 (N, 2) 输入的向后兼容。
        """
        if path is None or len(path) == 0:
            return 0.0
        p = np.asarray(path, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] == 0:
            return 0.0

        if p.shape[1] >= 3:
            last = float(p[-1, 2])
            if np.isfinite(last):
                return float(np_angle_wrap(last))
            # 末点角度异常：用最后 GOAL_HEADING_AVG_WINDOW 个有限点的圆均值兜底
            tail = p[-MapGenerator.GOAL_HEADING_AVG_WINDOW:, 2]
            tail = tail[np.isfinite(tail)]
            if tail.size > 0:
                return float(np_angle_wrap(np.arctan2(
                    float(np.mean(np.sin(tail))), float(np.mean(np.cos(tail))),
                )))
            return 0.0

        if len(p) >= 2:
            dx = p[-1, 0] - p[-2, 0]
            dy = p[-1, 1] - p[-2, 1]
            return float(np.arctan2(dy, dx))
        return 0.0

    def _compute_initial_heading(self, path, start_theta=None):
        if start_theta is not None:
            return start_theta
        if len(path) >= 2:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            return float(np.arctan2(dy, dx))
        return 0.0
