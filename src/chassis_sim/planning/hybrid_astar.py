# ruff: noqa: BLE001, I001, PLR1730, RUF059
# fmt: off
"""Hybrid A* path planner for Ackermann vehicles.

Implements search on (x, y, theta) continuous state space using motion
primitives and analytic expansion (Reeds-Shepp curves) — aligned with
Nav2's Smac Hybrid-A* planner.
"""
from __future__ import annotations
import math
import heapq
import numpy as np
try:
    # numba 是可选加速器（extra: planning）。core 安装（无 numba）下退化为
    # 等价的纯 Python 函数 —— 语义不变，只是慢（见 pyproject 的 planning extra）。
    from numba import jit
except ImportError:  # pragma: no cover - 由 tests/test_planning_core.py 子进程覆盖

    def jit(*args, **kwargs):
        # 恒等装饰器：同时支持 @jit 与 @jit(...) 两种写法。
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator
from chassis_sim.planning.rs_curve import reeds_shepp_path_pose

# Inline angle_wrap for numba compatibility (cannot call path_utils.angle_wrap
# from nopython JIT).
@jit(nopython=True, inline='always')
def _angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Obstacle pre-processing: convert dict list to numba-compatible arrays
# ---------------------------------------------------------------------------

def _obstacles_to_arrays(
    obstacles: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert obstacle dicts to (circles, rects) float arrays.

    circles: (N_c, 3)  [cx, cy, radius]
    rects:   (N_r, 6)  [cx, cy, half_l, half_w, cos_theta, sin_theta]
    """
    circles = []
    rects = []
    for obs in obstacles:
        shape = obs["shape"]
        ox, oy, otheta = float(obs["state"][0]), float(obs["state"][1]), float(obs["state"][2])
        if shape["name"] == "circle":
            circles.append((ox, oy, float(shape["radius"])))
        elif shape["name"] == "rectangle":
            half_l = float(shape["length"]) / 2.0
            half_w = float(shape["width"]) / 2.0
            rects.append((ox, oy, half_l, half_w, np.cos(otheta), np.sin(otheta)))
    return (
        np.array(circles, dtype=np.float64) if circles else np.zeros((0, 3), dtype=np.float64),
        np.array(rects, dtype=np.float64) if rects else np.zeros((0, 6), dtype=np.float64),
    )


# ── 机器人足迹膨胀默认值 (m) ──────────────────────────────────────
# 规划器把机器人当质点，必须显式膨胀障碍。车体 0.38×0.20 → 半宽 0.10、
# 半长 0.19、外接半径 0.215。
# 0.16 = 半宽 + 0.06 余量。实测折中（obstacles_dense，60 张地图各）：
#   0.16 → 最小中心线净空 0.161 m，几何专家到达率 20%，规划 66 ms/map
#   0.22 → 最小中心线净空 0.199 m，几何专家到达率 27%，规划 72 ms/map
# 但 0.22 会让 maze_grid（0.4 m 方块 + 1.0 m 栅格 → 0.6 m 走廊）在部分
# RNG 序列下 10 次重试全部无解（SAT 复验拒绝所有穿走廊路径）→ 默认仍取 0.16。
# 需要更大跟踪余量时，按阶段在 curriculum.yaml 里覆盖
# ``hybrid_astar.robot_inflation``（稀疏/中等障碍可放宽到 0.20-0.22）。
# 无论取值如何，最终由 MapGenerator.generate() 的 env 级 SAT 足迹复验兜底。
DEFAULT_ROBOT_INFLATION = 0.16


DEFAULT_CONFIG = {
    "xy_resolution": 0.5,
    "theta_bins": 16,

    "turning_radius": 0.25,
    "primitive_length": 0.5,
    "reverse_primitives": False,
    "analytic_interval": 5,
    "heuristic_scale": 1.1,
    "max_iterations": 2000,
    "num_points": 20,
    "obstacle_penalty_weight": 2.0,
    "obstacle_penalty_range": 1.0,
    # 世界边界 (xmin, xmax, ymin, ymax)。这里只是**兜底值**（= 默认 env.yaml
    # 的 10×10）；真实配置由 MapGenerator 解析后经 config 传入（P2-15：
    # 显式 curriculum 覆盖 > env.yaml world.width/height > 本兜底），
    # 搜索/启发式/解析扩展四处都读同一个 cfg["world_bounds"]。
    "world_bounds": [0.0, 10.0, 0.0, 10.0],
    "reverse_cost_multiplier": 1.5,
    "via_points": {"enabled": False, "grid": 3, "count": [1, 2], "clearance": 0.5},
    # ── 机器人足迹膨胀 (2026-08-17) ──────────────────────────────
    # 规划器检查的是"路径中心点"，而车体是 0.38×0.20 m。此膨胀半径把障碍
    # 在局部系两个方向各外扩 robot_inflation，使中心线保持净空。
    # 0.16 m ≈ 半宽 0.10 + 0.06 余量；最终仍有 env 级 SAT 足迹复验兜底。
    "robot_inflation": DEFAULT_ROBOT_INFLATION,
    # 解析扩展 (RS) 密集碰撞采样步长与点数上限（防粗采样漏检小障碍）
    "analytic_check_step": 0.05,
    "analytic_max_points": 3000,
    # 组装路径的输出采样间距（None = 保持旧行为 num_points 个点）
    "output_point_spacing": None,
    "output_max_points": 4000,
}

# curvature = 1 / turning_radius = 4.0 m⁻¹
_FORWARD_PRIMITIVES = [
    ("forward", 0.0, 0.5),
    ("forward", 4.0, 0.5),
    ("forward", -4.0, 0.5),
]

_BACKWARD_PRIMITIVES = [
    ("backward", 0.0, 0.5),
    ("backward", 4.0, 0.5),
    ("backward", -4.0, 0.5),
]


def _discretize(x: float, y: float, theta: float, cfg: dict) -> tuple:
    """Map continuous pose to grid index for closed-set lookup."""
    res = cfg["xy_resolution"]
    bins = cfg["theta_bins"]
    ti = int(_angle_wrap(theta) / (2 * math.pi / bins)) % bins
    return (int(x / res), int(y / res), ti)


def _integrate_primitive(
    x: float, y: float, theta: float,
    curvature: float, length: float, direction: str,
) -> tuple[float, float, float]:
    """Integrate a motion primitive from (x, y, theta).

    Args:
        curvature: 1/r (positive=left), 0 (straight), -1/r (right).
        length: arc length in meters.
        direction: 'forward' or 'backward'.

    Returns:
        (x', y', theta') after the primitive.
    """
    sign = 1.0 if direction == "forward" else -1.0

    if abs(curvature) < 1e-9:
        return (
            x + sign * length * np.cos(theta),
            y + sign * length * np.sin(theta),
            theta,
        )

    r = 1.0 / abs(curvature)
    # dtheta = direction * length * curvature: 物理学正确。
    # 前进-左转: dθ>0(cow) → 路径左转。后退-左转: dθ<0(cw) → 路径左转。
    # curvature 始终是路径曲率（+4=左转路径），不因方向改变。
    dtheta = sign * length * curvature
    theta_new = _angle_wrap(theta + dtheta)

    if curvature > 0:  # 左转路径 — 圆心在车体左侧
        cx = x - r * np.sin(theta)
        cy = y + r * np.cos(theta)
        nx = cx + r * np.sin(theta_new)
        ny = cy - r * np.cos(theta_new)
    else:  # 右转路径 — 圆心在车体右侧
        cx = x + r * np.sin(theta)
        cy = y - r * np.cos(theta)
        nx = cx - r * np.sin(theta_new)
        ny = cy + r * np.cos(theta_new)

    return (nx, ny, theta_new)


def _sample_primitive_points(
    x: float, y: float, theta: float,
    curvature: float, length: float, direction: str,
    n_samples: int,
) -> np.ndarray:
    """Generate *n_samples*+1 equally-spaced (x,y) points along a primitive.

    Returns (n_samples+1, 2) array for batch collision checking.
    """
    sign = 1.0 if direction == "forward" else -1.0
    fracs = np.linspace(0.0, 1.0, n_samples + 1)
    dists = fracs * length

    if abs(curvature) < 1e-9:
        return np.column_stack([
            x + sign * dists * np.cos(theta),
            y + sign * dists * np.sin(theta),
        ])

    r = 1.0 / abs(curvature)
    dthetas = sign * dists * curvature
    thetas = theta + dthetas

    if curvature > 0:
        cx = x - r * np.sin(theta)
        cy = y + r * np.cos(theta)
        nx = cx + r * np.sin(thetas)
        ny = cy - r * np.cos(thetas)
    else:
        cx = x + r * np.sin(theta)
        cy = y - r * np.cos(theta)
        nx = cx - r * np.sin(thetas)
        ny = cy + r * np.cos(thetas)

    return np.column_stack([nx, ny])


# ---------------------------------------------------------------------------
# JIT collision primitives (work on pre-converted arrays, not dicts)
# ---------------------------------------------------------------------------

@jit(nopython=True)
def _check_collision_point_arr(
    x: float, y: float,
    circles: np.ndarray, rects: np.ndarray,
    margin: float = 0.0,
) -> bool:
    """Check single point against pre-converted obstacle arrays (numba-fast).

    ⚠ 局部坐标必须是 R(−θ)：lx = c·dx + s·dy, ly = −s·dx + c·dy（c=cosθ, s=sinθ）。
    旧实现写成 lx = c·dx − s·dy, ly = s·dx + c·dy，等价于把矩形镜像到 −θ 朝向，
    与 env SAT / LiDAR 的真实几何不一致（θ≠0 时 ~28% 近障点判定错误）。
    """
    if not np.isfinite(x) or not np.isfinite(y):
        return True
    for i in range(circles.shape[0]):
        cx, cy, cr = circles[i, 0], circles[i, 1], circles[i, 2] + margin
        if (x - cx) * (x - cx) + (y - cy) * (y - cy) < cr * cr:
            return True
    for i in range(rects.shape[0]):
        cx, cy = rects[i, 0], rects[i, 1]
        hl, hw = rects[i, 2] + margin, rects[i, 3] + margin
        cos_t, sin_t = rects[i, 4], rects[i, 5]
        dx = x - cx
        dy = y - cy
        lx = dx * cos_t + dy * sin_t
        ly = -dx * sin_t + dy * cos_t
        if abs(lx) < hl and abs(ly) < hw:
            return True
    return False


@jit(nopython=True)
def _obstacle_distance_arr(
    x: float, y: float,
    circles: np.ndarray, rects: np.ndarray,
) -> float:
    """Min distance to obstacle boundary (numba-fast). Returns inf if none.

    局部坐标同 ``_check_collision_point_arr``：R(−θ)。
    """
    best = np.inf
    for i in range(circles.shape[0]):
        cx, cy, cr = circles[i, 0], circles[i, 1], circles[i, 2]
        d = np.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - cr
        if d < best:
            best = d
    for i in range(rects.shape[0]):
        cx, cy = rects[i, 0], rects[i, 1]
        hl, hw = rects[i, 2], rects[i, 3]
        cos_t, sin_t = rects[i, 4], rects[i, 5]
        dx = x - cx
        dy = y - cy
        lx = dx * cos_t + dy * sin_t
        ly = -dx * sin_t + dy * cos_t
        dx_in = max(0.0, abs(lx) - hl)
        dy_in = max(0.0, abs(ly) - hw)
        d = np.sqrt(dx_in * dx_in + dy_in * dy_in)
        if d < best:
            best = d
    return max(0.0, best)


def _check_collision_batch(
    points: np.ndarray,
    circles: np.ndarray, rects: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    """Vectorized collision check: (N,2) points → (N,) bool.

    矩形局部坐标用 R(−θ)（与 ``_check_collision_point_arr`` / env SAT 一致）。
    """
    n = len(points)
    result = np.zeros(n, dtype=bool)
    if n == 0:
        return result
    px, py = points[:, 0], points[:, 1]

    for i in range(circles.shape[0]):
        cx, cy, cr = circles[i, 0], circles[i, 1], circles[i, 2] + margin
        result |= (px - cx) ** 2 + (py - cy) ** 2 < cr * cr

    for i in range(rects.shape[0]):
        cx, cy = rects[i, 0], rects[i, 1]
        hl, hw = rects[i, 2] + margin, rects[i, 3] + margin
        cos_t, sin_t = rects[i, 4], rects[i, 5]
        dx = px - cx
        dy = py - cy
        lx = dx * cos_t + dy * sin_t
        ly = -dx * sin_t + dy * cos_t
        result |= (np.abs(lx) < hl) & (np.abs(ly) < hw)

    return result


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (dict-based obstacle interface)
# These delegate to the JIT array versions so existing callers keep working.
# ---------------------------------------------------------------------------

def _check_collision_point(
    x: float, y: float, obstacles: list[dict], margin: float = 0.0,
) -> bool:
    """Check if a single point collides with any obstacle (dict-based, compat)."""
    circles, rects = _obstacles_to_arrays(obstacles)
    return _check_collision_point_arr(x, y, circles, rects, margin)


def _check_path_collision(
    path: np.ndarray, obstacles: list[dict], margin: float = 0.0,
) -> bool:
    """Check if any point in path collides with any obstacle (dict-based, compat)."""
    circles, rects = _obstacles_to_arrays(obstacles)
    for row in path:
        px, py = float(row[0]), float(row[1])
        if _check_collision_point_arr(px, py, circles, rects, margin):
            return True
    return False


def _obstacle_distance(x: float, y: float, obstacles: list[dict]) -> float:
    """Minimum distance from (x, y) to any obstacle boundary (dict-based, compat)."""
    circles, rects = _obstacles_to_arrays(obstacles)
    return float(_obstacle_distance_arr(x, y, circles, rects))


def _obstacle_penalty(x: float, y: float, obstacles: list[dict], cfg: dict) -> float:
    """Soft penalty for proximity to obstacles (dict-based, compat)."""
    weight = cfg.get("obstacle_penalty_weight", 2.0)
    rng = cfg.get("obstacle_penalty_range", 1.0)
    d = _obstacle_distance(x, y, obstacles)
    if d >= rng:
        return 0.0
    return weight * (1.0 - d / rng)


def _obstacle_penalty_arr(
    x: float, y: float,
    circles: np.ndarray, rects: np.ndarray,
    cfg: dict,
) -> float:
    """Soft penalty for proximity to obstacles (array-based, for search loop)."""
    weight = cfg.get("obstacle_penalty_weight", 2.0)
    rng = cfg.get("obstacle_penalty_range", 1.0)
    d = _obstacle_distance_arr(x, y, circles, rects)
    if d >= rng:
        return 0.0
    return weight * (1.0 - d / rng)


def _build_dijkstra_heuristic(
    goal: tuple[float, float, float],
    obstacles: list[dict],
    cfg: dict,
    circles: np.ndarray | None = None,
    rects: np.ndarray | None = None,
) -> np.ndarray:
    """Build 2D Dijkstra distance table from goal (ignoring orientation).

    Uses 1m resolution (coarser than search grid) for speed.
    Returns (H, W) array where cell = shortest obstacle-avoiding
    distance to goal cell, or inf if unreachable.
    """
    bounds = cfg.get("world_bounds", [0.0, 10.0, 0.0, 10.0])
    res = 1.0  # coarser grid for heuristic
    nx = int((bounds[1] - bounds[0]) / res)
    ny = int((bounds[3] - bounds[2]) / res)

    gx = int((goal[0] - bounds[0]) / res)
    gy = int((goal[1] - bounds[2]) / res)

    INF = float("inf")
    dist = np.full((nx, ny), INF)
    if 0 <= gx < nx and 0 <= gy < ny:
        dist[gx, gy] = 0.0
    else:
        return dist  # goal outside world bounds — all cells unreachable

    # Use spatial hash if available, otherwise fall back to full obstacle scan
    _circles = circles if circles is not None else np.zeros((0, 3))
    _rects = rects if rects is not None else np.zeros((0, 6))

    heap = [(0.0, gx, gy)]
    # 4-connected neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        d, cx, cy = heapq.heappop(heap)
        if d > dist[cx, cy]:
            continue
        for dx, dy in neighbors:
            nx_i, ny_i = cx + dx, cy + dy
            if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                continue
            # Cell center in world coords
            wx = bounds[0] + (nx_i + 0.5) * res
            wy = bounds[2] + (ny_i + 0.5) * res
            if _check_collision_point_arr(wx, wy, _circles, _rects, margin=0.1):
                continue
            nd = d + np.hypot(dx * res, dy * res)
            if nd < dist[nx_i, ny_i]:
                dist[nx_i, ny_i] = nd
                heapq.heappush(heap, (nd, nx_i, ny_i))

    return dist


def _heuristic_lookup(
    x: float, y: float,
    dist_table: np.ndarray, cfg: dict,
    h_rs_floor: float = 0.0,
) -> float:
    """Heuristic: max(2D Dijkstra, RS floor). RS floor computed once at start."""
    h_dijkstra = float('inf')
    bounds = cfg.get("world_bounds", [0.0, 10.0, 0.0, 10.0])
    res = 1.0
    ix = int((x - bounds[0]) / res)
    iy = int((y - bounds[2]) / res)
    nx, ny = dist_table.shape
    if 0 <= ix < nx and 0 <= iy < ny:
        h_dijkstra = float(dist_table[ix, iy])
    return max(h_dijkstra, h_rs_floor)




def _try_analytic_expansion(
    x: float, y: float, theta: float,
    goal: tuple[float, float, float],
    obstacles: list[dict],
    cfg: dict,
    circles: np.ndarray | None = None,
    rects: np.ndarray | None = None,
) -> np.ndarray | None:
    """Try Reeds-Shepp curve from (x,y,theta) to goal.

    Returns (N,3) path if collision-free and in-bounds, None otherwise.

    ⚠ 2026-08-17 修复 (B2/B3)：
    - 碰撞采样必须 ≤ ``analytic_check_step``(默认 0.05 m)。旧实现固定用
      ``num_points``=20 个点检查整条 5–15 m 曲线（间距 0.25–0.75 m），
      小障碍被系统性漏检：实测"障碍精确落在曲线上"的放行率 r=0.1 m 时 23.3%
      （r=0.2 m 时 0%，因为间距恰好小于直径）。修复后两种半径均 0%。
    - 解析检查必须带 ``robot_inflation``，否则中心线贴障、车体必撞。
    - 返回的也是密集曲线：旧实现返回 20 点，下游按 0.05 m 线性插值时
      走的是弦线（R=0.25 m 圆弧 0.5 m 弦矢高 ~0.115 m）→ 切角穿障。
    """
    r = cfg.get("turning_radius", 0.25)
    n_coarse = cfg.get("num_points", 20)
    step = float(cfg.get("analytic_check_step", 0.05))
    max_pts = int(cfg.get("analytic_max_points", 3000))
    inflation = float(cfg.get("robot_inflation", DEFAULT_ROBOT_INFLATION))

    # 先用解析长度估算所需采样点数，再一次性生成密集曲线（避免粗采样漏检）
    from chassis_sim.planning.rs_curve import reeds_shepp_length as _rs_len
    try:
        _len_est = float(_rs_len((x, y, theta), goal, r))
    except Exception:
        # 长度估算失败 → 保守拒绝该扩展，绝不退化回粗采样检查
        return None
    if not np.isfinite(_len_est):
        return None  # 无解析解（长度 inf/NaN）
    if _len_est <= 1e-6:
        n_pts = n_coarse  # 退化曲线（起点即终点）：20 点已足够密集
    else:
        n_pts = int(np.ceil(_len_est / max(step, 1e-6))) + 2
        if n_pts > max_pts:
            # 病态曲线（超长/多 cusp）：拒绝该扩展，而不是放行一条未检查的路径
            return None
        n_pts = max(n_coarse, n_pts)

    rs_pose, segments = reeds_shepp_path_pose(
        (x, y, theta),
        (goal[0], goal[1], goal[2]),
        r, n_pts,
    )
    if rs_pose is None:
        return None
    # 无反向前进时拒绝含倒车段的路径（Stage 0/1 不需要倒车）
    use_reverse = cfg.get("reverse_primitives", False)
    if not use_reverse and any(s['type'] == 'backward' for s in segments):
        return None
    # World bounds check (RS curves can overshoot)
    bounds = cfg.get("world_bounds", [0.0, 10.0, 0.0, 10.0])
    margin = max(0.2, inflation)
    rs_xy = rs_pose[:, :2]
    if (np.min(rs_xy[:, 0]) < bounds[0] - margin or
        np.max(rs_xy[:, 0]) > bounds[1] + margin or
        np.min(rs_xy[:, 1]) < bounds[2] - margin or
        np.max(rs_xy[:, 1]) > bounds[3] + margin):
        return None

    # Collision check with pre-converted arrays (fast path) — 带足迹膨胀
    # ⚠ 若调用方未传 circles/rects，必须从 obstacles 现场转换：
    #   否则碰撞检查会静默变成空数组（no-op），直接放行穿障路径。
    if circles is None or rects is None:
        _c, _r = _obstacles_to_arrays(obstacles or [])
        _circles = _c if circles is None else circles
        _rects = _r if rects is None else rects
    else:
        _circles, _rects = circles, rects
    if not _check_collision_batch(rs_xy, _circles, _rects, margin=inflation).any():
        # rs_pose[:, 2] is NOSE heading from vehicle integration.
        # Keep as-is — pure_pursuit and env heading_err expect nose direction.
        return np.column_stack([rs_xy, rs_pose[:, 2]])
    return None




def hybrid_astar_search(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    obstacles: list[dict],
    config: dict | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Search for a kinematically-feasible, collision-free path.

    Args:
        start: (x, y, theta) start pose.
        goal: (x, y, theta) goal pose.
        obstacles: list of obstacle dicts (MapGenerator format).
        config: search parameters (merged with DEFAULT_CONFIG).
        rng: random generator (unused, reserved for reproducibility).

    Returns:
        (path, info)
        path: (N, 3) waypoints [x, y, theta] in world coords, or None on failure.
        info: dict with cost, nodes_expanded, segments, success.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    # Reset numpy error handling — pytest/ROS plugins may set it to 'raise'
    np.seterr(all="warn")

    # -- Pre-process obstacles once: arrays for JIT + spatial hash ----------
    circles, rects = _obstacles_to_arrays(obstacles)
    bounds = cfg.get("world_bounds", [0.0, 10.0, 0.0, 10.0])

    # Local tie-breaking counter (thread-safe per-call)
    _counter = [0]

    def _make_node():
        _counter[0] += 1
        return _counter[0]

    use_reverse = cfg.get("reverse_primitives", False)
    max_iter = cfg.get("max_iterations", 2000)
    heur_scale = cfg.get("heuristic_scale", 1.1)
    analytic_interval = cfg.get("analytic_interval", 5)
    # 机器人足迹膨胀：规划器把车当质点，必须显式把障碍外扩 (B3)
    inflation = float(cfg.get("robot_inflation", DEFAULT_ROBOT_INFLATION))

    _fwd_prims = list(_FORWARD_PRIMITIVES)
    _bwd_prims = list(_BACKWARD_PRIMITIVES) if use_reverse else []
    primitives = _fwd_prims + _bwd_prims

    # ── Primitive template precomputation ────────────────────────────
    # Precompute relative (dx, dy, dtheta) and collision sample points
    # for each primitive at origin (0,0,0), then translate+rotate per-expansion.
    # Avoids N×M heap allocations in the search loop.
    n_samples = 8  # collision check points per primitive
    _prim_templates: list[dict] = []
    for _p_dir, _p_curv, _p_len in primitives:
        _ep = _integrate_primitive(0.0, 0.0, 0.0, _p_curv, _p_len, _p_dir)
        _samps = _sample_primitive_points(0.0, 0.0, 0.0, _p_curv, _p_len, _p_dir, n_samples)
        _prim_templates.append({
            'dx': _ep[0], 'dy': _ep[1], 'dtheta': _ep[2],
            'samples': _samps,  # (n_samples+1, 2) at origin
            'direction': _p_dir, 'curvature': _p_curv, 'length': _p_len,
        })

    # Build heuristic table
    dist_table = _build_dijkstra_heuristic(
        goal, obstacles, cfg,
        circles=circles, rects=rects,
    )

    # RS floor: compute once from start→goal, use as global heuristic lower-bound.
    # Prevents HA* from degenerating to BFS near the goal on dense obstacle maps.
    from chassis_sim.planning.rs_curve import reeds_shepp_length as _rs_len
    _r = cfg.get("turning_radius", 0.25)
    _rs_floor = _rs_len(start, goal, _r) * 0.5  # discounted — obstacles not accounted

    # Heuristic for start node
    h_start = _heuristic_lookup(start[0], start[1], dist_table, cfg, _rs_floor)

    # Priority queue: (f, tiebreaker, g, x, y, theta, parent_key, seg_label)
    open_set = []
    heapq.heappush(open_set, (h_start, _make_node(), 0.0,
                              start[0], start[1], start[2], None, "start"))

    # Closed set: grid_key → best g (already expanded)
    closed = {}
    # Best g seen so far for each grid key (including open set)
    best_g: dict = {}
    # Parent map: grid_key → parent_grid_key
    parent = {}
    # Node coords: grid_key → (x, y, theta)
    _node_coords = {}
    # Node segments: grid_key → list[dict] of segments from parent
    _node_segments = {}
    # Node edges: grid_key → (parent_key, samples_x, samples_y, dtheta, theta0)
    # 用于回溯时重建 primitive 的真实圆弧几何（否则只能拿节点位姿连弦线，
    # 0.5 m 弦在 R=0.25 m 圆弧上的矢高可达 0.115 m → 切角穿障）
    _node_edge = {}

    start_key = _discretize(start[0], start[1], start[2], cfg)
    _node_coords[start_key] = (start[0], start[1], start[2])
    _node_segments[start_key] = []

    nodes_expanded = 0

    while open_set:
        f_val, _, g_val, x, y, theta, parent_key, _seg_label = heapq.heappop(open_set)
        cur_key = _discretize(x, y, theta, cfg)

        if cur_key in closed and closed[cur_key] <= g_val:
            continue
        closed[cur_key] = g_val

        if cur_key not in _node_coords:
            _node_coords[cur_key] = (x, y, theta)

        nodes_expanded += 1
        if nodes_expanded > max_iter:
            break

        # Analytic expansion (lazy RS — first valid mode)
        if nodes_expanded % analytic_interval == 0:
            rs_path = _try_analytic_expansion(
                x, y, theta, goal, obstacles, cfg,
                circles=circles, rects=rects,
            )
            if rs_path is not None:
                # Found! Extract path via backtrack
                full_path = _backtrack_and_assemble(
                    cur_key, parent, _node_coords, rs_path, cfg, goal,
                    node_edge=_node_edge,
                )
                return full_path, {
                    "cost": g_val,
                    "nodes_expanded": nodes_expanded,
                    "segments": _node_segments.get(cur_key, []),
                    "success": True,
                }

        # Expand motion primitives — use precomputed templates
        for _tmpl in _prim_templates:
            direction, curvature, length = _tmpl['direction'], _tmpl['curvature'], _tmpl['length']

            # Endpoint: translate+rotate template
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            nx = x + _tmpl['dx'] * cos_t - _tmpl['dy'] * sin_t
            ny = y + _tmpl['dx'] * sin_t + _tmpl['dy'] * cos_t
            ntheta = _angle_wrap(theta + _tmpl['dtheta'])
            if not (bounds[0] <= nx <= bounds[1] and bounds[2] <= ny <= bounds[3]):
                continue

            # Collision: translate+rotate template sample points (带足迹膨胀)
            s = _tmpl['samples']  # (n_pts, 2) at origin
            sx = x + s[:, 0] * cos_t - s[:, 1] * sin_t
            sy = y + s[:, 0] * sin_t + s[:, 1] * cos_t
            if _check_collision_batch(
                np.column_stack([sx, sy]), circles, rects, inflation,
            ).any():
                continue

            nkey = _discretize(nx, ny, ntheta, cfg)

            # Cost: primitive length + obstacle penalty
            step_cost = length + _obstacle_penalty_arr(nx, ny, circles, rects, cfg)
            if direction == "backward":
                step_cost *= cfg.get("reverse_cost_multiplier", 1.5)
            ng = g_val + step_cost

            if nkey in closed and closed[nkey] <= ng:
                continue
            if nkey in best_g and best_g[nkey] <= ng:
                continue  # already reached this cell with lower cost
            best_g[nkey] = ng

            nh = _heuristic_lookup(nx, ny, dist_table, cfg, _rs_floor)
            nf = ng + nh * heur_scale

            heapq.heappush(open_set, (nf, _make_node(), ng,
                                      nx, ny, ntheta, cur_key, direction))
            parent[nkey] = cur_key
            _node_coords[nkey] = (nx, ny, ntheta)
            # 记录该 primitive 的世界系采样点、总转角与起始朝向（theta 为展开时
            # 实际使用的父节点朝向），供回溯重建真实圆弧
            _node_edge[nkey] = (cur_key, sx, sy, _tmpl['dtheta'], float(theta))

            seg_type = "forward" if direction == "forward" else "backward"
            if curvature > 0:
                seg_dir = "left"
            elif curvature < 0:
                seg_dir = "right"
            else:
                seg_dir = "straight"
            _node_segments[nkey] = _node_segments.get(cur_key, []) + [{
                "type": seg_type, "length": length,
                "curvature": curvature, "direction": seg_dir,
            }]

    # Search failed
    return None, {
        "cost": float("inf"),
        "nodes_expanded": nodes_expanded,
        "segments": [],
        "success": False,
    }


def _backtrack_and_assemble(
    goal_key: tuple,
    parent: dict,
    coords: dict,
    rs_path: np.ndarray,
    cfg: dict,
    goal: tuple[float, float, float],
    node_edge: dict | None = None,
) -> np.ndarray:
    """Backtrack parent pointers and assemble with RS tail into final path.

    Returns (N, 3) array with columns [x, y, theta].

    ``node_edge``: grid_key → (parent_key, sx, sy, dtheta, theta0)，即该节点
    primitive 的世界系采样点（含 parent 位姿，故首点重复）、该 primitive 的总转角
    与实际起始朝向。提供时用采样点重建真实圆弧，避免"节点连弦线"导致的切角
    穿障 (2026-08-17 B2/B3)。
    """
    n_pts = cfg.get("num_points", 20)
    _osp = cfg.get("output_point_spacing", None)

    # Collect waypoints in reverse: (x, y, theta)
    waypoints = []
    seen = set()
    cur = goal_key
    chain = []
    while cur is not None:
        if cur in seen:
            break  # cycle detected — truncate
        seen.add(cur)
        chain.append(cur)
        cur = parent.get(cur)
    chain.reverse()

    for idx, node in enumerate(chain):
        edge = node_edge.get(node) if node_edge is not None else None
        if idx == 0 or edge is None:
            # 根节点（无入边）或缺少采样记录 → 退回节点位姿
            if node in coords:
                waypoints.append(coords[node])
            continue
        # 入边采样：跳过首点（与父节点位姿重复），末点即本节点位姿
        sx, sy, dtheta, theta0 = edge[1], edge[2], edge[3], edge[4]
        n_s = len(sx) - 1
        if n_s < 1:
            if node in coords:
                waypoints.append(coords[node])
            continue
        for i in range(1, n_s + 1):
            waypoints.append((float(sx[i]), float(sy[i]),
                              float(_angle_wrap(theta0 + dtheta * (i / n_s)))))

    # Append RS path tail with heading derived from segment direction.
    # Skip rs_path[0] — it duplicates the last backtracked waypoint
    # (both are the RS expansion node).
    if rs_path is not None and len(rs_path) > 1:
        rs_path = rs_path[1:]
        for i, p in enumerate(rs_path):
            px, py = float(p[0]), float(p[1])
            h = float(p[2]) if rs_path.shape[1] >= 3 else (
                goal[2] if i == len(rs_path) - 1 else 0.0)
            waypoints.append((px, py, h))

    # 输出点数：默认保持旧行为 (num_points)；设置 output_point_spacing 时按弧长
    # 加密，避免 20 点粗采样把真实曲率拉成弦线（下游还会按 0.05 m 再重采样）。
    if _osp:
        try:
            _total = 0.0
            for i in range(1, len(waypoints)):
                _total += float(np.hypot(waypoints[i][0] - waypoints[i - 1][0],
                                         waypoints[i][1] - waypoints[i - 1][1]))
            _cap = int(cfg.get("output_max_points", 4000))
            n_pts = int(min(_cap, max(n_pts, np.ceil(_total / float(_osp)) + 1)))
        except (TypeError, ValueError):
            pass

    return _resample_waypoints(waypoints, n_pts)


def _resample_waypoints(waypoints: list, n_pts: int) -> np.ndarray:
    """Resample waypoints to exactly n_pts via cumulative distance.

    Handles (x, y) or (x, y, theta) waypoints.
    Returns (N, 2) or (N, 3) array.
    """
    if len(waypoints) < 2:
        if len(waypoints) == 1:
            return np.tile(np.array([waypoints[0]]), (n_pts, 1))
        dim = len(waypoints[0]) if waypoints else 2
        return np.zeros((n_pts, dim))

    pts = np.array(waypoints)
    has_theta = pts.shape[1] >= 3
    xy = pts[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumsum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumsum[-1]

    if total < 1e-9:
        return np.tile(pts[0:1], (n_pts, 1))

    sample_dists = np.linspace(0, total, n_pts)
    out_dim = 3 if has_theta else 2
    result = np.empty((n_pts, out_dim))
    for i, d in enumerate(sample_dists):
        idx = np.searchsorted(cumsum, d, side="right") - 1
        idx = max(0, min(idx, len(pts) - 2))
        t = (d - cumsum[idx]) / max(seg_lens[idx], 1e-9)
        t = max(0.0, min(1.0, t))
        # Position: linear interpolation
        result[i, 0] = pts[idx, 0] + t * diffs[idx, 0]
        result[i, 1] = pts[idx, 1] + t * diffs[idx, 1]
        # Theta: circular interpolation
        if has_theta:
            t0 = pts[idx, 2]
            t1 = pts[idx + 1, 2]
            dtheta = _angle_wrap(t1 - t0)
            result[i, 2] = _angle_wrap(t0 + t * dtheta)
    return result
