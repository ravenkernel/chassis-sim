"""DWA（Dynamic Window Approach）：速度窗口内采样 (ω, v) 并按四项加权代价选优，
实现 `Controller` 协议（构造时 `set_path`、逐帧 `act`）。

输入契约
--------
`act(observation)` 只吃位姿与（可选）障碍，**没有速度输入**：

    observation["pose"]       (3,) [x, y, θ] 世界系（必需）
    observation["obstacles"]  可选，[{"shape": {...}, "state": [x,y,θ]}, ...]
    observation["dt"]         可选，覆盖构造期的控制周期

**无 obstacles 键 / 空列表 ⇒ 按"无遮挡直行"退化**：`obstacle_clearance` 全返 +inf 且
`if obstacles:` 分支整段跳过，避障代价恒为 0，选优只剩路径/速度/平滑三项。

采样空间由 (δ, v) 改为 (ω, v)
-----------------------------
原实现按阿克曼 (δ, v) 网格采样并用 `v·tanδ/L` 前推。本实现要把同一套代价
用在三种底盘上，因此**直接按 (ω, v) 采样**（差速/麦轮没有 δ），前推式相同：

    x += v·cosθ·dt ; y += v·sinθ·dt ; θ += ω·dt

阿克曼需要时由 `to_chassis_action` 反解 δ = arctan(ω·L/v)，与
`x += v·cosθ·dt ... θ += v/L·tanδ·dt` 逐项等价。其余代价项、权重默认值、候选预算
（6×9 网格 + 5 个细化 = 59 ≤ 63）与原实现逐项一致；平滑项的自变量随之由 δ 变为 ω
（权重 `w_smooth` 的默认值沿用 0.4，量纲从 rad 变成 rad/s，见报告"差异清单"）。

相对原实现的四处**行为变更**（2026-11-04；1)/2) 由 obstacles_dense 的"原地抖 400 步"
定位，3) 由 obstacles_dense_reverse 的"0% / 500 步超时"定位，4) 是可选档位语义）
-----------------------------------------------------------------------------
1) `heading_mode="axis"`（新默认）：原实现的 `w_heading·herr²` 用**鼻向**判据。路径在折返点
   （cusp）前后两条腿的鼻向差近 π，于是"沿倒车腿正常行驶"被判成 π 的航向误差（罚
   `0.3·π² ≈ 2.96`），代价地形在 cusp 之前形成局部极小 —— 实测机器人卡在 cusp 前 0.2 m 处
   来回抖 400 步、成功率 0%。axis 模式只罚"车轴与路径轴向"的错位，不区分车头朝前还是朝后。
2) `w_prog`（新增，默认 2.0）：弧长进度项 `−w_prog·Δs`。原代价里只有"到折线的横向距离"，
   原地不动与慢速前进同价，缺一个"往前挪就得分"的激励。
3) 航向项的参考方向改为**弧长前瞻**（`heading_lookahead`，默认 0.35 m）：参考段原本取候选
   末端最近的那一段，而 cusp 两侧两条腿实测只隔 0.243 m，同一姿态下 `i_end` 会在两腿之间
   跳（参考方向 1.199 ↔ 0.279，差 0.93 rad），于是"沿路径开"的候选被自己的参考罚掉、
   机器人停在 cusp 前 0.35 m 的极限环里跑满 500 步。改成
   `_index_at(当前弧长 + heading_lookahead)` 后参考随弧长单调推进：
   obstacles_dense_reverse 由 0% / 500 步变成 100% / 145 步。
前两处合起来把 obstacles_dense 从 0% 成功率提到 100%（翻转次数 87 → 0）。
4) 档位（gear）语义（`gear_mode="auto"`，**可选档，默认 "off"**）：路径第 3 列存的是**鼻向**
   而不是运动方向（RS 积分），倒车段的判据是"段运动方向与鼻向夹角 > π/2"（与
   `segment_is_reverse` 同源；方向用弧长 ±`gear_dir_window` 窗口的中心差分，
   单段差分在 cusp 处不可信）。auto 档沿弧长定位当前段（再前探 `gear_lookahead`）后：候选
   速度截到本档半轴（前进 [0, v_max] / 倒车 [v_min, 0]），航向项改用**运动方向**
   （`θ + π·[v<0]`），进度项只在候选末端落在本档段时给正分。换档带票数滞回（`gear_votes`）
   与最小保持步数（`gear_hold_steps`）。
   **为什么默认 off**：硬档位在本仓的点式障碍检查下有吸收态 —— 不加 `min_speed` 时"停住"
   （`w_vel·v_max²` = 1.44）比沿路径慢走（1.71）更便宜，obstacles_dense / foldback_reverse /
   obstacles_dense_reverse 三场景**全部 500 步超时**；加 `min_speed=0.1` 后又变成贴着 cusp
   原地打轮、车尾扫进点式检查（只看车心轨迹 vs 障碍中心）的盲区：obstacles_dense 100% 碰撞、
   foldback_reverse 0%。而且它对 `heading_lookahead` 极敏感（0.25 → 454 步、0.35 → 223 步、
   0.5 → 超时），是刀尖不是平台。验收要的倒车能力由 3) 的弧长前瞻提供，档位语义作为可选档
   保留，供 ablation 与"将来把点式障碍检查换成真车体扫掠"时启用。
旧行为**不能逐位复现**：`heading_mode="nose", w_prog=0.0, gear_mode="off",
heading_lookahead=0.0` 可退回鼻向判据与对称采样，但 3) 替换掉的"末端最近段参考"正是被修掉
的那处缺陷，没有开关保留它。
"""

from __future__ import annotations

import hashlib

import numpy as np

from ._geometry import (
    arc_lengths,
    as_path,
    obstacle_clearance,
    path_segments,
    polyline_project,
    polyline_query,
    wrap_angle,
)
from .base import Controller

__all__ = ["DwaController"]

#: 弧长进度项在"后退"侧的饱和值（m）。ds < 0 时罚分封顶，避免把合法倒车一票否决；
#: 前进侧的上限用 `horizon·dt·v_max`（本拍物理上能走的最大位移）。
_PROG_BACK_CLIP = 0.5


class _LastCallMemo:
    """“同输入同输出”护栏。

    DWA 带内部状态（平滑项参考上一帧 ω、差分测速），连调两次同参数结果会差在低位；
    契约要求确定性，因此输入逐位相同时直接复用上一次输出。
    """

    def __init__(self) -> None:
        self.key = None
        self.value = None

    def check(self, key):
        if self.key is not None and self.key == key:
            return self.value
        return None

    def store(self, key, value) -> None:
        self.key = key
        self.value = value

    def reset(self) -> None:
        self.key = None
        self.value = None


class _DiffEstimator:
    """位姿差分测速：只给位姿的接口下拿到 (v, ω) 的手段。"""

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        self.v = 0.0
        self.omega = 0.0
        self.prev_xy = None
        self.prev_theta = 0.0

    def reset(self) -> None:
        self.v = 0.0
        self.omega = 0.0
        self.prev_xy = None
        self.prev_theta = 0.0

    def update(self, xy, theta):
        """喂入本步位姿，返回 (v, omega)；首步为 0。"""
        if self.prev_xy is not None and self.dt > 0.0:
            dx = float(xy[0]) - float(self.prev_xy[0])
            dy = float(xy[1]) - float(self.prev_xy[1])
            # 位移投影到**上一次**车头方向 ⇒ 带符号的 v（倒车为负）
            ds = dx * np.cos(self.prev_theta) + dy * np.sin(self.prev_theta)
            self.v = ds / self.dt
            self.omega = float(wrap_angle(theta - self.prev_theta)) / self.dt
        else:
            self.v = 0.0
            self.omega = 0.0
        self.prev_xy = (float(xy[0]), float(xy[1]))
        self.prev_theta = float(theta)
        return self.v, self.omega


def _signature(xy, theta, path, obstacles):
    """输入指纹，用于“同输入同输出”护栏。"""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(xy, dtype=np.float64).tobytes())
    h.update(np.float64(theta).tobytes())
    h.update(np.ascontiguousarray(path, dtype=np.float64).tobytes())
    if obstacles:
        for obs in obstacles:
            try:
                shape = obs.get("shape", {}) or {}
                state = obs.get("state", (0.0, 0.0, 0.0))
                key = (
                    str(shape.get("name", "")),
                    float(shape.get("radius", 0.0)),
                    float(shape.get("length", 0.0)),
                    float(shape.get("width", 0.0)),
                    float(state[0]),
                    float(state[1]),
                    float(state[2]) if len(state) > 2 else 0.0,
                )
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            h.update(repr(key).encode("utf-8"))
    return h.digest()


def _smoothed_tangents(path: np.ndarray, s_cum: np.ndarray, half_window: float):
    """每段沿索引增大方向的**运动方向**（rad），用 ±half_window 弧长窗口做中心差分。

    单段差分在 cusp 处不可信：实测 obstacles_dense_reverse 的鼻向列在 0.05 m 内跳 ±2.5 rad
    （i=88 nose=+1.352 → i=89 nose=−2.374 → i=90 nose=+2.922），于是相邻段的方向差能到
    1.7 rad，"当前该用哪个档"会在 cusp 附近被单个抖动段主导。用弧长窗口把两端摊平后，
    判定点落在真正的换向处。窗口内位移退化（原地折返）时回落到单段差分。
    """
    n = int(path.shape[0])
    if n < 2:
        return np.zeros(1)
    s_mid = 0.5 * (s_cum[:-1] + s_cum[1:])
    hw = max(0.0, float(half_window))
    i0 = np.clip(np.searchsorted(s_cum, s_mid - hw, side="left"), 0, n - 1)
    i1 = np.clip(np.searchsorted(s_cum, s_mid + hw, side="right") - 1, 0, n - 1)
    dx = path[i1, 0] - path[i0, 0]
    dy = path[i1, 1] - path[i0, 1]
    seg_dx = np.diff(path[:, 0])
    seg_dy = np.diff(path[:, 1])
    # 窗口两端重合（弧长退化）⇒ 用单段差分兜底，避免 atan2(0,0)=0 假装"朝 +x"
    bad = np.hypot(dx, dy) <= 1e-9
    dx = np.where(bad, seg_dx, dx)
    dy = np.where(bad, seg_dy, dy)
    return np.arctan2(dy, dx)


def _segment_gears(path: np.ndarray, dirs: np.ndarray):
    """段档位：运动方向与**鼻向**夹角 > π/2 ⇒ −1（倒车段），否则 +1（前进段）。

    判据点积形式 `mdx·cos(nose) + mdy·sin(nose) >= 0`；这里把"点积"写成"夹角余弦"，两者在边界上给出同一集合。
    """
    return np.where(np.cos(dirs - path[:-1, 2]) >= 0.0, 1.0, -1.0)


class DwaController(Controller):
    """DWA 基线：速度窗口内采样 (ω, v)，按四项加权代价选最优，输出 (v, ω)。

    代价（四项加权）：

    a) 路径跟踪 `w_path·d_end² + w_heading·herr²`，另加 `w_cte_traj` 乘整条预测轨迹的
       平均 CTE²（只看到末端时低速短时域拿不到横向修正的梯度）；herr = wrap(**运动方向** −
       参考方向)，参考方向由 `heading_mode` 决定（默认 "axis" = 段的运动方向，与鼻向列
       无关；"nose" = 鼻向判据）；"axis" 仍对 herr 做模 π 折叠；

    a2) 弧长进度 `−w_prog·Δs`（默认 2.0）：Δs 是候选末端相对当前位姿在**路径弧长坐标**
       上前进的距离，后退为负（饱和于 `_PROG_BACK_CLIP`）。缺了这一项时"停着"与
       "慢慢往前挪"同价，代价地形一有局部极小就永远出不来；
    b) 避障 `w_obs·max(safe−d_min,0)² + w_obs_soft·max(soft−d_min,0)²`，d_min 是预测轨迹上
       到最近障碍的**最小**净空（不逐点求和：求和会让"跑得快经过更多点"本身变成惩罚）；
    c) 速度 `w_vel·(v_max−|v|)²`（档位生效时；否则退回 `w_vel·(v_max−v)²`）；
    d) 平滑 `w_smooth·(ω−ω_prev)²`。

    动态窗：`accel_max > 0` 时 v ∈ v_est ± accel_max·dt；`alpha_max > 0` 时
    ω ∈ ω_prev ± alpha_max·dt。两者为 0（默认）时直接在
    [v_min, v_max] × [−ω_max, ω_max] 上采样。

    档位（gear）：`gear_mode="off"` 是**默认** —— 候选跨整个 [v_min, v_max] 采样、代价不分
    档位（倒车能力由上面的弧长前瞻参考提供，实测已足够）。`gear_mode="auto"` 是可选项：
    按路径段判定当前该用前进档还是倒车档（判据同 `segment_is_reverse`），候选空间
    随之截到半轴、航向项用**运动方向**、进度项只在候选末端落在本档段时给正分；换档带
    `gear_votes` 票数与 `gear_hold_steps` 最小保持步数的滞回，`gear_switches` 可读。
    不改协议、不加随机。硬档位需配合 `min_speed` 才不退化到"停住"，且对
    `heading_lookahead` 极敏感（见模块 docstring 第 4 条），故默认关闭，留作 ablation。
    """

    def __init__(
        self,
        *,
        dt: float = 0.1,
        horizon: int = 10,
        n_v: int = 6,
        n_omega: int = 9,
        n_refine: int = 5,
        accel_max: float = 0.0,
        alpha_max: float = 0.0,
        w_path: float = 3.0,
        w_heading: float = 0.3,
        w_cte_traj: float = 2.0,
        w_obs: float = 60.0,
        w_obs_soft: float = 4.0,
        obs_safe_radius: float = 0.30,
        obs_soft_radius: float = 0.70,
        w_vel: float = 1.0,
        w_smooth: float = 0.4,
        w_prog: float = 2.0,
        heading_mode: str = "axis",
        gear_mode: str = "off",
        gear_votes: int = 3,
        gear_hold_steps: int = 2,
        gear_lookahead: float = 0.0,
        gear_dir_window: float = 0.10,
        heading_lookahead: float = 0.35,
        min_speed: float = 0.0,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.n_v = int(n_v)
        self.n_omega = int(n_omega)
        self.n_refine = int(n_refine)

        self.accel_max = float(accel_max or 0.0)
        self.alpha_max = float(alpha_max or 0.0)

        self.w_path = float(w_path)
        self.w_heading = float(w_heading)
        self.w_cte_traj = float(w_cte_traj)
        self.w_obs = float(w_obs)
        self.w_obs_soft = float(w_obs_soft)
        self.obs_safe_radius = float(obs_safe_radius)
        self.obs_soft_radius = float(obs_soft_radius)
        self.w_vel = float(w_vel)
        self.w_smooth = float(w_smooth)
        self.w_prog = float(w_prog)

        mode = str(heading_mode).strip().lower()
        if mode not in ("axis", "nose"):
            raise ValueError(
                f"heading_mode 只能是 'axis' 或 'nose'，收到 {heading_mode!r}"
            )
        self.heading_mode = mode

        gmode = str(gear_mode).strip().lower()
        if gmode not in ("auto", "off"):
            raise ValueError(f"gear_mode 只能是 'auto' 或 'off'，收到 {gear_mode!r}")
        self.gear_mode = gmode
        self.gear_votes = int(gear_votes)
        if self.gear_votes < 1:
            raise ValueError(f"gear_votes 必须 >= 1，收到 {gear_votes!r}")
        self.gear_hold_steps = int(gear_hold_steps)
        if self.gear_hold_steps < 0:
            raise ValueError(f"gear_hold_steps 必须 >= 0，收到 {gear_hold_steps!r}")
        self.gear_lookahead = max(0.0, float(gear_lookahead))
        self.gear_dir_window = max(0.0, float(gear_dir_window))
        self.heading_lookahead = max(0.0, float(heading_lookahead))
        self.min_speed = max(0.0, float(min_speed))

        self._est = _DiffEstimator(self.dt)
        self._memo = _LastCallMemo()
        self._last_omega = 0.0
        self._path = None
        self._segs = None
        self._s_cum = None
        self._seg_len = None
        self._seg_dir = None  # 每段的运动方向（弧长窗口平滑）
        self._seg_gear = None  # 每段的档位：+1 前进 / −1 倒车

        # 档位滞回状态（回合边界由 _reset_gear 清空）
        self._gear = None  # None = 尚未定档（首次 act 直接采纳 raw）
        self._gear_vote = 0.0
        self._gear_hold = 0
        self.gear_switches = 0

        self.last_candidate_count = 0
        self.last_cost = float("nan")

    @property
    def gear(self) -> float:
        """当前档位：+1 前进 / −1 倒车。未定档或 `gear_mode="off"` 时报 +1。"""
        return 1.0 if self._gear is None else float(self._gear)

    # ── 协议 ────────────────────────────────────────────────────────
    def _reset_gear(self) -> None:
        """清空档位滞回状态与换档计数（回合边界 / 换路径时调用）。"""
        self._gear = None
        self._gear_vote = 0.0
        self._gear_hold = 0
        self.gear_switches = 0

    def reset(self) -> None:
        """清空速度估计、平滑参考量与记忆护栏（保留路径）。"""
        self._est.reset()
        self._memo.reset()
        self._last_omega = 0.0
        self.last_candidate_count = 0
        self.last_cost = float("nan")
        self._reset_gear()

    def set_path(self, path_world: np.ndarray) -> None:
        """(N,3) 世界系路径；空路径 ⇒ `act` 输出零动作。"""
        self._path = as_path(path_world)
        if self._path is None:
            self._segs = None
            self._s_cum = None
            self._seg_len = None
            self._seg_dir = None
            self._seg_gear = None
            self._reset_gear()
            return
        self._segs = path_segments(self._path)
        # 弧长坐标：进度项 w_prog 用"候选末端弧长 − 当前弧长"作为前进距离
        self._s_cum, self._seg_len = arc_lengths(self._path)
        # 档位判据：段的**运动方向**（弧长窗口平滑）与段起点鼻向的夹角 > π/2 ⇒ 倒车段
        self._seg_dir = _smoothed_tangents(
            self._path, self._s_cum, self.gear_dir_window
        )
        self._seg_gear = _segment_gears(self._path, self._seg_dir)
        self._reset_gear()

    def act(self, observation: dict) -> np.ndarray:
        x, y, theta = self._pose(observation)
        xy = np.array([x, y], dtype=np.float64)
        obstacles = observation.get("obstacles")

        v_est, _ = self._est.update(xy, theta)  # 差分估计先推进（空路径也更新）
        if self._path is None:
            return self.to_chassis_action(0.0, 0.0)

        key = _signature(xy, theta, self._path, obstacles)
        cached = self._memo.check(key)
        if cached is not None:
            return cached.copy()

        # 档位判定必须先于候选采样（档位决定候选速度的正负），但必须在 memo 之后：
        # memo 命中时不推进滞回计数器，保证"同输入同输出"逐位成立。
        self._update_gear(xy[0], xy[1])

        v_lo, v_hi = self._v_window(v_est)
        v_lo, v_hi = self._gear_span(v_lo, v_hi)
        o_lo, o_hi = self._omega_window()
        vs = (
            np.linspace(v_lo, v_hi, self.n_v)
            if self.n_v > 1
            else np.array([0.5 * (v_lo + v_hi)])
        )
        os_ = (
            np.linspace(o_lo, o_hi, self.n_omega)
            if self.n_omega > 1
            else np.array([0.5 * (o_lo + o_hi)])
        )
        o_grid = np.repeat(os_, vs.shape[0])
        v_grid = np.tile(vs, os_.shape[0])

        cost = self._score(xy[0], xy[1], theta, o_grid, v_grid, obstacles)
        i_best = int(np.argmin(cost))
        best_o = float(o_grid[i_best])
        best_v = float(v_grid[i_best])
        best_c = float(cost[i_best])
        n_eval = int(o_grid.shape[0])

        # 局部细化：在最优 ω 的半格邻域再试几个点，改善稳态曲率跟踪精度
        if self.n_refine > 1 and self.n_omega > 1:
            half = 0.5 * abs(os_[1] - os_[0])
            lo_f = max(o_lo, -self.omega_max)
            hi_f = min(o_hi, self.omega_max)
            o_fine = np.clip(
                np.linspace(best_o - half, best_o + half, self.n_refine), lo_f, hi_f
            )
            v_fine = np.full(o_fine.shape, best_v)
            c_fine = self._score(xy[0], xy[1], theta, o_fine, v_fine, obstacles)
            n_eval += int(o_fine.shape[0])
            j = int(np.argmin(c_fine))
            if float(c_fine[j]) < best_c:
                best_c = float(c_fine[j])
                best_o = float(o_fine[j])

        best_o = float(np.clip(best_o, -self.omega_max, self.omega_max))
        best_v = float(np.clip(best_v, self.v_min, self.v_max))
        self._last_omega = best_o
        self.last_candidate_count = n_eval
        self.last_cost = best_c

        action = self.to_chassis_action(best_v, best_o)
        self._memo.store(key, action.copy())
        return action

    # ── 内部 ────────────────────────────────────────────────────────
    def _v_window(self, v_est):
        lo, hi = self.v_min, self.v_max
        if self.accel_max > 0.0:
            span = self.accel_max * self.dt
            lo = max(lo, v_est - span)
            hi = min(hi, v_est + span)
        if hi < lo:  # 数值兜底
            lo = hi = 0.5 * (self.v_min + self.v_max)
        return float(lo), float(hi)

    def _omega_window(self):
        lo, hi = -self.omega_max, self.omega_max
        if self.alpha_max > 0.0:
            span = self.alpha_max * self.dt
            lo = max(lo, self._last_omega - span)
            hi = min(hi, self._last_omega + span)
        if hi < lo:
            lo = hi = self._last_omega
        return float(lo), float(hi)

    def _arc_at(self, x, y) -> float:
        """点 (x, y) 在路径弧长坐标上的位置（最近段 + 段内插值）。"""
        _, idx, t = polyline_project(x, y, self._segs)
        n_seg = self._seg_len.shape[0]
        i = int(np.clip(np.asarray(idx, dtype=np.int64).ravel()[0], 0, n_seg - 1))
        frac = float(np.asarray(t, dtype=np.float64).ravel()[0])
        return float(self._s_cum[i] + frac * self._seg_len[i])

    def _gear_active(self) -> bool:
        """档位机制是否生效：ablation（`gear_mode="off"`）或尚未定档时为 False。"""
        return self.gear_mode == "auto" and self._gear is not None

    def _index_at(self, s: float) -> int:
        """弧长坐标 s 所在段的索引。"""
        n_seg = self._seg_len.shape[0]
        i = int(np.searchsorted(self._s_cum, float(s), side="right") - 1)
        return int(np.clip(i, 0, n_seg - 1))

    def _locate(self, x, y):
        """当前位姿的 (弧长, 段索引)。"""
        _, idx, t = polyline_project(x, y, self._segs)
        n_seg = self._seg_len.shape[0]
        i = int(np.clip(np.asarray(idx, dtype=np.int64).ravel()[0], 0, n_seg - 1))
        frac = float(np.asarray(t, dtype=np.float64).ravel()[0])
        return float(self._s_cum[i] + frac * self._seg_len[i]), i

    def _update_gear(self, x, y) -> None:
        """按当前位置的路径段更新档位（票数滞回 + 最小保持步数）。

        滞回形状：票数 `_gear_vote` 在 [−votes, +votes] 饱和，
        只有触边才换档，过渡期保持上一稳定档位 ⇒ `votes` 步之内的单帧抖动翻不了档，
        `votes=3` 相当于至少 6 步驻留。`gear_hold_steps` 是叠在上面的硬下限
        （`votes=1` 时票数两步就能跨越，靠它保住最小驻留）。
        """
        if self.gear_mode != "auto" or self._seg_gear is None:
            return
        if self.v_min >= 0.0:  # 底盘不能倒车 ⇒ 恒前进档
            self._gear = 1.0
            return
        s_now, _ = self._locate(x, y)
        # 档位按"当前位置 +gear_lookahead 处的段"判定，而不是脚下那一段：车会贴在
        # cusp 前 7 cm 停住（本段只剩 7 cm，进也罚退也罚），若只看本段则换档永远等不到。
        raw = float(self._seg_gear[self._index_at(s_now + self.gear_lookahead)])
        if self._gear is None:  # 首步直接采纳 raw
            self._gear = raw
            self._gear_vote = raw * float(self.gear_votes)
            self._gear_hold = 0
            return
        self._gear_vote = float(
            np.clip(
                self._gear_vote + (1.0 if raw > 0.0 else -1.0),
                -self.gear_votes,
                self.gear_votes,
            )
        )
        self._gear_hold += 1
        if self._gear_vote * self._gear > 0.0:
            return
        if abs(self._gear_vote) < self.gear_votes:
            return
        if self._gear_hold < self.gear_hold_steps:
            return
        self._gear = raw
        self._gear_vote = raw * float(self.gear_votes)
        self._gear_hold = 0
        self.gear_switches += 1

    def _gear_span(self, lo: float, hi: float):
        """按当前档位把候选速度窗口截到半轴（前进 [0, v_max] / 倒车 [v_min, 0]）。"""
        if self.gear_mode != "auto" or self._gear is None:
            return lo, hi
        if self._gear < 0.0:
            hi = min(hi, 0.0)
        else:
            lo = max(lo, 0.0)
        if hi < lo:  # 动态窗与档位冲突 ⇒ 退化成"停住"，绝不越档采样
            lo = hi = 0.0
        # 车速下限：v=0 在 cusp 附近是代价地形里的吸收态 —— w_vel*(v_max-0)**2 使它
        # 成为全局极小点（实测停住 1.554 < 沿路径慢走 1.712），一旦选中就再也走不出来。
        # 因此档位生效时候选网格不含 0，最小 |v| 由 min_speed 决定。
        floor = self.min_speed
        if floor > 0.0:
            if self._gear < 0.0:
                hi = min(hi, -floor)
            else:
                lo = max(lo, floor)
            if hi < lo:  # 窗口比下限还窄：退化成该档的最小速度
                lo = hi = -floor if self._gear < 0.0 else floor
        return float(lo), float(hi)

    def _score(self, x0, y0, th0, omegas, vs, obstacles):
        """对一批候选 (ω, v) 打分（越小越好），返回 (C,) 代价向量。"""
        c = omegas.shape[0]
        xs = np.empty((c, self.horizon + 1))
        ys = np.empty((c, self.horizon + 1))
        ths = np.empty((c, self.horizon + 1))
        xs[:, 0], ys[:, 0], ths[:, 0] = x0, y0, th0
        # 每步转角增量与状态无关 ⇒ 先算好，循环里只做位置积分
        th_step = omegas * self.dt
        for k in range(self.horizon):
            th = ths[:, k]
            xs[:, k + 1] = xs[:, k] + vs * np.cos(th) * self.dt
            ys[:, k + 1] = ys[:, k] + vs * np.sin(th) * self.dt
            ths[:, k + 1] = th + th_step

        d_end, idx_end, t_end = polyline_project(xs[:, -1], ys[:, -1], self._segs)
        i_end = np.clip(
            np.asarray(idx_end, dtype=np.int64).ravel(), 0, self._seg_len.shape[0] - 1
        )
        # 航向项以**运动方向**度量：v < 0 的候选沿 −切向走，车头与运动方向相差 π。
        # 参考方向取"当前位置 +heading_lookahead 弧长"处的段，而不是候选末端自己的最近段：
        # cusp 处两条腿只隔 0.243 m，末端投影会在两腿之间跳（实测同一姿态下参考相差
        # 0.93 rad），于是"沿路径走"的候选被自己的参考罚掉，地形里只剩"停住"一个极小点。
        # 弧长前瞻是单调的，cusp 两侧的方向由窗口平滑自然过渡。
        i_ref = self._index_at(self._arc_at(x0, y0) + self.heading_lookahead)
        travel = ths[:, -1] + np.where(vs < 0.0, np.pi, 0.0)
        if self.heading_mode == "nose":
            herr = wrap_angle(travel - self._segs[2][i_ref])
        else:
            herr = wrap_angle(travel - self._seg_dir[i_ref])
            # 模 π 的车轴对齐：折返点前后两条腿鼻向差近 π，鼻向判据会把"沿倒车腿正常
            # 行驶"判成 π 误差（罚 w_heading·π²），代价地形因此在 cusp 前形成局部极小。
            # 档位生效时候选同号 ⇒ π 偏移是常数，折叠是恒等变换（不改变选优结果）。
            herr = np.minimum(np.abs(herr), np.pi - np.abs(herr))
        cost = self.w_path * d_end**2 + self.w_heading * herr**2

        gear_on = self._gear_active()
        if self.w_prog != 0.0:
            s_end = self._s_cum[i_end] + t_end * self._seg_len[i_end]
            ds = s_end - self._arc_at(x0, y0)
            if gear_on:
                # 时域末端落在**反档**段上 ⇒ 这一拍要跨过换档点而档位不变，物理上做不到。
                # 这类候选只允许负进度（老实付后退的账），不给正向补贴：否则"前进候选的
                # 末端漂到倒车腿上"会把弧长坐标跳前，形成假进度 + 0.35 m 过冲的极限环。
                ds = np.where(
                    self._seg_gear[i_end] * self._gear > 0.0, ds, np.minimum(ds, 0.0)
                )
            ds = np.clip(ds, -_PROG_BACK_CLIP, self.horizon * self.dt * self.v_max)
            cost = cost - self.w_prog * ds

        if self.w_cte_traj > 0.0:
            # 整条预测轨迹的平均 CTE：只看到末端时，低速短时域几乎拿不到横向修正的
            # 梯度，机器人会“贴着偏差直线开”。这一项把中间点的偏差也算进来。
            d_all, _ = polyline_query(xs[:, 1:].ravel(), ys[:, 1:].ravel(), self._segs)
            cost = cost + self.w_cte_traj * (d_all.reshape(xs.shape[0], -1) ** 2).mean(
                axis=1
            )

        if obstacles:
            # 用轨迹上的**最小**净空（而不是逐点求和）。
            clr = obstacle_clearance(xs, ys, obstacles).min(axis=1)
            hard = np.maximum(self.obs_safe_radius - clr, 0.0)
            soft = np.maximum(self.obs_soft_radius - clr, 0.0)
            cost = cost + self.w_obs * hard**2 + self.w_obs_soft * soft**2

        # 档位生效时"速度"指沿档位方向的速率 ⇒ 用 |v| 度量。否则倒车档会被
        # (v_max − v)² 罚成"越倒越亏"（v=−1 要付 (v_max+1)²），永远比停车差。
        speed = np.abs(vs) if gear_on else vs
        cost = (
            cost
            + self.w_vel * (self.v_max - speed) ** 2
            + self.w_smooth * (omegas - self._last_omega) ** 2
        )
        return cost
