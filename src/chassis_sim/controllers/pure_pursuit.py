"""经典纯追踪（pure pursuit）：把参考路径当几何曲线，追一个弧长前瞻点。

几何：由位姿到折线的最近投影得到当前弧长 s，取弧长 s + L_d 处的路径点为目标 P，
车体系下目标方位角 `α = wrap(atan2(P−p) − θ)`，则

    ω = 2·v·sin(α) / L_d_eff      （L_d_eff = 车到目标点的**欧氏**距离）

线速度按曲率限速：`v = clamp(v_max / (1 + k·|ω|))`。ω 与 v 相互依赖，这里用
**两遍**求解（先用 v_max 估 ω 得 v，再用该 v 精算 ω），因此控制器对同一 (位姿, 路径)
是纯函数，不含上一拍的隐藏状态。

前瞻距离可自适应（`adaptive=True`）：`L_d = clip(lookahead + lookahead_gain·|v_last|,
lookahead_min, lookahead_max)`。默认关闭 —— 固定 L_d 的稳态跟踪误差与 L_d²/R 同阶，
自适应会让高速段误差变大。
"""

from __future__ import annotations

import numpy as np

from ._geometry import (
    arc_lengths,
    as_path,
    path_segments,
    point_at_arc,
    windowed_project,
    wrap_angle,
)
from .base import Controller

__all__ = ["PurePursuitController"]

# |α| 夹在 ±(π/2 − ε)：目标跑车身侧后方时 sin(α) 会随 |α| 增大而**减小**，
# 曲率反而变小（纯追踪的已知奇异）。夹住后保持最大转向。
_ALPHA_LIMIT = np.pi / 2.0 - 1e-3


class PurePursuitController(Controller):
    """纯追踪控制器。输出 (v, ω) 后由 `to_chassis_action` 适配三车型。

    构造参数（除 `**base_kw` 外全部可选）：

    lookahead          前瞻弧长 L_d [m]，默认 0.6
    adaptive           是否让 L_d 随车速自适应，默认 False
    lookahead_min/max  自适应区间 [m]，默认 [lookahead, lookahead]（即不放大）
    lookahead_gain     自适应斜率 [s]，默认 0.8
    speed_gain         限速系数 k，默认 0.5（k↑ ⇒ 弯道更慢）
    slowdown_distance  末段减速距离 [m]，默认 0.5（0 = 不减速）
    window_pts/back_pts 单调投影窗口（前向/后退段数），默认 40 / 10

    `**base_kw` 透传 `Controller`（chassis=/kind=/limits=/params=/action_space=）。
    """

    def __init__(
        self,
        *,
        lookahead: float = 0.6,
        adaptive: bool = False,
        lookahead_min: float | None = None,
        lookahead_max: float | None = None,
        lookahead_gain: float = 0.8,
        speed_gain: float = 0.5,
        slowdown_distance: float = 0.5,
        window_pts: int = 40,
        back_pts: int = 10,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        if lookahead <= 0.0:
            raise ValueError(f"lookahead 必须 > 0，实得 {lookahead}")
        self.lookahead = float(lookahead)
        self.adaptive = bool(adaptive)
        self.lookahead_min = float(
            lookahead if lookahead_min is None else lookahead_min
        )
        self.lookahead_max = float(
            lookahead if lookahead_max is None else lookahead_max
        )
        self.lookahead_gain = float(lookahead_gain)
        self.speed_gain = float(speed_gain)
        self.slowdown_distance = float(slowdown_distance)
        self.window_pts = int(window_pts)
        self.back_pts = int(back_pts)

        self._path = None
        self._segs = None
        self._s_cum = None
        self._seg_len = None
        self._s_total = 0.0
        self._anchor = -1
        self._last_v = 0.0

    # ── 协议 ────────────────────────────────────────────────────────
    def reset(self) -> None:
        """清空投影锚点与自适应参考速度（保留路径）。"""
        self._anchor = -1
        self._last_v = 0.0

    def set_path(self, path_world: np.ndarray) -> None:
        """(N,3) 世界系路径；空路径 ⇒ 控制器输出零动作。"""
        self._path = as_path(path_world)
        self._anchor = -1
        self._last_v = 0.0
        if self._path is None:
            self._segs = None
            self._s_cum = None
            self._seg_len = None
            self._s_total = 0.0
            return
        self._segs = path_segments(self._path)
        self._s_cum, self._seg_len = arc_lengths(self._path)
        self._s_total = float(self._s_cum[-1])

    def act(self, observation: dict) -> np.ndarray:
        x, y, theta = self._pose(observation)
        if self._path is None:
            return self.to_chassis_action(0.0, 0.0)

        # 单调投影：自交 / 首尾重合路径上不会锚回时间序更早的支路
        _, idx, t = windowed_project(
            x, y, self._segs, self._anchor, back=self.back_pts, fwd=self.window_pts
        )
        self._anchor = int(idx[0])
        s = float(self._s_cum[self._anchor] + t[0] * self._seg_len[self._anchor])

        lookahead = self._lookahead_for(self._last_v)
        tx, ty = point_at_arc(self._path, self._s_cum, s + lookahead)
        dx = tx - x
        dy = ty - y
        dist = float(np.hypot(dx, dy))
        if dist < 1e-9:
            self._last_v = 0.0
            return self.to_chassis_action(0.0, 0.0)

        alpha = float(wrap_angle(np.arctan2(dy, dx) - theta))
        alpha = float(np.clip(alpha, -_ALPHA_LIMIT, _ALPHA_LIMIT))

        # 两遍求解 v = v_max/(1 + k|ω|) 与 ω = 2v sinα / L_d 的不动点
        omega0 = 2.0 * self.v_max * np.sin(alpha) / dist
        v = self.v_max / (1.0 + self.speed_gain * abs(omega0))
        remaining = self._s_total - s
        if self.slowdown_distance > 0.0 and remaining < self.slowdown_distance:
            v = min(v, self.v_max * max(remaining, 0.0) / self.slowdown_distance)
        v = float(np.clip(v, max(self.v_min, 0.0), self.v_max))
        omega = 2.0 * v * np.sin(alpha) / dist
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))

        self._last_v = v
        return self.to_chassis_action(v, omega)

    # ── 内部 ────────────────────────────────────────────────────────
    def _lookahead_for(self, v_last: float) -> float:
        if not self.adaptive:
            return self.lookahead
        ld = self.lookahead + self.lookahead_gain * abs(float(v_last))
        return float(np.clip(ld, self.lookahead_min, self.lookahead_max))
