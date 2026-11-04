"""控制器协议：控制器内部按 (v, ω[, vy]) 规划，出口统一走 `to_chassis_action`。

一次实现跑三种车型：控制器只关心物理量 (v, ω)，由 `to_chassis_action` 做**车型逆映射**
成该车型的（归一化）action，再交给 `Chassis.command()` 还原成 `ChassisCommand`。
逆映射与各车型的正映射严格互逆（差速/麦轮逐位、阿克曼到 1e-16 量级）。

车型正映射的来源（**逆映射必须据此反解，不许凭空假设**）：

* 阿克曼：曲率模式 `δ = arctan(clip(a0*κ_max, ±κ_max) * L)`、
  线性模式 `δ = a0*δ_max`；速度 `a1 ≥ 0 → v = a1*v_max`、`a1 < 0 → v = a1*(-v_min)`；
  再由 `ω = v/L·tan(δ)` 得到横摆角速度。
* 差速 / 麦轮：归一化线性映射 `a = [v/v_max, (vy/vy_max,) ω/ω_max]`，负速度用
  `-v_min` 分段（与 `Chassis.command()` 同口径）。

物理限值 / 几何参数一律来自 `chassis.describe()`：`["limits"]` 提供
`{"v_min", "v_max", "omega_max", "vy_max"}`，`["params"]` 提供
`{"wheelbase", "delta_max", "curvature_action", "kappa_max"}`。缺键就地抛 KeyError
并点名（不静默取默认值）——静默默认值会让三种车型的边界悄悄漂移。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces

__all__ = ["Controller"]

# |v| 低于此值 ⇒ 阿克曼的 δ 无法由 ω 反解（ω = v/L·tan δ ≡ 0）：沿用上一拍 δ。
# 这是**物理**限制而不是数值技巧：ω 与 v 的比值才决定 δ，零速时任何 δ 都不产生 ω。
V_EPS = 1e-6

_ACTION_DIM = {"diff_drive": 2, "mecanum": 3, "ackermann": 2}

_LIMIT_KEYS = {
    "diff_drive": ("v_min", "v_max", "omega_max"),
    "ackermann": ("v_min", "v_max", "omega_max"),
    "mecanum": ("v_min", "v_max", "omega_max", "vy_max"),
}
_PARAM_KEYS = {
    "diff_drive": (),
    "mecanum": (),
    "ackermann": ("wheelbase", "delta_max", "curvature_action", "kappa_max"),
}


def _require(mapping, keys, where):
    """按键取值；缺键抛 KeyError 并在消息里点名缺的是哪个键。"""
    out = {}
    for key in keys:
        if key not in mapping:
            raise KeyError(
                f"{where} 缺少键 {key!r}（需要 {list(keys)}，实得 {sorted(mapping)}）"
            )
        out[key] = mapping[key]
    return out


def _ratio(x, limit):
    """物理量 → 归一化分量；limit <= 0（该自由度不可用）时退化为 0。"""
    return float(x) / float(limit) if float(limit) > 0.0 else 0.0


class Controller(ABC):
    """控制器基类：写一次 (v, ω, vy)，跑三种车型。

    构造方式二选一：

    * `Controller(chassis=chassis)` —— 鸭子类型，只用 `describe()` 与 `action_space()`；
    * `Controller(kind="diff_drive", limits={...}, params={...})` —— 无底盘对象时显式给参数。

    `action_space` 是**控制器的输出域**（= 该车型的 action 空间），`act()` 的返回值
    形状由它在构造时锁定。
    """

    def __init__(
        self,
        *,
        chassis=None,
        kind: str | None = None,
        action_space=None,
        limits: dict | None = None,
        params: dict | None = None,
    ) -> None:
        if chassis is not None:
            desc = chassis.describe()
            if kind is None:
                kind = desc.get("name", getattr(chassis, "name", None))
            if action_space is None:
                action_space = chassis.action_space()
            if limits is None:
                if "limits" not in desc:
                    raise KeyError(
                        f"{kind!r} 的 chassis.describe() 缺少键 'limits'"
                        f"（需要 {_LIMIT_KEYS.get(kind, ())}）"
                    )
                limits = desc["limits"]
            if params is None:
                if "params" not in desc:
                    raise KeyError(
                        f"{kind!r} 的 chassis.describe() 缺少键 'params'"
                        f"（需要 {_PARAM_KEYS.get(kind, ())}）"
                    )
                params = desc["params"]

        if kind is None:
            raise TypeError("Controller 需要 chassis= 或 kind= 之一来确定车型")
        if kind not in _ACTION_DIM:
            raise ValueError(f"未知车型 {kind!r}；支持 {sorted(_ACTION_DIM)}")
        if limits is None or params is None:
            raise TypeError("未传 chassis= 时必须显式给出 limits= 与 params=")

        self.kind = kind
        self.limits = dict(_require(limits, _LIMIT_KEYS[kind], "limits"))
        self.params = dict(_require(params, _PARAM_KEYS[kind], "params"))

        dim = _ACTION_DIM[kind]
        if action_space is None:
            action_space = spaces.Box(
                low=-np.ones(dim, dtype=np.float32),
                high=np.ones(dim, dtype=np.float32),
                dtype=np.float32,
            )
        shape = tuple(int(d) for d in action_space.shape)
        if shape != (dim,):
            raise ValueError(f"{kind} 的 action_space.shape={shape}，应为 ({dim},)")
        self.action_space = action_space

        self.v_min = float(self.limits["v_min"])
        self.v_max = float(self.limits["v_max"])
        self.omega_max = float(self.limits["omega_max"])
        self.vy_max = float(self.limits.get("vy_max", 0.0))
        self.wheelbase = float(self.params.get("wheelbase", 0.0))
        self.delta_max = float(self.params.get("delta_max", 0.0))
        self.curvature_action = bool(self.params.get("curvature_action", False))
        self.kappa_max = float(self.params.get("kappa_max", 0.0))

        if kind == "ackermann":
            if self.wheelbase <= 0.0:
                raise ValueError(f"阿克曼的 wheelbase 必须 > 0，实得 {self.wheelbase}")
            if self.curvature_action and self.kappa_max <= 0.0:
                raise ValueError(f"曲率模式需要 kappa_max > 0，实得 {self.kappa_max}")
            if not self.curvature_action and self.delta_max <= 0.0:
                raise ValueError(f"线性模式需要 delta_max > 0，实得 {self.delta_max}")

        # 阿克曼 v≈0 时沿用的上一拍转向角
        self._last_delta = 0.0

    # ── 子类必须实现 ────────────────────────────────────────────────
    @abstractmethod
    def reset(self) -> None:
        """清空控制器内部状态（不含 path）。"""

    @abstractmethod
    def set_path(self, path_world: np.ndarray) -> None:
        """喂入参考路径 (N,3) 世界系 [x, y, θ]；θ 是该点**车头鼻向**。"""

    @abstractmethod
    def act(self, observation: dict) -> np.ndarray:
        """观测 -> 该车型的 action（形状 = `self.action_space.shape`）。

        观测契约（ duck-typed，只读不写）：
            observation["pose"]        (3,) [x, y, θ] 世界系位姿（必需）
            observation["obstacles"]   可选，env 侧同构的障碍列表
            observation["dt"]          可选，DWA 覆盖控制周期
        """

    # ── 车型逆映射 ──────────────────────────────────────────────────
    def _clip(self, action) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )

    def _denorm_speed(self, v: float) -> float:
        """物理速度 → 归一化速度分量（与 command() 的分段正映射逐位互逆）。

        正映射：a >= 0 → v = a*v_max；a < 0 → v = a*(-v_min)。
        反解：v >= 0 → a = v/v_max；v < 0 → a = v/(-v_min)。
        v_min == 0（该车型不允许倒车）时负速度无法表达，退化为 0。
        """
        if v >= 0.0:
            return _ratio(v, self.v_max)
        return _ratio(v, -self.v_min)

    def to_chassis_action(self, v: float, omega: float, vy: float = 0.0) -> np.ndarray:
        """物理 (v, ω, vy) → 本车型的**归一化 action**（形状 = action_space.shape）。

        差速：`[v/v_max, ω/ω_max]`；麦轮：`[v/v_max, vy/vy_max, ω/ω_max]`；
        阿克曼：先由 (v, ω) 反解 `δ = arctan(ω·L/v)`（|v| < V_EPS 时取上一拍 δ，见
        `_last_delta` 的注释 —— 零速下 ω 恒为 0，任何 δ 都还原不出非零 ω），再由 δ 反解
        归一化转向：曲率模式 `a0 = tan(δ)/(κ_max·L)`（因正映射是 κ = a0·κ_max、
        δ = arctan(κ·L)，故 a0 = κ/κ_max = tan(δ)/(L·κ_max)）；线性模式 `a0 = δ/δ_max`。

        不可达的请求会被 action_space 的 [-1, 1] 边界**饱和**（例如阿克曼的
        `|ω| > |v|·κ_max`），这是物理上限而不是静默修正。
        """
        v = float(v)
        omega = float(omega)
        vy = float(vy)

        if self.kind == "diff_drive":
            return self._clip([self._denorm_speed(v), _ratio(omega, self.omega_max)])

        if self.kind == "mecanum":
            return self._clip(
                [
                    self._denorm_speed(v),
                    _ratio(vy, self.vy_max),
                    _ratio(omega, self.omega_max),
                ]
            )

        # ackermann
        if abs(v) >= V_EPS:
            delta = float(np.arctan(omega * self.wheelbase / v))
            self._last_delta = delta
        else:
            delta = self._last_delta
        if self.curvature_action:
            a0 = float(np.tan(delta) / (self.kappa_max * self.wheelbase))
        else:
            a0 = float(delta / self.delta_max)
        return self._clip([a0, self._denorm_speed(v)])

    # ── 观测解析 ────────────────────────────────────────────────────
    @staticmethod
    def _pose(observation: dict) -> tuple[float, float, float]:
        """从 observation 取世界位姿 (x, y, θ)。

        契约键是 `pose`（(3,) 的 [x, y, θ]）；为兼容不同 env 的拼装方式，也接受
        `robot_pose` 别名与 `x` / `y` / `theta` 三键。
        """
        for key in ("pose", "robot_pose"):
            if key in observation:
                arr = np.asarray(observation[key], dtype=np.float64).reshape(-1)
                if arr.size < 3:
                    raise ValueError(
                        f"observation[{key!r}] 需要 3 个分量，实得 {arr.size}"
                    )
                return float(arr[0]), float(arr[1]), float(arr[2])
        if {"x", "y", "theta"} <= set(observation):
            return (
                float(observation["x"]),
                float(observation["y"]),
                float(observation["theta"]),
            )
        raise KeyError(
            "observation 缺少位姿：需要 'pose'（或 'robot_pose' / x,y,theta 三键）"
        )
