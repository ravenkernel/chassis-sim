"""执行器/轮胎动力学（EnvConfig.dynamics 三级开关的 limits / wheel 档实现）。

三级开关（`EnvConfig.dynamics`，取值见 LEVELS）
--------------------------------------------------
`none`   纯运动学。env 侧**连 import 都跳过**（`dynamics.py` 不被加载、
          不被调用、不消耗任何随机数）⇒ 与加入本开关前逐位相同。
`limits` 速度/转角速率**硬限幅**（实现仍在 `envs.chassis_env._apply_limits_and_gate`，
          本模块只提供"0 = 用默认值"的解析规则与默认值本身）。
`wheel`  limits + 本模块的**一阶滞后 / 死区 / 轮滑**（phenomenological，不追求
          物理精确，但每个参数都有独立可解释的含义与可复现的离散式）。

物理域路径（`envs.chassis_env._step_impl`，顺序不可交换）
----------------------------------------------------------
    act ──chassis.command()──► (v, ω)              纯映射，无状态
        ──_apply_limits_and_gate()──► (v, ω)       硬限幅：|Δv| ≤ accel_max·dt
                                                   |Δδ| ≤ steer_rate_max·dt
                                                   （含 SafetyGate 紧急覆盖）
        ──ActuatorDynamics.step()──► (v′, ω′)      **仅 wheel 档**：滞后/死区/轮滑
        ──_encode_physical()──► act′
        ──bridge.step(act′)──► chassis.step()      积分（ω 用上一拍 δ，见下）

dynamics 与 SafetyGate 的先后：dynamics 在 gate **之后**。gate 是紧急覆盖，它的
契约是"这一拍就按这个 (δ, v) 执行"；把一阶滞后插进 gate 前面会让刹车被延迟
0.1~0.15 s，等于悄悄削弱闸门。代价是闸门触发的那一拍不受 |Δv| ≤ accel_max·dt 约束
（闸门前本来就如此）。训练时 gate=None，这条只影响评估。

为什么一阶滞后必须放在**限幅之后**（顺序不可交换）：
  滞后是单调收缩算子 `x′ = x + a·(x_cmd − x)`，a ∈ (0, 1] ⇒ `|x′ − x| ≤ |x_cmd − x|`。
  先限幅再滞后 ⇒ 输出仍满足 `|Δv′| ≤ accel_max·dt`（`_prev_action` 记的是**滞后后**
  的值，于是下一步的限幅基准也是滞后后的值）。反过来"先滞后再限幅"会把滞后项直接
  削掉（指令与上一拍输出的差被 clip 后，lag 只剩很小一截）⇒ wheel 档退化成 limits 档。

与 `chassis.step()` 既有「转向滞后一拍」的关系（**必读**）
-----------------------------------------------------------
`chassis/ackermann.py:84-93` 里 `omega = v/wheelbase·tan(self._prev_delta)` 用的是
**上一拍**的转角指令 ⇒ 底盘本来就有 1 拍采样延迟（`tests/test_chassis_actuator_lag.py`
用 golden 锁住，本次不动）。本模块的 ω 一阶滞后叠加在这 1 拍延迟**之上**，得到的是
「1 拍纯延迟 ⊕ 1 个一阶极点」，**不是两拍纯延迟**。

  判据（单位阶跃 ω_cmd=1，dt=0.1，omega_tau=0.10 ⇒ a=0.632）：
    · 第 1 步执行 ω=0      —— 纯延迟那一拍（tan(_prev_delta)，_prev_delta 仍是 0）；
    · 第 2 步 0<ω<终值     —— 一阶极点（0.632 而非 1）；
    · 两拍纯延迟的指纹会是「第 1、2 步都是 0、第 3 步直接到 1」——本实现不会。

默认值标定依据（accel_max / steer_rate_max / v_tau / omega_tau）
----------------------------------------------------------------
底盘口径：轴距 0.145 m、车体 0.38×0.20 m、δ_max=0.526 rad、v_max=1.2 m/s、
dt=0.1 s（10 Hz）、300~500 步/回合。

· `accel_max=1.5 m/s²`：0→v_max(1.2 m/s) 需 0.8 s = 8 个控制步；每步 Δv=0.15 m/s，
  约为 v_max 的 12.5%（可见而不过分）。小型室内底盘的常见量级是 1~3 m/s²，取偏保守
  的 1.5 让"限幅"是约束而不是把策略钉死；再小（<0.5）会让 300 步回合内都跑不到 v_max。
· `steer_rate_max=3.0 rad/s`：满舵 0.526 rad 用 0.175 s ≈ 2 步；每步 Δδ=0.3 rad 仍
  小于 δ_max（单步不会直接跳到满舵）。舵机/转向执行器带载的常见量级 3~5 rad/s，
  取下沿 3.0。
· `v_tau=0.15 s`：驱动电机 + 减速箱速度闭环的典型时间常数 50~200 ms，取中位 0.15 s
  = 1.5 个 dt ⇒ a = 1−e^(−0.1/0.15) = 0.4866，63% 上升点 = τ = 0.15 s（落在 0.1/0.2 s
  两个采样点之间，便于插值校验），95% 落在 3τ = 0.45 s ≈ 4.5 步。τ<dt 会让 a>0.63
  退化成"几乎无滞后"，τ>0.5 s 会让速度在回合内追不上指令。
· `omega_tau=0.10 s`：转向执行器通常比驱动快，取 1 个 dt ⇒ a = 0.632；与 v_tau 组合
  得到"转向先到位、速度后跟上"的阶跃响应，符合实际底盘。

（以上都是** phenomenological** 取值：为了让 wheel 档在 10 Hz、300~500 步的回合里
"看得出滞后、又不会把可完成的任务变成完不成"，不是任何具体硬件的辨识结果。）
"""

from __future__ import annotations

import math
from typing import Any

# 三级开关的合法取值（顺序即"从最简到最全"）。
LEVELS: tuple[str, ...] = ("none", "limits", "wheel")

# limits / wheel 档下 accel_max、steer_rate_max 为 0（= 未显式配置）时用的默认值。
# 标定依据见模块 docstring「默认值标定依据」。
DEFAULT_ACCEL_MAX: float = 1.5  # m/s²
DEFAULT_STEER_RATE_MAX: float = 3.0  # rad/s
# wheel 档的一阶滞后时间常数（标定依据同上）。
DEFAULT_V_TAU: float = 0.15  # s
DEFAULT_OMEGA_TAU: float = 0.10  # s


def resolve_level(level: Any) -> str:
    """规范化动力学等级，非法值立刻抛 ValueError（不静默降级成 none）。"""
    # 只接受 str：`str(None)` 会得到 "None" → "none" ⇒ 静默降级，必须挡住。
    if not isinstance(level, str):
        raise ValueError(  # noqa: TRY004 —— 与 config.py 的同类校验同一约定（对外的配置错误一律 ValueError）
            f"dynamics 必须是 {LEVELS} 之一，收到 {level!r}"
        )
    text = level.strip().lower()
    if text not in LEVELS:
        raise ValueError(f"dynamics 必须是 {LEVELS} 之一，收到 {level!r}")
    return text


def resolve_limits(
    level: Any, accel_max: float, steer_rate_max: float
) -> tuple[float, float]:
    """把「配置里的 0」解析成本级**实际生效**的限幅值。

    · `none`：原样返回。0 仍是 0 ⇒ 与加入本开关前逐位相同（限幅通路一点都不进）。
    · `limits`/`wheel`：<=0 视为"未显式配置" ⇒ 用 DEFAULT_*；>0 以用户值为准。
      反过来把 0 当"关闭"是不行的：选了这一级就是要它，关闭只能靠选 `none`。
    """
    _lvl = resolve_level(level)
    _a = float(accel_max)
    _s = float(steer_rate_max)
    if _lvl == "none":
        return _a, _s
    if not _a > 0.0:
        _a = DEFAULT_ACCEL_MAX
    if not _s > 0.0:
        _s = DEFAULT_STEER_RATE_MAX
    return _a, _s


class ActuatorDynamics:
    """wheel 档：速度/角速度一阶滞后 + 速度死区 + 轮滑（纯 numpy，无随机数）。

    离散式（与连续一阶系统 ẋ=(x_cmd−x)/τ 的零阶保持精确解一致）::

        a_v     = 1 − exp(−dt/τ_v)          # 每步收敛比例，与 dt、τ 一致地取值
        v      += (v_cmd − v)   * a_v
        omega  += (ω_cmd − omega) * a_ω
        v_out   = v * (1 − slip)            # 轮速 → 地速

    不用 `x += (x_cmd−x)·dt/τ` 的欧拉形式：dt/τ>2 时它会振荡甚至发散，而
    `1−exp(−dt/τ)` 对任意 dt,τ>0 都落在 (0,1) ⇒ 无条件稳定（任务书的"等价稳定形式"）。

    三个可选效应（默认全关 ⇒ 与 limits 档逐位相同）：
      · `v_tau / omega_tau`：时间常数，单位秒；越小越快。
      · `v_deadband`：|v_cmd| 小于该值时电机不转 ⇒ v_cmd=0 **且 ω_cmd=0**
        （与底盘 `omega = 0 if v == 0` 同语义：不动就没有偏航率）。
        ⚠️ 死区作用在**已限速**的目标上（本类排在限速器之后）⇒ `v_deadband >
        `accel_max·dt` 时静止车永远挣不脱死区（目标被锚在 0 上）。这是真实粘滞
        (stiction) 的形状，不是 bug：要能动就把死区设到 `accel_max·dt` 以下。
      · `wheel_slip`：轮滑/滑移率 s∈[0,1)，**只作用于线速度**（地速 = 轮速×(1−s)）。
        不作用于 ω ⇒ 同一 (v, ω) 指令下等效转弯半径变大 = 转向不足(understeer)，
        这正是轮滑在阿克曼底盘上的主要现象。状态量存的是**轮速**，滑移只影响出口。

    `reset()` 把状态归零；env 在 `reset()` 里调用 ⇒ 每回合从静止开始，
    两条同样动作的回合逐位可复现（本类没有任何随机源）。
    """

    def __init__(
        self,
        *,
        dt: float,
        v_tau: float = DEFAULT_V_TAU,
        omega_tau: float = DEFAULT_OMEGA_TAU,
        v_deadband: float = 0.0,
        wheel_slip: float = 0.0,
    ):
        self.dt = float(dt)
        self.v_tau = float(v_tau)
        self.omega_tau = float(omega_tau)
        self.v_deadband = float(v_deadband)
        self.wheel_slip = float(wheel_slip)
        if not self.dt > 0.0:
            raise ValueError(f"dynamics.dt 必须 > 0，收到 {dt}")
        if not self.v_tau > 0.0:
            raise ValueError(f"dynamics.v_tau 必须 > 0，收到 {v_tau}")
        if not self.omega_tau > 0.0:
            raise ValueError(f"dynamics.omega_tau 必须 > 0，收到 {omega_tau}")
        if self.v_deadband < 0.0:
            raise ValueError(f"dynamics.v_deadband 必须 >= 0，收到 {v_deadband}")
        if not 0.0 <= self.wheel_slip < 1.0:
            raise ValueError(
                f"dynamics.wheel_slip 必须在 [0, 1) 内（1 会让车永远不动），"
                f"收到 {wheel_slip}"
            )
        # 每步收敛比例（"等价稳定形式"）：1 − exp(−dt/τ) ∈ (0, 1)。
        self.alpha_v = 1.0 - math.exp(-self.dt / self.v_tau)
        self.alpha_omega = 1.0 - math.exp(-self.dt / self.omega_tau)
        self.reset()

    def reset(self) -> None:
        """状态归零（静止起步）。env 的每回合 reset 都会调用。"""
        self.v = 0.0
        self.omega = 0.0

    def step(self, v_cmd: float, omega_cmd: float) -> tuple[float, float]:
        """一拍：给指令 (v_cmd, ω_cmd)，返回**实际作用到积分器**的 (v, ω)。"""
        v_target = float(v_cmd)
        w_target = float(omega_cmd)
        if self.v_deadband > 0.0 and abs(v_target) < self.v_deadband:
            v_target = 0.0
            w_target = 0.0
        self.v += (v_target - self.v) * self.alpha_v
        self.omega += (w_target - self.omega) * self.alpha_omega
        return self.v * (1.0 - self.wheel_slip), self.omega

    def describe(self) -> dict:
        """wheel 档实际生效的参数（给 info / describe_dynamics 用）。"""
        return {
            "v_tau": self.v_tau,
            "omega_tau": self.omega_tau,
            "alpha_v": self.alpha_v,
            "alpha_omega": self.alpha_omega,
            "v_deadband": self.v_deadband,
            "wheel_slip": self.wheel_slip,
        }
