"""dynamics 三级开关（none / limits / wheel）的模块级与 env 级契约。

三级语义（实现见 src/chassis_sim/envs/dynamics.py）：

*  "none"   —— env **不 import** dynamics 模块、不调用、不消耗随机数；行为与加入
  本开关前逐位相同（info 里也没有 "dynamics" 键）。
*  "limits" —— 复用既有 _apply_limits_and_gate 的 Δv / Δδ 速率限幅；选了本级而
  accel_max / steer_rate_max 仍为 0 时取文档默认值 1.5 m/s² / 3.0 rad/s。
*  "wheel"  —— 在 limits 之上叠加一阶执行器滞后（v_tau / omega_tau）+ 死区 + 轮滑。

模块级的离散式：x += (x_cmd − x) · (1 − exp(−dt/τ))，因此第 n 步的阶跃响应
恰好是 x_n = x_cmd · (1 − (1−α)^n)（本文件按闭式逐位对拍，不是"看起来像"）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chassis_sim.chassis import make_chassis
from chassis_sim.config import EnvConfig
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.envs.dynamics import (
    DEFAULT_ACCEL_MAX,
    DEFAULT_OMEGA_TAU,
    DEFAULT_STEER_RATE_MAX,
    DEFAULT_V_TAU,
    LEVELS,
    ActuatorDynamics,
    resolve_level,
    resolve_limits,
)

DT = 0.1
V_TAU = DEFAULT_V_TAU  # 0.15
OMEGA_TAU = DEFAULT_OMEGA_TAU  # 0.10
ALPHA_V = 1.0 - math.exp(-DT / V_TAU)
ALPHA_OMEGA = 1.0 - math.exp(-DT / OMEGA_TAU)


def _run(env: ChassisSimEnv, actions, seed: int = 0):
    """同 seed 连跑 actions，返回 (obs 列表, reward 列表, 末步 info)。"""
    obs, _ = env.reset(seed=seed)
    obs_seq, rew_seq = [obs.copy()], []
    info = {}
    for a in actions:
        obs, rew, _term, _trunc, info = env.step(np.asarray(a, dtype=np.float32))
        obs_seq.append(obs.copy())
        rew_seq.append(rew)
    return obs_seq, rew_seq, info


def _ramp(n: int = 10, dim: int = 2) -> np.ndarray:
    """固定的确定性动作序列（不用随机源，避免把测试自身的 rng 混进来）。"""
    return np.linspace(-0.6, 0.6, n * dim).reshape(n, dim)


# --------------------------------------------------------------------------
# 模块级：一阶滞后的闭式、死区、轮滑、稳定性、参数校验
# --------------------------------------------------------------------------


class TestActuatorDynamics:
    def test_step_response_is_closed_form(self):
        """v_n 必须逐位等于 x_cmd·(1−(1−α)^n)，两条轴各有自己的 α。"""
        dyn = ActuatorDynamics(dt=DT, v_tau=V_TAU, omega_tau=OMEGA_TAU)
        assert dyn.alpha_v == pytest.approx(ALPHA_V, abs=0.0)
        assert dyn.alpha_omega == pytest.approx(ALPHA_OMEGA, abs=0.0)
        for n in range(1, 21):
            v, w = dyn.step(1.0, 2.0)
            assert v == pytest.approx(1.0 * (1.0 - (1.0 - ALPHA_V) ** n), abs=1e-15)
            assert w == pytest.approx(2.0 * (1.0 - (1.0 - ALPHA_OMEGA) ** n), abs=1e-15)

    def test_tau_ordering_is_observable(self):
        """alpha 由 tau 决定：tau 越小越跟手（同一 dt 下 alpha 更大）。"""
        fast = ActuatorDynamics(dt=DT, v_tau=0.05, omega_tau=0.05)
        slow = ActuatorDynamics(dt=DT, v_tau=0.50, omega_tau=0.50)
        for _ in range(3):
            v_fast, _ = fast.step(1.0, 0.0)
            v_slow, _ = slow.step(1.0, 0.0)
        assert v_fast > v_slow

    def test_63pct_rise_time_matches_tau_within_one_sample(self):
        """实测 63% 上升时间 − τ 的偏差必须 ≤ 1 个采样周期（离散化下界）。

        τ=0.15s / dt=0.1s 时解析 63% 时刻是 0.15s，而采样只能落在 0.2s
        （第 2 步 1−(1−α)²=0.7364 ≥ 0.6321）—— 偏差 0.05s = dt/2，属于采样抖动
        而非模型误差。
        """
        dyn = ActuatorDynamics(dt=DT, v_tau=V_TAU, omega_tau=OMEGA_TAU)
        for tau, axis in ((V_TAU, 0), (OMEGA_TAU, 1)):
            dyn.reset()
            n63 = None
            t = 0.0
            for _ in range(200):
                v, w = dyn.step(1.0, 1.0)
                t += DT
                if (v if axis == 0 else w) >= 1.0 - math.exp(-1.0):
                    n63 = t
                    break
            assert n63 is not None
            assert abs(n63 - tau) <= DT + 1e-12, f"tau={tau} 实测 63% 时刻 {n63}"

    def test_deadband_zeroes_both_axes(self):
        """死区内 v 目标为 0，且与底盘 omega = 0 if v == 0 同语义把 ω 也归零。"""
        dyn = ActuatorDynamics(dt=DT, v_tau=V_TAU, omega_tau=OMEGA_TAU, v_deadband=0.3)
        for _ in range(5):
            v, w = dyn.step(0.2, 1.0)
        assert v == 0.0 and w == 0.0
        for _ in range(30):
            v, w = dyn.step(0.5, 1.0)
        assert v > 0.4 and w > 0.9

    def test_wheel_slip_scales_exit_speed_only(self):
        """wheel_slip 只作用在出口线速度（转向不足），不改 ω 的滞后轨迹。"""
        clean = ActuatorDynamics(dt=DT, v_tau=V_TAU, omega_tau=OMEGA_TAU)
        slip = ActuatorDynamics(
            dt=DT, v_tau=V_TAU, omega_tau=OMEGA_TAU, wheel_slip=0.25
        )
        for _ in range(40):
            v_clean, w_clean = clean.step(1.0, 1.0)
            v_slip, w_slip = slip.step(1.0, 1.0)
        # 稳态：内部状态都到 1.0 ⇒ 出口 v 恰好打 0.75 折，ω 完全一致
        assert v_clean == pytest.approx(1.0, abs=1e-9)
        assert v_slip == pytest.approx(0.75, abs=1e-9)
        assert w_slip == pytest.approx(w_clean, abs=0.0)

    def test_state_is_bounded_no_overshoot_even_with_large_dt(self):
        """dt ≫ τ 时 alpha → 1（不会 >1），响应不振荡、不过冲。"""
        dyn = ActuatorDynamics(dt=10.0, v_tau=V_TAU, omega_tau=OMEGA_TAU)
        assert 0.0 < dyn.alpha_v <= 1.0 and 0.0 < dyn.alpha_omega <= 1.0
        prev = 0.0
        for _ in range(5):
            v, _ = dyn.step(1.0, 0.0)
            assert v <= 1.0 + 1e-12
            assert v >= prev
            prev = v

    def test_reset_clears_state(self):
        dyn = ActuatorDynamics(dt=DT, v_tau=V_TAU, omega_tau=OMEGA_TAU)
        for _ in range(5):
            dyn.step(1.0, 1.0)
        assert dyn.v > 0 and dyn.omega > 0
        dyn.reset()
        assert dyn.v == 0.0 and dyn.omega == 0.0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dt": 0.0}, "dt"),
            ({"dt": -0.1}, "dt"),
            ({"v_tau": 0.0}, "v_tau"),
            ({"omega_tau": -1.0}, "omega_tau"),
            ({"v_deadband": -0.1}, "v_deadband"),
            ({"wheel_slip": 1.0}, "wheel_slip"),
            ({"wheel_slip": -0.01}, "wheel_slip"),
        ],
    )
    def test_rejects_bad_params(self, kwargs, match):
        base = {"dt": DT}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ActuatorDynamics(**base)

    def test_describe_reports_effective_values(self):
        dyn = ActuatorDynamics(
            dt=DT, v_tau=0.2, omega_tau=0.3, v_deadband=0.05, wheel_slip=0.1
        )
        d = dyn.describe()
        assert d["v_tau"] == pytest.approx(0.2)
        assert d["omega_tau"] == pytest.approx(0.3)
        assert d["alpha_v"] == pytest.approx(1.0 - math.exp(-DT / 0.2))
        assert d["alpha_omega"] == pytest.approx(1.0 - math.exp(-DT / 0.3))
        assert d["v_deadband"] == pytest.approx(0.05)
        assert d["wheel_slip"] == pytest.approx(0.1)


class TestResolve:
    def test_levels_tuple_is_the_documented_order(self):
        assert LEVELS == ("none", "limits", "wheel")

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("none", "none"),
            ("LIMITS", "limits"),
            ("  wheel  ", "wheel"),
            ("WHEEL", "wheel"),
        ],
    )
    def test_resolve_level_normalizes(self, raw, expected):
        assert resolve_level(raw) == expected

    @pytest.mark.parametrize("raw", ["", "dyn", "full", None, 3])
    def test_resolve_level_rejects_unknown(self, raw):
        with pytest.raises(ValueError, match="dynamics 必须是"):
            resolve_level(raw)

    def test_resolve_limits_keeps_none_untouched(self):
        assert resolve_limits("none", 0.0, 0.0) == (0.0, 0.0)
        assert resolve_limits("none", 2.0, 5.0) == (2.0, 5.0)

    def test_resolve_limits_fills_documented_defaults(self):
        assert resolve_limits("limits", 0.0, 0.0) == (
            DEFAULT_ACCEL_MAX,
            DEFAULT_STEER_RATE_MAX,
        )
        assert resolve_limits("wheel", 0.0, 0.0) == (
            DEFAULT_ACCEL_MAX,
            DEFAULT_STEER_RATE_MAX,
        )
        # 用户显式给了非零值就以用户值为准
        assert resolve_limits("wheel", 1.0, 2.0) == (1.0, 2.0)
        # 单向缺省也各自回落
        assert resolve_limits("limits", 0.8, 0.0) == (0.8, DEFAULT_STEER_RATE_MAX)


# --------------------------------------------------------------------------
# env 级：none 逐位不变 / limits 速率界 / wheel 滞后指纹 / 确定性 / reset
# --------------------------------------------------------------------------


class TestNoneLevel:
    def test_none_is_bitwise_identical_to_default_env(self):
        """dynamics="none" 等价于不传该键（同 seed 同动作 40 步 obs/reward 逐位相等）。"""
        acts = _ramp(40)
        a = ChassisSimEnv(cfg=EnvConfig(dynamics="none"))
        b = ChassisSimEnv(cfg=EnvConfig())
        try:
            obs_a, rew_a, info_a = _run(a, acts, seed=11)
            obs_b, rew_b, info_b = _run(b, acts, seed=11)
        finally:
            a.close()
            b.close()
        for x, y in zip(obs_a, obs_b):
            np.testing.assert_array_equal(x, y)
        assert rew_a == rew_b
        assert set(info_a) == set(info_b)
        assert "dynamics" not in info_a
        assert a.describe_dynamics() == {
            "level": "none",
            "accel_max": 0.0,
            "steer_rate_max": 0.0,
        }

    def test_none_does_not_rate_limit_and_has_no_module_object(self):
        """none 档不构造 dynamics 对象、动作一步到位（limits 档才会分步爬升）。"""
        env = ChassisSimEnv(cfg=EnvConfig(dynamics="none"))
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
            assert env._dynamics is None
            assert float(info["mapped_action"][1]) == pytest.approx(1.2, abs=1e-6)
        finally:
            env.close()


class TestLimitsLevel:
    def test_defaults_are_filled_when_config_says_zero(self):
        env = ChassisSimEnv(cfg=EnvConfig(dynamics="limits"))
        try:
            d = env.describe_dynamics()
            assert d["level"] == "limits"
            assert d["accel_max"] == pytest.approx(DEFAULT_ACCEL_MAX)
            assert d["steer_rate_max"] == pytest.approx(DEFAULT_STEER_RATE_MAX)
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
            assert info["dynamics"] == d
        finally:
            env.close()

    def test_user_values_win_over_defaults(self):
        env = ChassisSimEnv(
            cfg=EnvConfig(dynamics="limits", accel_max=0.7, steer_rate_max=0.4)
        )
        try:
            d = env.describe_dynamics()
            assert d["accel_max"] == pytest.approx(0.7)
            assert d["steer_rate_max"] == pytest.approx(0.4)
        finally:
            env.close()

    def test_rate_bounds_hold_and_saturate(self):
        """accel_max=1.0 / steer_rate_max=2.0：|Δv| ≤ 0.1、|Δδ| ≤ 0.2，且两者都撞到界。

        "撞到界"是本用例的关键：只断言"不超界"会被一个把动作全压成 0 的坏实现
        骗过去，所以同时要求存在一步的 Δ 恰好等于界值。
        """
        accel, rate = 1.0, 2.0
        env = ChassisSimEnv(
            cfg=EnvConfig(dynamics="limits", accel_max=accel, steer_rate_max=rate)
        )
        try:
            env.reset(seed=0)
            rows = []
            prev_d = prev_v = 0.0
            for _ in range(14):
                _, _, _, _, info = env.step(np.array([0.5, 1.0], dtype=np.float32))
                d_act = float(info["mapped_action"][0])
                v_act = float(info["mapped_action"][1])
                rows.append((d_act, v_act, d_act - prev_d, v_act - prev_v))
                prev_d, prev_v = d_act, v_act
        finally:
            env.close()
        dv = [r[3] for r in rows]
        dd = [r[2] for r in rows]
        assert max(abs(x) for x in dv) <= accel * DT + 1e-6, f"Δv 越界: {dv}"
        assert max(abs(x) for x in dd) <= rate * DT + 1e-6, f"Δδ 越界: {dd}"
        assert any(abs(abs(x) - accel * DT) <= 1e-6 for x in dv), f"Δv 从未贴到界: {dv}"
        assert any(abs(abs(x) - rate * DT) <= 1e-6 for x in dd), f"Δδ 从未贴到界: {dd}"


class TestWheelLevel:
    def test_wheel_sits_on_top_of_limits(self):
        """wheel = limits 的速率界 + 一阶滞后：Δv 仍 ≤ accel·dt，但只有 α_v 那一份。"""
        accel = 1.0
        env = ChassisSimEnv(cfg=EnvConfig(dynamics="wheel", accel_max=accel))
        try:
            env.reset(seed=0)
            prev_v = 0.0
            deltas = []
            for _ in range(5):
                _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
                v_act = float(info["mapped_action"][1])
                deltas.append(v_act - prev_v)
                prev_v = v_act
        finally:
            env.close()
        for dv in deltas:
            assert dv <= accel * DT + 1e-9
            assert dv == pytest.approx(ALPHA_V * accel * DT, abs=1e-6), deltas
        assert max(deltas) < accel * DT - 1e-3

    def test_wheel_level_keeps_steer_rate_bound(self):
        """wheel 档必须**严格包含** limits 档的 δ 速率界（回归）。

        未夹之前实测 |Δδ| = 0.3821 rad > 界 0.3000 rad（默认档、单步满舵、只出现在
        第 0 步）：v 与 ω 的时间常数不同（0.15 vs 0.10），低速端 arctan(ω·L/v) 把
        ω 的滞后差放大成更大的转角差。修复 = 滞后后对 δ 再夹一次并用夹后 δ 反算 ω
        （转向执行器的速率限制在物理上位于模型最下游）。
        """
        rate = 3.0
        env = ChassisSimEnv(cfg=EnvConfig(dynamics="wheel", steer_rate_max=rate))
        try:
            env.reset(seed=0)
            prev_d = 0.0
            deltas = []
            for _ in range(20):
                _, _, _, _, info = env.step(np.array([0.5, 1.0], dtype=np.float32))
                d_act = float(info["mapped_action"][0])
                deltas.append(abs(d_act - prev_d))
                prev_d = d_act
        finally:
            env.close()
        assert max(deltas) <= rate * DT + 1e-6, f"Δδ 越界: {max(deltas)}"
        assert max(deltas) == pytest.approx(rate * DT, abs=1e-6), deltas

    def test_ackermann_has_one_sample_delay_plus_one_pole(self):
        """与 chassis.step() 既有「转向滞后一拍」的关系：1 拍延迟 ⊗ 1 个一阶极点。

        判据（能把"两拍纯延迟"排除掉的那个）：
          * 第 1 步 ω_odom == 0 —— 底盘用的是上一帧转角（延迟那一拍，未变）；
          * 第 2 步 ω_odom > 0 —— 若再叠一拍纯延迟这里仍是 0；
          * ω_odom 单调升向稳态 —— 极点那部分。
        限幅在 1000（每步 100 m/s / 100 rad）下完全不生效，剩下的只有动力学本身。
        """
        env = ChassisSimEnv(
            cfg=EnvConfig(dynamics="wheel", accel_max=1000.0, steer_rate_max=1000.0)
        )
        try:
            env.reset(seed=0)
            omegas = []
            for _ in range(12):
                env.step(np.array([0.5, 1.0], dtype=np.float32))
                omegas.append(float(env.bridge.get_odometry()[4]))
        finally:
            env.close()
        assert omegas[0] == 0.0, omegas
        assert omegas[1] > 0.0, f"第 2 步仍为 0 ⇒ 看起来是两拍纯延迟: {omegas}"
        # 极点那一部分：第 2 步远未到稳态，峰值出现在延迟之后（不要求严格单调 ——
        # δ 由 arctan(ω·L/v) 反解，v 与 ω 的滞后比例不同会让它轻微回落）。
        assert omegas[1] < 0.98 * max(omegas)
        assert max(omegas) == max(omegas[2:])

    def test_diff_drive_has_pole_without_extra_delay(self):
        """对照：差速底盘没有转向滞后 ⇒ 第 1 步 ω 就非零（同一 dynamics 代码路径）。

        这一条与上一条一起把"一拍延迟"钉在底盘上，而不是 dynamics 模块里。
        """
        wheel_kw = {"dynamics": "wheel", "accel_max": 1000.0, "steer_rate_max": 1000.0}
        ref_kw = {"dynamics": "limits", "accel_max": 1000.0, "steer_rate_max": 1000.0}
        ref = ChassisSimEnv(make_chassis("diff_drive"), cfg=EnvConfig(**ref_kw))
        env = ChassisSimEnv(make_chassis("diff_drive"), cfg=EnvConfig(**wheel_kw))
        try:
            ref.reset(seed=0)
            ref.step(np.array([1.0, 0.5], dtype=np.float32))
            omega_cmd = float(ref.bridge.get_odometry()[4])
            env.reset(seed=0)
            env.step(np.array([1.0, 0.5], dtype=np.float32))
            omega_step1 = float(env.bridge.get_odometry()[4])
        finally:
            ref.close()
            env.close()
        assert omega_step1 > 0.0
        assert omega_step1 == pytest.approx(ALPHA_OMEGA * omega_cmd, rel=1e-6)

    def test_deadband_drops_small_commands(self):
        """死区 0.1 m/s（< accel_max·dt=0.15，可挣脱贫区）：小指令被吞，大指令照常走。"""
        env = ChassisSimEnv(cfg=EnvConfig(dynamics="wheel", v_deadband=0.1))
        try:
            env.reset(seed=0)
            for _ in range(3):
                env.step(np.array([0.5, 0.05], dtype=np.float32))  # v_cmd=0.06 < 0.1
            odom = env.bridge.get_odometry()
            assert float(odom[3]) == pytest.approx(0.0, abs=1e-12)
            assert float(odom[4]) == pytest.approx(0.0, abs=1e-12)
            for _ in range(12):
                env.step(np.array([0.5, 1.0], dtype=np.float32))
            assert float(env.bridge.get_odometry()[3]) > 0.4
        finally:
            env.close()

    def test_deadband_above_accel_dt_locks_from_rest(self):
        """文档化的粘滞锁死：v_deadband > accel_max·dt 时静止车永不启动。

        机制：死区作用在**限速后**的目标上，而限速器把目标锚在实际速度上，
        实际速度又被死区钉成 0 ⇒ 每步目标都 <= accel_max·dt < 死区。
        这是真实粘滞的形状（不是 bug），所以这里把它锁成契约而不是留成惊喜。
        """
        env = ChassisSimEnv(
            cfg=EnvConfig(dynamics="wheel", accel_max=1.5, v_deadband=0.4)
        )
        try:
            env.reset(seed=0)
            for _ in range(20):
                env.step(np.array([0.5, 1.0], dtype=np.float32))
            assert float(env.bridge.get_odometry()[3]) == pytest.approx(0.0, abs=1e-12)
        finally:
            env.close()

    def test_slip_lowers_steady_speed_only(self):
        """轮滑 0.3：稳态线速度打 0.7 折。"""
        kw = {"dynamics": "wheel", "accel_max": 1000.0, "steer_rate_max": 1000.0}
        clean = ChassisSimEnv(cfg=EnvConfig(**kw))
        slip = ChassisSimEnv(cfg=EnvConfig(wheel_slip=0.3, **kw))
        try:
            for env in (clean, slip):
                env.reset(seed=0)
                for _ in range(20):
                    env.step(np.array([0.0, 1.0], dtype=np.float32))
            v_clean = float(clean.bridge.get_odometry()[3])
            v_slip = float(slip.bridge.get_odometry()[3])
            # 容差 1e-4：动作经 float32 编码往返，1.2 回来是 1.1999980
            assert v_clean == pytest.approx(1.2, abs=1e-4)
            assert v_slip == pytest.approx(0.7 * v_clean, rel=1e-4)
        finally:
            clean.close()
            slip.close()

    def test_reset_clears_actuator_state(self):
        env = ChassisSimEnv(cfg=EnvConfig(dynamics="wheel", accel_max=1.0))
        try:
            env.reset(seed=0)
            for _ in range(4):
                env.step(np.array([0.0, 1.0], dtype=np.float32))
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
            assert float(info["mapped_action"][1]) == pytest.approx(
                ALPHA_V * 0.1, abs=1e-6
            )
        finally:
            env.close()

    def test_describe_includes_lag_params(self):
        env = ChassisSimEnv(cfg=EnvConfig(dynamics="wheel"))
        try:
            d = env.describe_dynamics()
            assert d["level"] == "wheel"
            assert d["v_tau"] == pytest.approx(V_TAU)
            assert d["omega_tau"] == pytest.approx(OMEGA_TAU)
            assert d["alpha_v"] == pytest.approx(ALPHA_V)
            assert d["alpha_omega"] == pytest.approx(ALPHA_OMEGA)
            assert d["v_deadband"] == 0.0 and d["wheel_slip"] == 0.0
            assert d["accel_max"] == pytest.approx(DEFAULT_ACCEL_MAX)
        finally:
            env.close()


class TestDeterminism:
    @pytest.mark.parametrize("level", LEVELS)
    def test_two_runs_same_seed_are_identical(self, level):
        acts = _ramp(20)
        a = ChassisSimEnv(cfg=EnvConfig(dynamics=level))
        b = ChassisSimEnv(cfg=EnvConfig(dynamics=level))
        try:
            obs_a, rew_a, _ = _run(a, acts, seed=7)
            obs_b, rew_b, _ = _run(b, acts, seed=7)
        finally:
            a.close()
            b.close()
        worst = max(float(np.max(np.abs(x - y))) for x, y in zip(obs_a, obs_b))
        assert worst == 0.0, worst
        assert rew_a == rew_b


class TestConfigPlumbing:
    def test_invalid_level_raises_at_construction(self):
        with pytest.raises(ValueError, match="dynamics 必须是"):
            ChassisSimEnv(cfg=EnvConfig(dynamics="bogus"))

    def test_info_schema_growth_is_exactly_one_key(self):
        """wheel 档 info 只比 none 多一个 dynamics 键（与 limits 相同），传感器不进 obs。"""
        envs = [
            ChassisSimEnv(cfg=EnvConfig()),
            ChassisSimEnv(cfg=EnvConfig(dynamics="limits")),
            ChassisSimEnv(cfg=EnvConfig(dynamics="wheel")),
        ]
        try:
            keys = []
            for env in envs:
                env.reset(seed=0)
                _, _, _, _, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
                keys.append(set(info))
            assert keys[1] == keys[0] | {"dynamics"}
            assert keys[2] == keys[1]
        finally:
            for env in envs:
                env.close()


# --------------------------------------------------------------------------
# 构造器 dynamics= （2026-11-04）：与 sensors= 对称的 env 级覆盖入口
# --------------------------------------------------------------------------


class TestDynamicsKwarg:
    """ChassisSimEnv(dynamics=...) 覆盖 cfg 的档位；None（默认）逐位回到旧行为。"""

    def test_default_none_is_bitwise_identical_to_omitted(self):
        """dynamics=None 与不传该参数逐位相同（obs / reward / info 键集 / describe 字典）。"""
        acts = _ramp(30)
        a = ChassisSimEnv(cfg=EnvConfig())
        b = ChassisSimEnv(cfg=EnvConfig(), dynamics=None)
        try:
            obs_a, rew_a, info_a = _run(a, acts, seed=5)
            obs_b, rew_b, info_b = _run(b, acts, seed=5)
            desc_a = a.describe_dynamics()
            desc_b = b.describe_dynamics()
        finally:
            a.close()
            b.close()
        for x, y in zip(obs_a, obs_b):
            np.testing.assert_array_equal(x, y)
        assert rew_a == rew_b
        assert set(info_a) == set(info_b)
        assert "dynamics" not in info_a
        # 默认路径的字典必须与加入本参数前逐键逐值相同（没有 level_source 键）
        assert desc_a == desc_b
        assert desc_a == {
            "level": "none",
            "accel_max": 0.0,
            "steer_rate_max": 0.0,
        }

    def test_kwarg_wheel_overrides_cfg(self):
        """cfg 是 none，构造器给 wheel ⇒ level 变 wheel，info 里照常曝光 dynamics。"""
        env = ChassisSimEnv(cfg=EnvConfig(), dynamics="wheel")
        try:
            d = env.describe_dynamics()
            assert d["level"] == "wheel"
            assert d["level_source"] == "constructor"
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
            assert info["dynamics"] == d
        finally:
            env.close()

    def test_kwarg_none_overrides_cfg_and_ignores_limits(self):
        """cfg 说 wheel 且 accel_max>0，kwarg 说 none ⇒ 纯运动学：限幅整段不生效。"""
        cfg = EnvConfig(dynamics="wheel", accel_max=1.0, steer_rate_max=2.0)
        env = ChassisSimEnv(cfg=cfg, dynamics="none")
        try:
            d = env.describe_dynamics()
            assert d["level"] == "none"
            assert d["level_source"] == "constructor"
            assert (d["accel_max"], d["steer_rate_max"]) == (0.0, 0.0)
            assert env._dynamics is None
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
            assert "dynamics" not in info
            # none 档一步到位；限幅若还生效，首步只会到 accel_max·dt = 0.1
            assert float(info["mapped_action"][1]) == pytest.approx(1.2, abs=1e-6)
            # 冲突消解不改调用方传进来的 cfg
            assert cfg.dynamics == "wheel" and cfg.accel_max == 1.0
        finally:
            env.close()

    @pytest.mark.parametrize("raw", ["WHEEL ", " Wheel", "  wheel\t", "wHeEl"])
    def test_kwarg_accepts_case_and_whitespace(self, raw):
        env = ChassisSimEnv("ackermann", dynamics=raw)
        try:
            assert env.describe_dynamics()["level"] == "wheel"
            assert env.describe_dynamics()["level_source"] == "constructor"
        finally:
            env.close()

    def test_kwarg_none_accepts_case_and_whitespace(self):
        env = ChassisSimEnv("ackermann", dynamics=" NONE ")
        try:
            d = env.describe_dynamics()
            assert d["level"] == "none"
            assert d["level_source"] == "constructor"
        finally:
            env.close()

    @pytest.mark.parametrize("bad", ["bogus", "", "none-ish"])
    def test_invalid_kwarg_lists_all_levels(self, bad):
        with pytest.raises(ValueError) as excinfo:
            ChassisSimEnv("ackermann", dynamics=bad)
        msg = str(excinfo.value)
        for level in LEVELS:
            assert level in msg, msg
        assert repr(bad) in msg or bad in msg

    def test_kwarg_matches_cfg_when_no_conflict(self):
        """kwarg="limits" 与 cfg.dynamics="limits" 是同一条路径（只多一个 level_source）。"""
        acts = _ramp(20)
        a = ChassisSimEnv(cfg=EnvConfig(dynamics="limits"))
        b = ChassisSimEnv(cfg=EnvConfig(), dynamics="limits")
        try:
            obs_a, rew_a, _ = _run(a, acts, seed=3)
            obs_b, rew_b, _ = _run(b, acts, seed=3)
            desc_a = a.describe_dynamics()
            desc_b = dict(b.describe_dynamics())
        finally:
            a.close()
            b.close()
        for x, y in zip(obs_a, obs_b):
            np.testing.assert_array_equal(x, y)
        assert rew_a == rew_b
        assert desc_b.pop("level_source") == "constructor"
        assert desc_b == desc_a
