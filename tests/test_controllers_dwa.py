"""DWA 在无障碍直线上的稳定性：不反向、不震荡、候选数守住预算。

闭环用本文件自带的 `_StubDiff` 推演（**不 import chassis 包**）。
`v_min=-1.0` 显式放开倒车，这样"速度符号恒定"才是对代价函数的真检验 ——
若把 v_min 钳到 0，"不倒退"会退化成一句废话。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from chassis_sim.controllers import DwaController

DT = 0.1


@dataclass(frozen=True)
class _StubCommand:
    v: float
    omega: float
    vy: float = 0.0


class _StubDiff:
    name = "diff_drive"
    action_dim = 2

    def __init__(self, v_max=1.0, v_min=-1.0, omega_max=1.0):
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)

    def action_space(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def command(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v = a[0] * self.v_max if a[0] >= 0.0 else a[0] * (-self.v_min)
        return _StubCommand(float(v), float(a[1] * self.omega_max), 0.0)

    def describe(self):
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "action_shape": (2,),
            "limits": {
                "v_min": self.v_min,
                "v_max": self.v_max,
                "omega_max": self.omega_max,
            },
            "params": {},
        }


def _integrate(state, cmd, dt):
    x, y, th = state
    return (
        x + (cmd.v * np.cos(th) - cmd.vy * np.sin(th)) * dt,
        y + (cmd.v * np.sin(th) + cmd.vy * np.cos(th)) * dt,
        th + cmd.omega * dt,
    )


def _straight_path(length=30.0, step=0.05):
    xs = np.arange(0.0, length + step, step)
    return np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])


def test_dwa_straight_line_no_reverse_no_oscillation():
    chassis = _StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0)
    ctrl = DwaController(chassis=chassis, dt=DT)
    path = _straight_path()
    ctrl.reset()
    ctrl.set_path(path)

    state = (0.0, 0.0, 0.0)
    vs, omegas, counts = [], [], []
    for _ in range(100):  # 10 s @ dt=0.1
        action = ctrl.act({"pose": np.array(state)})
        cmd = chassis.command(action)
        counts.append(ctrl.last_candidate_count)
        vs.append(cmd.v)
        omegas.append(cmd.omega)
        state = _integrate(state, cmd, DT)

    vs = np.asarray(vs)
    omegas = np.asarray(omegas)
    print(
        "DWA 直线：v 前 3 =",
        np.round(vs[:3], 6),
        "| v 末值 =",
        round(float(vs[-1]), 6),
        "| min v =",
        round(float(vs.min()), 6),
        "| max|ω| =",
        round(float(np.abs(omegas).max()), 6),
        "| 候选数 max =",
        max(counts),
    )
    assert np.all(vs > 0.0), f"出现非正速度：min v = {vs.min():.6f}"
    assert np.all(np.isfinite(vs)) and np.all(np.isfinite(omegas))
    assert np.abs(omegas).max() < 0.05, (
        f"直线出现横摆抖动：max|ω| = {np.abs(omegas).max():.4f}"
    )
    assert max(counts) <= 63, f"候选数 {max(counts)} 超出 ≤63 预算"
    assert np.abs(state[1]) < 0.05, f"直线终点横向漂移 {state[1]:.4f} m"


def test_dwa_without_path_is_zero_action():
    chassis = _StubDiff()
    ctrl = DwaController(chassis=chassis)
    assert np.allclose(ctrl.act({"pose": np.zeros(3)}), 0.0)
    ctrl.set_path(np.empty((0, 3)))
    assert np.allclose(ctrl.act({"pose": np.zeros(3)}), 0.0)


def test_dwa_is_deterministic_under_identical_input():
    """同输入同输出（_LastCallMemo 护栏）。"""
    chassis = _StubDiff()
    ctrl = DwaController(chassis=chassis, dt=DT)
    ctrl.set_path(_straight_path())
    obs = {"pose": np.array([0.0, 0.0, 0.0])}
    a1 = ctrl.act(obs)
    a2 = ctrl.act(obs)
    assert np.array_equal(a1, a2)

# ── 反振荡/脱困改造（2026-11-04）───────────────────────────────────
# 背景：obstacles_dense 上原实现在路径折返点（cusp）前 0.2 m 处来回抖满 500 步、
# 成功率 0%。根因是 w_heading 用**鼻向**判据：cusp 前后两条腿的鼻向差近 π，
# "沿倒车腿正常行驶"被判成 π 的航向误差；且代价里没有"沿路径往前走"的激励。


def test_dwa_axis_heading_ignores_nose_flip():
    """折返点前后鼻向差 π：axis 模式对"鼻向列"不敏感，nose 模式罚 w_heading·π²。

    这正是 obstacles_dense 上原地抖 400 步的根因：机器人贴着 cusp 前的直腿前进，
    候选末端投影到 cusp 之后的倒车腿（鼻向差近 π），鼻向判据把它判成 π 的航向误差。
    """
    xs = np.linspace(0.0, 2.0, 41)
    nose_fwd = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    nose_flip = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, np.pi)])

    def _cost(path, mode):
        ctrl = DwaController(
            chassis=_StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0),
            dt=DT,
            w_cte_traj=0.0,
            w_vel=0.0,
            w_smooth=0.0,
            w_prog=0.0,
            heading_mode=mode,
        )
        ctrl.set_path(path)
        # 一个候选：从 (0.2,0) 朝 +x 走 1 m，末端 (1.2,0) 落在路径上 ⇒ d_end = 0
        return float(ctrl._score(0.2, 0.0, 0.0, np.array([0.0]), np.array([1.0]), None)[0])

    axis_fwd = _cost(nose_fwd, "axis")
    axis_flip = _cost(nose_flip, "axis")
    nose_fwd = _cost(nose_fwd, "nose")
    nose_flip_cost = _cost(nose_flip, "nose")
    print(
        "axis: nose=0 ->",
        round(axis_fwd, 6),
        "| nose=pi ->",
        round(axis_flip, 6),
        "|| nose 模式:",
        round(nose_fwd, 6),
        "|",
        round(nose_flip_cost, 6),
    )
    assert abs(axis_fwd - axis_flip) < 1e-12, "axis 模式不应受鼻向列翻转影响"
    assert abs(nose_fwd - axis_fwd) < 1e-12, "鼻向与车头一致时两种模式应相同"
    assert abs((nose_flip_cost - axis_flip) - 0.3 * np.pi**2) < 1e-9, (
        "nose 模式应把鼻向翻转罚 w_heading*pi^2，实际差 "
        f"{nose_flip_cost - axis_flip:.6f}"
    )


def test_dwa_progress_term_breaks_forward_backward_tie():
    """w_prog 给"沿弧长前进"正激励，并把后退罚分饱和在 _PROG_BACK_CLIP。"""
    path = _straight_path(length=5.0, step=0.05)
    ctrl = DwaController(
        chassis=_StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0),
        dt=DT,
        w_cte_traj=0.0,
        w_vel=0.0,
        w_smooth=0.0,
        w_prog=2.0,
    )
    ctrl.set_path(path)
    cost = ctrl._score(
        1.0, 0.0, 0.0, np.array([0.0, 0.0]), np.array([1.0, -1.0]), None
    )
    # 前进：ds = 2.0 - 1.0 = 1.0（== horizon*dt*v_max，未饱和）；后退：ds = -1.0 → 饱和到 -0.5
    assert abs(cost[0] - (-2.0)) < 1e-9, f"前进候选代价应 ≈ -w_prog*1.0，实际 {cost[0]:.6f}"
    assert abs(cost[1] - (2.0 * 0.5)) < 1e-9, f"后退候选代价应 ≈ +w_prog*0.5，实际 {cost[1]:.6f}"


def test_dwa_defaults_are_axis_and_progress():
    ctrl = DwaController(chassis=_StubDiff())
    assert ctrl.heading_mode == "axis"
    assert ctrl.w_prog == 2.0


def test_dwa_rejects_unknown_heading_mode():
    try:
        DwaController(chassis=_StubDiff(), heading_mode="both")
    except ValueError as exc:
        assert "heading_mode" in str(exc)
    else:  # pragma: no cover - 只为给出清晰失败信息
        raise AssertionError("heading_mode 取值非法时应抛 ValueError")


# ---------------------------------------------------------------------------
# 档位（gear）语义与航向前瞻：2026-11-04 追加
# 判据与主仓 AckermannEnv.segment_is_reverse 同源：路径第 3 列是**鼻向**，
# 段档位由"段运动方向与鼻向夹角 > π/2 ⇒ 倒车"决定。
# ---------------------------------------------------------------------------


def _nose_flip_path(length=3.0, step=0.05, rev_lo=1.0, rev_hi=2.0):
    """沿 +x 的直线，鼻向列在 [rev_lo, rev_hi) 翻成 π ⇒ 中段是倒车段。

    几何上是一条直线（运动方向恒 +x），只有鼻向列变了 —— 这正是"档位由鼻向判、
    不由几何判"的最小样本：只看几何怎么都推不出倒车段。
    """
    xs = np.arange(0.0, length + step, step)
    nose = np.where((xs >= rev_lo) & (xs < rev_hi), np.pi, 0.0)
    return np.column_stack([xs, np.zeros_like(xs), nose])


def _foldback_path(step=0.05, leg_len=1.95, back_dx=1.6, back_dy=0.25):
    """前进腿沿 +x 到 (leg_len, 0)，折返腿回到 (leg_len-back_dx, back_dy)。

    折返腿的鼻向取"运动方向 + π" ⇒ 它是倒车段。两条腿在 cusp 处相接、整段相距
    back_dy（0.25 m），用来复现"候选末端最近段在两条腿之间跳"。
    """
    ax = np.arange(0.0, leg_len + 0.5 * step, step)
    a = np.column_stack([ax, np.zeros_like(ax), np.zeros_like(ax)])
    n = 30
    t = np.linspace(0.0, 1.0, n + 1)
    b = np.column_stack(
        [
            (ax[-1] + step) - back_dx * t,
            back_dy * t,
            np.full(t.shape, np.arctan2(back_dy, -back_dx) + np.pi),
        ]
    )
    return np.vstack([a, b[1:]])


def test_dwa_segment_gears_from_nose_column():
    """段档位 = F/R/F 三段，倒车段弧长 ≈1 m，且运动方向不受鼻向列影响。"""
    ctrl = DwaController(chassis=_StubDiff())
    path = _nose_flip_path()
    ctrl.set_path(path)
    gear = np.asarray(ctrl._seg_gear)
    assert gear.shape[0] == path.shape[0] - 1
    assert set(np.unique(gear)) == {1.0, -1.0}
    flips = int(np.count_nonzero(np.diff(np.sign(gear))))
    assert flips == 2, f"档位序列应只有两次切换，实际 {flips} 次"
    rev_arc = float(np.sum(ctrl._seg_len[gear < 0]))
    assert 0.9 < rev_arc < 1.1, f"倒车段弧长应 ≈1.0 m，实际 {rev_arc:.4f} m"
    assert float(np.sum(ctrl._seg_len[gear > 0])) > 1.9
    assert np.allclose(ctrl._seg_dir, 0.0, atol=1e-12), "直线路径的运动方向应恒为 +x"


def test_dwa_gear_hysteresis_needs_full_votes():
    """票数滞回：同一侧连续 votes 步才换档；回程同样要攒满票（同主仓 GearVoteLag）。"""
    ctrl = DwaController(
        chassis=_StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0),
        dt=DT,
        gear_mode="auto",
        gear_votes=3,
        gear_hold_steps=2,
        gear_lookahead=0.0,
    )
    ctrl.set_path(_nose_flip_path())
    ctrl._update_gear(0.2, 0.0)  # 前进段首步：直接采纳 raw
    assert ctrl.gear == 1.0 and ctrl.gear_switches == 0
    for _ in range(5):  # 进入倒车段：票数 3→2→1→0→−1→−2，未触边
        ctrl._update_gear(1.5, 0.0)
    assert ctrl.gear == 1.0, "票数未触边不应换档"
    assert ctrl.gear_switches == 0
    ctrl._update_gear(1.5, 0.0)  # 第 6 次：票数 −3 触边且保持步数满足
    assert ctrl.gear == -1.0 and ctrl.gear_switches == 1
    ctrl._update_gear(0.2, 0.0)  # 回到前进段一步：票数 −2，仍不换回
    assert ctrl.gear == -1.0 and ctrl.gear_switches == 1
    for _ in range(5):
        ctrl._update_gear(0.2, 0.0)
    assert ctrl.gear == 1.0 and ctrl.gear_switches == 2


def test_dwa_gear_hold_steps_is_a_hard_floor():
    """votes=1 时票数两步就能跨越，靠 gear_hold_steps 保住最小驻留。"""
    ctrl = DwaController(
        chassis=_StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0),
        dt=DT,
        gear_mode="auto",
        gear_votes=1,
        gear_hold_steps=3,
        gear_lookahead=0.0,
    )
    ctrl.set_path(_nose_flip_path())
    ctrl._update_gear(0.2, 0.0)
    ctrl._update_gear(1.5, 0.0)
    assert ctrl.gear == 1.0
    ctrl._update_gear(1.5, 0.0)  # 票数已触边，但 hold=2 < 3
    assert ctrl.gear == 1.0, "保持步数未满足不应换档"
    ctrl._update_gear(1.5, 0.0)
    assert ctrl.gear == -1.0 and ctrl.gear_switches == 1


def test_dwa_gear_span_truncates_to_active_gear():
    """档位把候选速度截到半轴；min_speed 再去掉 0；off 模式原样返回（ablation）。"""
    path = _nose_flip_path()
    off = DwaController(
        chassis=_StubDiff(v_max=1.2, v_min=-1.0, omega_max=1.2), dt=DT
    )
    off.set_path(path)
    assert off.gear_mode == "off"
    assert off._gear_active() is False
    assert off._gear_span(-1.0, 1.2) == (-1.0, 1.2)

    auto = DwaController(
        chassis=_StubDiff(v_max=1.2, v_min=-1.0, omega_max=1.2),
        dt=DT,
        gear_mode="auto",
        gear_lookahead=0.0,
    )
    auto.set_path(path)
    auto._update_gear(0.2, 0.0)
    assert auto._gear_span(-1.0, 1.2) == (0.0, 1.2), "前进档不应采样负速度"
    for _ in range(6):
        auto._update_gear(1.5, 0.0)
    assert auto.gear == -1.0
    assert auto._gear_span(-1.0, 1.2) == (-1.0, 0.0), "倒车档不应采样正速度"

    floor = DwaController(
        chassis=_StubDiff(v_max=1.2, v_min=-1.0, omega_max=1.2),
        dt=DT,
        gear_mode="auto",
        gear_lookahead=0.0,
        min_speed=0.1,
    )
    floor.set_path(path)
    floor._update_gear(0.2, 0.0)
    assert floor._gear_span(-1.0, 1.2) == (0.1, 1.2)
    for _ in range(6):
        floor._update_gear(1.5, 0.0)
    assert floor._gear_span(-1.0, 1.2) == (-1.0, -0.1)


def test_dwa_gear_auto_forward_path_never_reverses():
    """前进场景里档位恒 +1、不换档、速度非负（验收 d 的离线版）。"""
    chassis = _StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0)
    ctrl = DwaController(chassis=chassis, dt=DT, gear_mode="auto")
    ctrl.set_path(_straight_path(length=30.0, step=0.05))
    state = (0.0, 0.0, 0.0)
    vs, gears, counts = [], [], []
    for _ in range(60):
        action = ctrl.act({"pose": np.array(state)})
        cmd = chassis.command(action)
        vs.append(cmd.v)
        gears.append(ctrl.gear)
        counts.append(ctrl.last_candidate_count)
        state = _integrate(state, cmd, DT)
    vs = np.asarray(vs)
    print("DWA auto 直线：v 前 3 =", np.round(vs[:3], 6), "| min v =", round(float(vs.min()), 6))
    assert np.all(vs >= 0.0), f"前进场景出现负速度：min v = {vs.min():.6f}"
    assert set(np.round(np.asarray(gears), 9)) == {1.0}
    assert ctrl.gear_switches == 0
    assert max(counts) <= 63
    assert state[0] > 1.5, f"60 步只走了 {state[0]:.3f} m"


def test_dwa_heading_reference_is_arc_based():
    """航向参考取"起始弧长 + heading_lookahead"处的段，与候选末端落在哪条腿无关。"""
    path = _foldback_path()
    kw = {
        "dt": DT,
        "w_path": 0.0,
        "w_cte_traj": 0.0,
        "w_vel": 0.0,
        "w_smooth": 0.0,
        "w_prog": 0.0,
        "w_heading": 1.0,
    }
    vs = np.array([0.1, 0.3, 0.5])
    zeros = np.zeros(3)
    look = DwaController(chassis=_StubDiff(), heading_lookahead=0.35, **kw)
    look.set_path(path)
    cost = look._score(1.9, 0.0, 0.0, zeros, vs, None)
    assert float(np.ptp(cost)) < 1e-12, f"航向参考随候选末端跳变：{cost}"
    d_back = float(np.arctan2(0.25, -1.6))
    expected = float((np.pi - abs(d_back)) ** 2)
    assert abs(float(cost[0]) - expected) < 1e-6, (
        f"参考应取弧长 1.9+0.35 处（折返腿）：{cost[0]:.6f} != {expected:.6f}"
    )

    near = DwaController(chassis=_StubDiff(), heading_lookahead=0.0, **kw)
    near.set_path(path)
    cost_near = near._score(1.9, 0.0, 0.0, zeros, vs, None)
    assert float(np.ptp(cost_near)) < 1e-12
    assert float(cost_near[0]) < float(cost[0]), (
        f"lookahead=0 应看到前进腿（代价更小）：{cost_near[0]:.6f} vs {cost[0]:.6f}"
    )


def test_dwa_gear_defaults_off_and_exposes_ablation():
    """默认 off：判据照算（可审计），但不参与决策；默认参数值与文档一致。"""
    ctrl = DwaController(chassis=_StubDiff())
    assert ctrl.gear_mode == "off"
    assert ctrl.gear_votes == 3
    assert ctrl.gear_hold_steps == 2
    assert ctrl.gear_lookahead == 0.0
    assert ctrl.heading_lookahead == 0.35
    assert ctrl.min_speed == 0.0
    ctrl.set_path(_nose_flip_path())
    assert ctrl._seg_gear is not None and bool(np.any(np.asarray(ctrl._seg_gear) < 0))
    assert ctrl._gear_active() is False
    assert ctrl.gear == 1.0 and ctrl.gear_switches == 0
    assert ctrl._gear_span(-1.0, 1.2) == (-1.0, 1.2)


def test_dwa_rejects_invalid_gear_params():
    cases = (
        ({"gear_mode": "hard"}, "gear_mode"),
        ({"gear_votes": 0}, "gear_votes"),
        ({"gear_hold_steps": -1}, "gear_hold_steps"),
    )
    for kwargs, needle in cases:
        try:
            DwaController(chassis=_StubDiff(), **kwargs)
        except ValueError as exc:
            assert needle in str(exc)
        else:  # pragma: no cover - 只为给出清晰失败信息
            raise AssertionError(f"{kwargs} 应抛 ValueError")


def test_dwa_gear_auto_is_deterministic_under_identical_input():
    """档位状态机不得引入任何随机性：两次独立运行逐位相同。"""
    path = _foldback_path()

    def run():
        chassis = _StubDiff(v_max=1.0, v_min=-1.0, omega_max=1.0)
        ctrl = DwaController(chassis=chassis, dt=DT, gear_mode="auto")
        ctrl.set_path(path)
        state = (0.3, 0.0, 0.0)
        acts, poses = [], []
        for _ in range(15):
            action = ctrl.act({"pose": np.array(state)})
            acts.append(np.array(action, copy=True))
            state = _integrate(state, chassis.command(action), DT)
            poses.append(np.array(state))
        return np.array(acts), np.array(poses), ctrl.gear_switches, ctrl.gear

    a1, p1, s1, g1 = run()
    a2, p2, s2, g2 = run()
    assert np.array_equal(a1, a2), "两次运行的动作不一致"
    assert np.array_equal(p1, p2), "两次运行的位姿不一致"
    assert s1 == s2 and g1 == g2

