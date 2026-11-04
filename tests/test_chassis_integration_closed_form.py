"""闭式积分 / 小 ω 退化 / substep / CCD / obs-reward 指纹的验收测试。

对应任务书的验收条目：
  ① 圆周闭合（max-radial 偏差 ≤ 1e-9 m，旧实现 5e-2 量级）
  ② dt 扫描表（dt = 0.1 / 0.05 / 0.01，旧 vs 新并列）
  ③ ω → 0 连续性（|ω| ≤ 1e-9/0.1 走退化分支；1e-6 落在截断项界内；ω=0 逐位相同）
  ④ 位姿积分与观测的 sha256 指纹（ω=0 的回合必须不变；转向回合的变化可归因到积分本身）
  ⑤ substep = 1 逐位等同旧路径（ω=0 回合的指纹即证据）；substep 对 Euler 是 1/n 机制；
     CCD 开关的薄墙对照 + "只允许多检出"的超集性质

旧实现（Euler）在这里以 `_euler_integrate()` 的形式保留为**测试内对照物**：
它不是被测代码的副本，而是验收表的基准（旧数字由它复现）。改 integrate() 时
不需要同步改它——它就该停在 2026-11-05 之前的那三行算术上。
"""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import (
    AckermannChassis,
    Chassis,
    ChassisCommand,
    DiffDriveChassis,
)
from chassis_sim.config import EnvConfig, RewardConfig
from chassis_sim.envs import ChassisSimEnv

# ══════════════════════════════════════════════════════════════════════
# 旧实现（Euler，2026-11-05 之前 base.py 的 integrate 原文）
# ══════════════════════════════════════════════════════════════════════


def _euler_integrate(chassis, x, y, theta, cmd, dt):
    """改动前的 integrate：一阶 Euler，乘加顺序照抄原件。"""
    v, omega, vy = cmd.v, cmd.omega, cmd.vy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return (
        x + (v * cos_t - vy * sin_t) * dt,
        y + (v * sin_t + vy * cos_t) * dt,
        theta + omega * dt,
    )


def _euler_method(self, x, y, theta, cmd, dt):
    """_euler_integrate 的绑定方法形态，供 monkeypatch 回 Chassis.integrate 用。"""
    return _euler_integrate(self, x, y, theta, cmd, dt)


def _pose_bytes(pose) -> bytes:
    return np.asarray(pose, dtype=np.float64).tobytes()


# ══════════════════════════════════════════════════════════════════════
# ① / ② 圆周：max-radial 偏差与同刻解析解位置误差
# ══════════════════════════════════════════════════════════════════════

# 改动前实测（/tmp/probe_pre.py，2026-11-05）：整圈半径 R = v/ω = 1 m，v = ω = 1
_LEGACY_MAX_RADIAL = {0.1: 5.042e-02, 0.05: 2.510e-02, 0.01: 5.004e-03}
_LEGACY_POS_ERR_AT_T = {0.1: 8.408e-04, 0.05: 4.204e-04, 0.01: 1.593e-05}


def _circle(integrate, dt: float, v: float = 1.0, omega: float = 1.0):
    """走满 2π/ω（步数 round 到整数）：返回 (max-radial 偏差, 同刻解析解位置误差, 步数)。"""
    ch = DiffDriveChassis(wheelbase=0.145, v_max=1.0, omega_max=1.0)
    cmd = ChassisCommand(v=v, omega=omega)
    radius = v / omega
    n = int(round(2.0 * math.pi / omega / dt))
    x = y = theta = 0.0
    worst = 0.0
    for _ in range(n):
        x, y, theta = integrate(ch, x, y, theta, cmd, dt)
        worst = max(worst, abs(math.hypot(x, y - radius) - radius))
    t_end = n * dt
    pos_err = math.hypot(
        x - radius * math.sin(omega * t_end),
        y - radius * (1.0 - math.cos(omega * t_end)),
    )
    return worst, pos_err, n


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.01])
def test_circle_error_dt_sweep_new_vs_legacy(dt):
    """新旧对照表：旧实现是 O(dt) 的一阶指纹，新实现只剩浮点噪声。

    旧：max-radial 5.042e-02 / 2.510e-02 / 5.004e-03（严格按 dt 线性减半）；
    新：3.331e-16 / 7.772e-16 / 1.554e-15 —— 三个 dt 之间只差浮点噪声。
    """
    old_radial, old_pos, n_old = _circle(_euler_integrate, dt)
    new_radial, new_pos, n_new = _circle(
        lambda ch, x, y, th, cmd, d: ch.integrate(x, y, th, cmd, d), dt
    )
    assert n_old == n_new == int(round(2.0 * math.pi / dt))

    # 旧实现的数字（验收表里逐字给出的那三个）
    assert old_radial == pytest.approx(_LEGACY_MAX_RADIAL[dt], rel=1e-3)
    assert old_pos == pytest.approx(_LEGACY_POS_ERR_AT_T[dt], rel=1e-3)

    # 新实现：验收 ① 的门槛是 1e-9 m，实测 1e-15 量级
    assert new_radial <= 1e-9
    assert new_pos <= 1e-9
    assert old_radial / new_radial > 1e9  # 至少 9 个数量级


def test_circle_error_is_dt_independent_for_closed_form():
    """② 的后半句：新实现在三个 dt 之间只差浮点噪声（不是"再小 3 倍"）。"""
    errs = []
    for dt in (0.1, 0.05, 0.01):
        radial, _, _ = _circle(
            lambda ch, x, y, th, cmd, d: ch.integrate(x, y, th, cmd, d), dt
        )
        errs.append(radial)
    assert max(errs) <= 1e-12
    assert max(errs) / min(errs) < 100.0  # 纯浮点噪声，无 dt 依赖


# ══════════════════════════════════════════════════════════════════════
# ③ ω → 0 连续性
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("omega", [1e-12, 1e-9, 1e-8])
def test_tiny_omega_takes_degenerate_branch_bitwise(omega):
    """|ω·dt| ≤ sqrt(eps) ≈ 1.49e-8 时走 Euler 退化分支 ⇒ 与旧实现**逐位**相同。"""
    ch = DiffDriveChassis()
    cmd = ChassisCommand(v=1.0, omega=omega, vy=0.2)
    dt = 0.1
    assert abs(omega * dt) <= 1.5e-8
    new = ch.integrate(0.3, 0.4, 0.7, cmd, dt)
    old = _euler_integrate(ch, 0.3, 0.4, 0.7, cmd, dt)
    assert _pose_bytes(new) == _pose_bytes(old)


@pytest.mark.parametrize("omega", [1e-12, 1e-9, 1e-6])
def test_omega_to_zero_continuity_is_bounded(omega):
    """与 Euler 的差 ≤ 理论上界：粗口径 |ω|·dt·R（= |v|·dt），尖锐口径 |v|·dt·|φ|/2。

    φ = ω·dt。φ/2 是"Euler 沿起点切向走 vs 闭式沿中点切向走"的方向差，
    位置差的主导项就是 |v|·dt·|φ|/2；实测 ω=1e-6 时为 5.021e-9（上界 5.0e-9，
    1% 余量足够——差的下一阶是曲率项 |v|dt·φ²/24 ≈ 4e-13）。
    """
    ch = DiffDriveChassis()
    cmd = ChassisCommand(v=1.0, omega=omega)
    dt = 0.1
    new = ch.integrate(0.3, 0.4, 0.7, cmd, dt)
    old = _euler_integrate(ch, 0.3, 0.4, 0.7, cmd, dt)

    dpos = math.hypot(new[0] - old[0], new[1] - old[1])
    radius = abs(cmd.v / omega)
    phi = abs(omega * dt)

    assert dpos <= abs(omega) * dt * radius + 1e-18  # 粗口径（任务书给的那个）
    assert dpos <= 1.01 * abs(cmd.v) * dt * phi / 2.0 + 1e-18  # 尖锐口径
    assert np.float64(new[2]).tobytes() == np.float64(old[2]).tobytes()  # θ 同式


def test_zero_omega_path_is_bitwise_identical_to_legacy():
    """ω 恰为 0：新式必须逐位等于旧式的 (x, y, θ) —— 乘加顺序一字不改。"""
    ch = DiffDriveChassis()
    checked = 0
    for v in (1.0, -0.37, 0.0, 1e-17):
        for vy in (0.0, 0.5, -0.42):
            for theta in (0.0, 0.7, -2.9, math.pi):
                for dt in (0.1, 0.05, 1.0 / 3.0):
                    cmd = ChassisCommand(v=v, omega=0.0, vy=vy)
                    new = ch.integrate(0.25, -1.5, theta, cmd, dt)
                    old = _euler_integrate(ch, 0.25, -1.5, theta, cmd, dt)
                    assert _pose_bytes(new) == _pose_bytes(old)
                    checked += 1
    assert checked == 4 * 3 * 4 * 3


# ══════════════════════════════════════════════════════════════════════
# ⑤ substep：默认 1 逐位同旧；对 Euler 是 1/n；对闭式积分没有额外收益
# ══════════════════════════════════════════════════════════════════════

_ACK_ACTIONS = [
    [0.20, 0.50],
    [-0.30, 0.80],
    [0.00, 0.40],
    [0.35, 0.90],
    [0.10, 0.60],
    [0.00, 0.20],
    [-0.25, 0.70],
]


def _bridge_poses(chassis, actions, dt: float = 0.1, **kwargs):
    bridge = SimpleBridge(
        chassis,
        dt=dt,
        world_bounds=(0.0, 10.0, 0.0, 10.0),
        domain_rand=False,
        seed=0,
        **kwargs,
    )
    bridge.reset(
        obstacles=[],
        path_waypoints=np.array([[1.0, 5.0, 0.0], [8.0, 5.0, 0.0]]),
        initial_heading=0.0,
    )
    poses = [bridge.pose()]
    for act in actions:
        bridge.step(np.asarray(act, dtype=np.float64))
        poses.append(bridge.pose())
    bridge.close()
    return np.asarray(poses, dtype=np.float64)


def test_substeps_one_is_bitwise_identical_to_default():
    """不传 substep 与显式传 1 必须逐位相同（默认路径没被"统一进循环"改动）。"""
    default = _bridge_poses(AckermannChassis(), _ACK_ACTIONS)
    explicit = _bridge_poses(AckermannChassis(), _ACK_ACTIONS, integration_substeps=1)
    assert default.tobytes() == explicit.tobytes()
    # 且与"完全不经过 substep 分支"的直算参考一致（同一 dt 一步积分）
    ch = AckermannChassis()
    x, y, th = 1.0, 5.0, 0.0
    ref = [(x, y, th)]
    for act in _ACK_ACTIONS:
        cmd = ch.step(np.asarray(act, dtype=np.float64))
        x, y, th = ch.integrate(x, y, th, cmd, 0.1)
        th = (th + np.pi) % (2.0 * np.pi) - np.pi
        ref.append((x, y, th))
    assert np.asarray(ref, dtype=np.float64).tobytes() == default.tobytes()


def _circle_via_bridge(monkeypatch, n_sub: int, dt: float = 0.1, n_steps: int = 63):
    """差速底盘在桥里走半径 1 m 的整圈，返回轨迹点（世界系）。"""
    ch = DiffDriveChassis(v_max=1.0, omega_max=1.0)
    bridge = SimpleBridge(
        ch,
        dt=dt,
        world_bounds=(0.0, 10.0, 0.0, 10.0),
        domain_rand=False,
        seed=0,
        integration_substeps=n_sub,
    )
    bridge.reset(
        obstacles=[],
        path_waypoints=np.array([[3.0, 5.0, 0.0], [8.0, 5.0, 0.0]]),
        initial_heading=0.0,
    )
    pts = []
    for _ in range(n_steps):
        bridge.step(np.array([1.0, 1.0]))  # v = 1, ω = 1
        pts.append(bridge.pose()[:2])
    bridge.close()
    return np.asarray(pts, dtype=np.float64)


def _max_radial_from_pts(pts: np.ndarray, centre=(3.0, 6.0), radius: float = 1.0):
    return float(
        np.max(np.abs(np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) - radius))
    )


def test_substep_halves_euler_error_by_one_over_n(monkeypatch):
    """substep 的机制：对 **Euler** 是严格 1/n（实测 5.042e-2 → 1.253e-2，比 4.02）。

    这条同时说明为什么"substep=4 让圆周误差再降一个量级"能成立**只对一阶方法**。
    """
    monkeypatch.setattr(Chassis, "integrate", _euler_method)
    err1 = _max_radial_from_pts(_circle_via_bridge(monkeypatch, 1))
    err4 = _max_radial_from_pts(_circle_via_bridge(monkeypatch, 4))
    assert err1 == pytest.approx(5.042e-02, rel=0.02)
    ratio = err1 / err4
    assert 3.5 <= ratio <= 4.5, f"Euler 下误差应按 1/n 收缩，实测比 {ratio:.3f}"


def test_substep_gains_nothing_for_closed_form(monkeypatch):
    """闭式积分对恒定 (v, ω) 本来就精确 ⇒ substep=4 与 1 只差浮点噪声（如实记录）。

    实测 max-radial：n_sub=1 → 3.331e-16、n_sub=2 → 7.772e-16、n_sub=4 → 6.661e-16；
    全部 ≤ 1e-12，且 n_sub 之间无单调改善。"再降一个量级"在闭式解上不可能，
    因为误差已经是 **每步一次浮点舍入** 的累加，不是截断误差。
    """
    err1 = _max_radial_from_pts(_circle_via_bridge(monkeypatch, 1))
    err4 = _max_radial_from_pts(_circle_via_bridge(monkeypatch, 4))
    assert err1 <= 1e-12 and err4 <= 1e-12
    assert abs(err1 - err4) <= 1e-12


# ══════════════════════════════════════════════════════════════════════
# ⑤ CCD：0.1 m 薄墙的开关对照 + 超集性质
# ══════════════════════════════════════════════════════════════════════


def _thin_wall_run(ccd: bool):
    """v_max=20、dt=0.1 ⇒ 每步 2 m，一步跨过 0.1 m 薄墙（x=1.0）。"""
    ch = AckermannChassis(v_max=20.0, curvature_action=False)
    bridge = SimpleBridge(
        ch,
        dt=0.1,
        world_bounds=(0.0, 10.0, 0.0, 10.0),
        domain_rand=False,
        seed=0,
        ccd=ccd,
    )
    bridge.reset(
        obstacles=[
            {
                "shape": {"name": "rectangle", "length": 0.1, "width": 2.0},
                "state": [1.0, 5.0, 0.0],
            }
        ],
        path_waypoints=np.array([[0.9, 5.0, 0.0], [9.0, 5.0, 0.0]]),
        initial_heading=0.0,
    )
    hits = []
    for _ in range(2):
        bridge.step(np.array([0.0, 1.0]))
        hits.append(bridge.check_collision())
    pose = bridge.pose()
    bridge.close()
    return hits, pose


def test_ccd_detects_thin_wall_that_plain_check_misses():
    """关：一步从 x=0.9 跳到 2.9，端点判定两边都没墙 ⇒ 静默穿模（对照物）。

    开：扫掠段在墙体内采到中间位姿 ⇒ 第一步即检出。
    """
    hits_off, pose_off = _thin_wall_run(ccd=False)
    assert hits_off == [False, False], "关掉 CCD 时这场景本就该穿过去（对照组）"
    assert pose_off[0] > 1.0, f"车已经越过墙（x={pose_off[0]:.3f}）"

    hits_on, _ = _thin_wall_run(ccd=True)
    assert hits_on[0] is True, "CCD 必须在第一步扫掠时就发现墙"


def test_ccd_only_adds_collisions_never_removes_them():
    """同一场景、同一动作序列：ccd=True 的检出步集合必须是 ccd=False 的超集。"""

    def run(ccd: bool):
        ch = AckermannChassis(v_max=1.0)
        bridge = SimpleBridge(
            ch,
            dt=0.1,
            world_bounds=(0.0, 10.0, 0.0, 10.0),
            domain_rand=False,
            seed=0,
            ccd=ccd,
        )
        bridge.reset(
            obstacles=[
                {
                    "shape": {"name": "rectangle", "length": 0.1, "width": 2.0},
                    "state": [2.0, 5.0, 0.0],
                }
            ],
            path_waypoints=np.array([[1.0, 5.0, 0.0], [8.0, 5.0, 0.0]]),
            initial_heading=0.0,
        )
        hits = []
        for _ in range(20):
            bridge.step(np.array([0.0, 1.0]))
            hits.append(bridge.check_collision())
        bridge.close()
        return hits

    off, on = run(False), run(True)
    first_off = next((i for i, h in enumerate(off) if h), None)
    first_on = next((i for i, h in enumerate(on) if h), None)
    assert first_off is not None, "普通速度下端点判定应当已经能检出这面墙"
    assert first_on is not None
    assert first_on <= first_off, "CCD 只允许**更早**检出，不允许漏检"
    assert all(on[i] or not off[i] for i in range(len(off))), "检出集合必须是超集"


# ══════════════════════════════════════════════════════════════════════
# 配置接线：EnvConfig 校验 / yaml 往返 / env → bridge 转发
# ══════════════════════════════════════════════════════════════════════


def test_envconfig_new_fields_defaults_and_validation(tmp_path):
    cfg = EnvConfig()
    assert cfg.integration_substeps == 1
    assert cfg.ccd is False

    for bad in (0, -1, 2.5, True, "4", None):
        with pytest.raises(ValueError):
            EnvConfig(integration_substeps=bad)

    back = EnvConfig.from_yaml(cfg.to_yaml(tmp_path / "env_default.yaml"))
    assert back.integration_substeps == 1 and back.ccd is False
    assert back == cfg

    tuned = EnvConfig(integration_substeps=4, ccd=True)
    back2 = EnvConfig.from_yaml(tuned.to_yaml(tmp_path / "env_tuned.yaml"))
    assert back2.integration_substeps == 4 and back2.ccd is True
    assert back2 == tuned


class _FixedScene:
    """确定性地图（不依赖 MapGenerator 的随机性）。"""

    stage_index = -1

    def __init__(
        self, path=None, obstacles=None, goal_heading=0.0, initial_heading=0.0
    ):
        self.path = _straight_path() if path is None else path
        self.obstacles = [] if obstacles is None else obstacles
        self.goal_heading = goal_heading
        self.initial_heading = initial_heading

    def generate(self, rng):
        return (self.obstacles, self.path, self.goal_heading, self.initial_heading)


def _straight_path(x0: float = 1.0, x1: float = 5.0, y: float = 5.0, n: int = 81):
    p = np.zeros((n, 3), dtype=np.float64)
    p[:, 0] = np.linspace(x0, x1, n)
    p[:, 1] = y
    return p


def test_env_forwards_substeps_and_ccd_to_self_made_bridge():
    """env 自造桥时把两个新配置转发下去；注入桥由调用方负责（不回写）。"""
    env = ChassisSimEnv(
        "ackermann",
        cfg=EnvConfig(integration_substeps=3, ccd=True, domain_rand_enabled=False),
        reward_cfg=RewardConfig(),
        map_generator=_FixedScene(),
        domain_rand=False,
    )
    try:
        assert env.bridge.integration_substeps == 3
        assert env.bridge.ccd is True
    finally:
        env.close()

    env2 = ChassisSimEnv(
        "ackermann",
        cfg=EnvConfig(domain_rand_enabled=False),
        reward_cfg=RewardConfig(),
        map_generator=_FixedScene(),
        domain_rand=False,
    )
    try:
        assert env2.bridge.integration_substeps == 1
        assert env2.bridge.ccd is False
    finally:
        env2.close()

    # 注入桥：cfg 里的开关**不**回写（桥的构造是调用方的事）
    chassis = AckermannChassis()
    injected = SimpleBridge(chassis, dt=0.1, domain_rand=False, seed=0)
    env3 = ChassisSimEnv(
        chassis,  # 同一个实例，满足"动作映射与物理同源"的构造期校验
        bridge=injected,
        cfg=EnvConfig(integration_substeps=4, ccd=True, domain_rand_enabled=False),
        reward_cfg=RewardConfig(),
        map_generator=_FixedScene(),
        domain_rand=False,
    )
    try:
        assert env3.bridge is injected
        assert injected.integration_substeps == 1 and injected.ccd is False
    finally:
        env3.close()


# ══════════════════════════════════════════════════════════════════════
# ④ obs / reward 指纹：ω=0 的回合必须逐位不变，转向回合的变化归因到积分
# ══════════════════════════════════════════════════════════════════════

_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.35}, "state": [4.5, 5.0, 0.0]},
    {
        "shape": {"name": "rectangle", "length": 0.6, "width": 0.4},
        "state": [6.0, 6.0, 0.3],
    },
]

_ENV_CFG = {
    "delta_max": 0.526,
    "kappa_max": 4.0,
    "curvature_action": True,
    "v_min": -1.0,
    "v_max": 1.0,
    "wheelbase": 0.145,
    "dt": 0.1,
    "domain_rand_enabled": False,
    "goal_end_wp_count": 3,
    "goal_end_wp_margin": 2,
}

# 改动前的指纹（2026-11-05，/tmp/probe_pre.py，本仓 src 上的旧 integrate）
#
# 这些数值只在本机口径（Linux / CPython 3.12 / numpy 2.x）成立：摘要里含路径点的
# trig 结果，libm 与 numpy 版本一变字节就变。所以默认走**同进程**的 Old-vs-New 对照，
# 记录值只在 CHASSIS_STRICT_GOLDEN=1 时再断言（跨环境 CI 不认绝对值）。
_STRICT_GOLDEN = os.environ.get("CHASSIS_STRICT_GOLDEN") == "1"

_LEGACY_DIGESTS = {
    "ackermann_straight": (
        "39ee385305b3a4720403702f56cd247a4acabef461a604137d93ad6f4d94900a",
        40,
    ),
    "diff_drive_straight": (
        "5b6642247aa235d1c8f751647104784691e4d3ec281564013e85c635e13bd4aa",
        36,
    ),
    "mecanum_straight": (
        "fab62cd4364728c8cc14a71131ad4ba9428183cefd12fb412c0eac71d66afe7e",
        32,
    ),
    "ackermann_turning": (
        "7f1fc8cc3981018636dc7a02f294b606a01e71104632d4ce7e48cf9d194721dc",
        36,
    ),
}

_STRAIGHT_ACK = [
    [0.0, float(v)] for v in (0.2, 0.5, 0.8, 1.0, 0.6, 0.3, 0.0, -0.3, -0.6, -1.0) * 4
]
_STRAIGHT_DIFF = [
    [float(v), 0.0] for v in (0.2, 0.5, 0.8, 1.0, 0.6, 0.3, 0.0, -0.3, -0.6) * 4
]
_STRAIGHT_MECANUM = [
    [float(v), 0.0, 0.0] for v in (0.2, 0.5, 0.8, 1.0, 0.6, 0.3, 0.0, -0.3) * 4
]
_TURNING_ACK = [
    [0.3, 0.6],
    [0.5, 0.8],
    [-0.2, 0.7],
    [-0.4, 0.5],
    [0.1, 0.9],
    [0.0, 0.4],
] * 6


def _digest_run(chassis_name: str, actions, poke_wheels: bool = False):
    """一回合的 sha256：reset obs + 每步 obs/reward/term/trunc。

    摘要口径与改动前的基线完全一致（同样场景、同样动作、同样 dtype），
    所以两条能直接对比。
    """
    scene = _FixedScene(path=_straight_path(), obstacles=_OBSTACLES)
    env = ChassisSimEnv(
        chassis_name,
        cfg=EnvConfig(**_ENV_CFG),
        reward_cfg=RewardConfig(),
        map_generator=scene,
        domain_rand=False,
    )
    h = hashlib.sha256()
    try:
        obs, info = env.reset(seed=0)
        h.update(np.ascontiguousarray(obs).tobytes())
        nrec = 0
        for act in actions:
            obs, rew, term, trunc, info = env.step(act)
            if poke_wheels:
                # 轮级层是只读观察：每步都调它，指纹必须一字不差
                env.bridge.chassis.wheel_kinematics(ChassisCommand(v=0.3, omega=0.2))
            h.update(np.ascontiguousarray(obs).tobytes())
            h.update(np.float64(rew).tobytes())
            h.update(bytes([bool(term), bool(trunc)]))
            nrec += 1
            if term or trunc:
                break
        return h.hexdigest(), nrec
    finally:
        env.close()


@pytest.mark.parametrize(
    "chassis_name,actions",
    [
        ("ackermann", _STRAIGHT_ACK),
        ("diff_drive", _STRAIGHT_DIFF),
        ("mecanum", _STRAIGHT_MECANUM),
    ],
)
def test_straight_line_obs_reward_digest_unchanged(chassis_name, actions, monkeypatch):
    """ω=0 的回合（全直线动作）必须与旧 Euler 实现**逐位**相同：退化分支没被改写。

    对照在同一进程里做（把 integrate 换回 _euler_method 再跑一遍），所以它证明的是
    "退化分支逐位未变"这条不变量本身，而不是"这台机器的 libm 和作者一致"。
    绝对值锚点见 _LEGACY_DIGESTS，只在本机口径 / CHASSIS_STRICT_GOLDEN=1 下核对。
    """
    new_digest, new_steps = _digest_run(chassis_name, actions)

    monkeypatch.setattr(Chassis, "integrate", _euler_method)
    old_digest, old_steps = _digest_run(chassis_name, actions)
    monkeypatch.undo()

    assert new_steps == old_steps
    assert new_digest == old_digest

    if _STRICT_GOLDEN:
        expected, expected_steps = _LEGACY_DIGESTS[f"{chassis_name}_straight"]
        assert (new_digest, new_steps) == (expected, expected_steps)


def test_turning_round_diff_is_attributable_to_integration_alone(monkeypatch):
    """转向回合：闭式积分改变了指纹，而换回旧 Euler 就逐位复现旧指纹。

    这条是"归因"的证据：同一份 env / 桥 / 奖励 / 观测代码，唯一变量是积分式。
    """
    new_digest, new_steps = _digest_run("ackermann", _TURNING_ACK)

    monkeypatch.setattr(Chassis, "integrate", _euler_method)
    old_digest, old_steps = _digest_run("ackermann", _TURNING_ACK)
    monkeypatch.undo()

    assert new_steps == old_steps
    assert new_digest != old_digest, "闭式积分必须改变 ω≠0 的轨迹（否则它没生效）"

    if _STRICT_GOLDEN:
        legacy_digest, legacy_steps = _LEGACY_DIGESTS["ackermann_turning"]
        assert (old_digest, old_steps) == (legacy_digest, legacy_steps)


def test_wheel_kinematics_calls_do_not_perturb_obs_reward_digest():
    """④ 的后半句：轮级层纯只读 —— 每步调用它，指纹与不调用时逐位相同。"""
    plain_digest, plain_steps = _digest_run("ackermann", _STRAIGHT_ACK)
    poked_digest, poked_steps = _digest_run(
        "ackermann", _STRAIGHT_ACK, poke_wheels=True
    )
    assert (poked_digest, poked_steps) == (plain_digest, plain_steps)
