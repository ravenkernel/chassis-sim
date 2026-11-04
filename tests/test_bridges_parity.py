"""桥对拍 golden：SimpleBridge ≡ 主仓 NumpySimBridge（A1 验收点）。

两件事：

(i) **生成阶段**（`/home/zhao/drl-sim` 存在时）——
    import `ros_bridge.numpy_bridge.NumpySimBridge`，用**本仓**
    `AckermannChassis.command()` 把归一化动作映射成旧桥认识的物理量
    `(δ, v)`（旧桥 `step()` 直接吃 [转向角, 速度]，不 clip 不映射），
    再逐步把位姿 / 双 LiDAR / 碰撞 / 到达 / 里程计全精度录进
    `tests/data/bridge_golden.json`。
    重录命令：`PYTHONPATH=src python tests/test_bridges_parity.py`

(ii) **断言阶段** —— 只依赖那份 JSON：同一串归一化动作重放
    `SimpleBridge(AckermannChassis())`，逐步比较。

三个场景（缺一不可）：

* `random_240` —— 2 圆 + 2 矩形障碍 + 拱形路径 + 240 步随机动作，
  覆盖射线求交 / SAT / 边界 / 里程计的主路径。
* `goal_60` —— 空场直行冲过终点，让 `goal_reached` 两侧都出现。
* `collision_40` —— 正前方放一个圆，让 `collision` 两侧都出现。
  后两个场景是**布尔断言的防空转闸门**：只有 True 与 False 都被覆盖，
  "布尔全等"才不是一句废话。

最后是**反向对照**：把 `step` 换成 `command`（不滞后）重放同一串动作，
位姿偏差必须 > 1e-3 m —— 否则说明这个对拍根本分辨不出转向滞后。

⚠️ 2026-11-05 起 `Chassis.integrate` 是**闭式**积分，而这份 golden 是旧桥
（Euler）录的。断言阶段因此分两支：

* `replay`（legacy）—— 把 `integrate` 钉回旧 Euler 再重放，与 golden 的
  位姿 / 双 LiDAR / 里程计 / 布尔量**逐通道**对拍；这一支锁死"桥的执行层没变"。
* `replay_closed` —— 生产实现重放，只用来量它与旧记录的差：
  θ 通道逐位相同、xy 差 ≈ 3.4e-2 m，恰好是换成闭式积分该有的差别。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis, Chassis
from chassis_sim.sensors import Lidar

_DRL_SIM = "/home/zhao/drl-sim"
_ENV_YAML = os.path.join(_DRL_SIM, "config", "env.yaml")
_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "bridge_golden.json"

SEED = 20261104
N_STEPS_RANDOM = 240
TOL_POSE = 1e-12  # 位姿"同一公式"；xy 在 ω=0 分支逐位相同（实测 0.0）
TOL_THETA = 1e-12  # θ 的 ulp 级噪声上限（跨 libm 实测 4.4e-16 ~ 2.7e-15）
TOL_REVERSE = 1e-3  # 反向对照的最小可分辨偏差（米）

# 与主仓 config/env.yaml 推导值逐位相同（world 10x10 / step_time 0.1 / 两处 margin）
_BRIDGE_PARAMS: dict = {
    "dt": 0.1,
    "world_bounds": (0.0, 10.0, 0.0, 10.0),
    "max_range": 10.0,
    "domain_rand": False,
    "goal_dist_threshold": 0.3,
    "goal_end_wp_count": 3,
    "goal_end_wp_margin": 2,
    "collision_margin": 0.03,
    "boundary_margin": 0.05,
}

_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.30}, "state": [2.60, 2.20, 0.0]},
    {"shape": {"name": "circle", "radius": 0.18}, "state": [4.10, 3.40, 0.0]},
    {
        "shape": {"name": "rectangle", "length": 0.90, "width": 0.35},
        "state": [3.40, 1.60, 0.50],
    },
    {
        "shape": {"name": "rectangle", "length": 0.50, "width": 0.50},
        "state": [5.00, 2.60, -0.30],
    },
]


# ═════════════════════════════════════════════════════════════════════
# 场景定义（生成阶段与重放阶段共用）
# ═════════════════════════════════════════════════════════════════════


def _curve_path() -> np.ndarray:
    """拱形路径，第 3 列是车头鼻向（切线方向）。"""
    s = np.linspace(0.0, 1.0, 60)
    x = 0.9 + 4.2 * s
    y = 0.9 + 1.6 * np.sin(np.pi * s)
    theta = np.arctan2(np.gradient(y), np.gradient(x))
    return np.column_stack([x, y, theta])


def _line_path(
    x0: float, y0: float, x1: float, y1: float, heading: float, n: int = 20
) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        [
            x0 + (x1 - x0) * s,
            y0 + (y1 - y0) * s,
            np.full(n, heading),
        ]
    )


def _scenarios() -> list:
    rng = np.random.default_rng(SEED)
    return [
        {
            "name": "random_240",
            "obstacles": [dict(o) for o in _OBSTACLES],
            "path": _curve_path(),
            "actions": rng.uniform(-1.0, 1.0, (N_STEPS_RANDOM, 2)),
        },
        {
            "name": "goal_60",
            "obstacles": [],
            "path": _line_path(2.0, 2.0, 5.0, 2.0, 0.0),
            "actions": np.tile(np.array([[0.0, 0.6]]), (60, 1)),
        },
        {
            "name": "collision_40",
            "obstacles": [
                {"shape": {"name": "circle", "radius": 0.25}, "state": [2.2, 3.0, 0.0]}
            ],
            "path": _line_path(1.0, 3.0, 6.0, 3.0, 0.0),
            "actions": np.tile(np.array([[0.0, 0.6]]), (40, 1)),
        },
    ]


# ── 浮点编解码：repr(float) 字符串，float() 精确回读 ─────────────────


def _s(value) -> str:
    return repr(float(value))


# 位级断言只在本机录制环境（numpy 1.26 + 本机 libm/CPU）内保证。
# 跨 CPU / numpy 的 arctan、tan 可能差 1~2 ulp —— 实测 GitHub runner 上
# 同一个动作的 δ 是 0.2448991490545638，录制时是 0.24489914905456378
# （差 2e-17，相对 8e-17）。这不是移植错了，所以默认按容差比，位级只在
# CHASSIS_STRICT_GOLDEN=1 时要求。
_STRICT_GOLDEN = os.environ.get("CHASSIS_STRICT_GOLDEN") == "1"


def _assert_float_matches(got: float, expected: str, ctx: str) -> None:
    """默认容差（rel 1e-12 / abs 1e-15）；严格档要求 repr 位级相同。"""
    if _STRICT_GOLDEN:
        got_s = _s(got)
        assert got_s == expected, (
            f"{ctx}: 位级不一致 got={got_s} want={expected}"
            "（CHASSIS_STRICT_GOLDEN=1 的位级断言只在本机录制环境内保证，"
            "跨 numpy / CPU 不保证）"
        )
        return
    want = float(expected)
    assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-15), (
        f"{ctx}: got={got!r} want={want!r}（rel_tol=1e-12, abs_tol=1e-15）"
    )


def _assert_array_matches(got: np.ndarray, exp: np.ndarray, ctx: str) -> None:
    """float32 数组同理：严格档逐元素位级相等，默认档按 float32 精度容差。"""
    n_diff = int(np.count_nonzero(got != exp))
    if _STRICT_GOLDEN:
        print(f"[对拍] {ctx}: dtype={got.dtype} 逐元素不等个数={n_diff}")
        assert n_diff == 0, f"{ctx} 有 {n_diff} 个元素不等（严格档）"
        return
    max_diff = float(np.max(np.abs(got.astype(np.float64) - exp.astype(np.float64))))
    print(
        f"[对拍] {ctx}: dtype={got.dtype} 逐元素不等个数={n_diff} max|Δ|={max_diff:.3e}"
    )
    assert np.allclose(got, exp, rtol=1e-6, atol=1e-6), (
        f"{ctx} 超出 float32 容差：max|Δ|={max_diff:.3e}（rtol=1e-6, atol=1e-6）"
    )


def _floats(rows) -> np.ndarray:
    return np.array([[float(c) for c in row] for row in rows], dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════
# 生成阶段（需要主仓）
# ═════════════════════════════════════════════════════════════════════


def _record() -> dict:
    sys.path.insert(0, _DRL_SIM)
    from ros_bridge.numpy_bridge import NumpySimBridge

    chassis = AckermannChassis()
    out: dict = {"scenarios": []}
    for scn in _scenarios():
        bridge = NumpySimBridge(env_config=_ENV_YAML)
        bridge.reset(obstacles=scn["obstacles"], path_waypoints=scn["path"])

        rec = {
            "name": scn["name"],
            "obstacles": scn["obstacles"],
            "path": [[_s(c) for c in row] for row in scn["path"]],
            "goal_heading": _s(bridge.goal_heading),
            "actions": [[_s(v) for v in a] for a in scn["actions"]],
            "mapped_delta_v": [],
            "poses": [],
            "lidar_front": [],
            "lidar_rear": [],
            "odometry": [],
            "collision": [],
            "goal_reached": [],
        }

        for action in scn["actions"]:
            # 旧桥 action = 物理量 (δ, v)：映射交给本仓底盘，模拟 env 的动作通路
            delta, v = chassis._map_action(action)
            rec["mapped_delta_v"].append([_s(delta), _s(v)])
            bridge.step(np.array([delta, v], dtype=np.float64))
            rec["poses"].append([_s(bridge._x), _s(bridge._y), _s(bridge._theta)])
            rec["lidar_front"].append([_s(d) for d in bridge.get_lidar_front()])
            rec["lidar_rear"].append([_s(d) for d in bridge.get_lidar_rear()])
            rec["odometry"].append([_s(v_) for v_ in bridge.get_odometry()])
            rec["collision"].append(bool(bridge.check_collision()))
            rec["goal_reached"].append(bool(bridge.check_goal_reached()))

        out["scenarios"].append(rec)

    out["meta"] = {
        "title": "SimpleBridge vs 主仓 NumpySimBridge 全量对拍（A1）",
        "reference": "ros_bridge/numpy_bridge.py（678 行）+ ros_bridge/sensor_provider.py",
        "env_yaml": _ENV_YAML,
        "chassis": "AckermannChassis() 默认参数",
        "chassis_params": {
            "wheelbase": _s(chassis.wheelbase),
            "delta_max": _s(chassis.delta_max),
            "v_min": _s(chassis.v_min),
            "v_max": _s(chassis.v_max),
            "curvature_action": str(bool(chassis.curvature_action)),
            "kappa_max": _s(chassis.kappa_max),
        },
        "float_encoding": "python repr(float) 字符串，float() 精确回读",
        "seed": SEED,
        "action_sampling": (
            f"numpy.random.default_rng({SEED}).uniform(-1, 1, ({N_STEPS_RANDOM}, 2))"
        ),
        "action_bridge": (
            "归一化动作经 AckermannChassis.command() 映射成旧桥认识的物理量 "
            "(δ, v)；旧桥 step() 不做任何映射"
        ),
        "recorder": "PYTHONPATH=src python tests/test_bridges_parity.py",
    }
    out["params"] = {
        "dt": _s(_BRIDGE_PARAMS["dt"]),
        "world_bounds": [_s(v) for v in _BRIDGE_PARAMS["world_bounds"]],
        "max_range": _s(_BRIDGE_PARAMS["max_range"]),
        "domain_rand": str(bool(_BRIDGE_PARAMS["domain_rand"])),
        "goal_dist_threshold": _s(_BRIDGE_PARAMS["goal_dist_threshold"]),
        "goal_end_wp_count": str(_BRIDGE_PARAMS["goal_end_wp_count"]),
        "goal_end_wp_margin": str(_BRIDGE_PARAMS["goal_end_wp_margin"]),
        "collision_margin": _s(_BRIDGE_PARAMS["collision_margin"]),
        "boundary_margin": _s(_BRIDGE_PARAMS["boundary_margin"]),
    }
    return out


def main() -> int:
    if not os.path.isdir(_DRL_SIM):
        print(f"无法录制：{_DRL_SIM} 不存在（golden 只在有主仓的机器上生成）")
        return 1
    data = _record()
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {_GOLDEN_PATH}（{_GOLDEN_PATH.stat().st_size} 字节）")
    return 0


# ═════════════════════════════════════════════════════════════════════
# 断言阶段（只依赖 JSON）
# ═════════════════════════════════════════════════════════════════════


def _golden_params(golden: dict) -> dict:
    p = golden["params"]
    return {
        "dt": float(p["dt"]),
        "world_bounds": tuple(float(v) for v in p["world_bounds"]),
        "max_range": float(p["max_range"]),
        "domain_rand": p["domain_rand"] == "True",
        "goal_dist_threshold": float(p["goal_dist_threshold"]),
        "goal_end_wp_count": int(p["goal_end_wp_count"]),
        "goal_end_wp_margin": int(p["goal_end_wp_margin"]),
        "collision_margin": float(p["collision_margin"]),
        "boundary_margin": float(p["boundary_margin"]),
    }


class _NoLagAckermann(AckermannChassis):
    """反向对照：step 直接等于 command（转向不滞后），用来证明对拍能分辨滞后。"""

    def step(self, action):
        return self.command(action)


def _euler_integrate(chassis, x, y, theta, cmd, dt):
    """改动前的 integrate（2026-11-05 之前）：一阶 Euler，乘加顺序照抄原件。

    bridge_golden.json 是旧桥的全通道记录，而旧桥用的是这条算术；把它钉回去
    才能逐通道对拍，否则 xy 差 3.4e-2 会淹掉真正要找的桥侧回归。
    """
    v, omega, vy = cmd.v, cmd.omega, cmd.vy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return (
        x + (v * cos_t - vy * sin_t) * dt,
        y + (v * sin_t + vy * cos_t) * dt,
        theta + omega * dt,
    )


def _replay(
    golden: dict, chassis_factory=AckermannChassis, *, legacy: bool = False
) -> list:
    if legacy:
        original = Chassis.integrate
        Chassis.integrate = _euler_integrate
        try:
            return _replay_inner(golden, chassis_factory, _golden_params(golden))
        finally:
            Chassis.integrate = original
    return _replay_inner(golden, chassis_factory, _golden_params(golden))


def _replay_inner(golden: dict, chassis_factory, params: dict) -> list:
    results = []
    for rec in golden["scenarios"]:
        chassis = chassis_factory()
        front, rear = chassis.lidar_mounts()
        bridge = SimpleBridge(
            chassis,
            sensors=[
                Lidar("front", mount=front, n_rays=180),
                Lidar("rear", mount=rear, n_rays=180),
            ],
            **params,
        )
        path = _floats(rec["path"])
        bridge.reset(obstacles=rec["obstacles"], path_waypoints=path)

        out = {
            "name": rec["name"],
            "poses": [],
            "lidar_front": [],
            "lidar_rear": [],
            "odometry": [],
            "collision": [],
            "goal_reached": [],
            "goal_heading": float(bridge.goal_heading),
        }
        for action in rec["actions"]:
            bridge.step(np.array([float(v) for v in action], dtype=np.float64))
            out["poses"].append(
                [float(bridge._x), float(bridge._y), float(bridge._theta)]
            )
            reading = bridge.get_sensor_reading()
            out["lidar_front"].append(reading["front"]["scan"])
            out["lidar_rear"].append(reading["rear"]["scan"])
            out["odometry"].append(bridge.get_odometry())
            out["collision"].append(bool(bridge.check_collision()))
            out["goal_reached"].append(bool(bridge.check_goal_reached()))

        out["poses"] = np.array(out["poses"], dtype=np.float64).reshape(-1, 3)
        out["lidar_front"] = np.array(out["lidar_front"], dtype=np.float32)
        out["lidar_rear"] = np.array(out["lidar_rear"], dtype=np.float32)
        out["odometry"] = np.array(out["odometry"], dtype=np.float32)
        results.append(out)
    return results


@pytest.fixture(scope="module")
def golden() -> dict:
    if not _GOLDEN_PATH.exists():
        pytest.skip(f"golden 缺失：{_GOLDEN_PATH}")
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replay(golden: dict) -> list:
    """按旧桥的积分式（Euler）重放：与 golden 逐通道对拍的那一支。"""
    return _replay(golden, legacy=True)


@pytest.fixture(scope="module")
def replay_closed(golden: dict) -> list:
    """生产实现（闭式积分）重放：只用来量它与旧记录的差。"""
    return _replay(golden)


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def test_golden_mapping_consistency(golden: dict) -> None:
    """守住 golden 的语义：录制的 (δ, v) 必须仍是本仓 command() 的映射结果。"""
    chassis = AckermannChassis()
    for rec in golden["scenarios"]:
        for action, expected in zip(rec["actions"], rec["mapped_delta_v"]):
            delta, v = chassis._map_action(np.array([float(x) for x in action]))
            _assert_float_matches(delta, expected[0], f"{rec['name']}.delta")
            _assert_float_matches(v, expected[1], f"{rec['name']}.v")


def test_pose_parity(golden: dict, replay: list) -> None:
    for rec, got in zip(golden["scenarios"], replay):
        exp = _floats(rec["poses"])
        d_xy = float(np.max(np.abs(got["poses"][:, :2] - exp[:, :2])))
        d_th = float(np.max(np.abs(_wrap(got["poses"][:, 2] - exp[:, 2]))))
        print(f"[对拍] {rec['name']}: max|Δx,Δy|={d_xy:.3e}  max|Δθ|={d_th:.3e}")
        assert d_xy <= TOL_POSE, f"{rec['name']}: max|Δx,Δy|={d_xy}"
        assert d_th <= TOL_POSE, f"{rec['name']}: max|Δθ|={d_th}"


def test_lidar_parity(golden: dict, replay: list) -> None:
    for rec, got in zip(golden["scenarios"], replay):
        for key in ("lidar_front", "lidar_rear"):
            exp = np.array(
                [[float(v) for v in row] for row in rec[key]], dtype=np.float64
            ).astype(np.float32)
            got_arr = got[key]
            assert got_arr.dtype == np.float32, key
            assert got_arr.shape == exp.shape, (key, got_arr.shape, exp.shape)
            _assert_array_matches(got_arr, exp, f"{rec['name']}.{key}")


def test_odometry_parity(golden: dict, replay: list) -> None:
    for rec, got in zip(golden["scenarios"], replay):
        exp = np.array(
            [[float(v) for v in row] for row in rec["odometry"]], dtype=np.float64
        ).astype(np.float32)
        got_arr = got["odometry"]
        assert got_arr.dtype == np.float32
        _assert_array_matches(got_arr, exp, f"{rec['name']}.odometry")


def test_boolean_parity(golden: dict, replay: list) -> None:
    seen = {"collision": set(), "goal_reached": set()}
    for rec, got in zip(golden["scenarios"], replay):
        for key in ("collision", "goal_reached"):
            exp = [bool(v) for v in rec[key]]
            assert got[key] == exp, f"{rec['name']}.{key} 布尔序列不等"
            seen[key].update(exp)
    # 防空转：两个布尔量都必须两边都出现过
    for key, values in seen.items():
        assert values == {True, False}, f"{key} 只出现过 {values}，断言会空转"


def test_goal_heading_parity(golden: dict, replay: list) -> None:
    for rec, got in zip(golden["scenarios"], replay):
        _assert_float_matches(
            got["goal_heading"], _s(rec["goal_heading"]), f"{rec['name']}.goal_heading"
        )


def test_closed_form_differs_only_in_xy(golden: dict, replay_closed: list) -> None:
    """生产闭式积分 vs 旧桥记录：θ 逐位相同，xy 的差只来自积分式。

    反向锁定：若有人把 integrate 改回 Euler，d_xy 会塌到 0，这条就红；
    若有人把闭式实现写错（例如 ω≠0 仍走 Euler），同样红。
    """
    for rec, got in zip(golden["scenarios"], replay_closed):
        exp = _floats(rec["poses"])
        d_xy = float(np.max(np.abs(got["poses"][:, :2] - exp[:, :2])))
        d_th = float(np.max(np.abs(_wrap(got["poses"][:, 2] - exp[:, 2]))))
        print(
            f"[闭式 vs 旧记录] {rec['name']}: max|Δx,Δy|={d_xy:.3e}  max|Δθ|={d_th:.3e}"
        )
        # θ 只承诺"同一公式 + ulp 级噪声"：ω 是 np.arctan(clip(κ)·L) 再经
        # np.tan 得来的（src/chassis_sim/chassis/ackermann.py:71 与 :88），
        # 两个超越函数的 libm 实现跨平台/版本差 1 ulp，240 步累积后实测
        # 4.44e-16（Windows）/ 2.22e-15（macOS）/ 2.66e-15（Linux py3.10-3.11）
        # —— CI run 41901753996。判据是数量级：改成 atan2(dy, dx) 之类"看起来
        # 等价"的写法，差会从 1e-15 跳到 1e-3 以上。
        assert d_th <= TOL_THETA, f"{rec['name']}: θ 通道被积分式改动了 {d_th}"
        if np.any(_wrap(np.diff(exp[:, 2])) != 0.0):
            # 转过向 ⇒ 闭式的 A/B 项与 Euler 不同，差必须看得见
            assert d_xy > TOL_POSE, (
                f"{rec['name']}: 闭式与 Euler 无差（{d_xy}）——积分式退化了"
            )
            if rec["name"] == "random_240":
                assert 3.0e-2 <= d_xy <= 3.7e-2, (
                    f"random_240 的差 {d_xy:.3e} 超出记录区间"
                )
        else:
            # ω≡0 的直线场景 ⇒ 闭式走退化分支，必须与旧 Euler 逐位相同
            assert d_xy == 0.0, f"{rec['name']}: ω=0 分支不再逐位相同（{d_xy}）"


def test_reverse_control_detects_lag(golden: dict) -> None:
    """反向对照：无滞后同款底盘 ⇒ 偏差必须显著（> 1e-3 m）。"""
    no_lag = {r["name"]: r for r in _replay(golden, _NoLagAckermann, legacy=True)}
    rec = golden["scenarios"][0]
    exp = _floats(rec["poses"])
    got = no_lag[rec["name"]]["poses"]
    dev = float(np.max(np.hypot(got[:, 0] - exp[:, 0], got[:, 1] - exp[:, 1])))
    print(f"[反向对照] {rec['name']} 无滞后 vs 旧桥: max 位移偏差={dev:.6f} m")
    assert dev > TOL_REVERSE, f"反向对照偏差仅 {dev}，对拍分辨不出滞后"


if __name__ == "__main__":
    raise SystemExit(main())
