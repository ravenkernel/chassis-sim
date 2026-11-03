"""阿克曼 golden 对拍：本仓 chassis 与主仓 NumpySimBridge 的位姿轨迹逐位一致。

两件事：

(i) **生成阶段**（``/home/zhao/drl-sim`` 存在时）——
    ``sys.path.insert(0, "/home/zhao/drl-sim")`` 后 import
    ``ros_bridge.numpy_bridge.NumpySimBridge``，用固定 seed 的随机 action 驱动它，
    把 action / 物理 (δ, v) / 位姿全精度录进 ``tests/data/ackermann_golden.json``。
    物理 (δ, v) 由**旧实现自己的映射块**产生：直接从
    ``/home/zhao/drl-sim/envs/ackermann_env.py`` 抽取第 820~833 行源码块 ``exec``
    （抽取失败才退回本文件内的逐字副本，并在 meta 里注明）。
    重录命令：``PYTHONPATH=src python tests/test_chassis_ackermann_parity.py``

(ii) **断言阶段** —— 只依赖那份 JSON：同一串 action 重放本仓
    ``AckermannChassis.command()`` + ``Chassis.integrate()``，逐步比较位姿。

两处语义差异（已按旧桥对齐，不是放宽）：
* **转向延迟**：旧桥 ``step()`` 用**上一帧**转角积分（``numpy_bridge.py:379-383``），
  ``self._omega = self._v / self._wheelbase * np.tan(prev_steer)``。本仓 ``Chassis``
  是无状态映射，故重放时把转向分量滞后一步送进 ``command()``（速度分量取当步），
  与旧桥的 1 步延迟逐位等价。
* **θ 归一化**：旧桥每步把 θ 折到 [-π, π]（``numpy_bridge.py:390``）；统一积分核按
  规格不做归一，故重放时显式归一，并且 θ 的误差按角度差比较。

容差 1e-12：旧桥全程 float64（numpy 标量，无 float32 中转），实测见测试打印。
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chassis_sim.chassis import AckermannChassis

_DRL_SIM = "/home/zhao/drl-sim"
_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "ackermann_golden.json"
_ACKERMANN_ENV = os.path.join(_DRL_SIM, "envs", "ackermann_env.py")

N_STEPS = 240
SEED = 20261104
TOL = 1e-12

# 与主仓 config/train.yaml 的 env 块 + config/env.yaml 的 robot.shape 同值
_PARAMS = {
    "wheelbase": 0.145,
    "delta_max": 0.526,
    "v_min": -1.0,
    "v_max": 1.0,
    "curvature_action": True,
    "kappa_max": 4.0,
}

# 旧实现映射块的起止标记（ackermann_env.py:820 与 :835）
_MAP_START = "# 将 tianshou 的 tanh 输出"
_MAP_END = "# 加速度限制"


def _chassis() -> AckermannChassis:
    return AckermannChassis(**_PARAMS)


def _actions() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.uniform(-1.0, 1.0, size=(N_STEPS, 2))


def _ang_diff(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _wrap(theta: float) -> float:
    """与 numpy_bridge.py:390 同式的 θ 归一化。"""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


# ── 旧实现的动作映射：源码抽取 ───────────────────────────────────────


def _old_mapping_from_source():
    """从 drl-sim 源码抽出映射块并 exec 成可调用对象；不可得返回 None。"""
    try:
        lines = Path(_ACKERMANN_ENV).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    start = next((i for i, line in enumerate(lines) if _MAP_START in line), None)
    end = next(
        (
            i
            for i, line in enumerate(lines)
            if _MAP_END in line and (start is None or i > start)
        ),
        None,
    )
    if start is None or end is None:
        return None
    block = textwrap.dedent("\n".join(lines[start:end])).rstrip()
    src = (
        "def _map(self, action, np):\n"
        + textwrap.indent(block, "    ")
        + "\n    return mapped_action\n"
    )
    namespace: dict = {}
    exec(src, namespace)  # noqa: S102 — 只读地执行旧仓源码块，用于对拍
    return namespace["_map"]


def _old_mapping(action, ns) -> tuple[float, float]:
    """drl-sim/envs/ackermann_env.py:822-833 的逐字副本（源码抽取不可用时的备份）。"""
    a = np.asarray(action, dtype=np.float64)
    mapped_action = a.copy()
    if ns.curvature_action:
        kappa = a[0] * ns.kappa_max
        mapped_action[0] = np.arctan(
            np.clip(kappa, -ns.kappa_max, ns.kappa_max) * ns.wheelbase
        )
    else:
        mapped_action[0] = a[0] * ns.delta_max

    if a[1] >= 0:
        mapped_action[1] = a[1] * ns.v_max  # [0, +1] → [0, v_max]
    else:
        mapped_action[1] = a[1] * (-ns.v_min)  # [-1, 0] → [v_min, 0]
    return float(mapped_action[0]), float(mapped_action[1])


def _mapping_ns() -> SimpleNamespace:
    ch = _chassis()
    return SimpleNamespace(
        wheelbase=ch.wheelbase,
        delta_max=ch.delta_max,
        v_min=ch.v_min,
        v_max=ch.v_max,
        curvature_action=ch.curvature_action,
        kappa_max=ch.kappa_max,
    )


def _mapped_actions(actions: np.ndarray) -> tuple[list, str]:
    """(δ, v) 序列 + 来源说明。"""
    ns = _mapping_ns()
    fn = _old_mapping_from_source()
    if fn is not None:
        pairs = [tuple(float(x) for x in fn(ns, a, np)) for a in actions]
        return pairs, "exec(" + _ACKERMANN_ENV + ":820-833)"
    return [_old_mapping(a, ns) for a in actions], "逐字副本（drl-sim 源码不可用）"


# ── 旧桥驱动 ─────────────────────────────────────────────────────────


def _import_bridge_cls():
    if _DRL_SIM not in sys.path:
        sys.path.insert(0, _DRL_SIM)
    from ros_bridge.numpy_bridge import NumpySimBridge

    return NumpySimBridge


def _new_bridge():
    cls = _import_bridge_cls()
    return cls(env_config=os.path.join(_DRL_SIM, "config", "env.yaml"))


def _bridge_poses(pairs) -> tuple[list, float]:
    bridge = _new_bridge()
    assert float(bridge._wheelbase) == _PARAMS["wheelbase"], (
        "旧桥 wheelbase="
        + repr(bridge._wheelbase)
        + " 与对拍参数 "
        + repr(_PARAMS["wheelbase"])
        + " 不一致"
    )
    dt = float(bridge.dt)
    poses = [(bridge._x, bridge._y, bridge._theta)]
    for delta, v in pairs:
        bridge.step(np.array([delta, v], dtype=np.float64))
        poses.append((bridge._x, bridge._y, bridge._theta))
    return poses, dt


# ── 本仓重放 ─────────────────────────────────────────────────────────


def _replay(actions: np.ndarray, dt: float) -> list:
    ch = _chassis()
    x = y = theta = 0.0
    poses = [(x, y, theta)]
    prev_a0 = 0.0  # 旧桥初始 self._steer = 0.0（numpy_bridge.py:228）
    for a in actions:
        cmd = ch.command(np.array([prev_a0, float(a[1])], dtype=np.float64))
        x, y, theta = ch.integrate(x, y, theta, cmd, dt)
        theta = _wrap(theta)
        prev_a0 = float(a[0])
        poses.append((x, y, theta))
    return poses


# ── JSON 读写 ────────────────────────────────────────────────────────


def _generate() -> dict:
    actions = _actions()
    pairs, mapping_source = _mapped_actions(actions)
    poses, dt = _bridge_poses(pairs)
    ch = _chassis()
    params = {
        k: (str(v) if isinstance(v, bool) else repr(float(v)))
        for k, v in _PARAMS.items()
    }
    return {
        "meta": {
            "what": "阿克曼 bit 级对拍 golden：旧 NumpySimBridge 的位姿轨迹",
            "source_bridge": os.path.join(_DRL_SIM, "ros_bridge", "numpy_bridge.py"),
            "source_env_yaml": os.path.join(_DRL_SIM, "config", "env.yaml"),
            "mapping_source": mapping_source,
            "bridge_dt": repr(dt),
            "bridge_float_dtype": "float64（numpy 标量，无 float32 中转）",
            "steer_delay": "旧桥 ω/θ 用上一帧转角（numpy_bridge.py:379-383）；重放时转向分量滞后一步",
            "theta_wrap": "旧桥每步折到 [-π,π]（numpy_bridge.py:390）；重放时同步归一",
            "float_encoding": "python repr(float) 字符串，float() 精确回读",
            "n_steps": N_STEPS,
            "seed": SEED,
            "action_sampling": "numpy.random.default_rng(seed).uniform(-1, 1, (n_steps, 2))",
            "action_convention": "归一化原始动作（旧 env.action_space = Box(2,[-1,-1],[1,1])）",
            "params": params,
            "chassis_describe": ch.describe(),
        },
        "actions": [[repr(float(v)) for v in a] for a in actions],
        "mapped_delta_v": [[repr(delta), repr(v)] for delta, v in pairs],
        "poses": [[repr(float(c)) for c in p] for p in poses],
    }


def _load() -> dict:
    if not _GOLDEN_PATH.is_file():
        raise FileNotFoundError(
            "缺少 golden："
            + str(_GOLDEN_PATH)
            + "；用 PYTHONPATH=src python tests/test_chassis_ackermann_parity.py 重录（需 "
            + _DRL_SIM
            + "）"
        )
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


def _floats(rows) -> np.ndarray:
    return np.array([[float(s) for s in row] for row in rows], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════
# 断言分支：只读 JSON，永不 skip
# ══════════════════════════════════════════════════════════════════════


def test_ackermann_parity_against_recorded_golden():
    data = _load()
    meta = data["meta"]
    actions = _floats(data["actions"])
    expected = _floats(data["poses"])
    dt = float(meta["bridge_dt"])

    assert len(actions) >= 200, "golden 步数 " + str(len(actions)) + " < 200"
    assert expected.shape == (len(actions) + 1, 3)

    # golden 记录的参数就是本测试构造的底盘参数
    ch = _chassis()
    expected_params = {
        k: (str(v) if isinstance(v, bool) else repr(float(v)))
        for k, v in ch._params().items()
    }
    assert meta["params"] == expected_params, (
        "golden 参数 "
        + repr(meta["params"])
        + " 与当前底盘 "
        + repr(expected_params)
        + " 不一致"
    )

    got = np.array(_replay(actions, dt), dtype=np.float64)
    assert got.shape == expected.shape

    err_xy = float(np.abs(got[:, :2] - expected[:, :2]).max())
    err_theta = float(np.abs(_ang_diff(got[:, 2], expected[:, 2])).max())
    print(
        "\n[parity] steps="
        + str(len(actions))
        + " dt="
        + repr(dt)
        + " max|Δx,Δy|="
        + format(err_xy, ".3e")
        + " max|Δθ|="
        + format(err_theta, ".3e")
        + " tol="
        + format(TOL, ".0e")
    )
    assert err_xy <= TOL, (
        "xy 最大绝对误差 " + format(err_xy, ".3e") + " > " + format(TOL, ".0e")
    )
    assert err_theta <= TOL, (
        "θ 最大角度误差 " + format(err_theta, ".3e") + " > " + format(TOL, ".0e")
    )


# ══════════════════════════════════════════════════════════════════════
# 重录分支：旧仓库不存在时 skip
# ══════════════════════════════════════════════════════════════════════

_skip_no_drl = pytest.mark.skipif(
    not os.path.isdir(_DRL_SIM), reason=_DRL_SIM + " 不存在：跳过重录/源码对拍分支"
)


@_skip_no_drl
def test_recorded_golden_is_live_old_bridge_output():
    """现场重放旧桥，证明 JSON 里的位姿就是旧实现的真实输出。"""
    data = _load()
    actions = _floats(data["actions"])
    expected = _floats(data["poses"])

    pairs, _ = _mapped_actions(actions)
    recorded_pairs = _floats(data["mapped_delta_v"])
    np.testing.assert_array_equal(np.array(pairs, dtype=np.float64), recorded_pairs)

    poses, dt = _bridge_poses(pairs)
    live = np.array(poses, dtype=np.float64)
    assert dt == float(data["meta"]["bridge_dt"])
    err = float(np.abs(_ang_diff(live[:, 2], expected[:, 2])).max())
    err_xy = float(np.abs(live[:, :2] - expected[:, :2]).max())
    print(
        "\n[live] max|Δx,Δy|="
        + format(err_xy, ".3e")
        + " max|Δθ|="
        + format(err, ".3e")
    )
    assert err_xy <= TOL and err <= TOL


@_skip_no_drl
def test_mapping_and_omega_match_old_source():
    """本仓 command() 的 (v, ω) 与旧源码映射块 + 旧 ω 公式逐位一致。"""
    fn = _old_mapping_from_source()
    if fn is None:
        pytest.skip("无法从 " + _ACKERMANN_ENV + " 抽取映射块（标记行变动）")

    ns = _mapping_ns()
    ch = _chassis()
    actions = _actions()
    worst_v = worst_w = 0.0
    for a in actions:
        delta, v = (float(x) for x in fn(ns, a, np))
        cmd = ch.command(a)
        worst_v = max(worst_v, abs(cmd.v - v))
        omega_src = 0.0 if v == 0.0 else v / ns.wheelbase * np.tan(delta)
        worst_w = max(worst_w, abs(cmd.omega - omega_src))
        assert cmd.vy == 0.0
    print(
        "\n[mapping] max|Δv|="
        + format(worst_v, ".3e")
        + " max|Δω|="
        + format(worst_w, ".3e")
    )
    assert worst_v == 0.0, "v 映射与旧源码不一致：最大偏差 " + format(worst_v, ".3e")
    assert worst_w == 0.0, "ω 与旧源码公式不一致：最大偏差 " + format(worst_w, ".3e")


# ══════════════════════════════════════════════════════════════════════
# 生成入口
# ══════════════════════════════════════════════════════════════════════


def main() -> int:
    if not os.path.isdir(_DRL_SIM):
        print("缺少 " + _DRL_SIM + "，无法重录 golden", file=sys.stderr)
        return 1
    data = _generate()
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        "wrote "
        + str(_GOLDEN_PATH)
        + " steps="
        + str(len(data["actions"]))
        + " dt="
        + data["meta"]["bridge_dt"]
        + " mapping="
        + data["meta"]["mapping_source"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
