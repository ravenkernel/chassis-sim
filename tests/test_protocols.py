"""Protocols：@runtime_checkable 的 isinstance 正反例 + 加载不拉重依赖。

正例用**真桥**（SimpleBridge + AckermannChassis）—— 协议只有对真实现成立才算数；
反例是缺成员的哑对象：runtime_checkable 只检查成员**存在**，不看签名，所以"少一个
成员"是唯一可靠的否定判据（拿签名不符的对象做反例只会证明协议不管签名）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.protocols import Bridge, SceneSource, SensorBridge

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAVY_MODULES: tuple[str, ...] = (
    "imageio",
    "matplotlib",
    "numba",
    "scipy",
    "shapely",
    "torch",
)

#: 干净子进程探针：protocols 只依赖 numpy / typing
_PROBE = """\
import sys
import chassis_sim
import chassis_sim.protocols  # noqa: F401

bad = [m for m in {heavy!r} if m in sys.modules]
print("heavy =", bad)
assert not bad, bad
"""


class _NoPoseBridge:
    """除 pose() 之外成员齐全：证明协议检查的是成员存在性，不是具体类型。"""

    sim_time = 0.0
    max_range = 10.0

    def step(self, action: Any) -> None:
        return None

    def reset(self, **kwargs: Any) -> None:
        return None

    def get_odometry(self) -> np.ndarray:
        return np.zeros(5)

    def obstacle_shapes(self) -> list[dict[str, Any]]:
        return []


class _SceneOnly:
    """只有场景三件套（pose / obstacle_shapes / world_bounds）的只读来源。"""

    world_bounds = (0.0, 10.0, 0.0, 10.0)

    def pose(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def obstacle_shapes(self) -> list[dict[str, Any]]:
        return []


class _NoWorldBounds:
    """有 pose / obstacle_shapes，但没有 world_bounds：不是 SceneSource。"""

    def pose(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def obstacle_shapes(self) -> list[dict[str, Any]]:
        return []


def test_simple_bridge_satisfies_bridge_and_sensor_bridge() -> None:
    bridge = SimpleBridge(AckermannChassis())

    print(
        "SimpleBridge -> Bridge:",
        isinstance(bridge, Bridge),
        "| SensorBridge:",
        isinstance(bridge, SensorBridge),
    )
    assert isinstance(bridge, Bridge)
    assert isinstance(bridge, SensorBridge)


def test_object_missing_a_member_is_not_a_bridge() -> None:
    dummy = _NoPoseBridge()

    print("缺 pose() 的哑对象 -> Bridge:", isinstance(dummy, Bridge))
    assert not isinstance(dummy, Bridge)


def test_scene_source_needs_world_bounds() -> None:
    only_scene = _SceneOnly()
    no_bounds = _NoWorldBounds()

    print(
        "场景三件套 -> SceneSource:",
        isinstance(only_scene, SceneSource),
        "| 缺 world_bounds:",
        isinstance(no_bounds, SceneSource),
    )
    assert isinstance(only_scene, SceneSource)
    assert not isinstance(no_bounds, SceneSource)


def test_importing_protocols_pulls_no_heavy_dependency() -> None:
    script = _PROBE.format(heavy=list(HEAVY_MODULES))
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "heavy = []" in proc.stdout, proc.stdout
