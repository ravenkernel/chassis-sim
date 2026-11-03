"""import 纯净性：只 import 本包时不得拉起可选重依赖（在**子进程**里验证）。

父进程（pytest）的 sys.modules 早就被别的测试污染了——render 测试会 import
matplotlib、planning 测试会 import numba——所以"事后检查父进程 sys.modules"
恒为假阴性/假阳性，验证必须在干净解释器里做。

子进程刻意去掉 PYTHONPATH，验证的是**安装态**（pip install -e .）的 import 图，
与验收命令逐字一致::

    env -u PYTHONPATH python -c "import chassis_sim, sys; \
        bad=[m for m in ('irsim','matplotlib','shapely','scipy','torch') \
             if m in sys.modules]; assert not bad, bad"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 这些包属于 optional extra（irsim -> ir-sim + matplotlib、viz -> matplotlib、
# planning 的 scipy/numba 系）或根本不该出现在核心包里（torch、shapely）。
HEAVY_MODULES: tuple[str, ...] = ("irsim", "matplotlib", "shapely", "scipy", "torch")

_PROBE = """\
import os
import sys
import {target}

bad = [m for m in {heavy!r} if m in sys.modules]
print("pid =", os.getpid())
print("n_modules =", len(sys.modules))
print("heavy =", bad)
assert not bad, bad
"""


def _probe_import(target: str) -> tuple[int, int, str]:
    """在无 PYTHONPATH 的子进程里 import target，返回 (pid, 模块数, stdout)。"""
    script = _PROBE.format(target=target, heavy=list(HEAVY_MODULES))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"子进程 import {target} 失败：\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    parsed: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" = ")
        if value:
            parsed[key.strip()] = value.strip()
    assert parsed.get("heavy") == "[]", proc.stdout
    return int(parsed["pid"]), int(parsed["n_modules"]), proc.stdout


def test_import_chassis_sim_pulls_no_heavy_dependency() -> None:
    pid, n_modules, stdout = _probe_import("chassis_sim")
    assert pid != os.getpid(), f"探针没跑在子进程里，父进程已污染：{stdout}"
    assert n_modules > 0


def test_import_bridges_pulls_no_heavy_dependency() -> None:
    pid, _, _ = _probe_import("chassis_sim.bridges")
    assert pid != os.getpid()


def test_import_chassis_registry_pulls_no_heavy_dependency() -> None:
    pid, _, _ = _probe_import("chassis_sim.chassis.registry")
    assert pid != os.getpid()
