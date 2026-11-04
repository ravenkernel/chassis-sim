"""path_extractor 的 import 必须自洽：干净解释器里能 import，也能真跑一次 extract()。

病（W4b 上报，本会话复核）：path_extractor 模块 曾是主仓
时代的绝对 import `from envs.path_utils import _arc_length`，每次 extract() 都执行；
干净安装（无 PYTHONPATH、sys.path 里没有 drl-sim）下 ⇒ ModuleNotFoundError。

chassis_env.py 曾用 `sys.modules.setdefault("envs.path_utils", ...)` 兜住它 ——
那种兜底已删除：它把根因藏进 import 顺序（先 import chassis_env 才不炸）。
本文件在**子进程**里独立验证 path_extractor，并静态检查两处都不会回潮。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = REPO_ROOT / "src" / "chassis_sim" / "envs"
# 1.0.0 起 PathExtractor 住在稳定公共路径 chassis_sim.paths（原 envs/path_extractor.py）
PATH_EXTRACTOR = REPO_ROOT / "src" / "chassis_sim" / "paths.py"

_PROBE = """\
import numpy as np
from chassis_sim.paths import PathExtractor

path = np.column_stack([
    np.linspace(0.0, 5.0, 60),
    np.zeros(60),
    np.zeros(60),
])
extractor = PathExtractor(lookahead=4.0, target_count=30, power=1.0)
out = extractor.extract(path, np.array([0.0, 0.0, 0.0]), velocity=0.5)
print("shape =", out.shape)
print("valid =", extractor.last_n_valid)
"""


def test_path_extractor_imports_and_extracts_in_clean_interpreter() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"子进程 import/extract 失败：\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "shape = (30, 3)" in proc.stdout, proc.stdout


def test_path_extractor_has_no_absolute_envs_import() -> None:
    text = PATH_EXTRACTOR.read_text(encoding="utf-8")
    bad = [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(text.splitlines(), 1)
        if line.strip().startswith(("from envs.", "import envs"))
    ]
    assert bad == [], bad


def test_envs_package_registers_no_sys_modules_alias() -> None:
    hits = [
        f"{path.name}:{lineno}: {line.strip()}"
        for path in sorted(ENVS_DIR.glob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "sys.modules" in line
    ]
    assert hits == [], hits
