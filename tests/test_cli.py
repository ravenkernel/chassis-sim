"""CLI 端到端：起子进程跑 `python -m chassis_sim ...`（不直接调 main）。

子进程刻意删掉 PYTHONPATH —— 验收命令本身就在 `env -u PYTHONPATH` 下跑，所以这里
验证的是**安装态**入口（`pip install -e .` 之后 console script 与 `python -m` 共用
同一份 cli.main）。

耗时：只有 record 用例真的画图（12 步 / every=4 ⇒ 3 帧，dpi 用 CLI 默认 =
figsize 6x6 @100dpi），其余用例只 list / --help / 报错，秒级。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAVY_MODULES: tuple[str, ...] = (
    "imageio",
    "matplotlib",
    "numba",
    "shapely",
    "torch",
)
VIZ_HINT = 'pip install "chassis-sim[viz]"'

#: 遮住可视化依赖的 sitecustomize：find_spec 抛 ImportError = 真的 import 不到
_SITECUSTOMIZE = """\
import sys

_BLOCKED = {"matplotlib", "imageio"}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ImportError(f"blocked for test: {name}")
        return None


sys.meta_path.insert(0, _Blocker())
"""

#: 干净子进程探针：import cli 不许把 env 本体或重依赖拉进来（--help 必须轻）
_PROBE = """\
import sys
import chassis_sim.cli  # noqa: F401

bad = [m for m in {heavy!r} if m in sys.modules]
print("heavy =", bad)
print("envs =", "chassis_sim.envs" in sys.modules)
assert not bad, bad
"""


def _env(**overrides: str) -> dict[str, str]:
    """复刻验收命令的环境：PYTHONPATH 一律删掉（验证安装态）。"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(overrides)
    return env


def _run(
    *args: str, env: dict[str, str] | None = None, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chassis_sim", *args],
        cwd=str(REPO_ROOT),
        env=_env() if env is None else env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _blocked_env(tmp_path: Path) -> dict[str, str]:
    """PYTHONPATH 指向一个遮住 matplotlib / imageio 的 sitecustomize。"""
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return _env(PYTHONPATH=str(blocker))


def test_list_scenarios_prints_one_name_per_line() -> None:
    proc = _run("list-scenarios")

    assert proc.returncode == 0, proc.stderr
    names = proc.stdout.split()
    assert "obstacles_dense" in names
    assert "circle" in names


def test_unknown_scenario_exits_two_and_lists_available() -> None:
    proc = _run("record", "--scenario", "not-a-scenario", "--out", "unused.gif")

    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "not-a-scenario" in proc.stderr
    assert "obstacles_dense" in proc.stderr  # 可用值就在消息里
    assert "Traceback" not in proc.stderr


def test_help_exits_zero() -> None:
    proc = _run("--help")

    assert proc.returncode == 0, proc.stderr
    for name in ("list-scenarios", "record", "eval", "render"):
        assert name in proc.stdout


def test_help_works_without_matplotlib(tmp_path: Path) -> None:
    proc = _run("--help", env=_blocked_env(tmp_path))

    assert proc.returncode == 0, proc.stderr
    assert "list-scenarios" in proc.stdout


def test_importing_cli_pulls_no_heavy_dependency() -> None:
    script = _PROBE.format(heavy=list(HEAVY_MODULES))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "heavy = []" in proc.stdout, proc.stdout
    assert "envs = False" in proc.stdout, proc.stdout


def test_record_help_shows_safe_defaults() -> None:
    """省略 --steps/--every 时的默认值必须写在 --help 里（1000 帧不是能给的默认）。"""
    proc = _run("record", "--help")

    assert proc.returncode == 0, proc.stderr
    assert "200 步" in proc.stdout, proc.stdout
    assert "4 步一帧" in proc.stdout, proc.stdout


def test_record_writes_a_real_gif(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image = pytest.importorskip("PIL.Image")
    out = tmp_path / "x.gif"

    proc = _run("record", "--steps", "12", "--every", "4", "--out", str(out))

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert out.is_file() and out.stat().st_size > 0
    assert out.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    with image.open(out) as gif:
        assert getattr(gif, "n_frames", 1) >= 2
    assert f"path: {out}" in proc.stdout
    assert "frames:" in proc.stdout
    assert "total_reward:" in proc.stdout
    steps = re.search(r"^steps: (\d+)$", proc.stdout, flags=re.MULTILINE)
    assert steps is not None, proc.stdout
    assert 1 <= int(steps.group(1)) <= 12


def test_record_without_matplotlib_exits_two_with_pip_hint(tmp_path: Path) -> None:
    out = tmp_path / "blocked.gif"

    proc = _run(
        "record",
        "--steps",
        "2",
        "--every",
        "1",
        "--out",
        str(out),
        env=_blocked_env(tmp_path),
    )

    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert VIZ_HINT in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out.exists()
