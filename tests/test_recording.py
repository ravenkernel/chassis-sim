"""顶层 record / snapshot：真跑一个 40 步回合合成 GIF、缺依赖报错、import 不拉重模块。

env 本体在 chassis_sim.envs（模块级 import 会让本文件被别人的重构挡住），
所以 env 一律在测试函数里现造。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from chassis_sim.recording import RecordingResult, record, snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAVY_MODULES: tuple[str, ...] = ("irsim", "matplotlib", "shapely", "scipy", "torch")

#: 子进程探针：import 本模块后重依赖一个都不许在 sys.modules 里
_PROBE = """\
import os
import sys
import {target}

bad = [m for m in {heavy!r} if m in sys.modules]
print("pid =", os.getpid())
print("heavy =", bad)
assert not bad, bad
"""

#: 40 步 / every=2 ⇒ 20 帧；内建直线场景 8 m，40 步最多走 4.8 m ⇒ 走不到终点、无碰撞
STEPS = 40
EVERY = 2


def _env(max_steps: int = STEPS):
    from chassis_sim.envs import ChassisSimEnv

    return ChassisSimEnv("ackermann", max_steps=max_steps)


def _controller(env):
    from chassis_sim.controllers import make_controller

    return make_controller("pp", chassis=env.unwrapped.chassis)


def test_import_recording_pulls_no_heavy_dependency() -> None:
    script = _PROBE.format(target="chassis_sim.recording", heavy=list(HEAVY_MODULES))
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
    assert f"pid = {os.getpid()}" not in proc.stdout  # 必须跑在干净子进程里


def test_record_writes_a_real_gif(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image = pytest.importorskip("PIL.Image")

    env = _env()
    try:
        result = record(
            env,
            _controller(env),
            out=tmp_path / "demo.gif",
            every=EVERY,
            fps=8,
            dpi=40.0,
            save_frames=True,
        )
    finally:
        env.close()

    assert isinstance(result, RecordingResult)
    gif = Path(result.path)
    assert gif == tmp_path / "demo.gif"
    assert gif.is_file()
    assert gif.stat().st_size > 0
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    assert result.steps == STEPS
    assert result.terminated is False
    assert result.truncated is True  # 触到 env 自己的 max_steps
    assert result.frames == STEPS // EVERY
    assert result.total_reward != 0.0

    with image.open(gif) as gif_image:
        assert getattr(gif_image, "n_frames", 1) == result.frames
        assert gif_image.size[0] > 0 and gif_image.size[1] > 0

    kept = sorted((tmp_path / "demo_frames").glob("*.png"))
    assert len(kept) == result.frames
    assert kept[-1].name == f"frame_{STEPS:05d}.png"


def test_record_drops_frames_when_not_saved(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")

    env = _env(max_steps=6)
    try:
        result = record(
            env, _controller(env), out=tmp_path / "short.gif", every=2, dpi=32.0
        )
    finally:
        env.close()

    assert result.steps == 6
    assert result.frames == 3
    assert Path(result.path).suffix == ".gif"
    assert not (tmp_path / "short_frames").exists()


def test_record_adds_gif_suffix(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")

    env = _env(max_steps=2)
    try:
        result = record(
            env, _controller(env), out=tmp_path / "noext", every=1, dpi=32.0
        )
    finally:
        env.close()

    assert Path(result.path) == tmp_path / "noext.gif"
    assert Path(result.path).is_file()


def test_record_requires_controller(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="controller"):
        record(_env(max_steps=2), out=tmp_path / "x.gif")


def test_record_rejects_bad_every_and_fps(tmp_path: Path) -> None:
    env = _env(max_steps=2)
    try:
        with pytest.raises(ValueError, match="every"):
            record(env, _controller(env), out=tmp_path / "x.gif", every=0)
        with pytest.raises(ValueError, match="fps"):
            record(env, _controller(env), out=tmp_path / "x.gif", fps=0)
    finally:
        env.close()


def test_record_without_imageio_raises_with_pip_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺 imageio 必须抛 ImportError（不许静默降级成帧目录），消息带 pip 提示。"""
    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "imageio.v2", None)

    env = _env(max_steps=2)
    try:
        with pytest.raises(ImportError) as excinfo:
            record(env, _controller(env), out=tmp_path / "x.gif", every=1, dpi=32.0)
    finally:
        env.close()

    message = str(excinfo.value)
    assert "imageio" in message
    assert 'pip install "chassis-sim[viz]"' in message
    assert not (tmp_path / "x.gif").exists()


def test_snapshot_writes_png_for_env_and_bridge(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    env = _env(max_steps=4)
    try:
        from_env = snapshot(env, tmp_path / "env.png", dpi=32.0)
        from_bridge = snapshot(env.bridge, tmp_path / "bridge.png", dpi=32.0)
    finally:
        env.close()

    assert from_env == str(env_path := tmp_path / "env.png")
    assert Path(from_env).is_file() and Path(from_env).stat().st_size > 0
    assert Path(from_env).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Path(from_bridge).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert env_path.is_file()


def test_snapshot_without_matplotlib_raises_with_pip_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    env = _env(max_steps=2)
    try:
        with pytest.raises(ImportError) as excinfo:
            snapshot(env, tmp_path / "x.png")
    finally:
        env.close()

    message = str(excinfo.value)
    assert "matplotlib" in message
    assert 'pip install "chassis-sim[viz]"' in message
