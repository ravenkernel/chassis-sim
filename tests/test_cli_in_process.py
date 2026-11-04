"""CLI 的**进程内**单元覆盖：直接调 `chassis_sim.cli.main`。

姊妹文件 `test_cli.py` 用子进程验证安装态真实入口（`python -m chassis_sim`），
这很好，但子进程的行不进当前进程的覆盖率 —— `cli.py` 全文件 0% 会把总量拖到
门槛以下。本文件在**当前进程**里把 `main()` 的各分支真跑一遍，与子进程用例互补：

* 正常路径：`list-scenarios` / `eval` / `render` / `record`
* argparse 自身的退出：`--help` = SystemExit(0)，缺子命令 = SystemExit(2)，
  非法 `--steps/--every/--fps` = SystemExit(2)
* 可预期的用户错误：未知场景 ⇒ 返回 2（不抛异常），消息里带可用场景

`render` / `record` 用 `pytest.importorskip` 保护 matplotlib / imageio / PIL：
缺可视化依赖时是**有说明的**跳过（不是静默），装了依赖的环境下会真画图
（验收跑的 `/tmp/rel100/venv` 与 `drl2` 都装了 matplotlib，两条都真跑）。
`render` 用 CLI 默认 600px、`record` 用 8 步 / every=4，刻意不加慢参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chassis_sim.cli import main

VIZ_HINT = 'pip install "chassis-sim[viz]"'


def _output(capsys: pytest.CaptureFixture[str]) -> str:
    """把 stdout + stderr 一次性取出来（argparse 的提示可能落在任一侧）。"""
    captured = capsys.readouterr()
    return captured.out + captured.err


# ── 正常路径 ──────────────────────────────────────────────────────
def test_list_scenarios_returns_zero_and_prints_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["list-scenarios"]) == 0

    names = capsys.readouterr().out.split()
    assert "obstacles_dense" in names
    assert "circle" in names


def test_eval_runs_one_episode_and_prints_markdown(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """eval 成功路径：--episodes 1 足矣（5 回合的默认值只是默认值）。

    同时屏蔽 numba：只装 core 的用户没有 planning extra，默认场景一旦是
    `requires: planning` 的（曾经就是 obstacles_dense）这里就会 exit 2 ——
    于是"core 安装跑得动 eval"这条约束在装了 numba 的环境里也一样守着。
    """
    monkeypatch.setitem(sys.modules, "numba", None)

    code = main(["eval", "--episodes", "1"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "| controller |" in captured.out
    assert "PurePursuitController" in captured.out


def test_render_writes_a_real_png(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")  # 缺 viz 依赖时给出说明，而不是静默失败
    out = tmp_path / "x.png"

    assert main(["render", "--out", str(out)]) == 0, capsys.readouterr().err
    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"\x89PNG"
    assert str(out) in capsys.readouterr().out


def test_record_writes_a_real_gif(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    image = pytest.importorskip("PIL.Image")
    out = tmp_path / "x.gif"

    assert main(["record", "--steps", "8", "--every", "4", "--out", str(out)]) == 0, (
        capsys.readouterr().err
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    with image.open(out) as gif:
        assert getattr(gif, "n_frames", 1) >= 2

    printed = capsys.readouterr().out
    assert f"path: {out}" in printed
    assert "frames:" in printed
    assert "total_reward:" in printed


# ── argparse 自己的退出码 ─────────────────────────────────────────
def test_help_exits_zero_and_shows_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    text = _output(capsys)
    assert "usage:" in text
    for name in ("list-scenarios", "record", "eval", "render"):
        assert name in text


def test_missing_subcommand_exits_two_with_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """子解析器是 required=True：不选子命令由 argparse 直接 exit 2。"""
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
    assert "usage:" in _output(capsys)


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--steps", "0"],
        ["--steps", "-3"],
        ["--steps", "not-a-number"],
        ["--every", "0"],
        ["--fps", "0"],
    ],
)
def test_illegal_positive_int_exits_two(
    bad_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """_positive_int 的两条分支（非整数 / 非正数）都由 argparse 转成 exit 2。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["record", *bad_args])

    assert excinfo.value.code == 2
    text = _output(capsys)
    assert "usage:" in text
    assert "argument --" in text


# ── 可预期的用户错误：返回 2，不抛异常 ───────────────────────────
def test_unknown_scenario_returns_two_with_actionable_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["eval", "--scenario", "nope_xyz"]) == 2  # 不抛异常 ⇒ 返回码

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "nope_xyz" in captured.err
    assert "可用场景" in captured.err
    assert "obstacles_dense" in captured.err  # 可用值就在消息里，可直接改命令
    assert VIZ_HINT not in captured.err  # 这是场景错误，不是缺依赖
    assert "Traceback" not in captured.err


def test_unknown_scenario_on_record_returns_two_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "unused.gif"

    assert main(["record", "--scenario", "nope_xyz", "--out", str(out)]) == 2

    captured = capsys.readouterr()
    assert "可用场景" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()  # 失败在动手之前，不留半截产物
