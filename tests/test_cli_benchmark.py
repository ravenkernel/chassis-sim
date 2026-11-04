"""benchmark 子命令的进程内覆盖（直接调 chassis_sim.cli.main）。

子进程版的 --help 已经在 tests/test_cli.py 里守着；这里补上真正跑一次
benchmark 的路径：表能打印、Markdown 能落盘、坏输入给出退出码 2 而不是
traceback。步数/回合数刻意压到最小（straight + 1 回合），不拖慢全量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chassis_sim.cli import main


def _output(capsys: pytest.CaptureFixture[str]) -> str:
    """把 stdout + stderr 一次性取出来（argparse 的提示可能落在任一侧）。"""
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_benchmark_help_lists_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["benchmark", "--help"])

    assert excinfo.value.code == 0
    text = _output(capsys)
    for flag in ("--scenarios", "--seeds", "--episodes", "--policies", "--out"):
        assert flag in text


def test_benchmark_prints_table_and_writes_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bench.md"

    code = main(
        [
            "benchmark",
            "--scenarios",
            "straight",
            "--seeds",
            "0",
            "1",
            "--episodes",
            "1",
            "--policies",
            "zero",
            "pp",
            "--max-steps",
            "30",
            "--out",
            str(out),
        ]
    )
    text = _output(capsys)

    assert code == 0, text
    assert "| 策略 |" in text  # 表格进了 stdout
    assert f"markdown: {out}" in text
    assert out.is_file()
    body = out.read_text(encoding="utf-8")
    assert "**False**" in body  # 固定地图 ⇒ maps_independent=False
    assert "协议未满" in body  # 只有 2 个 seed

    # 少于 3 个 seed 是协议未满，但不是错误：退出码仍是 0。
    assert "协议要求 >= 3" in text


def test_benchmark_out_dash_prints_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(
        [
            "benchmark",
            "--seeds",
            "0",
            "1",
            "2",
            "--episodes",
            "1",
            "--policies",
            "zero",
            "--max-steps",
            "20",
            "--out",
            "-",
        ]
    )
    text = _output(capsys)

    assert code == 0, text
    assert "| 策略 |" in text
    assert "markdown:" not in text
    assert not (tmp_path / "benchmark.md").exists()


def test_benchmark_rejects_unknown_policy(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["benchmark", "--policies", "teleop", "--episodes", "1", "--max-steps", "10"]
    )
    text = _output(capsys)

    assert code == 2
    assert "未知策略" in text and "zero, random, pp, dwa" in text


def test_benchmark_rejects_unknown_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "--scenarios", "no_such_scenario", "--episodes", "1"])
    text = _output(capsys)

    assert code == 2
    assert "no_such_scenario" in text


def test_benchmark_rejects_non_integer_seed(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "--seeds", "abc", "--episodes", "1"])
    text = _output(capsys)

    assert code == 2
    assert "--seeds" in text


def test_benchmark_accepts_comma_separated_lists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "comma.md"

    code = main(
        [
            "benchmark",
            "--scenarios",
            "straight",
            "--seeds",
            "0,1",
            "--policies",
            "zero,pp",
            "--episodes",
            "1",
            "--max-steps",
            "20",
            "--out",
            str(out),
        ]
    )
    text = _output(capsys)

    assert code == 0, text
    # 逗号写法归一后与空格写法等价：两个 seed 都进了表，两条策略都在。
    assert "seeds `[0, 1]`" in text
    assert text.count("| `zero` | straight |") >= 2
    assert text.count("| `pp` | straight |") >= 2


def test_benchmark_default_scenario_needs_no_planning_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """只装 core 的用户（缺 numba）也能跑默认 benchmark。"""
    monkeypatch.setitem(sys.modules, "numba", None)

    code = main(
        [
            "benchmark",
            "--episodes",
            "1",
            "--policies",
            "random",
            "--max-steps",
            "20",
            "--out",
            str(tmp_path / "core.md"),
        ]
    )

    assert code == 0, _output(capsys)
