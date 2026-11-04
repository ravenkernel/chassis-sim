"""3.0.0 的元数据自洽守卫：版本三源、extras、classifier 与 README 代码块。

覆盖三类"声明与实际不符"：同一个版本号写在三个文件里（会漂移）、dev extra
不带 CI 用到的闸门工具（本地 .[dev] 与 CI 就不是同一套环境）、README 的
python 块没人执行（文档里的示例可以腐烂而测试仍然全绿）。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import chassis_sim

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
CITATION = REPO / "CITATION.cff"
CHANGELOG = REPO / "CHANGELOG.md"
CHECKER = REPO / "tools" / "check_readme_blocks.py"
VERSION = "3.0.0"
GATE_TOOLS = ("mypy", "pytest-cov", "types-PyYAML")


def _array(text: str, name: str) -> list[str]:
    """3.10 没有 tomllib（也不为此新增 tomli 依赖）时的退路：抓数组字面量。"""
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE
    )
    assert match, f"pyproject.toml 里找不到 {name} = [...]"
    return re.findall(r'"([^"]+)"', match.group(1))


def _project() -> dict:
    """[project] 里本文件要用的四个键；3.10 走正则，3.11+ 走 tomllib。"""
    text = PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:
        version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert version, "pyproject.toml 缺少 project.version"
        return {
            "version": version.group(1),
            "extras": {
                "dev": _array(text, "dev"),
                "planning": _array(text, "planning"),
            },
            "classifiers": re.findall(
                r'^\s*"(Development Status[^"]*)"', text, re.MULTILINE
            ),
        }
    project = tomllib.loads(text)["project"]
    return {
        "version": project["version"],
        "extras": project["optional-dependencies"],
        "classifiers": project["classifiers"],
    }


def _lower_bound(specs: list[str], package: str) -> tuple[int, ...]:
    for spec in specs:
        if spec.startswith(package):
            match = re.search(r">=\s*([0-9.]+)", spec)
            assert match, f"{package} 的要求没有下界：{spec}"
            return tuple(int(part) for part in match.group(1).split("."))
    raise AssertionError(f"{package} 不在 extras 里：{specs}")


def test_version_matches_in_pyproject_package_and_citation():
    """版本三源：pyproject / __version__ / CITATION 必须是同一个 3.0.0。"""
    assert _project()["version"] == VERSION
    assert chassis_sim.__version__ == VERSION
    citation = CITATION.read_text(encoding="utf-8")
    match = re.search(r'^version:\s*"([^"]+)"', citation, re.MULTILINE)
    assert match, "CITATION.cff 的 version 必须是带引号的字符串"
    assert match.group(1) == VERSION


def test_planning_extra_requires_a_numpy2_capable_numba():
    """numba 0.60.0 是第一个带 NumPy 2 二进制支持的版本，0.59 及以下不行。"""
    floor = _lower_bound(_project()["extras"]["planning"], "numba")
    assert floor >= (0, 60), f"planning 的 numba 下界是 {floor}，低于 (0, 60)"


def test_dev_extra_carries_every_gate_tool_at_a_pinned_version():
    """照 README 装 .[dev] 就能跑 mypy 与 pytest --cov —— 三件工具都得在里面。"""
    dev = _project()["extras"]["dev"]
    for tool in GATE_TOOLS:
        specs = [spec for spec in dev if spec.startswith(tool)]
        assert specs, f"dev extra 缺少 {tool}：{dev}"
        assert any("==" in spec for spec in specs), (
            f"{tool} 必须钉死版本，否则 pip install -e .[dev] 与 CI 装的不是同一套：{specs}"
        )


def test_classifier_still_declares_a_stable_line():
    """3.0.0 起承诺 semver，classifier 保持 Production/Stable，不降级为 Beta。"""
    classifiers = _project()["classifiers"]
    assert "Development Status :: 5 - Production/Stable" in classifiers
    betas = [
        item
        for item in classifiers
        if item.startswith("Development Status") and "Beta" in item
    ]
    assert not betas, betas


def test_changelog_opens_with_the_current_release():
    text = CHANGELOG.read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
    assert headings, "CHANGELOG 没有任何 ## 段"
    assert headings[0] == VERSION, f"CHANGELOG 顶部是 {headings[0]!r}，应为 {VERSION!r}"
    body = text.split(f"## {VERSION}", 1)[1].split("\n## ", 1)[0]
    for section in ("### Breaking", "### Added", "### Changed", "### Fixed"):
        assert section in body, f"3.0.0 段缺少 {section}"


def test_changelog_records_the_semver_commitment():
    body = CHANGELOG.read_text(encoding="utf-8").split(f"## {VERSION}", 1)[1]
    assert "API 稳定线" in body
    assert "semver" in body


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _readme(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "README.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_checker_script_exists_and_documents_the_skip_marker():
    assert CHECKER.is_file(), f"缺少 {CHECKER.relative_to(REPO)}"
    proc = _run_checker("--help")
    assert proc.returncode == 0, proc.stderr
    assert "no-exec" in proc.stdout + proc.stderr


def test_checker_passes_a_good_block(tmp_path):
    _readme(tmp_path, "```python\nprint('ok')\n```\n")
    proc = _run_checker("--root", str(tmp_path), "README.md")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed, 0 skipped, 0 failed" in proc.stdout


def test_checker_fails_a_block_that_raises(tmp_path):
    """跑不过的块必须是失败，不是"格式问题"也不是静默跳过。"""
    _readme(tmp_path, '```python\nraise RuntimeError("boom")\n```\n')
    proc = _run_checker("--root", str(tmp_path), "README.md")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "RuntimeError: boom" in proc.stdout


def test_checker_counts_no_exec_as_skipped_not_as_passed(tmp_path):
    body = "```python\n# no-exec: 需要 GPU\nraise SystemExit(3)\n```\n"
    _readme(tmp_path, body)
    proc = _run_checker("--root", str(tmp_path), "README.md")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 passed, 1 skipped, 0 failed" in proc.stdout


def test_checker_rejects_a_readme_without_python_blocks(tmp_path):
    """一个 python 块都没有不是通过：文件被改名/挪走时必须红。"""
    _readme(tmp_path, "```bash\necho hi\n```\n")
    proc = _run_checker("--root", str(tmp_path), "README.md")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "一个 python 块都没有" in proc.stdout


def test_checker_rejects_an_unclosed_fence(tmp_path):
    _readme(tmp_path, "```python\nprint(1)\n")
    proc = _run_checker("--root", str(tmp_path), "README.md")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "没有闭合" in proc.stdout
